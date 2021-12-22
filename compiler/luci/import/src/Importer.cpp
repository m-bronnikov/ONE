/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci/Importer.h"
#include "CircleImportMetadata.h"
#include "PostImport.h"

#include "luci/Import/GraphBuilder.h"
#include "luci/Import/GraphBuilderContext.h"
#include "luci/Import/GraphBuilderRegistry.h"
#include "luci/Import/CircleReader.h"
#include "luci/Import/Nodes/CircleConst.h"
#include "luci/Import/Nodes/CircleInput.h"
#include "luci/Import/Nodes/CircleOutput.h"

#include <luci/IR/Module.h>
#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeID.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Plan/CircleNodeExecutionPlan.h>
#include <luci/Log.h>
#include <luci/LogHelper.h>

#include <oops/InternalExn.h>
#include <oops/UserExn.h>

#include <memory>

namespace
{

// TODO move this helper to Utils
std::ostream &operator<<(std::ostream &os, const luci::VectorWrapper<int32_t> &vect)
{
  uint32_t seq = 0;
  for (const auto &v : vect)
  {
    if (seq)
      os << ", ";
    os << v;
    seq++;
  }
  return os;
}

void convert_graph(const luci::GraphBuilderSource &source, luci::CircleReader &reader,
                   loco::Graph *graph)
{
  LOGGER(l);

  auto nodefinder = std::make_unique<luci::IndexNodeFinder>();
  auto tensoroutputs = std::make_unique<luci::IndexTensorOutputs>();

  luci::GraphBuilderContext gb_context(graph, &reader, nodefinder.get(), tensoroutputs.get());

  const auto operators = reader.operators();
  const auto tensors = reader.tensors();
  assert(!tensors.null());
  auto circle_metadata = std::make_unique<luci::CircleImportMetadata>(reader);

  // build a cache to identify if a tensor is output of an operator
  // if this is set, we should not create a CircleConst for this tensor
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const auto op = operators[i];
    assert(op != nullptr);
    const auto outputs = luci::wrap(op->outputs());

    for (uint32_t j = 0; j < outputs.size(); ++j)
    {
      auto tidx = outputs[j];
      tensoroutputs->enroll(tidx);
    }
  }

  // graph inputs; there are no input nodes in TFlite but just Tensors
  // creating virtual input nodes will make possible to connect nodes that uses them
  // all attributes of tensor should be copied to CircleInput node
  luci::CircleInputTensorBuilder input_builder;
  for (const auto input : reader.inputs())
  {
    auto *input_node = input_builder.build(input, &gb_context);

    INFO(l) << "[luci] NodeFinder INPUT(" << input << ") = " << input_node << std::endl;
    nodefinder->enroll(input, input_node);

    // input_node is also an output to a tensor
    tensoroutputs->enroll(input);
  }

  // Create CircleNodes for constant tensors.
  // NOTE Origin is intentionally not provided for constants.
  luci::CircleConstTensorBuilder const_builder;
  for (int32_t i = 0; i < tensors.size(); ++i)
  {
    const auto tensor = tensors[i];
    assert(tensor != nullptr);

    auto *const_node = const_builder.build(i, &gb_context);
    if (const_node != nullptr)
    {
      nodefinder->enroll(i, const_node);
      INFO(l) << "[luci] NodeFinder const_node(" << i << ") -> " << const_node << " "
              << luci::wrap(tensor->shape()) << std::endl;
    }
  }

  // Import the operators.
  // Note that operators in model are stored in execution order. This means that when importing
  // an operator, its input operators have already been imported. We exploit this fact to set up
  // node's inputs right after creating the node.
  auto origin_table = circle_metadata->origin_table();
  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const auto op = operators[i];
    assert(op != nullptr);
    circle::BuiltinOperator builtincode = reader.builtin_code(op);

    if (const auto *builder = source.lookup(builtincode))
    {
      // create temporary unpack API obj
      circle::OperatorT oper_t;
      op->UnPackTo(&oper_t);

      luci::GraphBuilder::ValidateArgs args(oper_t, reader);
      if (!builder->validate(args))
      {
        throw oops::UserExn("Invalid operator", reader.opcode_name(op));
      }

      auto built_op = builder->build(oper_t, &gb_context);
      set_node_id(built_op, i);
      if (origin_table.find(i) != origin_table.end())
        add_origin(built_op, origin_table.at(i));
      else
        add_origin(built_op, luci::single_origin(i, built_op->name()));
    }
    else
    {
      throw oops::UserExn("Not supported", reader.opcode_name(op));
    }
  }

  // graph outputs
  luci::CircleOutputTensorBuilder output_builder;
  for (auto output : reader.outputs())
  {
    auto output_node = output_builder.build(output, &gb_context);

    INFO(l) << "[luci] NodeFinder OUTPUT(" << output << ") = " << output_node << std::endl;
  }
}

class ValidateCollector final : public loco::ErrorListener
{
public:
  void notify(const loco::ErrorDetail<loco::ErrorCategory::MissingArgument> &d) override
  {
    LOGGER(l);
    INFO(l) << "[luci] GraphValidate error " << d.node() << "(" << d.index() << ")" << std::endl;
  }
};

} // namespace

namespace luci
{

Importer::Importer()
{
  // DO NOTHING
}

std::unique_ptr<loco::Graph> Importer::import(const circle::Model *model) const
{
  auto graph = loco::make_graph();

  const GraphBuilderSource *source_ptr = &GraphBuilderRegistry::get();

  if (_source != nullptr)
  {
    // Use user-defined GraphBuilderSource
    source_ptr = _source;
  }

  CircleReader reader;
  if (!reader.parse(model))
    return nullptr;

  if (reader.num_subgraph() != 1)
  {
    INTERNAL_EXN("Use 'importModule()' for multiple subgraphs");
  }
  if (!reader.select_subgraph(0))
    return nullptr;

  // Convert circle::Model to loco::Graph
  convert_graph(*source_ptr, reader, graph.get());

  LOGGER(l);
  VERBOSE(l, 3) << "--- graph dump begin -------------------------------------------";
  VERBOSE(l, 3) << "Name: " << graph->name();
  VERBOSE(l, 3) << fmt(graph.get());
  VERBOSE(l, 3) << "--- graph dump end ---------------------------------------------";

  assert(loco::valid(graph.get(), std::make_unique<ValidateCollector>()));

  return graph;
}

std::unique_ptr<Module> Importer::importModule(const circle::Model *model) const
{
  auto module = make_module();

  const GraphBuilderSource *source_ptr = &GraphBuilderRegistry::get();

  if (_source != nullptr)
  {
    // Use user-defined GraphBuilderSource
    source_ptr = _source;
  }

  CircleReader reader;
  if (!reader.parse(model))
    return nullptr;

  for (uint32_t g = 0; g < reader.num_subgraph(); ++g)
  {
    auto graph = loco::make_graph();

    if (!reader.select_subgraph(g))
      return nullptr;

    graph->name(reader.name());

    // Convert circle::Model to loco::Graph
    convert_graph(*source_ptr, reader, graph.get());

    LOGGER(l);
    VERBOSE(l, 3) << "--- graph dump begin -------------------------------------------";
    VERBOSE(l, 3) << "Name: " << graph->name();
    VERBOSE(l, 3) << fmt(graph.get());
    VERBOSE(l, 3) << "--- graph dump end ---------------------------------------------";

    assert(loco::valid(graph.get(), std::make_unique<ValidateCollector>()));

    module->add(std::move(graph));
  }

  post_import_graph(module.get(), reader);

  // Initialize 'source_table'
  auto circle_metadata = std::make_unique<luci::CircleImportMetadata>(reader);
  if (circle_metadata->source_table().size() > 0)
  {
    // If there is 'source_table' metadata in circle model, copy the table.
    module->source_table(circle_metadata->source_table());
  }
  else
  {
    // If there is no 'source_table' metadata in circle model,
    // create new table with circle nodes.
    std::map<uint32_t, std::string> table;

    // NOTE Only first subgraph is considered
    for (auto node : loco::all_nodes(module->graph(0)))
    {
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);

      // Virtual nodes may not have id
      if (!has_node_id(circle_node))
        continue;

      assert(table.find(get_node_id(circle_node)) == table.end());
      table.insert({get_node_id(circle_node), circle_node->name()});
    }

    module->source_table(table);
  }

  // Add execution_plan annotations
  if (circle_metadata->execution_plan_table().size() > 0)
  {
    auto execution_plan_table = circle_metadata->execution_plan_table();
    auto node_position = 0;
    for (auto node : loco::postorder_traversal(loco::output_nodes(module->graph())))
    {
      if (auto circle_node = dynamic_cast<luci::CircleNode *>(node))
      {
        if (execution_plan_table.count(node_position) == 0)
          continue;

        auto node_plan = execution_plan_table[node_position];
        assert(node_plan.size() > 0);

        luci::add_execution_plan(
          circle_node,
          luci::CircleNodeExecutionPlan(
            node_plan[0], std::vector<uint32_t>(node_plan.begin() + 1, node_plan.end())));
      }
      node_position++;
    }
  }

  return module;
}

} // namespace luci
