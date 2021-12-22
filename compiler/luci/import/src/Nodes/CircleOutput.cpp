/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleOutput.h"

#include <luci/IR/Nodes/CircleOutput.h>

#include <oops/UserExn.h>

#include <cassert>
#include <string>
#include <vector>

namespace
{

} // namespace

namespace luci
{

CircleNode *CircleOutputTensorBuilder::build(TensorIndex tensor_index,
                                             GraphBuilderContext *context) const
{
  assert(tensor_index >= 0);

  const auto graph = context->graph();
  const auto reader = context->reader();
  const auto nodefinder = context->nodefinder();

  const auto tensor = reader->tensors()[tensor_index];
  assert(tensor != nullptr);

  auto output_node = graph->nodes()->create<luci::CircleOutput>();
  assert(output_node != nullptr);

  auto output_from = nodefinder->node(tensor_index);
  if (output_from != nullptr)
    output_node->from(output_from);
  else
  {
    // NOTE loco::Graph requires all input node(s) to a node should exist.
    //      Here, CircleOutput needs an input node.
    //      We add a dummy node to make it happy.
    auto output_dummy = graph->nodes()->create<luci::CircleOutputDummy>();
    assert(output_dummy != nullptr);
    output_node->from(output_dummy);

    luci::copy_tensor_attributes(tensor, output_dummy);
    if (tensor->shape() == nullptr)
      output_dummy->shape_status(luci::ShapeStatus::NOSHAPE);
    else
      output_dummy->shape_status(luci::ShapeStatus::VALID);
  }

  // set the graph output name and node object
  auto graph_output = graph->outputs()->create();
  std::string tname = luci::tensor_name(tensor);
  assert(tname.length() > 0);
  graph_output->name(tname);

  luci::copy_tensor_attributes(tensor, output_node);

  // Set GraphInputOutputIndex for graph
  output_node->index(graph_output->index());

  const auto tensor_shape_signature = luci::wrap(tensor->shape_signature());
  const auto tensor_shape = luci::wrap(tensor->shape());
  assert(tensor_shape_signature.size() == 0 ||
         tensor_shape_signature.size() == tensor_shape.size());

  // Shape of Output
  auto output_shape = std::make_unique<loco::TensorShape>();
  const auto &output_dims = tensor_shape; // in NHWC
  output_shape->rank(output_dims.size());
  for (uint32_t r = 0; r < output_dims.size(); ++r)
  {
    if (tensor_shape_signature.size() > 0 && tensor_shape_signature.at(r) == -1)
      output_shape->dim(r).unset();
    else
      output_shape->dim(r).set(output_dims[r]);
  }
  graph_output->shape(std::move(output_shape));

  // Data type
  auto dtype = luci::luci_datatype(tensor->type());
  graph_output->dtype(dtype);

  return output_node;
}

} // namespace luci
