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

#include "luci/Import/Nodes/CircleInput.h"

#include <luci/IR/Nodes/CircleInput.h>

#include <oops/UserExn.h>

#include <cassert>
#include <string>
#include <vector>

namespace
{

} // namespace

namespace luci
{

CircleNode *CircleInputTensorBuilder::build(TensorIndex tensor_index,
                                            GraphBuilderContext *context) const
{
  assert(tensor_index >= 0);

  const auto graph = context->graph();
  const auto reader = context->reader();

  const auto tensor = reader->tensors()[tensor_index];
  assert(tensor != nullptr);

  auto input_node = graph->nodes()->create<luci::CircleInput>();
  assert(input_node != nullptr);

  luci::copy_tensor_attributes(tensor, input_node);
  if (tensor->shape() == nullptr)
    input_node->shape_status(luci::ShapeStatus::NOSHAPE);
  else
    input_node->shape_status(luci::ShapeStatus::VALID);

  // Name
  auto graph_input = graph->inputs()->create();
  graph_input->name(input_node->name());

  // Set GraphInputOutputIndex for graph
  input_node->index(graph_input->index());

  // Data type
  graph_input->dtype(input_node->dtype());

  const auto tensor_shape_signature = luci::wrap(tensor->shape_signature());
  const auto tensor_shape = luci::wrap(tensor->shape());
  assert(tensor_shape_signature.size() == 0 ||
         tensor_shape_signature.size() == tensor_shape.size());

  // Shape of GraphInput
  auto input_shape = std::make_unique<loco::TensorShape>();
  const auto &input_dims = tensor_shape; // in NHWC
  input_shape->rank(input_dims.size());
  for (uint32_t r = 0; r < input_dims.size(); ++r)
  {
    if (tensor_shape_signature.size() > 0 && tensor_shape_signature.at(r) == -1)
      input_shape->dim(r).unset();
    else
      input_shape->dim(r).set(input_dims[r]);
  }
  graph_input->shape(std::move(input_shape));

  return input_node;
}

} // namespace luci
