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

#include "luci/Import/Nodes/CircleSquare.h"

#include <luci/IR/Nodes/CircleSquare.h>

#include <loco.h>

namespace luci
{

bool CircleSquareGraphBuilder::validate(const ValidateArgs &args) const
{
  if (!GraphBuilder::validate(args, 1))
    return false;

  const auto &inputs = args.op.inputs;
  // Must be one of the following types
  // bfloat16, half (float16), float32, float64, complex64, complex128
  // Currently, circle supports float16, float32, complex64
  const auto tensors = args.reader.tensors();
  const auto tensor = tensors.at(inputs.at(0));
  assert(tensor != nullptr);
  switch (tensor->type())
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
    case circle::TensorType_FLOAT16:
    case circle::TensorType_FLOAT32:
    case circle::TensorType_FLOAT64:
    case circle::TensorType_COMPLEX64:
      break;
    default:
      return false;
  }

  return true;
}

CircleNode *CircleSquareGraphBuilder::build_node(const circle::OperatorT &,
                                                 const std::vector<CircleNode *> &inputs,
                                                 loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleSquare>();
  node->x(inputs.at(0));

  return node;
}

} // namespace luci
