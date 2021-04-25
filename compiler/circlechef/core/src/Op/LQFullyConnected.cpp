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

#include "LQFullyConnected.h"

#include "Convert.h"

flatbuffers::Offset<void> LQFullyConnectedChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_lq_fully_connected_options());

  circle::LQFullyConnectedOptionsBuilder lq_fully_connected_options_builder{fbb};
  lq_fully_connected_options_builder.add_weights_hidden_size(
    operation.lq_fully_connected_options().weights_hidden_size());
  lq_fully_connected_options_builder.add_fused_activation_function(
    as_circle_activation(operation.lq_fully_connected_options().activation()));

  return lq_fully_connected_options_builder.Finish().Union();
}

std::unique_ptr<OpChef>
LQFullyConnectedChefFactory::create(const circlechef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new LQFullyConnectedChef{operation}};
}
