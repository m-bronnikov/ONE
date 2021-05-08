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

#include "QuantizationObservers.h"

#include <luci/IR/CircleOpcode.h>

#include <cstring>

using DataType = luci_interpreter::DataType;

namespace lquantizer
{

InputSavingObserver::InputSavingObserver(Node2NodeMatcher &input2node) : _input2node(input2node)
{
  allocate_data_buffers();
}

const std::vector<float> &InputSavingObserver::input_data_for(const luci::CircleNode *node)
{
  assert(_node2idata.count(node));

  return _node2idata[node];
}

void InputSavingObserver::allocate_data_buffers()
{
  // allocate input vectors for each node
  for (auto it : _input2node)
  {
    // create storage
    _node2idata[it.second] = {};
  }
}

// postTensorWrite is only called for a node producing a tensor
void InputSavingObserver::postTensorWrite(const luci::CircleNode *node,
                                          const luci_interpreter::Tensor *tensor)
{
  // save input only if this node exist as a key in _input2node
  auto iter = _input2node.find(node);
  if (iter == _input2node.end())
  {
    return;
  }

  // check type of tensor
  assert(tensor->element_type() == DataType::FLOAT32);

  // number of input elements to save
  auto input_elements = tensor->shape().num_elements();

  // find buffer for writing
  auto &storage = _node2idata[iter->second];

  // reallocate buffer
  auto const start_of_writing = storage.size();
  storage.resize(start_of_writing + input_elements);

  // save data of tensor
  auto const *tensor_data = tensor->data<float>();
  auto *save_data = &storage[start_of_writing];

  std::memcpy(save_data, tensor_data, input_elements * sizeof(float));
}

} // namespace lquantizer
