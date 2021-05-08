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

#ifndef __LQUANTIZER_QUANTIZATION_OBSERVERS_H__
#define __LQUANTIZER_QUANTIZATION_OBSERVERS_H__

#include <luci_interpreter/Interpreter.h>
#include <luci_interpreter/core/Tensor.h>

#include <vector>
#include <unordered_map>

namespace lquantizer
{

using Node2NodeMatcher = std::unordered_map<const luci::CircleNode *, const luci::CircleNode *>;
using Node2InputDataMatcher = std::unordered_map<const luci::CircleNode *, std::vector<float>>;

/*
 * @brief Observer which saves input data for inputs matched to nodes by input2node matcher.
 **/
class InputSavingObserver : public luci_interpreter::ExecutionObserver
{
public:
  explicit InputSavingObserver(Node2NodeMatcher &);

  void postTensorWrite(const luci::CircleNode *, const luci_interpreter::Tensor *) override;

  const std::vector<float> &input_data_for(const luci::CircleNode *);

private:
  void allocate_data_buffers();

private:
  Node2NodeMatcher &_input2node;
  Node2InputDataMatcher _node2idata;
};

} // namespace lquantizer

#endif // __LQUANTIZER_QUANTIZATION_OBSERVERS_H__
