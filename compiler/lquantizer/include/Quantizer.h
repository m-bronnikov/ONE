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

#ifndef __QUANTIZER_H__
#define __QUANTIZER_H__

#include <luci/IR/Module.h>
#include <luci_interpreter/Interpreter.h>

#include "QuantizationObservers.h"

#include <memory>

namespace lquantizer
{

class Quantizer
{
public:
  Quantizer() = default;

  explicit Quantizer(uint32_t bits_per_input, uint32_t bit_ber_weight)
    : _input_encoding_bits(bits_per_input), _weights_encoding_bits(bit_ber_weight)
  {
    // Do nothing
  }

  explicit Quantizer(uint32_t bits_per_param) : Quantizer(bits_per_param, bits_per_param)
  {
    // Do nothing
  }

  ~Quantizer() = default;

  void initialize(const std::string &input_model_path);

  void train();

  void path_to_input_data(const std::string &);
  void save(const std::string &output_model_path);

private:
  void train_weights();
  void train_input();
  void uptrain_input();

private:
  luci::CircleNode *make_and_process_lq_node(loco::Graph *, luci::CircleNode *,
                                             const luci::CircleNode *);
  void make_binded_lqgraph();

private:
  std::unique_ptr<luci::Module> _fp_module;
  std::unique_ptr<luci::Module> _lq_module;

  Node2NodeMatcher _input_node2fp_node; // match FP input nodes to keys from _fp2lq_nodes
  Node2NodeMatcher _input_node2lq_node; // match LQ input nodes to values from _fp2lq_nodes

  Node2NodeMatcher _fp2lq_nodes; // matches source nodes to LQ nodes

  bool _use_random = true;
  std::string _path_to_data;

  uint32_t _input_encoding_bits = 2;
  uint32_t _weights_encoding_bits = 2;

  // TODO define ideal numbers of iterations for learning
  const uint32_t _train_batches = 128;
  const uint32_t _qem_iterations = 5;
  const uint32_t _train_epochs = 5;
};

} // namespace lquantizer

#endif // __QUANTIZER_H__
