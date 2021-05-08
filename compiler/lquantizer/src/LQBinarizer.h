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

#ifndef LQUANTIZER_LQBINARIZER_H
#define LQUANTIZER_LQBINARIZER_H

#include <luci_interpreter/core/Tensor.h>

namespace lquantizer
{

using Shape = luci_interpreter::Shape;

int32_t ceil_div(int32_t num, int32_t denom);

class LQBinarizer
{
  using Level = std::pair<float, int32_t>;

public:
  LQBinarizer(uint32_t data_vec_size, float *data_scales, uint32_t bits_per_value);

  const int32_t *data() { return data_binary.get(); }
  void quantize_and_pack(const float *data_vector);

public:
  void gradient_descent_scales(const float *data_vector);

private:
  int32_t bin_search_encoding(float value);

private:
  void unpack_binary(std::vector<int8_t> &buffer);
  void dequantize(const int8_t *bins, std::vector<float> &data, uint32_t size);

private:
  int32_t data_float_size;
  int32_t data_bin_size;
  int32_t encode_bits;
  std::unique_ptr<int32_t[]> data_binary;

  int32_t levels_count;
  std::unique_ptr<Level[]> quantization_levels;
  std::unique_ptr<float[]> quantization_thresholds;

  float *scales;

  const float l2_reg = 0.02f;
  const float descent_lr = 0.001f;
  const uint32_t descent_steps = 8;
  const uint32_t batch_size = 64;
};

} // namespace lquantizer

#endif // LQUANTIZER_LQBINARIZER_H
