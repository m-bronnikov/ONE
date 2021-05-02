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

#ifndef LUCI_INTERPRETER_KERNELS_LQBINARIZER_H
#define LUCI_INTERPRETER_KERNELS_LQBINARIZER_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class LQBinarizer
{
  using Level = std::pair<float, int32_t>;

public:
  LQBinarizer(int32_t data_vec_size, const Tensor *data_scales);

  const int32_t *data() { return data_binary.get(); }

  void quantize_and_pack(const float *data_vector);

private:
  int32_t bin_search_encoding(float value);

private:
  int32_t data_float_size;
  int32_t data_bin_size;
  int32_t encode_bits;
  std::unique_ptr<int32_t[]> data_binary;

  int32_t levels_count;
  std::unique_ptr<Level[]> quantization_levels;
  std::unique_ptr<float[]> quantization_thresholds;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_LQBINARIZER_H
