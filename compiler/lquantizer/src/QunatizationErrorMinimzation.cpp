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

#include "QunatizationErrorMinimzation.h"

#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace lquantizer
{

QEM::QEM(float const *fp_source_data, float const *fp_target_data, float *scales_data,
         uint32_t output_size, uint32_t hidden_size, uint32_t bits_per_value)
  : _fp_source_data(fp_source_data), _fp_target_data(fp_target_data), _scales_data(scales_data),
    _output_size(output_size), _hidden_size(hidden_size), _encode_bits(bits_per_value)
{
  // scales should be sorted
  for (uint32_t o = 0; o < _output_size; ++o)
  {
    float *scales = &scales_data[o * _encode_bits];
    std::sort(scales, scales + bits_per_value);
  }
}

void QEM::fit(uint32_t epochs)
{
  for (uint32_t o = 0; o < _output_size; ++o)
  {
    for (uint32_t e = 0; e < epochs; ++e)
    {
      // use source data for binarization and target data for error minimization
      float const *source_data = &_fp_source_data[o * _hidden_size];
      float const *target_data = &_fp_target_data[o * _hidden_size];
      float *scales = &_scales_data[o * _encode_bits];

      // step 1: encode fp data using scales
      LQBinarizer binarizer(_hidden_size, scales, _encode_bits);
      binarizer.quantize_and_pack(source_data);

      // step 2: update scales using least squares method
      binarizer.gradient_descent_scales(target_data);

      // step 3: scales must be sorted
      std::sort(scales, scales + _encode_bits);
    }
  }
}

void QEM::fill_binary(int32_t *bin_data)
{
  // size of fp and bin data for single output
  auto const hidden_size = static_cast<int32_t>(_hidden_size);
  const uint32_t bin_data_size = _encode_bits * ceil_div(hidden_size, 32);

  for (uint32_t o = 0; o < _output_size; ++o)
  {
    float const *source_data = &_fp_source_data[o * hidden_size];
    int32_t *target_bin_data = &bin_data[o * bin_data_size];
    float *scales = &_scales_data[o * _encode_bits];

    LQBinarizer binarizer(_hidden_size, scales, _encode_bits);
    binarizer.quantize_and_pack(source_data);

    std::memcpy(target_bin_data, binarizer.data(), bin_data_size * sizeof(int32_t));
  }
}

} // namespace lquantizer
