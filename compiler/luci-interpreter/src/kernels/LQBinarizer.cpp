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

#include "kernels/LQBinarizer.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

LQBinarizer::LQBinarizer(int32_t data_vec_size, const Tensor *data_scales)
{
  data_float_size = data_vec_size;
  data_bin_size = ceil_div(data_float_size, 32);
  encode_bits = data_scales->shape().dim(0);

  // count of different levels is 2^(encode_bits)
  levels_count = 1 << encode_bits;

  // allocate buffers
  data_binary = std::make_unique<int32_t[]>(data_bin_size * encode_bits);
  quantization_levels = std::make_unique<Level[]>(levels_count);
  quantization_thresholds = std::make_unique<float[]>(levels_count);

  // init quantization levels
  {
    const float *scales = data_scales->data<float>();
    for (int32_t i = 0; i < levels_count; ++i)
    {
      // TODO investigate faster variants to perform this
      float value = 0.0f;
      for (int32_t b = 0; b < encode_bits; ++b)
      {
        value += ((i >> b) & 1) ? scales[b] : -scales[b];
      }

      // store level with encoding
      quantization_levels[i] = {value, i};
    }
  }

  // sort levels (first and last value always has correct places
  auto sorter = [](Level lhs, Level rhs) { return lhs.first < rhs.first; };
  std::sort(quantization_levels.get() + 1, quantization_levels.get() + levels_count - 1, sorter);

  // compute quantization thresholds via sorted levels
  {
    quantization_thresholds[0] = std::numeric_limits<float>::lowest(); // -INF

    for (uint32_t i = 1; i < levels_count; ++i)
    {
      float left = quantization_levels[i - 1].first;
      float right = quantization_levels[i].first;
      quantization_thresholds[i] = (left + right) / 2.0f;
    }
  }
}

// this bin search returns t[l] for values in (t[l], t[l+1]]
int32_t LQBinarizer::bin_search_encoding(float value)
{
  int32_t left = 0, right = levels_count - 1;

  while (left < right)
  {
    uint32_t middle = (left + right + 1) >> 1;

    if (quantization_thresholds[middle] < value)
    {
      left = middle;
    }
    else
    {
      right = middle - 1;
    }
  }

  return quantization_levels[left].second;
}

void LQBinarizer::quantize_and_pack(const float *data_vector)
{
  Shape shape = {encode_bits, data_bin_size};
  memset(data_binary.get(), 0, shape.num_elements() * sizeof(int32_t));

  for (int32_t i = 0; i < data_float_size; ++i)
  {
    // encode value of input
    int32_t encoding = bin_search_encoding(data_vector[i]);

    // pack value to binary buffer
    {
      int32_t idx = i >> 5;    // divide to 32
      int32_t offset = i & 31; // mod of 32
      for (int32_t b = 0; b < encode_bits; ++b)
      {
        data_binary[calcOffset(shape, b, idx)] |= ((encoding >> b) & 1) << offset;
      }
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
