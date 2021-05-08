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

#include "LQBinarizer.h"

#include <stdexcept>
#include <limits>
#include <cstring>
#include <algorithm>

namespace lquantizer
{

int32_t ceil_div(int32_t num, int32_t denom)
{
  assert(denom > 0);
  return (num + denom - 1) / denom;
}

inline int32_t calcOffset(const Shape &shape, int32_t d0, int32_t d1)
{
  return d0 * shape.dim(1) + d1;
}

LQBinarizer::LQBinarizer(uint32_t data_vec_size, float *data_scales, uint32_t bits_per_value)
{
  assert(data_vec_size);
  data_float_size = static_cast<int32_t>(data_vec_size);
  data_bin_size = ceil_div(data_float_size, 32);
  encode_bits = static_cast<int32_t>(bits_per_value);

  scales = data_scales;

  // count of different levels is 2^(encode_bits)
  levels_count = 1 << encode_bits;

  // allocate buffers
  data_binary = std::make_unique<int32_t[]>(data_bin_size * encode_bits);
  quantization_levels = std::make_unique<Level[]>(levels_count);
  quantization_thresholds = std::make_unique<float[]>(levels_count);

  // init quantization levels
  {
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

void LQBinarizer::dequantize(const int8_t *bins, std::vector<float> &data, uint32_t size)
{
  // calculate data from binary
  for (int32_t i = 0; i < size; ++i)
  {
    data[i] = 0;
    for (int32_t b = 0; b < encode_bits; ++b)
    {
      data[i] += static_cast<float>(bins[i * encode_bits + b]) * scales[b];
    }
  }
}

// set bits to byte array (byte per bit)
void LQBinarizer::unpack_binary(std::vector<int8_t> &buffer)
{
  assert(buffer.size() == encode_bits * data_float_size);
  for (int32_t i = 0; i < data_float_size; ++i)
  {
    int8_t *bits = &buffer[i * encode_bits];

    int32_t idx = i >> 5;    // divide to 32
    int32_t offset = i & 31; // mod of 32

    for (int32_t b = 0; b < encode_bits; ++b)
    {
      int32_t bit_value = (data_binary[b * data_bin_size + idx] >> offset) & 1;
      bits[b] = (bit_value == 1) ? 1 : -1;
    }
  }
}

// iterative gradient descent to search optimal scales `v: argmin||Bxv - x||^2`, `x` - data_vector
void LQBinarizer::gradient_descent_scales(const float *data_vector)
{
  // transposed binary vector with single char for bit
  std::vector<int8_t> bin_trans(encode_bits * data_float_size);
  unpack_binary(bin_trans);

  // allocate buffer for dequantized values and gradient step
  std::vector<float> bin2fp(batch_size);

  for (int32_t step = 0; step < descent_steps; ++step)
  {
    // correct scales with dv
    for (uint32_t n = 0; n < data_float_size; n += batch_size)
    {
      uint32_t size = std::min(data_float_size - n, batch_size);
      const float *target_vals = &data_vector[n];

      dequantize(&bin_trans[encode_bits * n], bin2fp, size);

      for (uint32_t i = 0; i < size; ++i)
      {
        // define gradient and correct overall step
        int8_t *x = &bin_trans[(n + i) * encode_bits];
        float err = bin2fp[i] - target_vals[i];

        for (int32_t b = 0; b < encode_bits; ++b)
        {
          scales[b] -= descent_lr * (static_cast<float>(x[b]) * err +
                                     l2_reg * scales[b]); // add antigradient with L2
        }
      }
    }
  }
}

} // namespace lquantizer
