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

#ifndef LQUANTIZER_QEM_H
#define LQUANTIZER_QEM_H

#include "LQBinarizer.h"

namespace lquantizer
{

using Shape = luci_interpreter::Shape;

class QEM
{
public:
  QEM(float const *fp_source_data, float const *fp_target_data, float *scales_data,
      uint32_t output_size, uint32_t hidden_size, uint32_t bits_per_value);

  void fit(uint32_t epochs);
  void fill_binary(int32_t *bin_data);

private:
  float const *_fp_source_data;
  float const *_fp_target_data;
  float *_scales_data;

  const uint32_t _output_size;
  const uint32_t _hidden_size;
  const uint32_t _encode_bits;
};

} // namespace lquantizer

#endif // LQUANTIZER_QEM_H
