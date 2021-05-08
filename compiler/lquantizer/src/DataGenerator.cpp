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

#include "DataGenerator.h"
#include <random>

namespace lquantizer
{

void set_float_random_data(float *data, uint32_t size)
{
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  for (uint32_t i = 0; i < size; ++i)
  {
    data[i] = distribution(generator);
  }
}

bool DataGenerator::read_record(std::vector<float> &input_data)
{
  if (record_idx >= records_num)
  {
    return false;
  }

  if (random_data)
  {
    set_float_random_data(const_cast<float *>(input_data.data()), input_data.size());
  }
  else
  {
    if (not hdf5_importer.isRawData())
    {
      DataType dtype;
      Shape shape = {};
      hdf5_importer.readTensor(record_idx, input_idx, &dtype, &shape, input_data.data());

      // Check the type and the shape of the input data is valid
      assert(dtype == DataType::FLOAT32);
      assert(shape.num_elements() == input_data.size());
    }
    else
    {
      // Skip type/shape check for raw data
      hdf5_importer.readTensor(record_idx, input_idx, input_data.data());
    }
  }

  // update input and record idx
  ++input_idx;
  if (input_idx >= inputs_num)
  {
    input_idx = 0;
    ++record_idx;
  }

  return true;
}

} // namespace lquantizer
