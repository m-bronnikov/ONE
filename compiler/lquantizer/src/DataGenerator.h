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

#ifndef __LQUANTIZER_DATA_GENERATOR_H__
#define __LQUANTIZER_DATA_GENERATOR_H__

#include "HDF5Importer.h"
#include <limits>

namespace lquantizer
{

void set_float_random_data(float *data, uint32_t size);

class DataGenerator
{
public:
  explicit DataGenerator(bool is_random) : random_data(is_random)
  {
    reset();

    records_num = std::numeric_limits<int32_t>::max(); // infinity
  }

  // needed only if random_data is false
  void open_data_file(const std::string &path)
  {
    if (random_data)
    {
      throw std::runtime_error("Can't open file when random data option is chosen.");
    }

    hdf5_importer.open_file(path);
    hdf5_importer.importGroup();

    int32_t file_records_num = hdf5_importer.numRecords();
    if (file_records_num <= 0)
    {
      throw std::runtime_error("Can't open file without records.");
    }

    inputs_num = hdf5_importer.numInputs(0);
    records_num = file_records_num > records_num ? records_num : file_records_num; // min
  }

  // set number of inputs if random_data chosen
  void inputs_count(int32_t count)
  {
    if (not random_data)
    {
      throw std::runtime_error("Can't set count of inputs when random data option is chosen.");
    }

    inputs_num = count;
  }

  // return generator state to start position
  void reset()
  {
    input_idx = 0;
    record_idx = 0;
  }

  // set number which will generated if possible
  void set_required_records_num(int32_t num)
  {
    records_num = records_num > num ? num : records_num; // min
  }

  // return number of inputs in single record
  int32_t inputs_count() const { return inputs_num; }

  int32_t current_input_idx() const { return input_idx; }

  int32_t current_record_idx() const { return record_idx; }

  bool is_empty() const { return record_idx >= records_num; }

  // Returns false if required number of records is generated.
  bool read_record(std::vector<float> &input_data);

private:
  bool random_data = false;
  HDF5Importer hdf5_importer;

  int32_t input_idx = 0;
  int32_t inputs_num = 0;

  int32_t record_idx = 0;
  int32_t records_num = 0;
};

} // namespace lquantizer

#endif // __LQUANTIZER_DATA_GENERATOR_H__
