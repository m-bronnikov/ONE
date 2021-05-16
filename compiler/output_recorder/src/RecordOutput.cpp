/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "RecordOutput.h"
#include "HDF5Importer.h"

#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/IR/CircleQuantParam.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <chrono>

using Shape = luci_interpreter::Shape;
using DataType = luci_interpreter::DataType;

namespace
{

/**
 * @brief  getTensorSize will return size in bytes
 */
template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

/**
 * @brief  verifyTypeShape checks the type and the shape of CircleInput
 *         This throws an exception if type or shape does not match
 */
void verifyTypeShape(const luci::CircleInput *input_node, const DataType &dtype, const Shape &shape)
{
  // Type check
  if (dtype != input_node->dtype())
    throw std::runtime_error("Wrong input type.");

  if (shape.num_dims() != input_node->rank())
    throw std::runtime_error("Input rank mismatch.");

  for (uint32_t i = 0; i < shape.num_dims(); i++)
  {
    if (shape.dim(i) != input_node->dim(i).value())
      throw std::runtime_error("Input shape mismatch.");
  }
}

} // namespace

namespace record_output
{

RecordOutput::RecordOutput(const std::string &input_model_path, const std::string &input_data_path,
                           const std::string &output_dir_path)
  : _input_data_path(input_data_path), _output_dir_path(output_dir_path)
{
  // Load model from the file
  std::ifstream fs(input_model_path, std::ifstream::binary);
  if (fs.fail())
  {
    throw std::runtime_error("Cannot open model file \"" + input_model_path + "\".\n");
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(model_data.data()),
                                 model_data.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("ERROR: Failed to verify circle '" + input_model_path + "'");
  }

  _module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (_module == nullptr)
  {
    throw std::runtime_error("ERROR: Failed to load '" + input_model_path + "'");
  }

  // Initialize interpreter
  _interpreter = std::make_unique<luci_interpreter::Interpreter>(_module.get());
}

void RecordOutput::run()
{
  try
  {
    HDF5Importer importer(_input_data_path);
    importer.importGroup();

    bool is_raw_data = importer.isRawData();

    const auto num_records = importer.numRecords();
    if (num_records == 0)
      throw std::runtime_error("The input data file does not contain any record.");

    const auto input_nodes = loco::input_nodes(_module->graph());
    const auto num_inputs = input_nodes.size();

    uint32_t overall_time = 0;
    for (int32_t record_idx = 0; record_idx < num_records; record_idx++)
    {
      if (num_inputs != importer.numInputs(record_idx))
        throw std::runtime_error("Wrong number of inputs.");

      if (record_idx % 100 == 0)
        std::cout << "Recording " << record_idx << "'th data" << std::endl;

      for (int32_t input_idx = 0; input_idx < num_inputs; input_idx++)
      {
        const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
        assert(input_node->index() == input_idx);
        std::vector<char> input_data(getTensorSize(input_node));

        if (!is_raw_data)
        {
          DataType dtype;
          Shape shape(input_node->rank());
          importer.readTensor(record_idx, input_idx, &dtype, &shape, input_data.data());

          // Check the type and the shape of the input data is valid
          verifyTypeShape(input_node, dtype, shape);
        }
        else
        {
          // Skip type/shape check for raw data
          importer.readTensor(record_idx, input_idx, input_data.data());
        }

        _interpreter->writeInputTensor(input_node, input_data.data(), input_data.size());
      }

      auto start = std::chrono::steady_clock::now();
      _interpreter->interpret();
      auto end = std::chrono::steady_clock::now();
      overall_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

      // TODO support outputs_nodes.size() > 1
      auto output_nodes = loco::output_nodes(_module->graph());
      assert(output_nodes.size() == 1);

      auto output_node = loco::must_cast<luci::CircleOutput *>(output_nodes[0]);

      assert(output_node->dtype() == loco::DataType::FLOAT32);
      std::vector<float> read_data(getTensorSize(output_node) / sizeof(float));
      _interpreter->readOutputTensor(output_node, &read_data[0], getTensorSize(output_node));

      // write output as txt to file
      std::string output_file_path = _output_dir_path + "/" + std::to_string(record_idx) + ".data";
      std::ofstream output_file(output_file_path);

      for (int o = 0; o < read_data.size(); ++o)
      {
        output_file << read_data[o] << " ";
      }
      output_file << std::endl;
      output_file.close();
    }

    std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;
    std::cout << "Average time: " << static_cast<double>(overall_time) / num_records << std::endl;
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("HDF5 error occurred.");
  }
}

} // namespace record_output
