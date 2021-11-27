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

#include <luci_interpreter/import/GraphBuilderRegistry.h>
#include <luci_interpreter/Interpreter.h>

#include <luci/Importer.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>

#include "sample_model.h"

namespace
{

std::unique_ptr<luci::Module> import_model_file(const std::string &filename)
{
  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
  {
    throw std::runtime_error("Cannot open model file \"" + filename + "\".\n");
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());
  return luci::Importer().importModule(circle::GetModel(model_data.data()));
}

std::unique_ptr<luci::Module> import_model_constant_buffer(const uint8_t *buffer,
                                                           luci::GraphBuilderSource *custom_source)
{
  return luci::Importer(custom_source).importModule(circle::GetModel(buffer));
}

} // namespace

int entry(int argc, char **argv)
{
  // hardcoded values
  const std::string conv_model_filename = "conv2d.circle";
  const uint8_t *conv_model_const_pointer = conv2d_circle;
  const std::vector<uint32_t> input_shape = {1, 5, 5, 1};
  const std::vector<uint32_t> output_shape = {1, 5, 5, 2};
  using model_dtype = float;

  // size of input/output tensor in elements
  auto const mult = std::multiplies<model_dtype>();
  auto input_tensor_size = std::accumulate(begin(input_shape), end(input_shape), 1u, mult);
  auto output_tensor_size = std::accumulate(begin(output_shape), end(output_shape), 1u, mult);

  // allocate input buffer
  std::vector<model_dtype> input_data(input_tensor_size);

  // generate random input
  {
    std::random_device device;
    std::mt19937 engine{device()};
    std::uniform_real_distribution<model_dtype> distrib(-3, 3);

    auto const generator = [&distrib, &engine]() { return distrib(engine); };
    std::generate(begin(input_data), end(input_data), generator);
  }

  // interpret given module
  auto const interpret_module_and_write_to_output = [&](std::unique_ptr<luci::Module> &module) {
    // Create interpreter.
    luci_interpreter::Interpreter interpreter(module.get());

    // Set input.
    const auto input_nodes = loco::input_nodes(module->graph());
    assert(input_nodes.size() == 1);

    auto const input_node = loco::must_cast<luci::CircleInput *>(input_nodes[0]);
    interpreter.writeInputTensor(input_node, input_data.data(),
                                 input_tensor_size * sizeof(model_dtype));

    // Do inference.
    interpreter.interpret();

    // Get output.
    const auto output_nodes = loco::output_nodes(module->graph());
    assert(output_nodes.size() == 1);

    auto const output_node = loco::must_cast<luci::CircleOutput *>(output_nodes[0]);
    std::vector<model_dtype> output_data(output_tensor_size);
    interpreter.readOutputTensor(output_node, output_data.data(),
                                 output_tensor_size * sizeof(model_dtype));

    return output_data;
  };

  // Load model from the file, import with copying, execute and save to output_buffer
  std::vector<model_dtype> output_data_1;
  {
    auto module = import_model_file(conv_model_filename);
    if (not module)
      throw std::runtime_error("Fail to import model");

    output_data_1 = interpret_module_and_write_to_output(module);
  }

  // Load model from const pointer, import with copying, execute and save to output_buffer
  std::vector<model_dtype> output_data_2;
  {
    // default builder source not allows using constants from model's buffer
    auto default_source = &luci::GraphBuilderRegistry::get();
    auto module = import_model_constant_buffer(conv_model_const_pointer, default_source);
    if (not module)
      throw std::runtime_error("Fail to import model");

    output_data_2 = interpret_module_and_write_to_output(module);
  }

  // Load model from const pointer, import without copying, execute and save to output_buffer
  std::vector<model_dtype> output_data_3;
  {
    auto default_source = luci_interpreter::source_without_constant_copying();
    auto module = import_model_constant_buffer(conv_model_const_pointer, default_source.get());
    if (not module)
      throw std::runtime_error("Fail to import model");

    output_data_3 = interpret_module_and_write_to_output(module);
  }

  // check all tensors are equal
  for (uint32_t o = 0; o < output_tensor_size; ++o)
  {
    bool elements_are_equal = true;

    elements_are_equal &= (output_data_1[o] == output_data_2[o]);
    elements_are_equal &= (output_data_2[o] == output_data_3[o]);

    if (not elements_are_equal)
    {
      std::cout << "[TEST FAILED]" << std::endl;
      throw std::runtime_error("Output values are not same!");
      return EXIT_FAILURE;
    }
  }

  std::cout << "[TEST PASSED]" << std::endl;
  return EXIT_SUCCESS;
}
