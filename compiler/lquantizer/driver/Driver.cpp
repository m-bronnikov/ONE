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

#include "Quantizer.h"

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <luci/UserSettings.h>

void print_version(void)
{
  std::cout << "lquantizer version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(const int argc, char **argv)
{
  using namespace lquantizer;

  arser::Arser arser("Provide LQ post-training quantization for circle models");

  arser.add_argument("--version")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  arser.add_argument("--input_model")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Input model filepath");

  arser.add_argument("--input_data")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Input data filepath. If not given, lquantizer will run with randomly generated data. "
          "Note that the random dataset does not represent inference workload, leading to poor "
          "model accuracy.");

  arser.add_argument("--output_model")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Output model filepath");

  // TODO add ability to set different count of bits for inputs and weights
  arser.add_argument("--encode_bits")
    .nargs(1)
    .type(arser::DataType::INT32)
    .required(false)
    .help("Quantization bits count per parameter value");

  // TODO Add ability to set training steps and batch size

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  auto input_model_path = arser.get<std::string>("--input_model");
  auto output_model_path = arser.get<std::string>("--output_model");

  std::unique_ptr<Quantizer> lqzer;

  if (arser["--encode_bits"])
  {
    // if encode bits provided, set this value via constructor
    auto encode_bits = arser.get<int32_t>("--encode_bits");
    assert(encode_bits > 0);

    lqzer = std::make_unique<Quantizer>(encode_bits);
  }
  else
  {
    // use default constructor
    lqzer = std::make_unique<Quantizer>();
  }

  // Initialize interpreter and observer
  lqzer->initialize(input_model_path);

  if (arser["--input_data"])
  {
    auto input_data_path = arser.get<std::string>("--input_data");

    // Profile min/max while executing the given input data
    lqzer->path_to_input_data(input_data_path);
  }

  // training
  lqzer->train();

  // Save profiled values to the model
  lqzer->save(output_model_path);

  return EXIT_SUCCESS;
}
