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

#include <arser/arser.h>
#include <vconone/vconone.h>

void print_version(void)
{
  std::cout << "output_recorder version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

using namespace record_output;

int entry(const int argc, char **argv)
{
  arser::Arser arser("Run network on hdf5 file and write result.");

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
    .required(true)
    .help("Input data filepath.");

  arser.add_argument("--output_dir")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Dir to store output of network.");

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
  auto input_data_path = arser.get<std::string>("--input_data");
  auto output_dir_path = arser.get<std::string>("--output_dir");

  RecordOutput ro(input_model_path, input_data_path, output_dir_path);
  ro.run();

  return EXIT_SUCCESS;
}
