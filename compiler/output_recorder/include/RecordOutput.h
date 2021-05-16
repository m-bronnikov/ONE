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

#ifndef __RECORD_OUTPUT_H__
#define __RECORD_OUTPUT_H__

#include <luci/IR/Module.h>
#include <luci_interpreter/Interpreter.h>

#include <memory>

namespace record_output
{

class RecordOutput
{
public:
  explicit RecordOutput(const std::string &input_model_path, const std::string &input_data_path,
                        const std::string &output_data_path);

  ~RecordOutput() = default;

  void run();

private:
  std::unique_ptr<luci::Module> _module;
  std::unique_ptr<luci_interpreter::Interpreter> _interpreter;
  const std::string _input_data_path;
  const std::string _output_dir_path;
};

} // namespace record_output

#endif // __RECORD_OUTPUT_H__
