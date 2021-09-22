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

#include "micro/Interpreter.h"
#include "micro/SimpleMemoryManager.h"

#include "loader/ModuleLoader.h"

#include <stdexcept>

namespace micro
{

Interpreter::Interpreter(const luci::Module *module, micro::IMemoryManager *memory_manager)
{
  _runtime_module = std::make_unique<RuntimeModule>();

  if (memory_manager == nullptr)
  {
    _default_memory_manager = std::make_unique<SimpleMemoryManager>();
    _memory_manager = _default_memory_manager.get();
  }
  else
  {
    _memory_manager = memory_manager;
  }

  ModuleLoader loader(module, _runtime_module.get(), _memory_manager);
  loader.load();
}

Interpreter::~Interpreter() = default;

void Interpreter::writeInputTensor(const luci::CircleInput *input_node, const void *data,
                                   size_t data_size)
{
  Tensor *tensor = _runtime_module->getInputTensors()[input_node->index()];
  if (tensor == nullptr)
  {
    const std::string &name = input_node->name();
    throw std::runtime_error("Cannot find tensor for input node named \"" + name + "\".");
  }
  if (data != nullptr)
    tensor->writeData(data, data_size);
}

void Interpreter::readOutputTensor(const luci::CircleOutput *output_node, void *data,
                                   size_t data_size)
{
  Tensor *tensor = _runtime_module->getOutputTensors()[output_node->index()];
  if (tensor == nullptr)
  {
    const std::string &name = output_node->name();
    throw std::runtime_error("Cannot find tensor for output node named \"" + name + "\".");
  }
  if (data != nullptr)
    tensor->readData(data, data_size);
}

void Interpreter::interpret() { _runtime_module->execute(); }

} // namespace micro
