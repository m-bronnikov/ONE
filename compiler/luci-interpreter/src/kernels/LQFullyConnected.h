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

#ifndef LUCI_INTERPRETER_KERNELS_LQFULLYCONNECTED_H
#define LUCI_INTERPRETER_KERNELS_LQFULLYCONNECTED_H

#include "core/Kernel.h"
#include "core/KernelParams.h"
#include "kernels/LQBinarizer.h"

namespace luci_interpreter
{
namespace kernels
{

class LQFullyConnected : public KernelWithParams<LQFullyConnectedParams>
{
private:
  using QuantizedBinaryInput = LQBinarizer;

public:
  LQFullyConnected(const Tensor *input, const Tensor *input_scales, const Tensor *weights_scales,
                   const Tensor *weights_binary, const Tensor *bias, Tensor *output,
                   const LQFullyConnectedParams &params);

  const Tensor *input() const { return _inputs[0]; }
  const Tensor *input_scales() const { return _inputs[1]; }
  const Tensor *weights_scales() const { return _inputs[2]; }
  const Tensor *weights_binary() const { return _inputs[3]; }
  const Tensor *bias() const { return _inputs[4]; }
  Tensor *output() const { return _outputs[0]; }

  void configure() override;
  void execute() const override;

private:
  // Tensor for inner computations
  std::unique_ptr<QuantizedBinaryInput> input_binary;

  // Only float execution supported
  void evalFloat() const;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_LQFULLYCONNECTED_H