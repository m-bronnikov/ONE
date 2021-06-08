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

#include "kernels/LQFullyConnected.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

LQFullyConnected::LQFullyConnected(const Tensor *input, const Tensor *input_scales,
                                   const Tensor *weights_scales, const Tensor *weights_binary,
                                   const Tensor *bias, Tensor *output,
                                   const LQFullyConnectedParams &params)
  : KernelWithParams<LQFullyConnectedParams>(
      {input, input_scales, weights_scales, weights_binary, bias}, {output}, params)
{
}

void LQFullyConnected::configure()
{
  // check data types
  LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::FLOAT32);
  LUCI_INTERPRETER_CHECK(input_scales()->element_type() == DataType::FLOAT32);
  LUCI_INTERPRETER_CHECK(weights_scales()->element_type() == DataType::FLOAT32);
  LUCI_INTERPRETER_CHECK(weights_binary()->element_type() == DataType::S32);
  LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32);
  LUCI_INTERPRETER_CHECK(bias() == nullptr || bias()->element_type() == DataType::FLOAT32);

  const Shape &input_shape = input()->shape();
  const Shape &weights_shape = weights_binary()->shape();

  // check weights valid
  LUCI_INTERPRETER_CHECK(weights_scales()->shape().num_dims() == 2);
  LUCI_INTERPRETER_CHECK(weights_shape.num_dims() == 3);
  LUCI_INTERPRETER_CHECK(bias() == nullptr ||
                         bias()->shape().num_elements() == weights_shape.dim(0));
  LUCI_INTERPRETER_CHECK(weights_shape.dim(0) == weights_scales()->shape().dim(0));
  LUCI_INTERPRETER_CHECK(weights_shape.dim(1) == weights_scales()->shape().dim(1));
  LUCI_INTERPRETER_CHECK(weights_shape.dim(2) == ceil_div(params().hidden_size, 32));

  // encoding lens should be less than 32
  LUCI_INTERPRETER_CHECK(input_scales()->shape().dim(0) < 32);
  LUCI_INTERPRETER_CHECK(weights_scales()->shape().dim(1) < 32);

  // check multiplication is possible
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_shape.dim(1) == params().hidden_size);

  // output tensor init
  const int32_t batches = input_shape.dim(0);
  const int32_t output_vec_size = weights_shape.dim(0);
  output()->resize({batches, output_vec_size});

  // input binary init
  input_binary = std::make_unique<QuantizedBinaryInput>(input_shape.dim(1), input_scales());
}

void LQFullyConnected::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void LQFullyConnected::evalFloat() const
{
  const float *input_data = getTensorData<float>(input());
  const float *input_scales_data = getTensorData<float>(input_scales());
  const float *weight_scales_data = getTensorData<float>(weights_scales());
  const int32_t *input_binary_data = input_binary->data();
  const int32_t *weights_binary_data = getTensorData<int32_t>(weights_binary());
  const float *bias_data = getTensorData<float>(bias());
  float *output_data = getTensorData<float>(output());

  const Shape &weights_scales_shape = weights_scales()->shape();

  const int32_t input_encode_bits = input_scales()->shape().dim(0);
  const int32_t weights_encode_bits = weights_scales_shape.dim(1);

  const Shape &in_shape = input()->shape();
  const Shape &out_shape = output()->shape();
  const int32_t batches = out_shape.dim(0);
  const int32_t output_size = out_shape.dim(1);

  const int32_t hidden_size = static_cast<int32_t>(params().hidden_size);
  const int32_t real_size = weights_binary()->shape().dim(2);

  // TODO implement different versions of this function
  const auto bin_dot = [hidden_size, real_size](const int32_t *data_1, const int32_t *data_2) {
    int32_t positives = hidden_size - (real_size << 5); // hs - 32*rs
    for (int32_t i = 0; i < real_size; ++i)
    {
      positives += __builtin_popcount(~(data_1[i] ^ data_2[i]));
    }
    return (positives << 1) - hidden_size; // 2*positives - neurons_count
  };

  // execution of matmul
  for (int32_t batch = 0; batch < batches; ++batch)
  {
    // quantize batch of input data
    input_binary->quantize_and_pack(&input_data[calcOffset(in_shape, batch, 0)]);

    // pointer to output vector
    float *output_buffer = &output_data[calcOffset(out_shape, batch, 0)];

    // matrix multiplication between weights and input vector
    for (int32_t out_idx = 0; out_idx < output_size; ++out_idx)
    {
      // init value with bias if exist
      float output_total = 0.0f;

      // calculate overall output float value
      for (int32_t bi = 0; bi < input_encode_bits; ++bi)
      {
        // input computation data
        float inp_s = input_scales_data[bi];
        const int32_t *inp_bin_line = &input_binary_data[bi * real_size];

        for (int32_t bw = 0; bw < weights_encode_bits; ++bw)
        {
          int32_t w_offset = calcOffset(weights_scales_shape, out_idx, bw);
          float w_s = weight_scales_data[w_offset];
          const int32_t *w_bin_line = &weights_binary_data[w_offset * real_size];

          // add to total
          output_total += inp_s * w_s * bin_dot(inp_bin_line, w_bin_line);
        }
      }

      output_buffer[out_idx] = output_total;
    }

    // add bias if exist
    if (bias_data)
    {
      for (int32_t out_idx = 0; out_idx < output_size; ++out_idx)
      {
        output_buffer[out_idx] += bias_data[out_idx];
      }
    }
  }

  // compute activation
  computeActivationInplace(params().activation, output_data, out_shape.num_elements());
}

} // namespace kernels
} // namespace luci_interpreter
