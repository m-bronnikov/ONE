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

#include "kernels/LQFullyConnected.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(LQFullyConnectedTest, Simple)
{
  // inputs
  Shape input_shape{1, 5};
  std::vector<float> input_vec{
    0.5, 1.2, 2.3, -1.0, 0.0, // batch 0
    //  TODO add second batch
  };
  Shape input_scales_shape{2}; // 2 bit to encode input
  std::vector<float> input_scales_vec{
    0.12, 1.7, // sorted
  };
  Shape weights_scales_shape{4, 3}; // 3 bit to encode weight
  std::vector<float> weights_scales_vec{
    0.11, 0.23, 0.31, // neuro 1
    0.23, 0.41, 0.53, // neuro 2
    0.13, 0.22, 0.46, // neuro 3
    0.32, 0.33, 0.35, // neuro 4
  };
  Shape weights_binary_shape{4, 3, 1};
  // each encoding here should be less than 2^hs = 2^5 = 32:
  std::vector<int32_t> weights_binary_vec{
    // neuro 1
    7,  // bit 1
    13, // bit 2
    20, // bit 3
    // neuro 2
    4,  // bit 1
    15, // bit 2
    3,  // bit 3
    // neuro 3
    31, // bit 1
    17, // bit 2
    11, // bit 3
    // neuro 4
    22, // bit 1
    19, // bit 2
    2,  // bit 3
  };
  Shape bias_shape{4};
  std::vector<float> bias_vec{-1.1, -5.0, -0.3, 2.8};

  // output
  std::initializer_list<int32_t> output_shape = {1, 4};
  std::vector<float> output_vec{
    -0.2014, -0.1546, 0.1526, 4.2936, // batch 0
    // TODO calculate second batch
  };

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_vec);
  Tensor input_scales_tensor =
    makeInputTensor<DataType::FLOAT32>(input_scales_shape, input_scales_vec);
  Tensor weights_scales_tensor =
    makeInputTensor<DataType::FLOAT32>(weights_scales_shape, weights_scales_vec);
  Tensor weights_binary_tensor =
    makeInputTensor<DataType::S32>(weights_binary_shape, weights_binary_vec);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_vec);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LQFullyConnectedParams params{};
  params.activation = Activation::NONE;
  params.hidden_size = 5;

  LQFullyConnected kernel(&input_tensor, &input_scales_tensor, &weights_scales_tensor,
                          &weights_binary_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(output_vec));
}

TEST(LQFullyConnectedTest, SimpleWithoutBatch)
{
  // inputs
  Shape input_shape{1, 5};
  std::vector<float> input_vec{
    0.5, 1.2, 2.3, -1.0, 0.0, // batch 0
    //  TODO add second batch
  };
  Shape input_scales_shape{2}; // 2 bit to encode input
  std::vector<float> input_scales_vec{
    0.12, 1.7, // sorted
  };
  Shape weights_scales_shape{4, 3}; // 3 bit to encode weight
  std::vector<float> weights_scales_vec{
    0.11, 0.23, 0.31, // neuro 1
    0.23, 0.41, 0.53, // neuro 2
    0.13, 0.22, 0.46, // neuro 3
    0.32, 0.33, 0.35, // neuro 4
  };
  Shape weights_binary_shape{4, 3, 1};
  // each encoding here should be less than 2^hs = 2^5 = 32:
  std::vector<int32_t> weights_binary_vec{
    // neuro 1
    7,  // bit 1
    13, // bit 2
    20, // bit 3
    // neuro 2
    4,  // bit 1
    15, // bit 2
    3,  // bit 3
    // neuro 3
    31, // bit 1
    17, // bit 2
    11, // bit 3
    // neuro 4
    22, // bit 1
    19, // bit 2
    2,  // bit 3
  };

  // output
  std::initializer_list<int32_t> output_shape = {1, 4};
  std::vector<float> output_vec{
    0.8986, 4.8454, 0.4526, 1.4936, // batch 0
    // TODO calculate second batch
  };

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_vec);
  Tensor input_scales_tensor =
    makeInputTensor<DataType::FLOAT32>(input_scales_shape, input_scales_vec);
  Tensor weights_scales_tensor =
    makeInputTensor<DataType::FLOAT32>(weights_scales_shape, weights_scales_vec);
  Tensor weights_binary_tensor =
    makeInputTensor<DataType::S32>(weights_binary_shape, weights_binary_vec);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LQFullyConnectedParams params{};
  params.activation = Activation::RELU;
  params.hidden_size = 5;

  LQFullyConnected kernel(&input_tensor, &input_scales_tensor, &weights_scales_tensor,
                          &weights_binary_tensor, nullptr, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(output_vec));
}

TEST(LQFullyConnectedTest, InvalidBinaryWeightsShape_NEG)
{
  // inputs
  Shape input_shape{1, 5};
  std::vector<float> input_vec{
    0.5, 1.2, 2.3, -1.0, 0.0, // batch 0
  };
  Shape input_scales_shape{2}; // 2 bit to encode input
  std::vector<float> input_scales_vec{
    0.12, 1.7, // sorted
  };
  Shape weights_scales_shape{4, 3}; // 3 bit to encode weight
  std::vector<float> weights_scales_vec{
    0.11, 0.23, 0.31, // neuro 1
    0.23, 0.41, 0.53, // neuro 2
    0.13, 0.22, 0.46, // neuro 3
    0.32, 0.33, 0.35, // neuro 4
  };
  Shape weights_binary_shape{3, 2, 1};
  // each encoding here should be less than 2^hs = 2^5 = 32:
  std::vector<int32_t> weights_binary_vec{
    // neuro 1
    7,  // bit 1
    13, // bit 2
    // neuro 2
    4,  // bit 1
    15, // bit 2
    // neuro 3
    31, // bit 1
    17, // bit 2
  };

  // output
  std::vector<float> output_vec{
    0.8986, 4.8454, 0.4526, 1.4936, // batch 0
  };

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_vec);
  Tensor input_scales_tensor =
    makeInputTensor<DataType::FLOAT32>(input_scales_shape, input_scales_vec);
  Tensor weights_scales_tensor =
    makeInputTensor<DataType::FLOAT32>(weights_scales_shape, weights_scales_vec);
  Tensor weights_binary_tensor =
    makeInputTensor<DataType::S32>(weights_binary_shape, weights_binary_vec);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LQFullyConnectedParams params{};
  params.activation = Activation::RELU;
  params.hidden_size = 5;

  LQFullyConnected kernel(&input_tensor, &input_scales_tensor, &weights_scales_tensor,
                          &weights_binary_tensor, nullptr, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(LQFullyConnectedTest, NotSupportedInputShape_NEG)
{
  // inputs
  Shape input_shape{1, 2, 5};
  std::vector<float> input_vec{
    0.5, 1.2, 2.3, -1.0, 0.0, // batch 0
    0.5, 1.2, 2.3, -1.0, 0.0, // batch 1
  };
  Shape input_scales_shape{2}; // 2 bit to encode input
  std::vector<float> input_scales_vec{
    0.12, 1.7, // sorted
  };
  Shape weights_scales_shape{4, 3}; // 3 bit to encode weight
  std::vector<float> weights_scales_vec{
    0.11, 0.23, 0.31, // neuro 1
    0.23, 0.41, 0.53, // neuro 2
    0.13, 0.22, 0.46, // neuro 3
    0.32, 0.33, 0.35, // neuro 4
  };
  Shape weights_binary_shape{4, 3, 1};
  // each encoding here should be less than 2^hs = 2^5 = 32:
  std::vector<int32_t> weights_binary_vec{
    // neuro 1
    7,  // bit 1
    13, // bit 2
    20, // bit 3
    // neuro 2
    4,  // bit 1
    15, // bit 2
    3,  // bit 3
    // neuro 3
    31, // bit 1
    17, // bit 2
    11, // bit 3
    // neuro 4
    22, // bit 1
    19, // bit 2
    2,  // bit 3
  };

  // output
  std::vector<float> output_vec{
    0.8986, 4.8454, 0.4526, 1.4936, // batch 0
    0.8986, 4.8454, 0.4526, 1.4936, // batch 1
  };

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_vec);
  Tensor input_scales_tensor =
    makeInputTensor<DataType::FLOAT32>(input_scales_shape, input_scales_vec);
  Tensor weights_scales_tensor =
    makeInputTensor<DataType::FLOAT32>(weights_scales_shape, weights_scales_vec);
  Tensor weights_binary_tensor =
    makeInputTensor<DataType::S32>(weights_binary_shape, weights_binary_vec);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LQFullyConnectedParams params{};
  params.activation = Activation::RELU;
  params.hidden_size = 5;

  LQFullyConnected kernel(&input_tensor, &input_scales_tensor, &weights_scales_tensor,
                          &weights_binary_tensor, nullptr, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
