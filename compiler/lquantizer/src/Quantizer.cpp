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

#include "Quantizer.h"
#include "HDF5Importer.h"
#include "QunatizationErrorMinimzation.h"
#include "DataGenerator.h"

#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/IR/CircleQuantParam.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace
{
using namespace lquantizer;

// TODO Create builder for LQ prototypes based on FP nodes
luci::CircleLQFullyConnected *make_lq_fully_connected(loco::Graph *g, luci::CircleNode *node,
                                                      const uint32_t input_bits_encode,
                                                      const uint32_t weight_bits_encode)
{
  // obtain source and target FC nodes
  auto const *fc_node = loco::must_cast<luci::CircleFullyConnected *>(node);
  auto *lq_node = g->nodes()->create<luci::CircleLQFullyConnected>();

  // name of fc node
  auto const name = fc_node->name();

  // TODO support rank > 2 for input and output
  assert(fc_node->rank() == 2);

  // TODO delete this because shape and dtype sets via Shape/Data inference pass
  {
    // pass dtype
    lq_node->dtype(loco::DataType::FLOAT32);

    // pass shape
    lq_node->rank(fc_node->rank());
    lq_node->dim(0).set(fc_node->dim(0).value());
    lq_node->dim(1).set(fc_node->dim(1).value());
    lq_node->shape_status(luci::ShapeStatus::VALID);
  }

  // pass activation
  lq_node->fusedActivationFunction(fc_node->fusedActivationFunction());

  // pass input
  lq_node->input(fc_node->input());

  // pass bias
  lq_node->bias(fc_node->bias());

  // create and pass input_scales
  auto *input_scales = g->nodes()->create<luci::CircleConst>();
  {
    // pass shape
    input_scales->rank(1);
    input_scales->dim(0).set(input_bits_encode);
    input_scales->shape_status(luci::ShapeStatus::VALID);

    // allocate data
    input_scales->dtype(loco::DataType::FLOAT32);
    input_scales->size<loco::DataType::FLOAT32>(input_bits_encode);

    // set name
    input_scales->name(name + "/input_scales");
  }
  lq_node->input_scales(input_scales);

  auto const *weights = loco::must_cast<luci::CircleNode *>(fc_node->weights());
  assert(weights->rank() == 2);

  // define and pass hidden_size
  auto const hidden_size = weights->dim(1).value();
  lq_node->weights_hidden_size(hidden_size);

  // create and pass weights_scales
  auto *weights_scales = g->nodes()->create<luci::CircleConst>();
  {
    // define shape
    auto const output_size = weights->dim(0).value();
    auto const bits = weight_bits_encode;

    // pass shape
    weights_scales->rank(2);
    weights_scales->dim(0).set(output_size);
    weights_scales->dim(1).set(bits);
    weights_scales->shape_status(luci::ShapeStatus::VALID);

    // allocate data
    weights_scales->dtype(loco::DataType::FLOAT32);
    weights_scales->size<loco::DataType::FLOAT32>(output_size * bits);

    // set name
    weights_scales->name(name + "/weights_scales");
  }
  lq_node->weights_scales(weights_scales);

  // create and pass weights_binary
  auto *weights_binary = g->nodes()->create<luci::CircleConst>();
  {
    // define shape (first 2 dimensions must be equal)
    auto const output_size = weights_scales->dim(0).value();
    auto const bits = weights_scales->dim(1).value();
    auto const real_size = ceil_div(hidden_size, 32);

    // pass shape
    weights_binary->rank(3);
    weights_binary->dim(0).set(output_size);
    weights_binary->dim(1).set(bits);
    weights_binary->dim(2).set(real_size);
    weights_binary->shape_status(luci::ShapeStatus::VALID);

    // allocate data
    weights_binary->dtype(loco::DataType::S32);
    weights_binary->size<loco::DataType::S32>(output_size * bits * real_size);

    // set name
    weights_binary->name(name + "/weights_binary");
  }
  lq_node->weights_binary(weights_binary);

  // set node name
  lq_node->name(name + "/LQFullyConnected");

  return lq_node;
}

} // namespace

namespace lquantizer
{

/*
 * @brief Create prototype for given CircleNode
 *
 * This method creates and replaces node with LQ prototype if possible.
 */
luci::CircleNode *Quantizer::make_and_process_lq_node(loco::Graph *g, luci::CircleNode *node,
                                                      const luci::CircleNode *fp_node)
{
  luci::CircleNode *lq_node = nullptr;

  // lq works only with float data
  if (node->dtype() != loco::DataType::FLOAT32)
  {
    return nullptr;
  }

  switch (node->opcode())
  {
    case luci::CircleOpcode::FULLY_CONNECTED:
    {
      // create fc node
      auto *lq_fc = make_lq_fully_connected(g, node, _input_encoding_bits, _weights_encoding_bits);
      auto const *fp_fc = loco::must_cast<const luci::CircleFullyConnected *>(fp_node);

      // only FullyConnected with const weights supported
      if (dynamic_cast<luci::CircleConst *>(fp_fc->weights()) == nullptr)
      {
        return nullptr;
      }

      // extract circle input
      auto const *lq_input = loco::must_cast<luci::CircleNode *>(lq_fc->input());
      auto const *fp_input = loco::must_cast<luci::CircleNode *>(fp_fc->input());

      // match inputs with fc nodes
      _input_node2lq_node[lq_input] = lq_fc;
      _input_node2fp_node[fp_input] = fp_fc;

      // save node
      lq_node = lq_fc;
      break;
    }
    default:
      // Other LQ ops not implemented, yet
      return nullptr;
  }

  // replace node with LQ node in target graph
  loco::replace(node).with(lq_node);

  // bind FP node to LQ node
  _fp2lq_nodes[fp_node] = lq_node;

  return lq_node;
}

/*
 * @brief Create LQ model binded and based on source FP model.
 *
 * This method create clone of source model, but replace some FP nodes to exist LQ prototypes.
 */
void Quantizer::make_binded_lqgraph()
{
  assert(_lq_module->size() == _fp_module->size());

  for (auto gn = 0u; gn < _fp_module->size(); ++gn)
  {
    auto fp_g = _fp_module->graph(gn);
    auto lq_g = _lq_module->graph(gn);

    // Note: order of nodes should be same because `postorder_traversal()` is stable.
    auto const fp_nodes = loco::postorder_traversal(output_nodes(fp_g));
    auto const exchange_nodes = loco::postorder_traversal(output_nodes(lq_g));
    assert(fp_nodes.size() == exchange_nodes.size());

    for (auto n = 0u; n < fp_nodes.size(); ++n)
    {
      auto const *fp_node = dynamic_cast<const luci::CircleNode *>(fp_nodes[n]);
      if (not fp_node)
      {
        // only circle nodes can be converted to LQ
        continue;
      }

      auto *exchange_node = loco::must_cast<luci::CircleNode *>(exchange_nodes[n]);
      assert(fp_node->opnum() == exchange_node->opnum());

      auto *lq_node = make_and_process_lq_node(lq_g, exchange_node, fp_node);
      if (lq_node == nullptr)
      {
        // LQ prototype not exist for this node, do nothing
        continue;
      }

      // update graph and bind FP with LQ
      loco::replace(exchange_node).with(lq_node);
      _fp2lq_nodes[fp_node] = lq_node;
    }
  }
}

/*
 * @brief Setter for path to input data
 *
 * If this function not called, quantizer will use random fake data.
 */
void Quantizer::path_to_input_data(const std::string &path)
{
  _path_to_data = path;
  _use_random = false; // use data from file if path passed
}

/*
 * @brief Initialize Quantizer
 *
 * This function import source graph and creates cloned graph with LQ nodes instead Full Precision.
 */
void Quantizer::initialize(const std::string &input_model_path)
{
  // Load model from file as binary
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

  // import model to source and target IR's
  _fp_module = luci::Importer().importModule(circle::GetModel(model_data.data()));
  _lq_module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (_fp_module == nullptr || _lq_module == nullptr)
  {
    throw std::runtime_error("ERROR: Failed to load model from '" + input_model_path + "'");
  }

  make_binded_lqgraph();
}

void Quantizer::train()
{
  train_weights();
  train_input();
  uptrain_input();
}

// train tries to learn input scales to minimize error between `fp_input` and
// `dequntize(quantize(lq_input))`
void Quantizer::uptrain_input()
{
  DataGenerator generator(_use_random);

  const auto lq_input_nodes = loco::input_nodes(_lq_module->graph());
  const auto fp_input_nodes = loco::input_nodes(_lq_module->graph());

  const auto inputs_count = static_cast<int32_t>(lq_input_nodes.size());
  assert(inputs_count == loco::input_nodes(_fp_module->graph()).size());

  if (_use_random)
  {
    generator.inputs_count(inputs_count);

    // TODO set this value via constructor instead hardcode
    generator.set_required_records_num(static_cast<int32_t>(_train_batches) * 3);
  }
  else
  {
    generator.open_data_file(_path_to_data);
    assert(generator.inputs_count() == inputs_count);
  }

  // uptraining
  for (uint32_t e = 0; e < _train_epochs; ++e)
  {
    // start read data from start state
    generator.reset();

    while (not generator.is_empty())
    {
      // create interpreter in order to save lq input data for nodes
      auto lq_interpreter = std::make_unique<luci_interpreter::Interpreter>(_lq_module.get());
      auto lq_observer = std::make_unique<InputSavingObserver>(_input_node2lq_node);
      lq_interpreter->attachObserver(lq_observer.get());

      // create interpreter in order to save fp input data for nodes
      auto fp_interpreter = std::make_unique<luci_interpreter::Interpreter>(_fp_module.get());
      auto fp_observer = std::make_unique<InputSavingObserver>(_input_node2fp_node);
      fp_interpreter->attachObserver(fp_observer.get());

      // Step 1. Save target and source inputs for quantizators uptraining
      for (int32_t b = 0; b < _train_batches && not generator.is_empty(); ++b)
      {
        // fill input tensors before execution
        for (int32_t i = 0; i < inputs_count; ++i)
        {
          // obtain input node
          auto lq_input_node = loco::must_cast<const luci::CircleInput *>(lq_input_nodes[i]);
          auto fp_input_node = loco::must_cast<const luci::CircleInput *>(fp_input_nodes[i]);

          // compute input buffer size
          uint32_t num_elements = 1u;
          for (uint32_t d = 0; d < lq_input_node->rank(); d++)
          {
            assert(lq_input_node->dim(d).known());
            num_elements *= lq_input_node->dim(d).value();
          }

          // read data for input
          std::vector<float> input_data(num_elements);
          if (not generator.read_record(input_data))
          {
            throw std::runtime_error("Input data broken!");
          }

          lq_interpreter->writeInputTensor(lq_input_node, input_data.data(),
                                           num_elements * sizeof(float));
          fp_interpreter->writeInputTensor(fp_input_node, input_data.data(),
                                           num_elements * sizeof(float));
        }

        // execute interpreters(store inputs of nodes)
        lq_interpreter->interpret();
        fp_interpreter->interpret();
      }

      // Step 2. Learn input quantizers of lq nodes.
      for (auto it : _fp2lq_nodes)
      {
        auto *lq_node = loco::must_cast<const luci::CircleLQFullyConnected *>(it.second);
        auto const *fp_node = loco::must_cast<const luci::CircleFullyConnected *>(it.first);

        auto *input_scales = loco::must_cast<luci::CircleConst *>(lq_node->input_scales());
        auto *input_scales_data = &(input_scales->scalar<loco::DataType::FLOAT32>());

        auto const &lq_input = lq_observer->input_data_for(lq_node);
        auto const *lq_input_data = lq_input.data();

        auto const &fp_input = fp_observer->input_data_for(fp_node);
        auto const *fp_input_data = fp_input.data();

        assert(lq_input.size() == fp_input.size());

        auto const bits_per_value = input_scales->dim(0).value();
        assert(bits_per_value == _input_encoding_bits);

        // update input_scales using lq inputs to minimize error with fp inputs
        QEM coach(lq_input_data, fp_input_data, input_scales_data, 1, lq_input.size(),
                  bits_per_value);
        coach.fit(_qem_iterations);
      }
    }
  }
}

// train tries to learn input scales to minimize error between `fp_input` and
// `dequntize(quantize(fp_input))`
void Quantizer::train_input()
{
  DataGenerator generator(_use_random);

  const auto input_nodes = loco::input_nodes(_fp_module->graph());
  const auto inputs_count = static_cast<int32_t>(input_nodes.size());
  assert(inputs_count == loco::input_nodes(_lq_module->graph()).size());

  if (_use_random)
  {
    generator.inputs_count(inputs_count);

    // TODO set this value via constructor instead hardcode
    generator.set_required_records_num(static_cast<int32_t>(_train_batches) * 3);
  }
  else
  {
    generator.open_data_file(_path_to_data);
    assert(generator.inputs_count() == inputs_count);
  }

  // training
  for (uint32_t e = 0; e < _train_epochs; ++e)
  {
    // start read data from start state
    generator.reset();

    while (not generator.is_empty())
    {
      // create interpreter in order to save fp input data for nodes
      auto interpreter = std::make_unique<luci_interpreter::Interpreter>(_fp_module.get());
      auto observer = std::make_unique<InputSavingObserver>(_input_node2fp_node);
      interpreter->attachObserver(observer.get());

      // Step 1. Save target inputs for quantizators training
      for (int32_t b = 0; b < _train_batches && not generator.is_empty(); ++b)
      {
        // fill input tensors before execution
        for (int32_t i = 0; i < inputs_count; ++i)
        {
          assert(generator.current_input_idx() == i);

          // obtain input node
          auto input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[i]);

          // compute input buffer size
          uint32_t num_elements = 1u;
          for (uint32_t d = 0; d < input_node->rank(); d++)
          {
            assert(input_node->dim(d).known());
            num_elements *= input_node->dim(d).value();
          }

          // read data for input
          std::vector<float> input_data(num_elements);
          if (not generator.read_record(input_data))
          {
            throw std::runtime_error("Input data broken!");
          }

          interpreter->writeInputTensor(input_node, input_data.data(),
                                        num_elements * sizeof(float));
        }

        // execute interpreter(store inputs of nodes)
        interpreter->interpret();
      }

      // Step 2. Learn input quantizers of lq nodes.
      for (auto it : _fp2lq_nodes)
      {
        auto *lq_node = loco::must_cast<const luci::CircleLQFullyConnected *>(it.second);
        auto const *fp_node = loco::must_cast<const luci::CircleFullyConnected *>(it.first);

        auto *input_scales = loco::must_cast<luci::CircleConst *>(lq_node->input_scales());
        auto *input_scales_data = &(input_scales->scalar<loco::DataType::FLOAT32>());

        auto const &input = observer->input_data_for(fp_node);
        auto const *input_data = input.data();

        auto const bits_per_value = input_scales->dim(0).value();
        assert(bits_per_value == _input_encoding_bits);

        // update input_scales using fp inputs to minimize error with fp inputs
        QEM coach(input_data, input_data, input_scales_data, 1, input.size(), bits_per_value);
        coach.fit(_qem_iterations);
      }
    }
  }
}

void Quantizer::train_weights()
{
  for (auto it : _fp2lq_nodes)
  {
    auto *lq_node = loco::must_cast<const luci::CircleLQFullyConnected *>(it.second);
    auto const *fp_node = loco::must_cast<const luci::CircleFullyConnected *>(it.first);
    auto const *weights = loco::must_cast<const luci::CircleConst *>(fp_node->weights());

    auto *weights_scales = loco::must_cast<luci::CircleConst *>(lq_node->weights_scales());
    auto *weights_binary = loco::must_cast<luci::CircleConst *>(lq_node->weights_binary());

    // init scales as random
    auto *scales_data = &(weights_scales->scalar<loco::DataType::FLOAT32>());
    auto const output_size = weights_scales->dim(0).value();
    auto const bits_per_value = weights_scales->dim(1).value();
    assert(bits_per_value == _weights_encoding_bits);

    set_float_random_data(scales_data, output_size * bits_per_value);

    // weights
    auto const hidden_size = weights->dim(1).value();
    assert(weights->dim(0).value() == output_size);
    auto const *weights_data = &(weights->scalar<loco::DataType::FLOAT32>());

    // train weights scales on fp weights to minimize error with fp weights
    QEM coach(weights_data, weights_data, scales_data, output_size, hidden_size, bits_per_value);
    coach.fit(_qem_iterations * _train_epochs);

    // fill binary weights
    auto binary_data = &(weights_binary->scalar<loco::DataType::S32>());
    coach.fill_binary(binary_data);
  }
}

/*
 * @brief Save LQ Network to file
 */
void Quantizer::save(const std::string &output_model_path)
{
  // Export to output Circle file
  luci::CircleExporter exporter;

  luci::CircleFileExpContract contract(_lq_module.get(), output_model_path);

  if (!exporter.invoke(&contract))
  {
    throw std::runtime_error("ERROR: Failed to export '" + output_model_path + "'");
  }
}

} // namespace lquantizer
