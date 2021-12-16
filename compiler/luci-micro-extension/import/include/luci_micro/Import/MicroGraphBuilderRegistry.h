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

#ifndef __LUCI_MICRO_IMPORT_MICRO_GRAPH_BUILDER_REGISTRY_H__
#define __LUCI_MICRO_IMPORT_MICRO_GRAPH_BUILDER_REGISTRY_H__

#include <luci/Import/GraphBuilderRegistry.h>

#include <vector>

namespace luci_micro
{
using namespace luci;

/**
 * @brief Memory optimized class to return graph builder for Circle nodes
 */
class MicroGraphBuilderRegistry final : public GraphBuilderSource
{
public:
  MicroGraphBuilderRegistry();

public:
  /**
   * @brief Returns constant node from given tensor's index and builder context
   */
  CircleNode *create_const(GraphBuilderContext *context, int32_t tensor_index) const final;

  /**
   * @brief Returns registered GraphBuilder pointer for operator or
   *        nullptr if not registered
   */
  const GraphBuilderBase *lookup(const circle::BuiltinOperator &op) const final
  {
    return _builders.at(uint32_t(op)).get();
  }

  static MicroGraphBuilderRegistry &get()
  {
    static MicroGraphBuilderRegistry me;
    return me;
  }

public:
  void add(const circle::BuiltinOperator op, std::unique_ptr<GraphBuilderBase> &&builder)
  {
    _builders.at(uint32_t(op)) = std::move(builder);
  }

private:
  const GraphBuilderSource *_parent = nullptr;

private:
  std::vector<std::unique_ptr<GraphBuilderBase>> _builders;
};

} // namespace luci_micro

#endif // __LUCI_MICRO_IMPORT_MICRO_GRAPH_BUILDER_REGISTRY_H__