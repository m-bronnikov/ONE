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

#ifndef __LUCI_IR_CIRCLELQFULLYCONNECTED_H__
#define __LUCI_IR_CIRCLELQFULLYCONNECTED_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/AttrFusedActFunc.h"
#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief LQ_FULLY_CONNECTED in Circle
 */
class CircleLQFullyConnected final
  : public FixedArityNode<5, CircleNodeImpl<CircleOpcode::LQ_FULLY_CONNECTED>>,
    public CircleNodeMixin<CircleNodeTrait::FusedActFunc>,
    public CircleNodeMixin<CircleNodeTrait::Bias>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *input_scales(void) const { return at(1)->node(); }
  void input_scales(loco::Node *node) { at(1)->node(node); }

  loco::Node *weights_scales(void) const { return at(2)->node(); }
  void weights_scales(loco::Node *node) { at(2)->node(node); }

  loco::Node *weights_binary(void) const { return at(3)->node(); }
  void weights_binary(loco::Node *node) { at(3)->node(node); }

  loco::Node *bias(void) const override { return at(4)->node(); }
  void bias(loco::Node *node) override { at(4)->node(node); }

public:
  int32_t weights_hidden_size(void) const { return _weights_hidden_size; }
  void weights_hidden_size(int32_t weights_hidden_size)
  {
    _weights_hidden_size = weights_hidden_size;
  }

private:
  int32_t _weights_hidden_size = 0;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLELQFULLYCONNECTED_H__