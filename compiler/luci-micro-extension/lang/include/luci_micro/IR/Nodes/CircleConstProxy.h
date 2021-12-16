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

#ifndef __LUCI_MICRO_IR_CIRCLE_CONST_PROXY_H__
#define __LUCI_MICRO_IR_CIRCLE_CONST_PROXY_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

#include <loco/IR/DataTypeTraits.h>

namespace luci_micro
{
using namespace luci;

/**
* @brief Class for read-only connection to tensor data
* @note  This will not be exported as a specific op. CircleConstProxy has access to provided data
* and provides reading access to user.
*/
class CircleConstProxy final : public FixedArityNode<0, CircleNodeImpl<CircleOpcode::CIRCLECONST>>
{
public:
 template <loco::DataType DT> uint32_t size(void) const;

 template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &at(uint32_t n) const;
 template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &scalar(void) const;

 // Note: this function makes reference to remote data buffer, CircleConstProxy not owns this data
 void bind_buffer(const uint8_t *data, uint32_t size);
 const uint8_t *data() const;
 uint32_t buffer_size() const;

private:
 struct ReferenceBuffer
 {
   const uint8_t *data = nullptr;
   uint32_t size = 0;
 };

private:
 ReferenceBuffer _buffer;
};

} // namespace luci_micro

#endif // __LUCI_MICRO_IR_CIRCLE_CONST_PROXY_H__