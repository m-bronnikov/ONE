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

#ifndef MICRO_TEST_MEMORY_MANAGER_H
#define MICRO_TEST_MEMORY_MANAGER_H

#include "micro/MemoryManager.h"

namespace micro
{
// Memory Manager for using in kernels tests. This eliminates the need to manually delete the
// allocated memory in tests. This mem_manager remembers all its allocations and in destructor
// delete all allocations.
class TestMemoryManager : public IMemoryManager
{
public:
  void allocate_memory(micro::Tensor &tensor) final;
  void release_memory(micro::Tensor &tensor) final;

  ~TestMemoryManager() override
  {
    for (auto allocation : allocations)
    {
      delete[] allocation;
    }
  }

private:
  std::vector<uint8_t *> allocations;
};

} // namespace micro

#endif // MICRO_TEST_MEMORY_MANAGER_H
