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

#ifndef CIRCLE_EXECUTION_PLANNER_H
#define CIRCLE_EXECUTION_PLANNER_H

#include "TargetPlatform.h"
#include "IScratchpadHelper.h"
#include "ScratchpadHelperLinux.h"
#include "ScratchpadHelperMCU.h"
#include "ScratchpadHelperCMSISNN.h"
#include <luci/IR/Module.h>
#include <luci/Plan/CircleNodeExecutionPlan.h>

namespace circle_planner
{
// struct for additional information for the node. it helps build allocations plan for nodes.
struct AllocationNodeInformation
{

  AllocationNodeInformation()
  {
    offset = 0;
    size = 0;
    node_num = -1;
    first_node = -1;
    last_node = -1;
    is_temp = false;
    breadth = 0;
  }
  // memory offset from the beginning of the buffer
  uint32_t offset;
  // node required size
  uint32_t size;
  // the value assigned to the node
  uint32_t node_num;
  // the value of the node_num of the node when current node first use.
  // Used to build the usage interval of the current node
  uint32_t first_node;
  // the value of the node_num of the node when current node last use.
  // Used to build the usage interval of the current node
  uint32_t last_node;
  // is the current node temporary or not
  bool is_temp;
  // Breadth is a sum of live tensors sizes at the moment of execution of given node
  uint32_t breadth;

  bool operator<(const AllocationNodeInformation &other) const { return offset < other.offset; }
};

class ExecutionPlanner
{
public:
  ExecutionPlanner() = delete;
  explicit ExecutionPlanner(loco::Graph *graph) : _graph(graph)
  {
    _scratchpad_helper = std::make_unique<ScratchpadHelperLinux>();
  }

  explicit ExecutionPlanner(loco::Graph *graph, TargetPlatform target_platform) : _graph(graph)
  {
    switch (target_platform.platform_type)
    {
      case LINUX:
        _scratchpad_helper = std::make_unique<ScratchpadHelperLinux>();
        break;
      case MCU:
        _scratchpad_helper = std::make_unique<ScratchpadHelperMCU>();
        break;
      case CMSISNN:
        _scratchpad_helper = std::make_unique<ScratchpadHelperCMSISNN>(target_platform.use_dsp);
        break;
      default:
        assert(false && "Use unsupported platform");
    }
  };

  // Method provides execution plan, which contains execution order and
  // memory offsets for all nodes in _graph.
  // This plan writes in nodes annotation information with help of CircleNodeExecutionPlan class.
  void make_execution_plan();

  // Method change planning mode:
  // is_null_consts = true - constants are no longer taken into account when planning
  // is_null_inputs = true - input are no longer taken into account when planning
  // is_null_scratchpads = true - scratchpads are no longer taken into account when planning
  void change_planning_mode(bool is_null_consts, bool is_null_inputs, bool is_null_scratchpads)
  {
    _is_null_consts = is_null_consts;
    _is_null_inputs = is_null_inputs;
    _is_null_scratchpads = is_null_scratchpads;
  };

private:
  // Method gets default execution order plan and saves it in _ordered_nodes vector.
  // There can be different variants of execution order and this method provides main one.
  void get_default_execution_order_plan();

  // Method provides nodes with usage interval information.
  void get_usage_interval();

  // Method dumps execution plan information.
  void dump_inform();

  // Method finds required offsets for all nodes from _ordered_nodes, using greedy by size approach.
  // It saves offsets in _offsets vector.
  // Return: required size of buffer.
  uint32_t get_offsets_with_greedy_by_size();

  // Realization of greedy by size approach (algorithm is mentioned in
  // "EFFICIENT MEMORY MANAGEMENT FOR DEEP NEURAL NET INFERENCE" paper) to find offsets for nodes.
  uint32_t greedy_by_size_approach();

  // Method creates and fills _alloc_node_inform_vector with usage interval inform and node's sizes.
  // null_consts = true - size of const nodes will be equal 0;
  // null_inputs = true - size of input nodes will be equal 0;
  // null_scratchpad = true - size of scratchpad nodes will be equal 0;
  // It using if we don't want to take input(const or scratchpads) nodes into account
  // when determining offsets and calculating the required buffer size. This is uses for
  // experiments.
  void create_alloc_node_inform_vector(bool null_consts = false, bool null_inputs = false,
                                       bool null_scratchpad = false);

  // Stores allocation additional information for the all nodes from _graph.
  std::vector<AllocationNodeInformation> _alloc_node_inform_vector;

  // Stores nodes in execution order.
  std::vector<loco::Node *> _ordered_nodes;

  // Stores nodes memory offsets in arena buffer.
  std::vector<std::vector<uint32_t>> _offsets;

  // Stores positions of nodes in _ordered_nodes vector,
  // where node in i'th position in this vector first use.
  // For example, if i'th position of _alloc_node stores j value, then
  // the node from _ordered_nodes in j'th position is the node when we should allocate (first use)
  // the node from _ordered_nodes in i'th position.
  std::vector<uint32_t> _alloc_node;

  // Stores positions of nodes in _ordered_nodes vector,
  // where node in i'th position in this vector last use.
  // For example, if i'th position of _alloc_node stores j value, then
  // the node from _ordered_nodes in j'th position is the node when we can deallocate (last use)
  // the node from _ordered_nodes in i'th position.
  std::vector<uint32_t> _dealloc_node;

  loco::Graph *_graph;

  // Calculate size of scratchpad tensors for current platform
  std::unique_ptr<IScratchpadHelper> _scratchpad_helper;

  // Required memory size.
  uint32_t _required_size = 0;

  // Flags for choosing different planning modes:
  // _is_null_consts = true - constants are no longer taken into account when planning
  // _is_null_inputs = true - input are no longer taken into account when planning
  // _is_null_scratchpads = true - scratchpads are no longer taken into account when planning
  bool _is_null_consts = false;
  bool _is_null_inputs = false;
  bool _is_null_scratchpads = false;
};

} // namespace circle_planner

#endif // CIRCLE_EXECUTION_PLANNER_H
