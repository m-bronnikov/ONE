set(SOURCES
        BinaryOpCommon.h
        Utils.h
        Utils.cpp
        ${TensorFlowSource_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc
        "${MICRO_INCLUDE_DIR}/micro/TestMemoryManager.h"
        ${MICRO_SOURCE_DIR}/TestMemoryManager.cpp
        "${MICRO_INCLUDE_DIR}/micro/SimpleMemoryManager.h"
        ${MICRO_SOURCE_DIR}/SimpleMemoryManager.cpp)

macro(REGISTER_KERNEL NODE)
  list(APPEND SOURCES "${NODE}.h")
  list(APPEND SOURCES "${NODE}.cpp")
endmacro(REGISTER_KERNEL)

include(${KERNEL_REGISTER_FILE})

add_library(${MICRO_KERNELS} STATIC ${SOURCES})
set_target_properties(${MICRO_KERNELS} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${MICRO_KERNELS} PUBLIC ${MICRO_SOURCE_DIR})
target_include_directories(${MICRO_KERNELS} SYSTEM PRIVATE
        "${TensorFlowRuySource_DIR}"
        "${TensorFlowGEMMLowpSource_DIR}"
        "${TensorFlowEigenSource_DIR}"
        "${TensorFlowSource_DIR}")

if (NOT LUCI_MICRO)
  target_link_libraries(${MICRO_KERNELS}
          PUBLIC ${MICRO_CORE}
          PRIVATE nncc_common Threads::Threads)
else ()
  target_link_libraries(${MICRO_KERNELS}
          PUBLIC ${MICRO_CORE})
endif()

add_pal_to_target(${MICRO_KERNELS})

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

macro(REGISTER_KERNEL NODE)
  list(APPEND TEST_SOURCES "${NODE}.test.cpp")
endmacro(REGISTER_KERNEL)

include(${KERNEL_REGISTER_FILE})

list(APPEND TEST_SOURCES TestUtils.h TestUtils.cpp)

GTest_AddTest(${MICRO_KERNELS}_test ${TEST_SOURCES})
target_link_libraries(${MICRO_KERNELS}_test ${MICRO_KERNELS})
