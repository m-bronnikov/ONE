set(SOURCES
    "${MICRO_INCLUDE_DIR}/micro/core/DataType.h"
    "${MICRO_INCLUDE_DIR}/micro/core/Tensor.h"
    Kernel.h
    KernelParams.h
    RuntimeGraph.h
    RuntimeGraph.cpp
    RuntimeModule.h
    Tensor.cpp)

add_library(${MICRO_CORE} STATIC ${SOURCES})
set_target_properties(${MICRO_CORE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${MICRO_CORE} PUBLIC "${MICRO_INCLUDE_DIR}")
target_include_directories(${MICRO_CORE} PUBLIC "${MICRO_SOURCE_DIR}")
target_link_libraries(${MICRO_CORE} PUBLIC luci_lang)
target_link_libraries(${MICRO_CORE} PRIVATE nncc_common)
