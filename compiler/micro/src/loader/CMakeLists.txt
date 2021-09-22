set(SOURCES
    GraphLoader.h
    GraphLoader.cpp
    KernelBuilderHelper.h
    KernelBuilderHelper.cpp
    KernelBuilder.h
    KernelBuilder.cpp
    ModuleLoader.h
    ModuleLoader.cpp
    RuntimeToIR.h
    nodes/Builders.h)

# include kernel specific builders
macro(REGISTER_KERNEL NODE)
  list(APPEND SOURCES "nodes/${NODE}.cpp")
endmacro(REGISTER_KERNEL)
include(${KERNEL_REGISTER_FILE})

add_library(${MICRO_LOADER} STATIC ${SOURCES})
set_target_properties(${MICRO_LOADER} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${MICRO_LOADER} PUBLIC "${MICRO_PAL_DIR}")
target_include_directories(${MICRO_LOADER} PUBLIC "${MICRO_SOURCE_DIR}")

target_link_libraries(${MICRO_LOADER}
        PUBLIC luci_lang ${MICRO_CORE}
        PRIVATE ${MICRO_KERNELS} nncc_common)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

set(TEST_SOURCES KernelBuilder.test.cpp)

GTest_AddTest(${MICRO_LOADER}_test ${TEST_SOURCES})
target_link_libraries(${MICRO_LOADER}_test ${MICRO_LOADER})
