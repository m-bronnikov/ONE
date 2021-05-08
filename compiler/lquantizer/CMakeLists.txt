nnas_find_package(HDF5 COMPONENTS STATIC QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build lquantizer: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(lquantizer ${DRIVER} ${SOURCES})
target_include_directories(lquantizer PRIVATE include)
target_include_directories(lquantizer PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(lquantizer ${HDF5_CXX_LIBRARIES})
target_link_libraries(lquantizer arser)
target_link_libraries(lquantizer safemain)
target_link_libraries(lquantizer luci_import)
target_link_libraries(lquantizer luci_env)
target_link_libraries(lquantizer luci_export)
target_link_libraries(lquantizer luci_interpreter)
target_link_libraries(lquantizer vconone)
target_link_libraries(lquantizer nncc_coverage)

install(TARGETS lquantizer DESTINATION bin)

#if(NOT ENABLE_TEST)
#  return()
#endif(NOT ENABLE_TEST)

# file(GLOB_RECURSE TESTS "tests/*.test.cpp")

# nnas_find_package(GTest REQUIRED)
# GTest_AddTest(lquantizer_function_test "${TESTS}")
# target_include_directories(lquantizer_function_test PRIVATE include)
# target_link_libraries(lquantizer_function_test nncc_coverage)
