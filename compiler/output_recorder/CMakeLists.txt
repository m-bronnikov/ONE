nnas_find_package(HDF5 COMPONENTS STATIC QUIET)

if(NOT HDF5_FOUND)
  message(STATUS "Build output_recorder: FAILED (missing HDF5)")
  return()
endif(NOT HDF5_FOUND)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(output_recorder ${DRIVER} ${SOURCES})
target_include_directories(output_recorder PRIVATE include)
target_include_directories(output_recorder PRIVATE ${HDF5_INCLUDE_DIRS})

target_link_libraries(output_recorder ${HDF5_CXX_LIBRARIES})
target_link_libraries(output_recorder arser)
target_link_libraries(output_recorder safemain)
target_link_libraries(output_recorder luci_import)
target_link_libraries(output_recorder luci_env)
target_link_libraries(output_recorder luci_export)
target_link_libraries(output_recorder luci_interpreter)
target_link_libraries(output_recorder vconone)
target_link_libraries(output_recorder nncc_coverage)

install(TARGETS output_recorder DESTINATION bin)
