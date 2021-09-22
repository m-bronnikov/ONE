include(${MICRO_PAL_DIR}/pal.cmake)

initialize_pal()

if (NOT PAL_INITIALIZED)
  return()
endif()

message(STATUS "MICRO BEGIN")

set(MICRO_BINARY "micro${MICRO_SUFFIX}")
set(MICRO_CORE "micro_core${MICRO_SUFFIX}")
set(MICRO_KERNELS "micro_kernels${MICRO_SUFFIX}")
set(MICRO_LOADER "micro_loader${MICRO_SUFFIX}")

add_subdirectory(core)
message(STATUS "MICRO CORE")
add_subdirectory(kernels)
message(STATUS "MICRO KERNELS")
add_subdirectory(loader)
message(STATUS "MICRO LOADER")
add_subdirectory(direct_loader)
message(STATUS "MICRO DIRECT LOADER")

message(STATUS "LUCI INTERPTER INITALIZED")

set(SOURCES
    "${MICRO_INCLUDE_DIR}/micro/Interpreter.h"
    Interpreter.cpp "${MICRO_INCLUDE_DIR}/micro/SimpleMemoryManager.h" SimpleMemoryManager.cpp
        "${MICRO_INCLUDE_DIR}/micro/TestMemoryManager.h" TestMemoryManager.cpp)

if (NOT MICRO_STATIC)
  add_library(${MICRO_BINARY} SHARED ${SOURCES})
else ()
  add_library(${MICRO_BINARY} STATIC ${SOURCES})
endif ()

target_include_directories(${MICRO_BINARY} PUBLIC "${MICRO_INCLUDE_DIR}")
target_include_directories(${MICRO_BINARY} PRIVATE "${MICRO_SOURCE_DIR}")
target_link_libraries(${MICRO_BINARY}
    PUBLIC luci_lang ${MICRO_LOADER} ${MICRO_CORE}
    PRIVATE nncc_common)

install(TARGETS ${MICRO_BINARY} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
