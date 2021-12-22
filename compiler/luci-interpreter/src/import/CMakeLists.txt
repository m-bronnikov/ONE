set(LUCI_INTERPRETER_IMPORT_INCLUDE_DIR "${LUCI_INTERPRETER_INCLUDE_DIR}/luci_interpreter/import")

set(SOURCES
    "${LUCI_INTERPRETER_IMPORT_INCLUDE_DIR}/GraphBuilderRegistry.h"
    GraphBuilderRegistry.cpp)

# include specific builders headers
file(GLOB_RECURSE NODES "${LUCI_INTERPRETER_IMPORT_INCLUDE_DIR}/Nodes/*.h")
list(APPEND SOURCES ${NODES})
# include specific builders sources
file(GLOB_RECURSE NODES "Nodes/*.cpp")
list(APPEND SOURCES ${NODES})

add_library(${LUCI_INTERPRETER_IMPORT} STATIC ${SOURCES})
if (NOT NNCC_LIBRARY_NO_PIC)
  set_target_properties(${LUCI_INTERPRETER_IMPORT} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(NOT NNCC_LIBRARY_NO_PIC)

target_include_directories(${LUCI_INTERPRETER_IMPORT} PUBLIC "${LUCI_INTERPRETER_IMPORT_INCLUDE_DIR}")
target_link_libraries(${LUCI_INTERPRETER_IMPORT} PUBLIC luci_import)
