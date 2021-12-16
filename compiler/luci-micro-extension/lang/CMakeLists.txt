file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(luci_micro_lang SHARED ${SOURCES})
target_include_directories(luci_micro_lang PRIVATE src)
target_include_directories(luci_micro_lang PUBLIC include)

target_link_libraries(luci_micro_lang PUBLIC luci_lang)

install(TARGETS luci_micro_lang DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

if(NOT ENABLE_TEST)
    return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_micro_lang_test ${TESTS})
target_include_directories(luci_micro_lang_test PRIVATE src)
target_link_libraries(luci_micro_lang_test luci_micro_lang)
