file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(luci_micro_import SHARED ${SOURCES})
target_include_directories(luci_micro_import PRIVATE src)
target_include_directories(luci_micro_import PUBLIC include)

target_link_libraries(luci_micro_import PUBLIC luci_import)
target_link_libraries(luci_micro_import PRIVATE luci_log)
target_link_libraries(luci_micro_import PRIVATE luci_micro_lang)

install(TARGETS luci_micro_import DESTINATION lib)
install(DIRECTORY include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
