# 
# CMake congifuration for lib-WAI
#

set(target lib-WAI)

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAI.h
    )

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/source/WAI.cpp
    )

add_library(${target}
    ${headers}
    ${sources}
    )

add_library(${META_PROJECT_NAME}::${target}
    ALIAS
    ${target}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    )

target_include_directories(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/source

    INTERFACE
    )

target_compile_definitions(${target}
	PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_DEFINITIONS}

    INTERFACE
    )

target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}

    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
    )
