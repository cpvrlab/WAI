# 
# CMake configuration for app-Demo-SLProject application
#

set(target app-Demo-SLProject)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions
    SL_STARTSCENE=SID_Minimal)

file(GLOB headers
    ${WAI_ROOT}/examples/app-Demo-SLProject/include/AppDemoGui.h
    )

file(GLOB sources
    ${WAI_ROOT}/examples/app-Demo-SLProject/source/AppDemoGui.cpp
    ${WAI_ROOT}/examples/app-Demo-SLProject/source/AppDemoLoad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoMainGLFW.cpp
    )

add_executable(${target}
    ${headers}
    ${sources}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "examples"
    )

target_include_directories(${target}
    PRIVATE
    ${WAI_ROOT}/examples/app-Demo-SLProject/include/
    PUBLIC
    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE
    lib-SLProject
    PUBLIC
    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}
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

