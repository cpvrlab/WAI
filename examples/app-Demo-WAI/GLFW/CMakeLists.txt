# 
# CMake configuration for app-Demo-WAI application
#

set(target app-Demo-WAI)

set(compile_definitions
    SL_STARTSCENE=SID_Minimal)

file(GLOB headers
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGui.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosDialog.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiTrackedMapping.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiMapStorage.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosMapNodeTransform.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosTracking.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAISceneView.h
    )

file(GLOB sources
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGui.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosDialog.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiTrackedMapping.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiMapStorage.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosMapNodeTransform.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAISceneView.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosTracking.cpp
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
    ${WAI_ROOT}/examples/app-Demo-WAI/source/

    PUBLIC

    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${META_PROJECT_NAME}::lib-WAI
    lib-SLProject

    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}

    PUBLIC
    ${DEFAULT_COMPILE_DEFINITIONS}
    WAI_DLL
    WAI_DLL_IMPORTS

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
