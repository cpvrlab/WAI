#
# CMake configuration for app-Demo-WAI application
#

set(target app-Demo-WAI)

set(compile_definitions
    SL_STARTSCENE=SID_Minimal)

file(GLOB headers
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/AppDemoGui.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/AppDemoGuiInfosDialog.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/AppDemoGuiMenu.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/AppDemoGuiPrefs.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiAbout.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosFrameworks.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosMapNodeTransform.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosScene.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosSensors.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosTracking.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiMapStorage.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiProperties.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSceneGraph.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsDebugTiming.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsTiming.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsVideo.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTrackedMapping.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTransform.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiUIPrefs.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiVideoStorage.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAI.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDirectories.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAIScene.h
    ${WAI_ROOT}/examples/source/WAICalibration.h
    ${WAI_ROOT}/examples/source/WAIMapIO.h
    ${WAI_ROOT}/examples/source/WAIMapStorage.h
    ${WAI_ROOT}/examples/source/WAIUtils.h
    )

file(GLOB sources
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/AppDemoGui.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/AppDemoGuiInfosDialog.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/AppDemoGuiMenu.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/AppDemoGuiPrefs.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiAbout.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosFrameworks.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosMapNodeTransform.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosScene.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosSensors.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiInfosTracking.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiMapStorage.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiProperties.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSceneGra.cpp.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsDebugTiming.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsTiming.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsVideo.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTrackedMapping.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTransform.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiUIPrefs.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiVideoStorage.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAI.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAIScene.cpp
    ${WAI_ROOT}/examples/source/WAICalibration.cpp
    ${WAI_ROOT}/examples/source/WAIMapIO.cpp
    ${WAI_ROOT}/examples/source/WAIMapStorage.cpp
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
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui/
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/
    ${WAI_ROOT}/examples/source
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
