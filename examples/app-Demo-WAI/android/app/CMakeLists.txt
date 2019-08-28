# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(target native-lib)

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
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiSceneGraph.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsDebugTiming.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsTiming.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiStatsVideo.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTrackedMapping.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiTransform.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiUIPrefs.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab/AppDemoGuiVideoStorage.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAI.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDirectories.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAIScene.cpp
    ${WAI_ROOT}/examples/source/WAICalibration.cpp
    ${WAI_ROOT}/examples/source/WAIMapIO.cpp
    ${WAI_ROOT}/examples/source/WAIMapStorage.cpp
    ${WAI_ROOT}/examples/source/WAIUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoAndroidJNI.cpp
    )


add_library(${target}
    SHARED
    ${sources}
    ${headers}
    )

target_include_directories(${target}
    PRIVATE

    PUBLIC
    ${WAI_ROOT}/examples/source
    ${WAI_ROOT}/examples/app-Demo-WAI/source
    ${WAI_ROOT}/examples/app-Demo-WAI/source/Gui
    ${WAI_ROOT}/examples/app-Demo-WAI/source/GuiPrefab
    ${WAI_ROOT}/lib-WAI/source
    ${OpenCV_INCLUDE_DIR}
    ${g2o_INCLUDE_DIR}
    ${WAI_ROOT}/thirdparty/eigen
    ${WAI_ROOT}/thirdparty/DBoW2
    ${DEFAULT_INCLUDE_DIRECTORIES}
    )

target_link_libraries(${target}
    PRIVATE
        lib-WAI
        lib-SLProject

    PUBLIC
        libc++_shared.so
        ${log-lib}
        z

    INTERFACE
    )

#
# Copy APK contents
#
file(GLOB_RECURSE TEXTURES
    ${WAI_ROOT}/thirdparty/SLProject/data/images/textures/CompileError.png
    ${WAI_ROOT}/thirdparty/SLProject/data/images/textures/cursor.png
    ${WAI_ROOT}/thirdparty/SLProject/data/images/textures/LiveVideoError.png
    ${WAI_ROOT}/thirdparty/SLProject/data/images/textures/LogoCPVR_256L.png
    )

file(GLOB_RECURSE VIDEOS
    ${WAI_ROOT}/data/videos/street3.mp4
    )

file(GLOB_RECURSE FONTS
    ${WAI_ROOT}/thirdparty/SLProject/data/images/fonts/*.png
    ${WAI_ROOT}/thirdparty/SLProject/data/images/fonts/*.ttf
    )

# If you add new models you must delete ${CMAKE_CURRENT_LIST_DIR}/src/main/assets
file(GLOB_RECURSE MODELS
    ${WAI_ROOT}/data/models/*
    )

file(GLOB_RECURSE SHADERS
    ${WAI_ROOT}/thirdparty/SLProject/data/shaders/*.vert
    ${WAI_ROOT}/thirdparty/SLProject/data/shaders/*.frag
    )

file(GLOB_RECURSE CALIBRATIONS
    ${WAI_ROOT}/data/calibrations/calib_in_params.yml
    ${WAI_ROOT}/data/calibrations/aruco_detector_params.yml
    ${WAI_ROOT}/data/calibrations/cam_calibration_main.xml
    ${WAI_ROOT}/data/calibrations/ORBvoc.bin
    )

set(APK_ASSETS
    ${CMAKE_CURRENT_LIST_DIR}/src/main/assets)

file(MAKE_DIRECTORY
    ${APK_ASSETS}/images
    ${APK_ASSETS}/images/fonts
    ${APK_ASSETS}/images/textures
    ${APK_ASSETS}/videos
    ${APK_ASSETS}/models
    ${APK_ASSETS}/shaders
    ${APK_ASSETS}/calibrations
    ${APK_ASSETS}/config
    )

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/images/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/images/textures)
file(COPY ${VIDEOS}         DESTINATION ${APK_ASSETS}/videos)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
