# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(target native-lib)

file(GLOB headers
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGui.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosDialog.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiTrackedMapping.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiMapStorage.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosMapNodeTransform.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosTracking.h
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAISceneView.h
    ${WAI_ROOT}/examples/source/WAIFileSystem.h
    ${WAI_ROOT}/examples/source/WAIMapIO.h
    ${WAI_ROOT}/examples/source/WAIMapStorage.h
    ${WAI_ROOT}/examples/source/WAIUtils.h
    )

file(GLOB sources
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGui.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosDialog.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiTrackedMapping.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiMapStorage.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppWAISceneView.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosMapNodeTransform.cpp
    ${WAI_ROOT}/examples/app-Demo-WAI/source/AppDemoGuiInfosTracking.cpp
    ${WAI_ROOT}/examples/source/WAIFileSystem.cpp
    ${WAI_ROOT}/examples/source/WAIMapIO.cpp
    ${WAI_ROOT}/examples/source/WAIMapStorage.cpp
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
    ${WAI_ROOT}/data/images/textures/CompileError.png
    ${WAI_ROOT}/data/images/textures/brick0512_C.png
    ${WAI_ROOT}/data/images/textures/brick*.jpg
    ${WAI_ROOT}/data/images/textures/Checkerboard0512_C.png
    ${WAI_ROOT}/data/images/textures/Chess0256_C.bmp
    ${WAI_ROOT}/data/images/textures/earth*.jpg
    ${WAI_ROOT}/data/images/textures/grass0512_C.jpg
    ${WAI_ROOT}/data/images/textures/gray_0256_C.jpg
    ${WAI_ROOT}/data/images/textures/MuttenzerBox*.png
    ${WAI_ROOT}/data/images/textures/rusty-metal_2048*.png
    ${WAI_ROOT}/data/images/textures/Pool*.png
    ${WAI_ROOT}/data/images/textures/Testmap_0512_C.png
    ${WAI_ROOT}/data/images/textures/tile1_0256_C.jpg
    ${WAI_ROOT}/data/images/textures/tree1_1024_C.png
    ${WAI_ROOT}/data/images/textures/wood*.jpg
    ${WAI_ROOT}/data/images/textures/cursor.png
    ${WAI_ROOT}/data/images/textures/Vision*.png
    ${WAI_ROOT}/data/images/textures/LiveVideoError.png
    ${WAI_ROOT}/data/images/textures/LogoCPVR_256L.png
    ${WAI_ROOT}/data/images/textures/features_stones.png
    ${WAI_ROOT}/data/images/textures/Desert*_C.jpg
    ${WAI_ROOT}/data/images/textures/i*_0000b.png
    )

file(GLOB_RECURSE VIDEOS
    ${WAI_ROOT}/data/videos/street3.mp4
    )

file(GLOB_RECURSE FONTS
    ${WAI_ROOT}/data/images/fonts/*.png
    ${WAI_ROOT}/data/images/fonts/*.ttf
    )

# If you add new models you must delete ${CMAKE_CURRENT_LIST_DIR}/src/main/assets
file(GLOB_RECURSE MODELS
    ${WAI_ROOT}/data/models/*
    )

file(GLOB_RECURSE SHADERS
    ${WAI_ROOT}/data/shaders/*.vert
    ${WAI_ROOT}/data/shaders/*.frag
    )

file(GLOB_RECURSE CALIBRATIONS
    ${WAI_ROOT}/data/calibrations/calib_in_params.yml
    ${WAI_ROOT}/data/calibrations/aruco_detector_params.yml
    ${WAI_ROOT}/data/calibrations/street1_manip.json
    ${WAI_ROOT}/data/calibrations/orb-slam-state-altstadtbiel1_manip.json
    ${WAI_ROOT}/data/calibrations/orb-slam-state-bern1-manip1.json
    ${WAI_ROOT}/data/calibrations/orb-slam-state-buero1.json
    ${WAI_ROOT}/data/calibrations/test.yml
    ${WAI_ROOT}/data/calibrations/cam_calibration_main_huawei_p10_640_480.xml
    ${WAI_ROOT}/data/calibrations/ORBvoc.bin
    )

file(GLOB_RECURSE CONFIG
    ${WAI_ROOT}/data/config/dummyFile.txt
    )

set(APK_ASSETS
    ${CMAKE_CURRENT_LIST_DIR}/src/main/assets)

file(MAKE_DIRECTORY
    ${APK_ASSETS}/fonts
    ${APK_ASSETS}/textures
    ${APK_ASSETS}/videos
    ${APK_ASSETS}/models
    ${APK_ASSETS}/shaders
    ${APK_ASSETS}/calibrations
    ${APK_ASSETS}/config
    )

file(COPY ${FONTS}          DESTINATION ${APK_ASSETS}/fonts)
file(COPY ${TEXTURES}       DESTINATION ${APK_ASSETS}/textures)
file(COPY ${VIDEOS}         DESTINATION ${APK_ASSETS}/videos)
file(COPY ${MODELS}         DESTINATION ${APK_ASSETS}/models)
file(COPY ${SHADERS}        DESTINATION ${APK_ASSETS}/shaders)
file(COPY ${CALIBRATIONS}   DESTINATION ${APK_ASSETS}/calibrations)
file(COPY ${CONFIG}         DESTINATION ${APK_ASSETS}/config)
