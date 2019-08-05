# 
# CMake options downloading and installing prebuilt libs
#

# 
# Download and install prebuilts from pallas.bfh.ch
#

set(OpenCV_VERSION)
set(OpenCV_DIR)
set(OpenCV_LINK_DIR)
set(OpenCV_INCLUDE_DIR)

set(OpenCV_LINK_LIBS
    opencv_aruco
    opencv_calib3d
    opencv_features2d
    opencv_face
    opencv_flann
    opencv_highgui
    opencv_imgcodecs
    opencv_objdetect
    opencv_video
    opencv_imgproc
    opencv_videoio
    opencv_xfeatures2d
    opencv_core
    )

set(OpenCV_LIBS)

set(g2o_DIR)
set(g2o_INCLUDE_DIR)
set(g2o_LINK_DIR)
set(g2o_LINK_LIBS
    g2o_core
    g2o_solver_dense
    g2o_solver_eigen
    g2o_stuff
    g2o_types_sba
    g2o_types_sim3
    g2o_types_slam3d
    g2o_types_slam3d_addons
    #g2o_opengl_helper
    )

set(PREBUILT_PATH "${WAI_ROOT}/thirdparty/prebuilt")
set(PREBUILT_URL "http://pallas.bfh.ch/libs/SLProject/_lib/prebuilt")

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(OpenCV_VERSION "3.4.1")
    set(OpenCV_DIR "${PREBUILT_PATH}/linux_opencv_${OpenCV_VERSION}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_LIBS ${OpenCV_LINK_LIBS})
    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

    set(g2o_DIR ${PREBUILT_PATH}/linux_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE})
    set(g2o_LIBS ${g2o_LINK_LIBS})

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    set(OpenCV_VERSION "3.4.1")
    set(OpenCV_PREBUILT_DIR "win64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/lib")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
    endif ()

    string(REPLACE "." "" OpenCV_LIBS_POSTFIX ${OpenCV_VERSION})

    foreach(lib ${OpenCV_LINK_LIBS})
        set(OpenCV_LIBS
            ${OpenCV_LIBS}
            optimized ${lib}${OpenCV_LIBS_POSTFIX}
            debug ${lib}${OpenCV_LIBS_POSTFIX}d)
    endforeach(lib)

    # Set working dir for VS
    set(DEFAULT_PROJECT_OPTIONS ${DEFAULT_PROJECT_OPTIONS}
        VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    file(GLOB usedCVLibs_Debug
        ${OpenCV_DIR}/lib/opencv_aruco*d.dll
        ${OpenCV_DIR}/lib/opencv_calib3d*d.dll
        ${OpenCV_DIR}/lib/opencv_core*d.dll
        ${OpenCV_DIR}/lib/opencv_features2d*d.dll
        ${OpenCV_DIR}/lib/opencv_face*d.dll
        ${OpenCV_DIR}/lib/opencv_flann*d.dll
        ${OpenCV_DIR}/lib/opencv_highgui*d.dll
        ${OpenCV_DIR}/lib/opencv_imgproc*d.dll
        ${OpenCV_DIR}/lib/opencv_imgcodecs*d.dll
        ${OpenCV_DIR}/lib/opencv_objdetect*d.dll
        ${OpenCV_DIR}/lib/opencv_video*d.dll
        ${OpenCV_DIR}/lib/opencv_videoio*d.dll
        ${OpenCV_DIR}/lib/opencv_xfeatures2d*d.dll
        )
    file(GLOB usedCVLibs_Release
        ${OpenCV_DIR}/lib/opencv_aruco*.dll
        ${OpenCV_DIR}/lib/opencv_calib3d*.dll
        ${OpenCV_DIR}/lib/opencv_core*.dll
        ${OpenCV_DIR}/lib/opencv_features2d*.dll
        ${OpenCV_DIR}/lib/opencv_face*.dll
        ${OpenCV_DIR}/lib/opencv_flann*.dll
        ${OpenCV_DIR}/lib/opencv_highgui*.dll
        ${OpenCV_DIR}/lib/opencv_imgproc*.dll
        ${OpenCV_DIR}/lib/opencv_imgcodecs*.dll
        ${OpenCV_DIR}/lib/opencv_objdetect*.dll
        ${OpenCV_DIR}/lib/opencv_video*.dll
        ${OpenCV_DIR}/lib/opencv_videoio*.dll
        ${OpenCV_DIR}/lib/opencv_xfeatures2d*.dll
        )

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
        file(COPY ${usedCVLibs_Debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${usedCVLibs_Release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    endif()

    #G2O
    set(g2o_DIR ${PREBUILT_PATH}/win64_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/lib)

    foreach(lib ${g2o_LINK_LIBS})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
            IMPORTED_IMPLIB_DEBUG "${g2o_LINK_DIR}/${lib}_d.lib"
            IMPORTED_IMPLIB "${g2o_LINK_DIR}/${lib}.lib"
            IMPORTED_LOCATION_DEBUG "${g2o_LINK_DIR}/${lib}_d.dll"
            IMPORTED_LOCATION "${g2o_LINK_DIR}/${lib}.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${g2o_INCLUDE_DIR}"
        )
        set(g2o_LIBS
            ${g2o_LIBS}
            ${lib}
        )
    endforeach(lib)
    
    set(g2o_PREBUILT_ZIP "win64_g2o.zip")
    set(g2o_URL ${PREBUILT_URL}/${g2o_PREBUILT_ZIP})
      
    if (NOT EXISTS "${g2o_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}" "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
    endif ()

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #-----------------------------
    # Download first for iOS
    set(OpenCV_VERSION "3.4.0")
    set(OpenCV_PREBUILT_DIR "iosV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
    endif ()

    # Now download for MacOS
    set(OpenCV_VERSION "3.4.1")
    set(OpenCV_PREBUILT_DIR "mac64_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
    endif ()

    foreach(lib ${OpenCV_LINK_LIBS})
        set(OpenCV_LIBS
            ${OpenCV_LIBS}
            optimized ${lib}
            debug ${lib})
    endforeach(lib)

    file(GLOB usedCVLibs_Debug
        ${OpenCV_DIR}/Debug/libopencv_aruco*.dylib
        ${OpenCV_DIR}/Debug/libopencv_calib3d*.dylib
        ${OpenCV_DIR}/Debug/libopencv_core*.dylib
        ${OpenCV_DIR}/Debug/libopencv_features2d*.dylib
        ${OpenCV_DIR}/Debug/libopencv_face*.dylib
        ${OpenCV_DIR}/Debug/libopencv_flann*.dylib
        ${OpenCV_DIR}/Debug/libopencv_highgui*.dylib
        ${OpenCV_DIR}/Debug/libopencv_imgproc*.dylib
        ${OpenCV_DIR}/Debug/libopencv_imgcodecs*.dylib
        ${OpenCV_DIR}/Debug/libopencv_objdetect*.dylib
        ${OpenCV_DIR}/Debug/libopencv_video*.dylib
        ${OpenCV_DIR}/Debug/libopencv_videoio*.dylib
        ${OpenCV_DIR}/Debug/libopencv_xfeatures2d*.dylib
        )
    file(GLOB usedCVLibs_Release
        ${OpenCV_DIR}/Release/libopencv_aruco*.dylib
        ${OpenCV_DIR}/Release/libopencv_calib3d*.dylib
        ${OpenCV_DIR}/Release/libopencv_core*.dylib
        ${OpenCV_DIR}/Release/libopencv_features2d*.dylib
        ${OpenCV_DIR}/Release/libopencv_face*.dylib
        ${OpenCV_DIR}/Release/libopencv_flann*.dylib
        ${OpenCV_DIR}/Release/libopencv_highgui*.dylib
        ${OpenCV_DIR}/Release/libopencv_imgproc*.dylib
        ${OpenCV_DIR}/Release/libopencv_imgcodecs*.dylib
        ${OpenCV_DIR}/Release/libopencv_objdetect*.dylib
        ${OpenCV_DIR}/Release/libopencv_video*.dylib
        ${OpenCV_DIR}/Release/libopencv_videoio*.dylib
        ${OpenCV_DIR}/Release/libopencv_xfeatures2d*.dylib
        )

    if(${CMAKE_GENERATOR} STREQUAL Xcode)
        file(COPY ${usedCVLibs_Debug} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
        file(COPY ${usedCVLibs_Release} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    endif()

    #G2O
    set(g2o_DIR ${PREBUILT_PATH}/mac64_g2o)
    set(g2o_PREBUILT_ZIP "mac64_g2o.zip")
    set(g2o_URL ${PREBUILT_URL}/${g2o_PREBUILT_ZIP})
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE})

    #message(STATUS "g2o_DIR: ${g2o_DIR}")
    #message(STATUS "g2o_LINK_DIR: ${g2o_LINK_DIR}")
    #message(STATUS "g2o_URL: ${g2o_URL}")

    if (NOT EXISTS "${g2o_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}" "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
    endif ()

    foreach(lib ${g2o_LINK_LIBS})
        add_library(lib${lib} SHARED IMPORTED)
        set_target_properties(lib${lib} PROPERTIES IMPORTED_LOCATION "${g2o_LINK_DIR}/lib${lib}.dylib")
        #message(STATUS "IMPORTED_LOCATION: ${g2o_LINK_DIR}/lib${lib}.dylib")
        set(g2o_LIBS
            ${g2o_LIBS}
            lib${lib}
            #optimized ${lib}
            #debug ${lib}
            )
    endforeach(lib)

    #message(STATUS "g2o_LIBS: ${g2o_LIBS}")

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID") #---------------------------
    set(OpenCV_VERSION "3.4.1")
    set(OpenCV_PREBUILT_DIR "andV8_opencv_${OpenCV_VERSION}")
    set(OpenCV_DIR "${PREBUILT_PATH}/${OpenCV_PREBUILT_DIR}")
    set(OpenCV_LINK_DIR "${OpenCV_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI}")
    set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
    set(OpenCV_PREBUILT_ZIP "${OpenCV_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${OpenCV_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${OpenCV_PREBUILT_ZIP}" "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${OpenCV_PREBUILT_ZIP}")
    endif ()

    set(OpenCV_LINK_LIBS
        ${OpenCV_LINK_LIBS}
        cpufeatures
        IlmImf
        libjasper
        libjpeg
        libpng
        libprotobuf
        libtiff
        libwebp
        tegra_hal)

    foreach(lib ${OpenCV_LINK_LIBS})
        add_library(lib_${lib} STATIC IMPORTED)
        set_target_properties(lib_${lib} PROPERTIES IMPORTED_LOCATION ${OpenCV_LINK_DIR}/lib${lib}.a)
        set(OpenCV_LIBS
            ${OpenCV_LIBS}
            lib_${lib})
    endforeach(lib)

    set(OpenCV_LIBS_DEBUG ${OpenCV_LIBS})

    #G2O
    set(g2o_PREBUILT_DIR "andV8_g2o")
    set(g2o_DIR ${PREBUILT_PATH}/andV8_g2o)
    set(g2o_INCLUDE_DIR ${g2o_DIR}/include)
    set(g2o_LINK_DIR ${g2o_DIR}/${CMAKE_BUILD_TYPE}/${ANDROID_ABI})
    set(g2o_PREBUILT_ZIP "${g2o_PREBUILT_DIR}.zip")

    if (NOT EXISTS "${g2o_DIR}")
        file(DOWNLOAD "${PREBUILT_URL}/${g2o_PREBUILT_ZIP}" "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
            "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}"
            WORKING_DIRECTORY "${PREBUILT_PATH}")
        file(REMOVE "${PREBUILT_PATH}/${g2o_PREBUILT_ZIP}")
    endif ()

    foreach(lib ${g2o_LINK_LIBS})
        add_library(lib_${lib} SHARED IMPORTED)
        set_target_properties(lib_${lib} PROPERTIES
            IMPORTED_LOCATION "${g2o_LINK_DIR}/lib${lib}.so"
        )
        set(g2o_LIBS
            ${g2o_LIBS}
            lib_${lib}
        )
    endforeach(lib)
endif()
#==============================================================================

link_directories(${OpenCV_LINK_DIR})
link_directories(${g2o_LINK_DIR})
