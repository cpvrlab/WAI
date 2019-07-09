cmake_minimum_required (VERSION 3.3 FATAL_ERROR)

project (FeatureMatching C CXX)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_BUILD_TYPE Debug)

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/app.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtractKeypoints.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gui_tools.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/matching.h
    ${CMAKE_CURRENT_SOURCE_DIR}/orb_descriptor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools.h
    )
file(GLOB sources 
    ${CMAKE_CURRENT_SOURCE_DIR}/FeatureMatching.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtractKeypoints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gui_tools.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matching.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/orb_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools.cpp
    )

add_executable( FeatureMatching ${sources} ${headers} )
target_link_libraries( FeatureMatching ${OpenCV_LIBS} )

