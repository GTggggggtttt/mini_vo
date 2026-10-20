cmake_minimum_required(VERSION 3.10)

project(mini_vo)


# ================================================================
# C++ 标准
# ================================================================

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)


# ================================================================
# 编译警告
# ================================================================

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
)


# ================================================================
# 查找第三方库
# ================================================================

find_package(Eigen3 REQUIRED)

# Frame 中已经使用 cv::Mat，
# 因此现在正式引入 OpenCV。
find_package(OpenCV REQUIRED)


# ================================================================
# mini_vo 核心库
# ================================================================
add_library(mini_vo
    src/camera.cpp
    src/frame.cpp
    src/dataset.cpp
    src/stereo_camera.cpp
    src/frontend.cpp
    src/mappoint.cpp
    src/map.cpp
    src/feature.cpp
    src/temporal_matcher.cpp
    src/pnp_tracker.cpp
    src/map_initializer.cpp
    src/multi_frame_tracker.cpp
    src/stereo_map_updater.cpp
    src/keyframe.cpp
    src/keyframe_selector.cpp
)


# ================================================================
# 头文件目录
# ================================================================

target_include_directories(mini_vo
    PUBLIC

    ${PROJECT_SOURCE_DIR}/include

    ${OpenCV_INCLUDE_DIRS}
)


# ================================================================
# mini_vo 依赖
# ================================================================

target_link_libraries(mini_vo
    PUBLIC

    Eigen3::Eigen

    ${OpenCV_LIBS}
)


# ================================================================
# Camera Test
# ================================================================

add_executable(test_camera
    test/test_camera.cpp
)


target_link_libraries(test_camera
    PRIVATE

    mini_vo
)


# ================================================================
# Frame Test
# ================================================================

add_executable(test_frame
    test/test_frame.cpp
)


target_link_libraries(test_frame
    PRIVATE

    mini_vo
)

#dataset_test
add_executable(test_dataset
test/test_dataset.cpp)

target_link_libraries(test_dataset
 PRIVATE
 mini_vo
)

# ================================================================
# Calibration Test
# ================================================================
add_executable(test_calibration
test/test_calibration.cpp)

target_link_libraries(test_calibration
PRIVATE
mini_vo
)

# ================================================================
# Stereo Rectification Test
# ================================================================

add_executable(
    test_stereo_rectification
    test/test_stereo_rectification.cpp
)

target_link_libraries(
    test_stereo_rectification
    PRIVATE
    mini_vo
)

# ================================================================
# Feature Test
# ================================================================

add_executable(
    test_feature
    test/test_feature.cpp
)

target_link_libraries(
    test_feature
    PRIVATE
    mini_vo
)


# ================================================================
# Stereo Matching Test
# ================================================================

add_executable(
    test_stereo_matching
    test/test_stereo_matching.cpp
)

target_link_libraries(
    test_stereo_matching
    PRIVATE
    mini_vo
)

# ================================================================
# Stereo Depth Test
# ================================================================

add_executable(
    test_stereo_depth
    test/test_stereo_depth.cpp
)

target_link_libraries(
    test_stereo_depth
    PRIVATE
    mini_vo
)

# ================================================================
# Stereo Matching V2 Test
# ================================================================

add_executable(
    test_stereo_matchingv2
    test/test_stereo_matchingv2.cpp
)

target_link_libraries(
    test_stereo_matchingv2
    PRIVATE
    mini_vo
)

add_executable(
    test_feature_relation
    test/test_feature_relation.cpp
)

target_link_libraries(
    test_feature_relation
    PRIVATE
    mini_vo
)

# ================================================================
# Initial Map Test
# ================================================================
add_executable(test_map_initialization
test/test_map_initialization.cpp)

target_link_libraries(
    test_map_initialization
    PRIVATE
    mini_vo
)

# ================================================================
# test_feature_descriptor
# ================================================================
add_executable(
    test_feature_descriptor
    test/test_feature_descriptor.cpp
)

target_link_libraries(
    test_feature_descriptor
    PRIVATE
    mini_vo
)

# ================================================================
# Temporal Matching V1
# ================================================================

add_executable(
    test_temporal_matching
    test/test_temporal_matching.cpp
)

target_link_libraries(
    test_temporal_matching
    PRIVATE
    mini_vo
)

# ================================================================
# PnP Tracking V1
# ================================================================

add_executable(
    test_pnp_tracking
    test/test_pnp_tracking.cpp
)

target_link_libraries(
    test_pnp_tracking
    PRIVATE
    mini_vo
)


# ================================================================
# Multi-frame Tracking V1
# ================================================================

add_executable(
    test_multi_frame_tracking
    test/test_multi_frame_tracking.cpp
)

target_link_libraries(
    test_multi_frame_tracking
    PRIVATE
    mini_vo
)

# ================================================================
# Multi-frame Tracking V2
# ================================================================

add_executable(
    test_multi_frame_tracking_v2
    test/test_multi_frame_tracking_v2.cpp
)

target_link_libraries(
    test_multi_frame_tracking_v2
    PRIVATE
    mini_vo
)

# ================================================================
# KeyFrame V1 Unit Test
# ================================================================

add_executable(
    test_keyframe
    test/test_keyframe.cpp
)

target_link_libraries(
    test_keyframe
    PRIVATE
    mini_vo
)


# ================================================================
# KeyFrame Tracking Integration Test
# ================================================================

add_executable(
    test_keyframe_tracking
    test/test_keyframe_tracking.cpp
)

target_link_libraries(
    test_keyframe_tracking
    PRIVATE
    mini_vo
)
