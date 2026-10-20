#include <iostream>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "mini_vo/frame.h"
#include "mini_vo/keyframe.h"
#include "mini_vo/keyframe_selector.h"
#include "mini_vo/map.h"


void Check(
    bool condition,
    const char* message
)
{
    /*
     * 不使用 assert()：
     *
     * 因为 Release + NDEBUG 构建
     * 可能禁用 assert 检查。
     */
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}


int main()
{
    /*
     * ============================================================
     * Test 1: Map owns KeyFrame
     * ============================================================
     */
    mini_vo::Map map;

    std::weak_ptr<mini_vo::Frame>
        weak_frame;


    {
        auto frame0 =
            std::make_shared<mini_vo::Frame>(
                0,
                0.0,
                cv::Mat{},
                cv::Mat{}
            );


        weak_frame = frame0;


        auto keyframe0 =
            std::make_shared<mini_vo::KeyFrame>(
                0,
                frame0
            );


        Check(
            map.InsertKeyFrame(keyframe0),
            "InsertKeyFrame failed."
        );


        Check(
            !map.InsertKeyFrame(keyframe0),
            "Duplicate KeyFrame was accepted."
        );


        Check(
            map.keyFrameCount() == 1,
            "Incorrect KeyFrame count."
        );


        Check(
            map.GetKeyFrame(0)->frameId() == 0,
            "Incorrect source Frame ID."
        );


        /*
         * 销毁局部 shared_ptr。
         *
         * Map 仍然持有 KeyFrame，
         * KeyFrame 仍然持有 Frame。
         */
        frame0.reset();

        keyframe0.reset();
    }


    /*
     * 因为 Map -> KeyFrame -> Frame
     * 仍然存在强引用链，所以 Frame 没有销毁。
     */
    Check(
        !weak_frame.expired(),
        "Frame was destroyed while KeyFrame is in Map."
    );


    /*
     * Map::Clear() 释放持有的 KeyFrame。
     *
     * 当前测试没有其它 KeyFrame / Frame 强引用。
     */
    map.Clear();


    Check(
        weak_frame.expired(),
        "Frame lifetime did not end after Map::Clear()."
    );


    std::cout
        << "KeyFrame ownership test passed.\n";


    /*
     * ============================================================
     * Test 2: KeyFrame Selection
     * ============================================================
     */
    auto reference_frame =
        std::make_shared<mini_vo::Frame>(
            0,
            0.0,
            cv::Mat{},
            cv::Mat{}
        );


    mini_vo::KeyFrame reference_kf(
        0,
        reference_frame
    );


    mini_vo::KeyFrameSelectionConfig config;

    config.min_frame_gap = 2;

    config.max_frame_gap = 6;

    config.min_translation_m = 0.10;

    config.min_rotation_rad = 0.15;

    config.low_inlier_threshold = 100;


    mini_vo::KeyFrameSelector selector(
        config
    );


    mini_vo::PnPTrackingStats stats;

    stats.success = true;

    stats.final_inliers = 200;


    /*
     * Frame1：
     *
     * 即使运动很大，
     * 也不允许小于 min_frame_gap。
     */
    mini_vo::Frame frame1(
        1,
        0.1,
        cv::Mat{},
        cv::Mat{}
    );


    Eigen::Matrix4d pose =
        Eigen::Matrix4d::Identity();

    pose(0, 3) = -0.20;

    frame1.setPose(pose);


    Check(
        !selector.ShouldInsert(
            reference_kf,
            frame1,
            stats
        ),
        "Minimum Frame gap was ignored."
    );


    /*
     * Frame2：
     *
     * 满足 min_frame_gap，
     * 且相机中心已经平移 0.2 m。
     */
    mini_vo::Frame frame2(
        2,
        0.2,
        cv::Mat{},
        cv::Mat{}
    );


    frame2.setPose(pose);


    Check(
        selector.ShouldInsert(
            reference_kf,
            frame2,
            stats
        ),
        "Translation threshold did not trigger."
    );


    /*
     * Frame3：
     *
     * 没有明显运动，
     * Inlier 数量较多，
     * 不需要新 KeyFrame。
     */
    mini_vo::Frame frame3(
        3,
        0.3,
        cv::Mat{},
        cv::Mat{}
    );


    Check(
        !selector.ShouldInsert(
            reference_kf,
            frame3,
            stats
        ),
        "Unnecessary KeyFrame insertion."
    );


    /*
     * 跟踪内点数量降低。
     */
    stats.final_inliers = 60;


    Check(
        selector.ShouldInsert(
            reference_kf,
            frame3,
            stats
        ),
        "Low Inlier condition did not trigger."
    );


    /*
     * PnP 失败：
     * 即使满足插入条件，也不能成为 KeyFrame。
     */
    stats.success = false;


    Check(
        !selector.ShouldInsert(
            reference_kf,
            frame3,
            stats
        ),
        "Failed tracking created a KeyFrame."
    );


    std::cout
        << "KeyFrame selector test passed.\n";


    std::cout
        << "\nAll KeyFrame V1 tests passed.\n";


    return 0;
}