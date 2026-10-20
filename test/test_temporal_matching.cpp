#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include "mini_vo/dataset.h"
#include "mini_vo/feature.h"
#include "mini_vo/frame.h"
#include "mini_vo/frontend.h"
#include "mini_vo/map.h"
#include "mini_vo/mappoint.h"
#include "mini_vo/stereo_camera.h"
#include "mini_vo/temporal_matcher.h"


int main(int argc, char** argv)
{
    /*
     * ============================================================
     * 1. Dataset
     * ============================================================
     */
    if (argc != 2)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <euroc_dataset_path>\n";

        return 1;
    }


    mini_vo::Dataset dataset(argv[1]);

    if (!dataset.Init())
    {
        std::cerr << "Dataset Init failed.\n";
        return 1;
    }


    auto left_camera = dataset.leftCamera();
    auto right_camera = dataset.rightCamera();

    if (!left_camera || !right_camera)
    {
        std::cerr << "Camera calibration unavailable.\n";
        return 1;
    }


    mini_vo::StereoCamera stereo_camera(
        left_camera,
        right_camera
    );


    /*
     * ============================================================
     * 2. 读取连续的 Frame0 / Frame1
     * ============================================================
     *
     * Dataset::NextFrame() 返回 shared_ptr<Frame>。
     *
     * 两个 Frame 都必须持续存活，
     * 因为后面 TemporalMatcher 会访问它们的 Feature。
     */
    auto frame0 = dataset.NextFrame();
    auto frame1 = dataset.NextFrame();

    if (!frame0 || !frame1)
    {
        std::cerr << "Cannot read two frames.\n";
        return 1;
    }


    /*
     * 第一帧左相机坐标系定义为世界坐标系。
     */
    frame0->setPose(
        Eigen::Matrix4d::Identity()
    );


    /*
     * ============================================================
     * 3. Stereo Rectification
     * ============================================================
     */
    if (!stereo_camera.InitRectification(
            frame0->leftImage().size()))
    {
        std::cerr << "Rectification init failed.\n";
        return 1;
    }


    cv::Mat left0, right0;
    cv::Mat left1, right1;


    if (!stereo_camera.Rectify(
            frame0->leftImage(),
            frame0->rightImage(),
            left0,
            right0))
    {
        std::cerr << "Frame0 rectification failed.\n";
        return 1;
    }


    if (!stereo_camera.Rectify(
            frame1->leftImage(),
            frame1->rightImage(),
            left1,
            right1))
    {
        std::cerr << "Frame1 rectification failed.\n";
        return 1;
    }


    /*
     * ============================================================
     * 4. Frontend
     * ============================================================
     */
    constexpr double min_depth = 0.5;
    constexpr double max_depth = 50.0;


    const double fb =
        stereo_camera.fx()
        * stereo_camera.rectifiedBaseline();


    mini_vo::Frontend frontend(1000);


    frontend.SetStereoMatchingParameters(
        0.75f,
        2.0f,
        static_cast<float>(fb / max_depth),
        static_cast<float>(fb / min_depth)
    );


    /*
     * ============================================================
     * 5. Frame0 Stereo Matching
     * ============================================================
     */
    if (!frontend.ProcessStereo(
            left0,
            right0))
    {
        std::cerr << "Frame0 stereo matching failed.\n";
        return 1;
    }


    /*
     * Frame0 保存自己的 Feature shared_ptr。
     *
     * 以后 Frontend 处理 Frame1 时，
     * Frame0 Feature 不会因此销毁。
     */
    frame0->setLeftFeatures(
        frontend.leftFeatures()
    );

    frame0->setRightFeatures(
        frontend.rightFeatures()
    );


    /*
     * 建立 Feature -> Frame 的弱引用关系。
     */
    for (const auto& feature :
         frame0->leftFeatures())
    {
        if (feature)
        {
            feature->setFrame(frame0);
        }
    }


    for (const auto& feature :
         frame0->rightFeatures())
    {
        if (feature)
        {
            feature->setFrame(frame0);
        }
    }


    /*
     * ============================================================
     * 6. Initial Map
     * ============================================================
     *
     * 本测试只初始化第一帧地图。
     *
     * 后面正式工程中，
     * 应将这部分提取成独立初始化模块，
     * 避免多个测试重复实现。
     */
    mini_vo::Map map;

    unsigned long next_mappoint_id = 0;


    /*
     * 保证一个右目 Feature
     * 不会对应两个不同 MapPoint。
     */
    std::vector<unsigned char> right_used(
        frame0->rightFeatureCount(),
        0
    );


    std::size_t duplicate_right_count = 0;


    for (const auto& match :
         frontend.stereoMatches())
    {
        if (match.queryIdx < 0 ||
            match.trainIdx < 0)
        {
            continue;
        }


        const std::size_t li =
            static_cast<std::size_t>(
                match.queryIdx
            );

        const std::size_t ri =
            static_cast<std::size_t>(
                match.trainIdx
            );


        if (li >= frame0->leftFeatureCount() ||
            ri >= frame0->rightFeatureCount())
        {
            continue;
        }


        if (right_used[ri])
        {
            ++duplicate_right_count;
            continue;
        }


        const auto& left_feature =
            frame0->leftFeatures()[li];

        const auto& right_feature =
            frame0->rightFeatures()[ri];


        if (!left_feature || !right_feature)
        {
            continue;
        }


        const cv::Point2f& pl =
            left_feature->keypoint().pt;

        const cv::Point2f& pr =
            right_feature->keypoint().pt;


        /*
         * --------------------------------------------------------
         * Disparity -> Depth
         * --------------------------------------------------------
         */
        const double disparity =
            static_cast<double>(pl.x)
            - static_cast<double>(pr.x);


        if (!std::isfinite(disparity) ||
            disparity <= 0.0)
        {
            continue;
        }


        const double depth =
            fb / disparity;


        if (!std::isfinite(depth) ||
            depth < min_depth ||
            depth > max_depth)
        {
            continue;
        }


        /*
         * --------------------------------------------------------
         * Rectified pixel -> Pc
         * --------------------------------------------------------
         */
        const Eigen::Vector2d pixel(
            pl.x,
            pl.y
        );


        const Eigen::Vector3d point_camera =
            stereo_camera.PixelToCamera(
                pixel,
                depth
            );


        if (!point_camera.allFinite() ||
            point_camera.z() <= 0.0)
        {
            continue;
        }


        /*
         * --------------------------------------------------------
         * Pc -> Pw
         * --------------------------------------------------------
         *
         * 当前第一帧：
         *
         *      T_cw = I
         *
         * 因此：
         *
         *      Pw = Pc
         */
        const Eigen::Vector3d point_world =
            point_camera;


        /*
         * 创建地图点。
         */
        auto map_point =
            std::make_shared<mini_vo::MapPoint>(
                next_mappoint_id,
                point_world
            );


        /*
         * Map 开始通过 shared_ptr 拥有 MapPoint。
         */
        if (!map.InsertMapPoint(map_point))
        {
            std::cerr << "MapPoint insertion failed.\n";
            return 1;
        }


        /*
         * 建立双目 Feature 对 MapPoint 的观测关系。
         */
        left_feature->setMapPoint(map_point);
        right_feature->setMapPoint(map_point);


        right_used[ri] = 1;
        ++next_mappoint_id;
    }


    std::cout
        << "\n========== Frame0 Initial Map ==========\n"
        << "Stereo matches       : "
        << next_mappoint_id + duplicate_right_count
        << '\n'
        << "Duplicate right      : "
        << duplicate_right_count
        << '\n'
        << "Created MapPoints    : "
        << map.size()
        << '\n';


    if (map.size() < 20)
    {
        std::cerr << "Too few initial MapPoints.\n";
        return 1;
    }


    /*
     * ============================================================
     * 7. Descriptor 生命周期测试
     * ============================================================
     *
     * 在 Frontend 处理 Frame1 之前，
     * 保存 Frame0 中一个已有 MapPoint 的 Feature descriptor。
     */
    mini_vo::Feature::Ptr saved_feature0;


    for (const auto& feature :
         frame0->leftFeatures())
    {
        if (feature &&
            feature->hasMapPoint() &&
            feature->hasDescriptor())
        {
            saved_feature0 = feature;
            break;
        }
    }


    if (!saved_feature0)
    {
        std::cerr << "No valid Frame0 descriptor.\n";
        return 1;
    }


    const cv::Mat descriptor_before =
        saved_feature0->descriptor().clone();


    /*
     * ============================================================
     * 8. Frontend 处理 Frame1
     * ============================================================
     *
     * 这里复用同一个 Frontend。
     *
     * 它内部的 left/right descriptors 会变成
     * Frame1 的内容。
     *
     * 但 Frame0::Feature 自己持有的 descriptor
     * 不应该受到影响。
     */
    if (!frontend.ExtractStereoFeatures(
            left1,
            right1))
    {
        std::cerr << "Frame1 feature extraction failed.\n";
        return 1;
    }


    frame1->setLeftFeatures(
        frontend.leftFeatures()
    );

    frame1->setRightFeatures(
        frontend.rightFeatures()
    );


    for (const auto& feature :
         frame1->leftFeatures())
    {
        if (feature)
        {
            feature->setFrame(frame1);
        }
    }


    for (const auto& feature :
         frame1->rightFeatures())
    {
        if (feature)
        {
            feature->setFrame(frame1);
        }
    }


    /*
     * ============================================================
     * 9. 验证 Frame0 Descriptor 仍然有效
     * ============================================================
     */
    if (!saved_feature0->hasDescriptor())
    {
        std::cerr << "Frame0 descriptor lost.\n";
        return 1;
    }


    const double descriptor_difference =
        cv::norm(
            descriptor_before,
            saved_feature0->descriptor(),
            cv::NORM_HAMMING
        );


    if (descriptor_difference != 0.0)
    {
        std::cerr << "Frame0 descriptor changed.\n";
        return 1;
    }


    std::cout
        << "\nDescriptor lifetime test: passed\n";


    /*
     * ============================================================
     * 10. Temporal Matching V1
     * ============================================================
     */
    mini_vo::TemporalMatcher temporal_matcher(
        0.75f
    );


    const int temporal_count =
        temporal_matcher.Match(
            *frame0,
            *frame1
        );


    const auto& stats =
        temporal_matcher.stats();


    const auto& temporal_matches =
        temporal_matcher.matches();


    std::cout
        << "\n========== Temporal Matching V1 ==========\n"
        << "Reference Frame ID  : "
        << frame0->id()
        << '\n'
        << "Current Frame ID    : "
        << frame1->id()
        << '\n'
        << '\n'
        << "Reference Features  : "
        << stats.reference_features
        << '\n'
        << "With MapPoint       : "
        << stats.reference_with_mappoint
        << '\n'
        << "Reference Descriptor: "
        << stats.reference_descriptors
        << '\n'
        << "Current Features    : "
        << stats.current_features
        << '\n'
        << "Current Descriptor  : "
        << stats.current_descriptors
        << '\n'
        << '\n'
        << "KNN Candidates      : "
        << stats.knn_candidates
        << '\n'
        << "Ratio Passed        : "
        << stats.ratio_passed
        << '\n'
        << "Unique Passed       : "
        << stats.unique_passed
        << '\n'
        << "Final Matches       : "
        << temporal_count
        << '\n';


    /*
     * ============================================================
     * 11. 构造 PnP 所需的 3D-2D 对应关系
     * ============================================================
     *
     * 这一步只收集数据，
     * 暂时不调用 solvePnPRansac()。
     *
     *   object_points:
     *       MapPoint 世界坐标 Pw
     *
     *   image_points:
     *       Frame1 校正左图像素
     */
    std::vector<cv::Point3f>
        object_points;


    std::vector<cv::Point2f>
        image_points;


    object_points.reserve(
        temporal_matches.size()
    );

    image_points.reserve(
        temporal_matches.size()
    );


    /*
     * 检查一个 Current Feature
     * 是否被重复匹配。
     */
    std::vector<unsigned char> current_used(
        frame1->leftFeatureCount(),
        0
    );


    double distance_sum = 0.0;


    for (const auto& match :
         temporal_matches)
    {
        /*
         * 当前 matcher 输出的索引，
         * 应该已经映射回原始 Feature 容器。
         */
        if (match.queryIdx < 0 ||
            match.trainIdx < 0)
        {
            std::cerr << "Negative match index.\n";
            return 1;
        }


        const std::size_t reference_index =
            static_cast<std::size_t>(
                match.queryIdx
            );


        const std::size_t current_index =
            static_cast<std::size_t>(
                match.trainIdx
            );


        if (reference_index >=
                frame0->leftFeatureCount() ||
            current_index >=
                frame1->leftFeatureCount())
        {
            std::cerr << "Invalid temporal match index.\n";
            return 1;
        }


        if (current_used[current_index])
        {
            std::cerr << "Duplicate current Feature.\n";
            return 1;
        }


        current_used[current_index] = 1;


        const auto& reference_feature =
            frame0->leftFeatures()[reference_index];


        const auto& current_feature =
            frame1->leftFeatures()[current_index];


        if (!reference_feature ||
            !current_feature)
        {
            std::cerr << "Null matched Feature.\n";
            return 1;
        }


        /*
         * 从 Frame0 Feature 找到地图点。
         */
        auto map_point =
            reference_feature->mapPoint();


        if (!map_point)
        {
            std::cerr << "Reference MapPoint expired.\n";
            return 1;
        }


        const Eigen::Vector3d& Pw =
            map_point->position();


        if (!Pw.allFinite())
        {
            std::cerr << "Invalid world point.\n";
            return 1;
        }


        /*
         * 3D：世界坐标系
         */
        object_points.emplace_back(
            static_cast<float>(Pw.x()),
            static_cast<float>(Pw.y()),
            static_cast<float>(Pw.z())
        );


        /*
         * 2D：Frame1 rectified left pixel
         */
        image_points.push_back(
            current_feature->keypoint().pt
        );


        /*
         * 这里只统计 Hamming distance。
         */
        distance_sum += match.distance;


        /*
         * 特别注意：
         *
         * 当前 Feature1 还不能直接绑定 MapPoint。
         *
         * 因为 descriptor matching 可能存在错误匹配。
         *
         * 必须等下一步 PnP RANSAC 确认是 inlier，
         * 才能正式建立关联。
         */
        if (current_feature->hasMapPoint())
        {
            std::cerr
                << "Current Feature was unexpectedly "
                << "associated with a MapPoint.\n";

            return 1;
        }
    }


    const double mean_hamming =
        temporal_matches.empty()
        ? 0.0
        : distance_sum /
            static_cast<double>(
                temporal_matches.size()
            );


    std::cout
        << "\n========== PnP Input Preparation ==========\n"
        << "3D World Points  : "
        << object_points.size()
        << '\n'
        << "2D Image Points  : "
        << image_points.size()
        << '\n'
        << "Mean Hamming     : "
        << mean_hamming
        << '\n';


    /*
     * PnP 至少需要一定数量的 3D-2D 对应。
     *
     * 对本次测试要求至少 6 个，
     * 但实际稳定跟踪通常希望明显多于这个数量。
     */
    if (object_points.size() < 6 ||
        object_points.size() != image_points.size())
    {
        std::cerr
            << "Insufficient or inconsistent "
            << "3D-2D correspondences.\n";

        return 1;
    }


    /*
     * ============================================================
     * 12. 准备可视化 KeyPoint
     * ============================================================
     *
     * 不能再使用：
     *
     *   frontend.leftKeypoints()
     *
     * 来获取 Frame0 KeyPoint，
     * 因为 Frontend 已经处理了 Frame1。
     *
     * 所以从持久化的 Frame Feature 中恢复。
     */
    std::vector<cv::KeyPoint>
        keypoints0;

    std::vector<cv::KeyPoint>
        keypoints1;


    keypoints0.reserve(
        frame0->leftFeatureCount()
    );

    keypoints1.reserve(
        frame1->leftFeatureCount()
    );


    for (const auto& feature :
         frame0->leftFeatures())
    {
        keypoints0.push_back(
            feature->keypoint()
        );
    }


    for (const auto& feature :
         frame1->leftFeatures())
    {
        keypoints1.push_back(
            feature->keypoint()
        );
    }


    /*
     * ============================================================
     * 13. Draw Temporal Matches
     * ============================================================
     */
    cv::Mat match_image;


    cv::drawMatches(
        left0,
        keypoints0,

        left1,
        keypoints1,

        temporal_matches,

        match_image,

        cv::Scalar::all(-1),
        cv::Scalar::all(-1),

        std::vector<char>(),

        cv::DrawMatchesFlags::
            NOT_DRAW_SINGLE_POINTS
    );


    cv::imshow(
        "Temporal Matching V1",
        match_image
    );


    std::cout
        << "\nTemporal Matching V1 test passed.\n"
        << "Press any key to exit.\n";


    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}