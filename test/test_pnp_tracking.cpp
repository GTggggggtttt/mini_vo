#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mini_vo/dataset.h"
#include "mini_vo/feature.h"
#include "mini_vo/frame.h"
#include "mini_vo/frontend.h"
#include "mini_vo/map.h"
#include "mini_vo/mappoint.h"
#include "mini_vo/pnp_tracker.h"
#include "mini_vo/stereo_camera.h"
#include "mini_vo/temporal_matcher.h"

/*
 * ================================================================
 * InitializeFirstFrameMap 专门用于准备第一帧地图
 * ================================================================
 *
 * 测试辅助函数：
 *
 *   Frame0
 *     ↓
 *   Stereo Matching
 *     ↓
 *   Disparity / Depth
 *     ↓
 *   MapPoint Pw
 *     ↓
 *   Initial Map
 *
 *
 * 注意：
 *
 *   这是为了让 test_pnp_tracking.cpp
 *   能独立运行。
 *
 *   暂时不是正式生产用 MapInitializer。
 */

bool InitializeFirstFrameMap(const mini_vo::Frame::Ptr &frame0,
                              const cv::Mat &left_rectified,
                              const cv::Mat &right_rectified,
                              const mini_vo::StereoCamera &stereo_camera,
                              mini_vo::Frontend &frontend,
                              mini_vo::Map &map)
{
    // 1.第一帧定义世界坐标系
    frame0->setPose(Eigen::Matrix4d::Identity());

    // 2.stereo matching  双目匹配恢复深度
    if (!frontend.ProcessStereo(left_rectified, right_rectified))
    {
        return false;
    }

    // 3.frame0 保存 Feature 相当于第一帧的初始化 提取第一批特征点
    frame0->setLeftFeatures(frontend.leftFeatures());
    frame0->setRightFeatures(frontend.rightFeatures());

    // 3.2 建议feature -> Frame弱引用
    for (const auto &feature : frame0->leftFeatures())
    {
        if (feature)
        {
            feature->setFrame(frame0);
        }
    }
    for (const auto &feature : frame0->rightFeatures())
    {
        if (feature)
        {
            feature->setFrame(frame0);
        }
    }

    // 4.stereo geometry
    constexpr double min_depth = 0.5;
    constexpr double max_depth = 50.0;

    const double fb = stereo_camera.fx() * stereo_camera.rectifiedBaseline();

    /*
     * 每个 Right Feature 最多用于
     * 初始化一个 MapPoint。
     */
    std::vector<unsigned char> right_used(frame0->rightFeatureCount(), 0);
    unsigned long next_map_point_id = 0;
    std::size_t duplicate_right = 0;

    // 5.stereo_match -> mappoint
    for (const auto &match : frontend.stereoMatches())
    {
        if (match.queryIdx < 0 || match.trainIdx < 0)
        {
            continue;
        }

        // 这两个是啥来着 10.11--19.26
        const std::size_t li = static_cast<std::size_t>(match.queryIdx);
        const std::size_t ri = static_cast<std::size_t>(match.trainIdx);

        if (li >= frame0->leftFeatureCount() || ri >= frame0->rightFeatureCount())
        {
            continue;
        }

        if (right_used[ri])
        {
            ++duplicate_right;
            continue;
        }

        const auto &left_feature = frame0->leftFeatures()[li];
        const auto &right_feature = frame0->rightFeatures()[ri];

        if (!left_feature || !right_feature)
        {
            continue;
        }

        // 像素坐标
        const cv::Point2f &pl = left_feature->keypoint().pt;
        const cv::Point2f &pr = right_feature->keypoint().pt;

        /*
         * Disparity:
         *
         *   d = uL - uR
         */
        const double disparity = static_cast<double>(pl.x) - static_cast<double>(pr.x);

        if (!std::isfinite(disparity) || disparity <= 0.0)
        {
            continue;
        }

        // 计算depth Z= fb/d
        const double depth = fb / disparity;
        if (!std::isfinite(depth) || depth < min_depth || depth > max_depth)
        {
            continue;
        }

        /*
         * Rectified Pixel -> Pc
         */
        const Eigen::Vector2d pixel(pl.x, pl.y);
        const Eigen::Vector3d Pc = stereo_camera.PixelToCamera(pixel, depth);

        if (!Pc.allFinite() || Pc.z() <= 0.0)
        {
            continue;
        }

        /*
         * 第一帧：
         * 第一帧的左目坐标就是世界坐标系
         *   T_cw = I
         *
         * 所以：
         *
         *   Pw = Pc
         */
        const Eigen::Vector3d Pw = Pc;
        auto map_point = std::make_shared<mini_vo::MapPoint>(next_map_point_id, Pw);

        if (!map.InsertMapPoint(map_point))
        {
            std::cerr << "MapPoint insertion failed." << std::endl;
            return false;
        }

        // 双目观测关联同一个MapPoint
        left_feature->setMapPoint(map_point);
        right_feature->setMapPoint(map_point);
        right_used[ri] = 1;
        ++next_map_point_id;
    }

    std::cout
        << "\n========== Initial Map ==========\n"
        << "Stereo Matches    : "
        << frontend.stereoMatches().size()
        << '\n'
        << "Duplicate Right   : "
        << duplicate_right
        << '\n'
        << "Created MapPoints : "
        << map.size()
        << '\n';

    return map.size() >= 20;
}

// main函数
int main(int argc, char **argv)
{
    // 1.dataset
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << "<euroc_dataset_path>\n";
        return 1;
    }

    mini_vo::Dataset dataset(argv[1]);

    if (!dataset.Init())
    {
        std::cerr << "Dataset initialization failed.\n";
        return 1;
    }

    // 2. Carmera / StereoCamera
    auto left_camera = dataset.leftCamera();
    auto right_camera = dataset.rightCamera();

    if (!left_camera || !right_camera)
    {
        std::cerr << "Camera calibration unavailable.\n";
        return 1;
    }

    mini_vo::StereoCamera stereo_camera(left_camera, right_camera);

    // 3.读取连续的两帧
    auto frame0 = dataset.NextFrame();
    auto frame1 = dataset.NextFrame();
    if (!frame0 || !frame1)
    {
        std::cerr << "Cannot read two frames.\n";
        return 1;
    }

    // 4.stereo rectification
    if (!stereo_camera.InitRectification(frame0->leftImage().size()))
    {
        std::cerr << "Rectification init failed.\n";
        return 1;
    }

    cv::Mat left0, right0;
    cv::Mat left1, right1;

    if (!stereo_camera.Rectify(frame0->leftImage(), frame0->rightImage(), left0, right0))
    {
        std::cerr << "Frame0 rectification failed.\n";
        return 1;
    }

    if (!stereo_camera.Rectify(frame1->leftImage(), frame1->rightImage(), left1, right1))
    {
        std::cerr << "Frame1 rectification failed.\n";
        return 1;
    }

    // 5.进入前端流程 frontend
    constexpr double min_depth = 0.5;
    constexpr double max_depth = 50.0;

    const double fb = stereo_camera.fx() * stereo_camera.rectifiedBaseline();

    mini_vo::Frontend frontend(1000);

    frontend.SetStereoMatchingParameters(0.75f, 2.0f, static_cast<float>(fb / max_depth), static_cast<float>(fb / min_depth));

    // 6. 初始化Frame0m map
    mini_vo::Map map;

    if (!InitializeFirstFrameMap(frame0, left0, right0, stereo_camera, frontend, map))
    {
        std::cerr << "Initial Map failed.\n";
        return 1;
    }

    /*
     * ============================================================
     * 7. 提取 Frame1 左图 ORB
     * ============================================================
     *
     * PnP 只需要当前左图上的 2D Feature。
     *
     * 因此这里不需要对 Frame1 再做
     * Stereo Matching。
     *
     * ExtractFeatures() 使用 Frontend 单图接口。
     */
    const int current_feature_count = frontend.ExtractFeatures(left1);
    if (current_feature_count <= 0)
    {
        std::cerr << "Frame1 ORB Extraction failed.\n";
        return 1;
    }

    // Frame1 开始拥有这些featrue
    frame1->setLeftFeatures(frontend.features());

    // 反过来 featrue->弱引用
    for (const auto &feature : frame1->leftFeatures())
    {
        if (feature)
        {
            feature->setFrame(frame1);
        }
    }

    // 8.Temporal Matching
    mini_vo::TemporalMatcher temporal_matcher(0.75f);

    const int temporal_count = temporal_matcher.Match(*frame0, *frame1);

    const auto &temporal_stats = temporal_matcher.stats();

    std::cout
        << "\n========== Temporal Matching ==========\n"
        << "Reference With MapPoint : "
        << temporal_stats.reference_with_mappoint
        << '\n'
        << "KNN Candidates          : "
        << temporal_stats.knn_candidates
        << '\n'
        << "Ratio Passed            : "
        << temporal_stats.ratio_passed
        << '\n'
        << "Unique Passed           : "
        << temporal_stats.unique_passed
        << '\n'
        << "Temporal Matches        : "
        << temporal_count
        << '\n';

    if (temporal_count < 6)
    {
        std::cerr << "Too few temporal matches.\n";
        return 1;
    }

    /*
     * ============================================================
     * 9. PnP Tracking V1
     * ============================================================
     */
    mini_vo::PnPConfig config;
    config.min_correspondences = 6;
    config.min_inliers = 15;
    config.ransac_iterations = 200;
    config.reprojection_error_px = 3.0;
    config.confidence = 0.99;
    config.min_inlier_ratio = 0.5;

    mini_vo::PnPTracker pnp_tracker(config);

    /*
     * 核心调用：
     *
     * Frame0 MapPoint Pw
     *            +
     * Frame1 2D Feature
     *            ↓
     *       solvePnPRansac
     *            ↓
     *         T_cw(1)
     */

    const bool tracking_success = pnp_tracker.Track(*frame0, *frame1, temporal_matcher.matches(), stereo_camera);

    const auto &stats = pnp_tracker.stats();

    /*
     * ============================================================
     * 10. PnP Statistics
     * ============================================================
     */
    std::cout
        << "\n========== PnP Tracking V1 ==========\n"
        << "Temporal Matches       : "
        << stats.temporal_matches
        << '\n'
        << "Valid 3D-2D Pairs      : "
        << stats.valid_correspondences
        << '\n'
        << "RANSAC Inliers         : "
        << stats.ransac_inliers
        << '\n'
        << "Verified Inliers       : "
        << stats.final_inliers
        << '\n'
        << "Inlier Ratio           : "
        << stats.inlier_ratio
        << '\n'
        << "Mean Reprojection Error: "
        << stats.mean_reprojection_error_px
        << " px\n"
        << "Max Reprojection Error : "
        << stats.max_reprojection_error_px
        << " px\n";

    if (!tracking_success)
    {
        std::cerr << "\n PnP Tracking failed: " << pnp_tracker.lastError() << std::endl;
        return 1;
    }

    // 11.检查Frame1 Pose
    const Eigen::Matrix4d &T_cw = frame1->pose();
    if (!T_cw.allFinite())
    {
        std::cerr << "Frame1 has invaild Pose.\n";
        return 1;
    }

    std::cout << "\nFrame1 T_cw:\n"
              << T_cw << '\n';

    /*
     * ============================================================
     * 12. Camera Center in World
     * ============================================================
     *
     * Frame::pose() 保存：
     *
     *   T_cw : World -> Camera
     *
     * 但相机在世界中的位姿是：
     *
     *   T_wc = inverse(T_cw)
     *
     * 所以相机中心世界坐标：
     *
     *   C_w = T_wc.translation()
     */
    const Eigen::Matrix4d T_wc = T_cw.inverse();
    const Eigen::Vector3d camera_center_world = T_wc.block<3, 1>(0, 3);

    std::cout << "\nFrame1 Camera Center (World):\n"
              << camera_center_world.transpose() << '\n';

    /*
     * 因为 Frame0 定义为世界坐标原点，
     * 所以此时 Camera Center 的 norm
     * 是从 Frame0 到 Frame1 的平移距离。
     */
    const double translation_distance = camera_center_world.norm();
    std::cout << "Fram0 -> Frame1 Translation: " << translation_distance << "m\n";

    /*
     * ============================================================
     * 13. 验证 Inlier Feature -> MapPoint
     * ============================================================
     *
     * 每一条 PnP Inlier Match：
     *
     *   Frame0 Feature
     *      -> MapPoint
     *
     *   Frame1 Feature
     *      -> 同一个 MapPoint
     *
     * 必须一致。
     */
    std::size_t verified_relations = 0;
    for (const auto &match : pnp_tracker.inlierMatches())
    {
        const auto &feature0 = frame0->leftFeatures().at(static_cast<std::size_t>(match.queryIdx));
        const auto &feature1 = frame1->leftFeatures().at(static_cast<std::size_t>(match.trainIdx));

        const auto map_point0 = feature0->mapPoint();
        const auto map_point1 = feature1->mapPoint();

        if (!map_point0 || !map_point1 || map_point0.get() != map_point1.get())
        {
            std::cerr << "Inlier MapPoint association mismatch.\n";
            return 1;
        }

        ++verified_relations;
    }

    if (verified_relations != stats.final_inliers)
    {
        std::cerr << "Inlier relation count mismatch.\n";
        return 1;
    }
    std::cout << "\nVerified MapPoint Relations: " << verified_relations << '\n';

    /*
     * ============================================================
     * 14. 从 Frame Feature 恢复 KeyPoint
     * ============================================================
     *
     * 不再依赖 Frontend 当前的 KeyPoint 容器。
     */
    std::vector<cv::KeyPoint> keypoints0;
    std::vector<cv::KeyPoint> keypoints1;

    keypoints0.reserve(frame0->leftFeatureCount());
    keypoints1.reserve(frame1->leftFeatureCount());

    for (const auto &feature : frame0->leftFeatures())
    {
        keypoints0.push_back(feature->keypoint());
    }

    for (const auto &feature : frame1->leftFeatures())
    {
        keypoints1.push_back(feature->keypoint());
    }

    // 15.可视化ransac inlier matches
    cv::Mat match_image;
    cv::drawMatches(left0, keypoints0, left1, keypoints1, pnp_tracker.inlierMatches(), match_image, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const std::string text = "PnP Inliers:" + std::to_string(stats.final_inliers);

    cv::putText(match_image, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::imshow("PnP Tracking V1 -RANSAC Inliers", match_image);

    std::cout
        << "\n=====================================\n"
        << "PnP Tracking V1 passed.\n"
        << "Press any key to exit.\n"
        << "=====================================\n";

        cv::waitKey(0);
        cv::destroyAllWindows();
        return 0;
}
