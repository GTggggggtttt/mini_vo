#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mini_vo/dataset.h"
#include "mini_vo/frame.h"
#include "mini_vo/feature.h"
#include "mini_vo/frontend.h"
#include "mini_vo/map.h"
#include "mini_vo/mappoint.h"
#include "mini_vo/stereo_camera.h"

int main(int argc, char **argv)
{
    // 1.command line
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << "<euroc_dataset_path>" << std::endl;
        return 1;
    }

    const std::string dataset_path = argv[1];

    /*
     * ============================================================
     * 2. Dataset Initialization
     * ============================================================
     */
    mini_vo::Dataset dataset(dataset_path);
    if (!dataset.Init())
    {
        std::cerr << "Dataset initialization failed." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 3. Camera Calibration
     * ============================================================
     */
    auto left_camera = dataset.leftCamera();
    auto right_camera = dataset.rightCamera();

    if (!left_camera || !right_camera)
    {
        std::cerr << "Camera calibration unavailable." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 4. StereoCamera
     * ============================================================
     */
    mini_vo::StereoCamera stereo_camera(left_camera, right_camera);

    /*
     * ============================================================
     * 5. 读取第一帧
     * ============================================================
     *
     * 当前 Initial Map 只使用第一帧。
     */
    mini_vo::Frame::Ptr frame = dataset.NextFrame();
    if (!frame)
    {
        std::cerr << "Cannot read first frame." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 6. 定义世界坐标系
     * ============================================================
     *
     * 我们规定：
     *
     *      第一帧左相机坐标系
     *
     *          ==
     *
     *      世界坐标系
     *
     *
     * 因此：
     *
     *      T_cw = I
     *
     *
     * 满足：
     *
     *      Pc = T_cw * Pw
     *
     * 所以第一帧：
     *
     *      Pc = Pw
     */
    frame->setPose(Eigen::Matrix4d::Identity());

    /*
     * ============================================================
     * 7. 初始化 Stereo Rectification
     * ============================================================
     */
    if (!stereo_camera.InitRectification(frame->leftImage().size()))
    {
        std::cerr << "Stereo rectification initialization failed." << std::endl;
        return 1;
    }
    /*
     * ============================================================
     * 8. Stereo Matching 参数
     * ============================================================
     *
     * 当前接受深度范围：
     *
     *      0.5m ~ 50m
     *
     *
     * 根据：
     *
     *      Z = fb / d
     *
     * 得：
     *
     *      d = fb / Z
     */
    constexpr double min_depth = 0.5;
    constexpr double max_depth = 50.0;

    const double fb = stereo_camera.fx() * stereo_camera.rectifiedBaseline();
    const float min_disparity = static_cast<float>(fb / max_depth);
    const float max_disparity = static_cast<float>(fb / min_depth);

    /*
     * ============================================================
     * 9. Frontend
     * ============================================================
     */
    mini_vo::Frontend frontend(1000);
    frontend.SetStereoMatchingParameters(0.75f, 2.0f, min_disparity, max_disparity);

    /*
     * ============================================================
     * 10. Stereo Rectification
     * ============================================================
     */
    cv::Mat left_rectified;
    cv::Mat right_rectified;
    if (!stereo_camera.Rectify(frame->leftImage(), frame->rightImage(), left_rectified, right_rectified))
    {
        std::cerr << "Stereo rectification failed." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 11. Stereo Feature Extraction + Matching
     * ============================================================
     */
    if (!frontend.ProcessStereo(left_rectified, right_rectified))
    {
        std::cerr << "Stereo frontend failed." << std::endl;
        return 1;
    }

    /*
     * Frontend 中当前这一帧产生的：
     *
     *      Feature
     *      KeyPoint
     *      Stereo Matches
     */
    const auto &left_features = frontend.leftFeatures();
    const auto &right_features = frontend.rightFeatures();
    const auto &left_keypoints = frontend.leftKeypoints();
    const auto &right_keypoints = frontend.rightKeypoints();
    const auto &stereo_matches = frontend.stereoMatches();
    const auto &match_stats = frontend.stereoMatchStats();
    /*
     * ============================================================
     * 12. 基础一致性检查
     * ============================================================
     *
     * Frontend 必须保持：
     *
     *      features[i]
     *          ↕
     *      keypoints[i]
     *          ↕
     *      descriptor row(i)
     */
    if (left_features.size() != left_keypoints.size())
    {
        std::cerr << "Left Feature / KeyPoint size mismatch." << std::endl;
        return 1;
    }
    if (right_features.size() != right_keypoints.size())
    {
        std::cerr << "Right Feature / KeyPoint size mismatch." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 13. 建立 Feature -> Frame 关系
     * ============================================================
     *
     * Feature 内部保存：
     *
     *      weak_ptr<Frame>
     *
     * 所以这里不会让 Feature 拥有 Frame。
     */
    for (const auto &feature : left_features)
    {
        if (feature)
        {
            feature->setFrame(frame);
        }
    }

    for (const auto &feature : right_features)
    {
        if (feature)
        {
            feature->setFrame(frame);
        }
    }

    /*
     * ============================================================
     * 14. Frame 开始拥有 Feature
     * ============================================================
     *
     * Frame 中保存：
     *
     *      shared_ptr<Feature>
     *
     *
     * 所以 ownership：
     *
     *      Frame
     *        |
     *        | shared_ptr
     *        v
     *      Feature
     *
     *
     * Feature 反向：
     *
     *      weak_ptr<Frame>
     */
    frame->setLeftFeatures(left_features);
    frame->setRightFeatures(right_features);

    /*
     * ============================================================
     * 15. 创建 Map
     * ============================================================
     *
     * Map 将成为 MapPoint 的主要 owner。
     */
    mini_vo::Map map;

    /*
     * ============================================================
     * 16. MapPoint ID
     * ============================================================
     *
     * 当前测试从 0 开始顺序分配。
     *
     * 后面正式 Frontend / Map 初始化类
     * 再统一管理 ID。
     */
    unsigned long next_mappoint_id = 0;

    /*
     * ============================================================
     * 17. Right Feature 使用标记
     * ============================================================
     *
     * 这一点非常重要。
     *
     * 当前：
     *
     *      knnMatch(left -> right)
     *
     * 并不能保证：
     *
     *      一个 right feature
     *
     * 只被一个 left feature 匹配。
     *
     *
     * 如果：
     *
     *      Left A -> Right X
     *      Left B -> Right X
     *
     * 然后我们分别创建：
     *
     *      MapPoint 1
     *      MapPoint 2
     *
     * 那么 Right X 的：
     *
     *      setMapPoint()
     *
     * 会被第二次覆盖，
     * 产生数据关联不一致。
     *
     *
     * 所以初始化地图时，
     * 第一版明确要求 right feature 唯一。
     */
    std::vector<bool> right_feature_used(right_features.size(), false);

    /*
     * 接受用于初始化地图的 matches。
     *
     * 后面用于 drawMatches 可视化。
     */
    std::vector<cv::DMatch> accepted_matches;
    accepted_matches.reserve(stereo_matches.size());

    /*
     * ============================================================
     * 18. 统计数据
     * ============================================================
     */
    std::size_t invalid_index_count = 0;
    std::size_t duplicate_right_count = 0;
    std::size_t invalid_depth_count = 0;
    std::size_t inserted_map_points = 0;
    double depth_sum = 0.0;
    double min_valid_depth = std::numeric_limits<double>::max();
    double max_valid_depth = 0.0;
    double reprojection_error_sum = 0.0;
    double max_reprojection_error = 0.0;

    /*
     * ============================================================
     * 19. Stereo Match -> MapPoint
     * ============================================================
     */
    for (const auto &match : stereo_matches) // 依次取出stereo_matches中的每个元素，并且把
    {
        /*
         * --------------------------------------------------------
         * 19.1 检查 OpenCV match index
         * --------------------------------------------------------
         */
        if (match.queryIdx < 0 || match.trainIdx < 0)
        {
            ++invalid_index_count;
            continue;
        }

        const std::size_t left_index = static_cast<std::size_t>(match.queryIdx);
        const std::size_t right_index = static_cast<std::size_t>(match.trainIdx);
        if (left_index >= left_features.size() || left_index >= left_keypoints.size() ||
            right_index >= right_features.size() || right_index >= right_keypoints.size())
        {
            ++invalid_index_count;
            continue;
        }

        /*
         * --------------------------------------------------------
         * 19.2 一个 Right Feature 只初始化一个 MapPoint
         * --------------------------------------------------------
         */
        if (right_feature_used[right_index])
        {
            ++duplicate_right_count;
            continue;
        }

        const auto &left_feature = left_features[left_index];
        const auto &right_feature = right_features[right_index];
        if (!left_feature || !right_feature)
        {
            continue;
        }

        /*
         * 当前是新 Frame，
         * 理论上 Feature 还不应该关联 MapPoint。
         */
        if (left_feature->hasMapPoint() || right_feature->hasMapPoint())
        {
            continue;
        }

        /*
         * --------------------------------------------------------
         * 19.3 获取左右像素
         * --------------------------------------------------------
         */
        const cv::Point2f &left_pixel = left_keypoints[left_index].pt;
        const cv::Point2f &right_pixel = right_keypoints[right_index].pt;

        /*
         * --------------------------------------------------------
         * 19.4 Disparity
         * --------------------------------------------------------
         *
         * rectified stereo：
         *
         *      d = uL - uR
         */
        const double disparity = static_cast<double>(left_pixel.x - right_pixel.x);
        if (!std::isfinite(disparity) || disparity <= 0.0)
        {
            ++invalid_depth_count;
            continue;
        }

        /*
         * --------------------------------------------------------
         * 19.5 Depth
         * --------------------------------------------------------
         *
         *          fx * baseline
         *      Z = -------------
         *           disparity
         */
        const double depth = fb / disparity;
        if (!std::isfinite(depth) || depth < min_depth || depth > max_depth)
        {
            ++invalid_depth_count;
            continue;
        }

        /*
         * --------------------------------------------------------
         * 19.6 Rectified pixel -> Pc  像素坐标到相机坐标
         * --------------------------------------------------------
         */
        const Eigen::Vector2d pixel(static_cast<double>(left_pixel.x), static_cast<double>(left_pixel.y));
        const Eigen::Vector3d point_camera = stereo_camera.PixelToCamera(pixel, depth);

        // point 必须位于相机正前方
        if (!point_camera.allFinite() || point_camera.z() <= 0.0)
        {
            ++invalid_depth_count;
            continue;
        }

        /*
         * ========================================================
         * 19.7 Pc -> Pw
         * ========================================================
         *
         * Pose 约定：
         *
         *      Pc = T_cw * Pw
         *
         * 所以一般情况下：
         *
         *      Pw = inverse(T_cw) * Pc
         *
         *
         * 当前第一帧：
         *
         *      T_cw = I
         *
         * 所以数学上：
         *
         *      Pw = Pc
         *
         *
         * 这里仍然写完整变换，
         * 是为了明确坐标系语义。
         */
        Eigen::Vector4d point_camera_h;
        point_camera_h << point_camera, 1.0;

        const Eigen::Matrix4d T_wc = frame->pose().inverse();
        const Eigen::Vector4d point_world_h = T_wc * point_camera_h;

        /*
         * 刚体齐次变换的 w 应该仍然为 1。
         *
         * 这里仍做标准齐次归一化。
         */
        if (std::abs(point_world_h.w()) < 1e-12)
        {
            continue;
        }

        const Eigen::Vector3d point_world = point_world_h.head<3>() / point_world_h.w();

        /*
         * --------------------------------------------------------
         * 19.8 创建 MapPoint
         * --------------------------------------------------------
         *
         * MapPoint.position_
         *
         * 必须始终保存：
         *
         *      Pw
         */
        auto map_point = std::make_shared<mini_vo::MapPoint>(next_mappoint_id, point_world);

        /*
         * --------------------------------------------------------
         * 19.9 插入 Map
         * --------------------------------------------------------
         *
         * 插入成功以后：
         *
         *      Map
         *
         * 通过 shared_ptr 拥有 MapPoint。
         */
        if (!map.InsertMapPoint(map_point))
        {
            std::cerr << "Failed to insert MapPoint " << next_mappoint_id << std::endl;
            continue;
        }

        /*
         * --------------------------------------------------------
         * 19.10 建立 Feature -> MapPoint association
         * --------------------------------------------------------
         *
         * 内部是 weak_ptr：
         *
         *      Feature 不拥有 MapPoint。
         */
        left_feature->setMapPoint(map_point);
        right_feature->setMapPoint(map_point);

        /*
         * 标记当前 right feature 已经使用。
         */
        right_feature_used[right_index] = true;

        /*
         * --------------------------------------------------------
         * 19.11 重投影验证
         * --------------------------------------------------------
         *
         * MapPoint 保存的是 Pw。
         *
         * 根据：
         *
         *      Pc = T_cw * Pw
         *
         * 再投影回左图。
         */
        Eigen::Vector4d point_world_check_h;
        point_world_check_h << map_point->position(), 1.0;
        const Eigen::Vector4d point_camera_check_h = frame->pose() * point_world_check_h;

        if (std::abs(point_camera_check_h.w()) < 1e-12)
        {
            continue;
        }

        const Eigen::Vector3d point_camera_check = point_camera_check_h.head<3>() / point_camera_check_h.w();
        const Eigen::Vector2d reprojected_pixel = stereo_camera.CameraToPixel(point_camera_check);
        const double reprojection_error = (pixel - reprojected_pixel).norm();

        /*
         * --------------------------------------------------------
         * 19.12 Statistics
         * --------------------------------------------------------
         */
        depth_sum += depth;
        min_valid_depth = std::min(min_valid_depth, depth);
        max_valid_depth = std::max(max_valid_depth, depth);
        reprojection_error_sum += reprojection_error;
        max_reprojection_error = std::max(max_reprojection_error, reprojection_error);
        accepted_matches.push_back(match);

        ++inserted_map_points;
        ++next_mappoint_id;
    }
        /*
         * ============================================================
         * 20. 统计结果
         * ============================================================
         */
        double mean_depth = 0.0;
        double mean_reprojection_error = 0.0;
        if (inserted_map_points > 0)
        {
            mean_depth = depth_sum / static_cast<double>(inserted_map_points);
            mean_reprojection_error = reprojection_error_sum / static_cast<double>(inserted_map_points);
        }
        else
        {
            min_valid_depth = 0.0;
        }

        /*
         * ============================================================
         * 21. 输出 Pipeline
         * ============================================================
         */
        std::cout
            << "\n=============================================\n"
            << "           Initial Map Test\n"
            << "=============================================\n";

        std::cout << "\nFrame ID: " << frame->id() << '\n';
        std::cout << "Timestamp: " << frame->timestamp() << '\n';
        std::cout << "\nT_cw:\n"
                  << frame->pose() << '\n';
        std::cout << "\nStereo Camera: \n";
        std::cout << "fx     =" << stereo_camera.fx() << '\n';
        std::cout << "baseline = " << stereo_camera.rectifiedBaseline() << '\n';
        std::cout << "f*b  = " << fb << '\n';
        std::cout << "\nStereo Matching Pipeline:\n";
        std::cout << "Left Features      :" << match_stats.left_features << '\n';
        std::cout << "Right Features      :" << match_stats.right_features << '\n';
        std::cout << "KNN Candidates       :" << match_stats.knn_candidates << '\n';
        std::cout << "Ratio Passed          :" << match_stats.ratio_passed << '\n';
        std::cout << "Epipolar Passed          :" << match_stats.epipolar_passed << '\n';
        std::cout << "Disparity Passed          :" << match_stats.disparity_passed << '\n';
        std::cout << "\nFrame Ownership:\n";
        std::cout << "Frame Left Features : " << frame->leftFeatureCount() << '\n';
        std::cout << "Frame right Features :" << frame->rightFeatureCount() << '\n';
        std::cout << "\nMap Initialization:\n";
        std::cout << "Stereo Matches        : " << stereo_matches.size() << '\n';
        std::cout << "Invalid Index         : " << invalid_index_count << '\n';
        std::cout << "Duplicate Right       : " << duplicate_right_count << '\n';
        std::cout << "Invalid Depth         : " << invalid_depth_count << '\n';
        std::cout << "Created MapPoints     : " << inserted_map_points << '\n';
        std::cout << "Map::size()           : " << map.size() << '\n';
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nDepth Statistics:\n";
        std::cout << "Min Depth  : " << min_valid_depth << " m\n";
        std::cout << "Mean Depth : " << mean_depth << " m\n";
        std::cout << "Max Depth  : " << max_valid_depth << " m\n";
        std::cout << "\nReprojection:\n";
        std::cout << "Mean Error : " << mean_reprojection_error << " px\n";
        std::cout << "Max Error  : " << max_reprojection_error << " px\n";

        /*
         * ============================================================
         * 22. 基本验收
         * ============================================================
         */
        if (map.empty())
        {
            std::cerr << "\nInitial map is empty." << std::endl;
            return 1;
        }

        if (map.size() != inserted_map_points)
        {
            std::cerr << "\nMap size mismatch." << std::endl;
            return 1;
        }

        /*
         * 这里没有规定：
         *
         *      必须 100 个
         *      必须 300 个
         *
         * 因为具体数量随帧和参数变化。
         *
         * 但是至少应该有一定数量的有效初始化点。
         */
        constexpr std::size_t minimum_map_points = 20;
        if (map.size() < minimum_map_points)
        {
            std::cerr << "\nToo few initial MapPoints: " << map.size() << std::endl;
            return 1;
        }

        /*
         * 我们当前这个重投影属于：
         *
         * pixel
         *   -> depth
         *   -> Pc
         *   -> Pw
         *   -> Pc
         *   -> pixel
         *
         * 在第一帧 Identity Pose 下，
         * 应该非常接近 0。
         */
        if (mean_reprojection_error > 1e-6)
        {
            std::cerr << "\nUnexpected reprojection error. Please check." << std::endl;
            return 1;
        }

        /*
         * ============================================================
         * 23. 打印前几个 MapPoint
         * ============================================================
         */
        std::cout << "\nFirst MapPoints:\n";
        std::size_t print_count = 0;
        constexpr std::size_t max_print_points = 10;

        for (const auto &item : map.mapPoints())
        {
            /*
             * unordered_map 中：
             *
             * item.first
             *      = MapPoint ID
             *
             * item.second
             *      = shared_ptr<MapPoint>
             */
            const auto &map_point = item.second;
            if (!map_point)
            {
                continue;
            }
            std::cout << "MapPoint " << map_point->id() << " | Pw = [" << map_point->position().transpose() << "]\n";

            ++print_count;
            if (print_count >= max_print_points)
            {
                break;
            }
        }

        /*
         * ============================================================
         * 24. 验证 Feature -> Frame / MapPoint
         * ============================================================
         *
         * 找第一个已经建立地图关联的 Left Feature。
         */
        bool relation_test_passed = false;
        for (const auto &feature : frame->leftFeatures())
        {
            if (!feature || !feature->hasMapPoint())
            {
                continue;
            }

            auto feature_frame = feature->frame();
            auto feature_map_point = feature->mapPoint();
            if (!feature_frame || !feature_map_point)
            {
                continue;
            }

            std::cout << "\nRelation Test:\n";
            std::cout << "Feature belongs to Frame " << feature_frame->id() << '\n';
            std::cout << "Feature observes MapPoint " << feature_map_point->id() << '\n';

            relation_test_passed = true;
            break;
        }

        if (!relation_test_passed)
        {
            std::cerr << "\nFeature relation test failed." << std::endl;
            return 1;
        }

        /*
         * ============================================================
         * 25. 可视化 Initial Map 使用的 Stereo Matches
         * ============================================================
         */
        cv::Mat match_image;
        cv::drawMatches(left_rectified, left_keypoints, right_rectified, right_keypoints, accepted_matches, match_image, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        const std::string text = " Initial MapPoints: " + std::to_string(map.size());
        cv::putText(match_image, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Initial Map Stereo Matches", match_image);

        /*
         * ============================================================
         * 26. 在左图显示建立出来的 MapPoint observation
         * ============================================================
         */
        cv::Mat map_image;
        cv::cvtColor(left_rectified, map_image, cv::COLOR_GRAY2BGR);
        for (const auto &match : accepted_matches)
        {
            const cv::Point2f &point = left_keypoints[static_cast<std::size_t>(match.queryIdx)].pt;
            cv::circle(map_image, point,3, cv::Scalar(0, 255, 0), -1);
        }

        cv::putText(map_image,text,cv::Point(20,30),cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(0,255,0),2);
        cv::imshow("Initial Map Points",map_image);
        
        
        
        std::cout<<"\n============================================\n"
        <<"Initial map initialization passed.\n"
        <<"Press any key to exit. \n"
        <<"======================================================";
        
        cv::waitKey(0);
        cv::destroyAllWindows();
        
        
        return 0 ;
    
}
