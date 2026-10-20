#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mini_vo/dataset.h"
#include "mini_vo/frame.h"
#include "mini_vo/multi_frame_tracker.h"
#include "mini_vo/stereo_camera.h"

/*
 * ================================================================
 * GetCameraCenterWorld
 * ================================================================
 *
 * Frame 保存：
 *
 *      T_cw
 *
 * 相机世界位姿：
 *
 *      T_wc = inverse(T_cw)
 *
 * 世界坐标系中的相机中心：
 *
 *      C_w = T_wc.translation()
 */

Eigen::Vector3d GetCameraCenterWorld(const mini_vo::Frame &frame)
{
    const Eigen::Matrix4d T_wc = frame.pose().inverse();
    return T_wc.block<3, 1>(0, 3);
}

/*
 * ================================================================
 * WriteTumPose
 * ================================================================
 *
 * TUM Trajectory 格式：
 *
 * timestamp tx ty tz qx qy qz qw
 *
 * 注意：
 *
 * 这里写入的是 T_wc：
 *
 *      Camera -> World
 *
 * 而不是 PnP 直接得到的 T_cw。
 *
 * timestamp 使用 Dataset 已转换的秒。
 */
void WriteTumPose(std::ofstream &file, const mini_vo::Frame &frame)
{
    const Eigen::Matrix4d T_wc = frame.pose().inverse();
    const Eigen::Vector3d translation = T_wc.block<3, 1>(0, 3);
    const Eigen::Matrix3d rotation = T_wc.block<3, 3>(0, 0);

    Eigen::Quaterniond quaternion(rotation);
    quaternion.normalize();

    file << std::fixed << std::setprecision(9) << frame.timestamp() << " " << translation.x() << " "
         << translation.y() << " " << translation.z() << " "
         << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << " " << quaternion.w() << "\n";
}

/*
 * ================================================================
 * main
 * ================================================================
 */
int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " <euroc_dataset_path> [frame_count]\n";
        return 1;
    }

    std::size_t requested_frames = 30;

    if (argc == 3)
    {
        try
        {
            std::size_t consumed = 0;
            const std::string argument = argv[2];

            // 只接受十进制数字，避免负数被转换为很大的无符号整数。
            if (argument.empty() ||
                argument.find_first_not_of("0123456789") != std::string::npos)
            {
                throw std::invalid_argument("Frame count must contain only digits.");
            }

            const unsigned long long frame_count = std::stoull(argument, &consumed);

            if (consumed != argument.size() || frame_count < 2 ||
                frame_count > std::numeric_limits<std::size_t>::max())
            {
                throw std::invalid_argument("Frame count is out of range.");
            }

            requested_frames = static_cast<std::size_t>(frame_count);
        }
        catch (const std::exception &)
        {
            std::cerr << "Invalid frame count." << '\n';
            return 1;
        }
    }

    // 1. Dataset
    mini_vo::Dataset dataset(argv[1]);
    if (!dataset.Init())
    {
        std::cerr << "Dataset initialization failed.\n";
        return 1;
    }

    // 2.Stereo Camera
    auto camera_left = dataset.leftCamera();
    auto camera_right = dataset.rightCamera();

    if (!camera_left || !camera_right)
    {
        std::cerr << "Camera calibration unavailable.\n";
        return 1;
    }

    mini_vo::StereoCamera stereo_camera(camera_left, camera_right);

    // 3.获取第一帧
    auto frame0 = dataset.NextFrame();
    if (!frame0)
    {
        std::cerr << "Cannot read Frame0.\n";
        return 1;
    }

    // stereo Rectification 初始化只需要一次
    if (!stereo_camera.InitRectification(frame0->leftImage().size()))
    {
        std::cerr << "Stereo rectification init failed.\n";
        return 1;
    }

    // 4. MultiFrameTracker Config
    mini_vo::MultiFrameTrackingConfig config;

    // 4.1 ORB数量
    config.num_features = 1000;

    // 4.2 Temporal Matching Ratio Test.
    config.temporal_ratio = 0.75f;

    // 4.3 第一帧地图初始化
    config.map_init.min_depth = 0.5;
    config.map_init.max_depth = 50.0;
    config.map_init.ratio_threshold = 0.75f;
    config.map_init.max_vertical_error = 2.0f;
    config.map_init.min_map_points = 20;

    // pnp tracking 与之前通过的pnp V1参数保持一致
    config.pnp.min_correspondences = 6;
    config.pnp.min_inliers = 15;
    config.pnp.ransac_iterations = 200;
    config.pnp.reprojection_error_px = 3.0;
    config.pnp.confidence = 0.99;
    config.pnp.min_inlier_ratio = 0.5;

    // 5.创建MultiFrameTracker
    mini_vo::MultiFrameTracker tracker(stereo_camera, config);

    // 6. Frame0 Initial Map
    if (!tracker.Initialize(frame0))
    {
        std::cerr << "Tracker initialization failed:" << tracker.lastError() << '\n';
        return 1;
    }

    const auto &init_stats = tracker.initializationStats();

    std::cout
        << "\n========== Initial Map ==========\n"

        << "Frame ID          : "
        << frame0->id()
        << '\n'

        << "Stereo Matches    : "
        << init_stats.stereo_matches
        << '\n'

        << "Duplicate Right   : "
        << init_stats.duplicate_right
        << '\n'

        << "Created MapPoints : "
        << init_stats.created_map_points
        << '\n'

        << "Map::size()       : "
        << tracker.map().size()
        << '\n';

    /*
     * ============================================================
     * 7. 轨迹输出
     * ============================================================
     *
     * 使用文本文件，方便以后：
     *
     *   Python 绘图
     *   evo 评估
     *   Ground Truth 对比
     *
     * 注意：
     * EuRoC 真值通常以机体/IMU 坐标系给出，
     * 严格比较前要处理 camera-body 外参。
     */
    const std::string trajectory_path = "trajectory_multiframe_v1.txt";
    std::ofstream trajectory_file(trajectory_path);

    if (!trajectory_file.is_open())
    {
        std::cerr << "Cannot open trajectory file.\n";
        return 1;
    }

    // 第一帧T_wc = I。
    WriteTumPose(trajectory_file, *frame0);

    // 8. Multi-frame Tracking Loop
    std::cout
        << "\n========== Multi-frame Tracking V1 ==========\n"

        << "Requested Frames : "
        << requested_frames
        << '\n'

        << "Tracking Strategy: Previous Frame\n"

        << "Map Strategy     : Fixed Initial Map\n\n";

    // 第一帧初始化成功，已处理一帧
    std::size_t processed_frames = 1;
    bool tracking_failed = false;

    while (processed_frames < requested_frames)
    {
        // 8.1 读取新的 EuRoC Frame
        auto current_frame = dataset.NextFrame();

        if (!current_frame)
        {
            std::cerr << "Dataset ended before requested frame count.\n";
            tracking_failed = true;
            break;
        }
        /*
         * --------------------------------------------------------
         * 8.2 保存上一帧的 shared_ptr
         * --------------------------------------------------------
         *
         * tracker.Track() 成功后会：
         *
         *    last_frame_ = current_frame
         *
         * 因此提前保存 previous_frame，
         * 以便计算两帧相机中心距离。
         */
        auto previous_frame = tracker.lastFrame();

        // 8.3 执行跟踪=======
        const bool success = tracker.Track(current_frame);

        if (!success)
        {
            std::cerr
                << "\n[Tracking Lost]\n"
                << "Frame ID : "
                << current_frame->id()
                << '\n'
                << "Reason   : "
                << tracker.lastError()
                << '\n';

            const auto &failed_pnp =
                tracker.pnpStats();

            std::cerr
                << "PnP correspondences : "
                << failed_pnp.valid_correspondences
                << '\n'
                << "PnP final inliers   : "
                << failed_pnp.final_inliers
                << '\n';

            tracking_failed = true;

            break;
        }

        /*
         * --------------------------------------------------------
         * 8.4 获取 Matching / PnP 统计
         * --------------------------------------------------------
         */
        const auto &temporal_stats = tracker.temporalStats();
        const auto &pnp_stats = tracker.pnpStats();

        // 8.5 Camera Center
        const Eigen::Vector3d previous_center = GetCameraCenterWorld(*previous_frame);
        const Eigen::Vector3d current_center = GetCameraCenterWorld(*current_frame);

        // 两帧之间的估计平移距离
        const double step_translation = (current_center - previous_center).norm();

        // 8.6 打印当前帧
        std::cout
            << "Frame "
            << std::setw(4)
            << current_frame->id()

            << " | RefMap "
            << temporal_stats.reference_with_mappoint

            << " | Match "
            << pnp_stats.temporal_matches

            << " | Inliers "
            << pnp_stats.final_inliers

            << " | Ratio "
            << std::fixed
            << std::setprecision(3)
            << pnp_stats.inlier_ratio

            << " | Reproj "
            << pnp_stats.mean_reprojection_error_px

            << " px"

            << " | Step "
            << step_translation

            << " m"

            << " | Cw ["
            << current_center.transpose()
            << "]"

            << '\n';

        /*
         * --------------------------------------------------------
         * 8.7 保存成功估计的位姿
         * --------------------------------------------------------
         */

        WriteTumPose(trajectory_file, *current_frame);
        ++processed_frames;
    }

    // 9.关闭轨迹文件
    trajectory_file.close();

    // close() 会刷新缓冲区；此时检查才能发现最后一次写入失败。
    if (!trajectory_file)
    {
        std::cerr << "Trajectory write failed.\n";
        return 1;
    }

    // 10.汇总
    std::cout
        << "\n========== Tracking Summary ==========\n"

        << "Requested Frames    : "
        << requested_frames
        << '\n'

        << "Processed Frames    : "
        << processed_frames
        << '\n'

        << "Tracked Frames      : "
        << tracker.trackedFrames()
        << '\n'

        << "MapPoints           : "
        << tracker.map().size()
        << '\n'

        << "Trajectory File     : "
        << trajectory_path
        << '\n';

    /*
     * V1 验收：
     *
     * 如果未能完成要求的帧数，
     * 返回非零退出码。
     *
     * 轨迹文件仍然保留已经成功的部分，
     * 用于 Debug。
     */
    if (tracking_failed || processed_frames != requested_frames)
    {
        std::cerr << "\nMulti-frame Tracking V1 "
                  << "did not complete requested frames.\n";
        return 1;
    }

    std::cout << "\nMulti-frame Tracking V1 passed.\n";
    return 0;
}
