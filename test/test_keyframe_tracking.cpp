#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mini_vo/dataset.h"
#include "mini_vo/frame.h"
#include "mini_vo/multi_frame_tracker.h"
#include "mini_vo/stereo_camera.h"


/*
 * ================================================================
 * Camera Center in World
 * ================================================================
 *
 * Frame 保存：
 *
 *   T_cw
 *
 * 相机世界位置：
 *
 *   C_w = translation(T_wc)
 *
 *   T_wc = inverse(T_cw)
 */
Eigen::Vector3d GetCameraCenterWorld(
    const mini_vo::Frame& frame
)
{
    const Eigen::Matrix4d T_wc =
        frame.pose().inverse();


    return T_wc.block<3, 1>(
        0,
        3
    );
}


/*
 * ================================================================
 * TUM Trajectory Writer
 * ================================================================
 *
 * 每行：
 *
 * timestamp tx ty tz qx qy qz qw
 *
 * 注意：
 *
 *   输出 T_wc
 *
 *   而不是 T_cw。
 */
void WriteTumPose(
    std::ofstream& file,
    const mini_vo::Frame& frame
)
{
    const Eigen::Matrix4d T_wc =
        frame.pose().inverse();


    const Eigen::Vector3d translation =
        T_wc.block<3, 1>(
            0,
            3
        );


    const Eigen::Matrix3d rotation =
        T_wc.block<3, 3>(
            0,
            0
        );


    Eigen::Quaterniond quaternion(
        rotation
    );


    quaternion.normalize();


    file
        << std::fixed
        << std::setprecision(9)

        << frame.timestamp() << " "

        << translation.x() << " "
        << translation.y() << " "
        << translation.z() << " "

        << quaternion.x() << " "
        << quaternion.y() << " "
        << quaternion.z() << " "
        << quaternion.w() << "\n";
}


/*
 * ================================================================
 * Main
 * ================================================================
 */
int main(int argc, char** argv)
{
    /*
     * 使用：
     *
     * ./test_multi_frame_tracking_v2 DATASET_PATH [FRAME_COUNT]
     */
    if (argc < 2 || argc > 3)
    {
        std::cerr
            << "Usage: "
            << argv[0]
            << " <euroc_dataset_path> [frame_count]\n";

        return 1;
    }


    /*
     * 默认先测试 30 帧。
     */
    std::size_t requested_frames = 30;


    if (argc == 3)
    {
        try
        {
            const std::string argument =
                argv[2];


            if (argument.empty() ||
                argument.front() == '-')
            {
                throw std::invalid_argument(
                    "Invalid frame count."
                );
            }


            std::size_t consumed = 0;


            requested_frames =
                static_cast<std::size_t>(
                    std::stoull(
                        argument,
                        &consumed
                    )
                );


            if (consumed != argument.size() ||
                requested_frames < 2)
            {
                throw std::invalid_argument(
                    "Frame count must be >= 2."
                );
            }
        }
        catch (const std::exception&)
        {
            std::cerr
                << "Invalid frame count.\n";

            return 1;
        }
    }


    /*
     * ============================================================
     * 1. Dataset
     * ============================================================
     */
    mini_vo::Dataset dataset(
        argv[1]
    );


    if (!dataset.Init())
    {
        std::cerr
            << "Dataset initialization failed.\n";

        return 1;
    }


    /*
     * ============================================================
     * 2. Camera
     * ============================================================
     */
    auto left_camera =
        dataset.leftCamera();


    auto right_camera =
        dataset.rightCamera();


    if (!left_camera || !right_camera)
    {
        std::cerr
            << "Camera calibration unavailable.\n";

        return 1;
    }


    mini_vo::StereoCamera stereo_camera(
        left_camera,
        right_camera
    );


    /*
     * ============================================================
     * 3. First Frame
     * ============================================================
     */
    auto frame0 =
        dataset.NextFrame();


    if (!frame0)
    {
        std::cerr
            << "Cannot read first Frame.\n";

        return 1;
    }


    /*
     * ============================================================
     * 4. Stereo Rectification Initialization
     * ============================================================
     */
    if (!stereo_camera.InitRectification(
            frame0->leftImage().size()))
    {
        std::cerr
            << "Stereo rectification init failed.\n";

        return 1;
    }


    /*
     * ============================================================
     * 5. Multi-frame V2 Config
     * ============================================================
     */
    mini_vo::MultiFrameTrackingConfig config;


    /*
     * ORB。
     */
    config.num_features = 1000;


    /*
     * Temporal Matching。
     */
    config.temporal_ratio = 0.75f;


    /*
     * 初始地图。
     *
     * 与 V1 保持一致。
     */
    config.map_init.min_depth = 0.5;

    config.map_init.max_depth = 50.0;

    config.map_init.ratio_threshold = 0.75f;

    config.map_init.max_vertical_error = 2.0f;

    config.map_init.min_map_points = 20;


    /*
     * PnP。
     *
     * 与之前通过的 V1 测试保持一致。
     */
    config.pnp.min_correspondences = 6;

    config.pnp.min_inliers = 15;

    config.pnp.ransac_iterations = 200;

    config.pnp.reprojection_error_px = 3.0;

    config.pnp.confidence = 0.99;

    config.pnp.min_inlier_ratio = 0.5;


    /*
     * ============================================================
     * 6. Enable V2
     * ============================================================
     *
     * 这是关键开关。
     */
    config.enable_map_replenishment = true;


    /*
     * ============================================================
     * 7. Map Update Config
     * ============================================================
     */
    config.map_update.min_depth = 0.5;

    config.map_update.max_depth = 50.0;


    /*
     * 每帧最多补充 150 个新地图点。
     *
     * 这是上限，不代表每帧一定能创建 150 个。
     */
    config.map_update.max_new_points_per_frame = 150;


    config.map_update.stereo_ratio_threshold = 0.75f;

    config.map_update.max_vertical_error = 2.0f;

    config.map_update.max_right_reprojection_error_px = 2.0;


    /*
     * ============================================================
     * 8. MultiFrameTracker
     * ============================================================
     */
    mini_vo::MultiFrameTracker tracker(
        stereo_camera,
        config
    );


    /*
     * ============================================================
     * 9. Initial Map
     * ============================================================
     */
    if (!tracker.Initialize(frame0))
    {
        std::cerr
            << "Tracker initialization failed: "
            << tracker.lastError()
            << '\n';

        return 1;
    }


    const auto& init_stats =
        tracker.initializationStats();


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
     * 10. Trajectory File
     * ============================================================
     *
     * V2 使用独立文件。
     *
     * 不覆盖之前 V1 的轨迹。
     */
    const std::string trajectory_path =
        "trajectory_multiframe_v2.txt";


    std::ofstream trajectory_file(
        trajectory_path
    );


    if (!trajectory_file.is_open())
    {
        std::cerr
            << "Cannot open trajectory file.\n";

        return 1;
    }


    /*
     * Frame0 Pose。
     */
    WriteTumPose(
        trajectory_file,
        *frame0
    );


    /*
     * ============================================================
     * 11. Main Tracking Loop
     * ============================================================
     */
    std::cout
        << "\n========== Multi-frame Tracking V2 ==========\n"

        << "Requested Frames  : "
        << requested_frames
        << '\n'

        << "Tracking Strategy: Previous Frame\n"

        << "Map Strategy     : Stereo Replenishment\n"

        << "Max New Points   : "
        << config.map_update.max_new_points_per_frame
        << " / Frame\n\n";


    std::size_t processed_frames = 1;

    bool failed = false;


    /*
     * 统计整个跟踪过程中创建的新地图点数量。
     */
    std::size_t total_new_map_points = 0;


    while (processed_frames < requested_frames)
    {
        /*
         * --------------------------------------------------------
         * 11.1 Next Frame
         * --------------------------------------------------------
         */
        auto current_frame =
            dataset.NextFrame();


        if (!current_frame)
        {
            std::cerr
                << "Dataset ended.\n";

            failed = true;
            break;
        }


        /*
         * 当前帧成功跟踪之前：
         *
         * previous_frame 仍然是上一帧的 owner。
         */
        auto previous_frame =
            tracker.lastFrame();


        /*
         * 记录地图更新前大小。
         */
        const std::size_t previous_map_size =
            tracker.map().size();


        /*
         * --------------------------------------------------------
         * 11.2 Tracking + Map Update
         * --------------------------------------------------------
         */
        const bool success =
            tracker.Track(
                current_frame
            );


        if (!success)
        {
            const auto& temporal =
                tracker.temporalStats();


            std::cerr
                << "\n[Tracking Lost]\n"

                << "Frame ID     : "
                << current_frame->id()
                << '\n'

                << "Reason       : "
                << tracker.lastError()
                << '\n'

                << "Temporal KNN : "
                << temporal.knn_candidates
                << '\n'

                << "Ratio Passed : "
                << temporal.ratio_passed
                << '\n'

                << "Unique Passed: "
                << temporal.unique_passed
                << '\n';


            failed = true;

            break;
        }


        /*
         * Map update 失败与 Pose Tracking 失败
         * 是两个不同的状态。
         *
         * 本测试为了保证地图结果可靠，
         * 遇到地图更新内部错误时停止运行。
         */
        if (!tracker.mapUpdateSucceeded())
        {
            std::cerr
                << "\n[Map Update Error]\n"

                << "Frame ID : "
                << current_frame->id()
                << '\n'

                << "Reason   : "
                << tracker.lastMapUpdateError()
                << '\n';


            failed = true;

            break;
        }


        /*
         * --------------------------------------------------------
         * 11.3 Statistics
         * --------------------------------------------------------
         */
        const auto& temporal =
            tracker.temporalStats();


        const auto& pnp =
            tracker.pnpStats();


        const auto& update =
            tracker.mapUpdateStats();


        /*
         * --------------------------------------------------------
         * 11.4 Camera Motion
         * --------------------------------------------------------
         */
        const Eigen::Vector3d previous_center =
            GetCameraCenterWorld(
                *previous_frame
            );


        const Eigen::Vector3d current_center =
            GetCameraCenterWorld(
                *current_frame
            );


        const double step =
            (
                current_center - previous_center
            ).norm();


        /*
         * --------------------------------------------------------
         * 11.5 Map Growth
         * --------------------------------------------------------
         */
        const std::size_t current_map_size =
            tracker.map().size();


        /*
         * 必须严格符合：
         *
         * MapSizeAfter
         *     =
         * MapSizeBefore + NewMapPoints
         *
         * 当前 V2 不删除地图点。
         */
        if (current_map_size <
                previous_map_size ||
            current_map_size - previous_map_size !=
                update.new_map_points)
        {
            std::cerr
                << "Map size / update count mismatch.\n";

            failed = true;
            break;
        }


        total_new_map_points +=
            update.new_map_points;


        /*
         * --------------------------------------------------------
         * 11.6 Print
         * --------------------------------------------------------
         */
        std::cout
            << "Frame "
            << std::setw(4)
            << current_frame->id()

            << " | RefMap "
            << temporal.reference_with_mappoint

            << " | Match "
            << pnp.temporal_matches

            << " | Inliers "
            << pnp.final_inliers

            << " | Ratio "
            << std::fixed
            << std::setprecision(3)
            << pnp.inlier_ratio

            << " | Reproj "
            << pnp.mean_reprojection_error_px
            << " px"

            << " | Stereo "
            << update.stereo_matches

            << " | New "
            << update.new_map_points

            << " | Map "
            << current_map_size

            << " | Step "
            << step
            << " m"

            << '\n';


        /*
         * --------------------------------------------------------
         * 11.7 Save Trajectory
         * --------------------------------------------------------
         */
        WriteTumPose(
            trajectory_file,
            *current_frame
        );


        ++processed_frames;
    }


    /*
     * ============================================================
     * 12. Close File
     * ============================================================
     */
    trajectory_file.close();


    /*
     * ============================================================
     * 13. Summary
     * ============================================================
     */
    std::cout
        << "\n========== Multi-frame V2 Summary ==========\n"

        << "Requested Frames     : "
        << requested_frames
        << '\n'

        << "Processed Frames     : "
        << processed_frames
        << '\n'

        << "Tracked Frames       : "
        << tracker.trackedFrames()
        << '\n'

        << "Initial MapPoints    : "
        << init_stats.created_map_points
        << '\n'

        << "Total New MapPoints  : "
        << total_new_map_points
        << '\n'

        << "Final MapPoints      : "
        << tracker.map().size()
        << '\n'

        << "Trajectory File      : "
        << trajectory_path
        << '\n';


    if (failed ||
        processed_frames != requested_frames)
    {
        std::cerr
            << "\nMulti-frame Tracking V2 "
            << "did not complete requested frames.\n";

        return 1;
    }


    /*
     * 地图大小统计必须一致。
     */
    if (tracker.map().size() !=
        init_stats.created_map_points +
        total_new_map_points)
    {
        std::cerr
            << "Final MapPoint count mismatch.\n";

        return 1;
    }


    std::cout
        << "\nMulti-frame Tracking V2 passed.\n";


    return 0;
}