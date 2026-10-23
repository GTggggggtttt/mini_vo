#include <cstddef>
#include <exception>
#include <iostream>
#include <string>

#include "common/euroc_test_context.h"
#include "common/tracker_test_config.h"
#include "common/tracking_test_runner.h"

#include "mini_vo/keyframe.h"
#include "mini_vo/mappoint.h"
#include "mini_vo/multi_frame_tracker.h"


int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <euroc_dataset_path> [frames]\n";

        return 1;
    }

    std::size_t requested_frames = 30;

    if (argc == 3)
    {
        try
        {
            const std::string text = argv[2];

            std::size_t consumed = 0;

            const auto value =
                std::stoull(text, &consumed);

            if (text.empty() ||
                text.front() == '-' ||
                consumed != text.size() ||
                value < 2)
            {
                throw std::invalid_argument(
                    "Invalid frame count."
                );
            }

            requested_frames =
                static_cast<std::size_t>(value);
        }
        catch (const std::exception&)
        {
            std::cerr << "Invalid frame count.\n";
            return 1;
        }
    }

    /*
     * ============================================================
     * 1. 公共测试初始化
     * ============================================================
     */
    mini_vo::test::EurocTestContext context(argv[1]);

    if (!context.Initialize())
    {
        std::cerr << context.error << '\n';
        return 1;
    }

    auto config =
        mini_vo::test::MakeKeyFrameConfig();

    mini_vo::MultiFrameTracker tracker(
        *context.stereo_camera,
        config
    );


    /*
     * ============================================================
     * 2. 运行集成测试
     * ============================================================
     */
    const auto result =
        mini_vo::test::RunTrackingSequence(
            context,
            tracker,
            requested_frames,
            "trajectory_observation_v1.txt",

            /*
             * 每次 Track 成功后执行这里的检查。
             *
             * 前三个参数分别是：
             *
             *   previous Frame
             *   current Frame
             *   tracker
             */
            [](const auto& previous,
               const auto& current,
               const auto& tracker) -> bool
            {
                (void)previous;

                std::size_t total_observations = 0;

                std::size_t observed_map_points = 0;

                /*
                 * =================================================
                 * 3. 检查 MapPoint -> KeyFrame -> Feature
                 * =================================================
                 */
                for (const auto& item :
                     tracker.map().mapPoints())
                {
                    const auto& map_point =
                        item.second;

                    if (!map_point)
                    {
                        return false;
                    }

                    if (map_point->observationCount() > 0)
                    {
                        ++observed_map_points;
                    }

                    for (const auto& obs_item :
                         map_point->observations())
                    {
                        const unsigned long keyframe_id =
                            obs_item.first;

                        const auto& observation =
                            obs_item.second;

                        /*
                         * Observation 的 weak_ptr
                         * 应该能锁定 Map 持有的 KeyFrame。
                         */
                        auto keyframe =
                            observation.keyframe.lock();

                        if (!keyframe ||
                            keyframe->id() != keyframe_id)
                        {
                            return false;
                        }

                        /*
                         * Observation 指向的 KeyFrame
                         * 应该仍然在当前 Map 里。
                         */
                        if (tracker.map().GetKeyFrame(
                                keyframe_id) != keyframe)
                        {
                            return false;
                        }

                        const auto& features =
                            keyframe->frame()->leftFeatures();

                        const std::size_t feature_index =
                            observation.left_feature_index;

                        if (feature_index >= features.size())
                        {
                            return false;
                        }

                        const auto& feature =
                            features[feature_index];

                        if (!feature)
                        {
                            return false;
                        }

                        /*
                         * 最重要的一致性检查：
                         *
                         * KeyFrame -> Feature -> MapPoint
                         * 必须与 MapPoint -> Observation
                         * 记录相同的对象。
                         */
                        if (feature->mapPoint().get()
                            != map_point.get())
                        {
                            return false;
                        }

                        ++total_observations;
                    }
                }

                /*
                 * =================================================
                 * 4. 输出统计
                 * =================================================
                 */
                std::cout
                    << "Frame "
                    << current->id()

                    << " | Inliers "
                    << tracker.pnpStats().final_inliers

                    << " | New "
                    << tracker.mapUpdateStats().new_map_points

                    << " | MapPoints "
                    << tracker.map().size()

                    << " | KeyFrames "
                    << tracker.map().keyFrameCount()

                    << " | Observed MapPoints "
                    << observed_map_points

                    << " | KF Observations "
                    << total_observations

                    << '\n';

                return true;
            }
        );


    /*
     * ============================================================
     * 5. 测试汇总
     * ============================================================
     */
    std::cout
        << "\n========== Observation V1 Summary ==========\n"
        << "Processed Frames : "
        << result.processed_frames
        << '\n'
        << "MapPoints        : "
        << tracker.map().size()
        << '\n'
        << "KeyFrames        : "
        << tracker.map().keyFrameCount()
        << '\n';


    if (!result.success)
    {
        std::cerr
            << "Observation tracking failed: "
            << result.error
            << '\n';

        return 1;
    }

    std::cout
        << "\nMapPoint Observation V1 passed.\n";

    return 0;
}