#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "common/euroc_test_context.h"
#include "common/trajectory_writer.h"

#include "mini_vo/multi_frame_tracker.h"


namespace mini_vo::test {


struct TrackingTestResult
{
    bool success = false;

    std::size_t processed_frames = 0;

    std::string error;
};


/*
 * ================================================================
 * RunTrackingSequence
 * ================================================================
 *
 * on_frame(previous, current, tracker)
 *
 * 每次成功跟踪后调用。
 *
 * 回调返回：
 *   true  本帧额外检查通过
 *   false 测试失败，停止执行
 *
 * 使用模板：
 *   可以直接传入 lambda
 *   不必引入 std::function 的类型擦除
 */
template <typename OnFrame>
TrackingTestResult RunTrackingSequence(
    EurocTestContext& context,
    MultiFrameTracker& tracker,
    std::size_t requested_frames,
    const std::string& trajectory_path,
    OnFrame&& on_frame
)
{
    TrackingTestResult result;

    if (requested_frames < 2)
    {
        result.error = "Frame count must be >= 2.";
        return result;
    }

    if (!context.frame0 || !context.stereo_camera)
    {
        result.error = "EurocTestContext is not ready.";
        return result;
    }

    if (!tracker.Initialize(context.frame0))
    {
        result.error = tracker.lastError();
        return result;
    }

    std::ofstream trajectory(trajectory_path);

    if (!trajectory.is_open())
    {
        result.error = "Cannot open trajectory file.";
        return result;
    }

    WriteTumPose(trajectory, *context.frame0);

    result.processed_frames = 1;

    while (result.processed_frames < requested_frames)
    {
        auto current = context.dataset.NextFrame();

        if (!current)
        {
            result.error = "Dataset ended.";
            return result;
        }

        /*
         * 保留上一帧的 shared_ptr。
         *
         * Track() 成功后，tracker.lastFrame()
         * 将变成 current。
         */
        auto previous = tracker.lastFrame();

        if (!tracker.Track(current))
        {
            result.error =
                "Frame " + std::to_string(current->id()) +
                ": " + tracker.lastError();

            return result;
        }

        /*
         * Pose Tracking 和地图更新分别检查。
         */
        if (!tracker.mapUpdateSucceeded())
        {
            result.error = tracker.lastMapUpdateError();
            return result;
        }

        /*
         * 需要已实现 KeyFrame V1 的相关接口。
         * 若尚未接入 KeyFrame，可暂时移除此检查。
         */
        if (!tracker.keyFrameUpdateSucceeded())
        {
            result.error = tracker.lastKeyFrameError();
            return result;
        }

        /*
         * 执行各集成测试自身的断言与统计。
         */
        if (!on_frame(previous, current, tracker))
        {
            result.error =
                "Custom validation failed at Frame " +
                std::to_string(current->id());

            return result;
        }

        WriteTumPose(trajectory, *current);

        ++result.processed_frames;
    }

    /*
     * 检查 I/O。
     *
     * 注意：
     * ofstream 的析构会关闭文件，
     * 但我们还需要确认写入没有报告错误。
     */
    trajectory.flush();

    if (!trajectory)
    {
        result.error = "Trajectory write failed.";
        return result;
    }

    result.success = true;

    return result;
}


} // namespace mini_vo::test