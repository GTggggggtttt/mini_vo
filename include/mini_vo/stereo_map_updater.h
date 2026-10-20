#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>

#include "mini_vo/frame.h"
#include "mini_vo/map.h"
#include "mini_vo/stereo_camera.h"

namespace mini_vo
{
    /*
     * ================================================================
     * StereoMapUpdateConfig
     * ================================================================
     *
     * 控制当前帧创建新 MapPoint 的条件。
     *
     * 当前 V1：
     *
     *   1. Depth Range
     *   2. Right Reprojection Error
     *   3. Max New Points Per Frame
     *
     * 这些都是工程参数，不是数学定理。
     */
    struct StereoMapUpdateConfig
    {
        double min_depth = 0.5;
        double max_depth = 50.0;
        double max_right_reprojection_error_px = 2.0;
        std::size_t max_new_points_per_frame = 150;

        // 传递给 Frontend::MatchStereo ()的参数。
        float stereo_ratio_threshold = 0.75f;
        float max_vertical_error = 2.0f;
    };

    // StereoMapUpdateStats 记录补图过程的过滤结果
    struct StereoMapUpdateStats
    {
        std::size_t stereo_matches = 0;
        std::size_t already_associated = 0;
        std::size_t duplicate_right = 0;
        std::size_t invalid_index = 0;
        std::size_t invalid_geometry = 0;
        std::size_t new_map_points = 0;
        /* data */
    };

    /*
     * ================================================================
     * StereoMapUpdater
     * ================================================================
     *
     * 当前职责：
     *
     *   在 PnP 成功以后，
     *   利用当前 Frame 的有效双目匹配，
     *   创建新的 World MapPoint。
     *
     *
     * 输入：
     *
     *   Current Frame
     *
     *       必须已有有效 Pose T_cw
     *       必须已有左右 Feature
     *
     *   Stereo Matches
     *
     *       queryIdx -> current.leftFeatures()
     *       trainIdx -> current.rightFeatures()
     *
     *   StereoCamera
     *
     *       Rectified geometry
     *
     *   Map
     *
     *       拥有全部 MapPoint
     *
     *
     * 输出：
     *
     *   1. New MapPoint
     *
     *   2. Map 插入新地图点
     *
     *   3. Left/Right Feature 建立关联
     *
     *
     * 不负责：
     *
     *   ORB 提取
     *   Stereo Descriptor Matching
     *   Temporal Matching
     *   PnP
     *   Pose Optimization
     *
     *
     * 当前为单线程版本。
     *  ======================= 什么时候可以升级为多线程版本=============================
     */
    class StereoMapUpdater
    {
    public:
        explicit StereoMapUpdater(const StereoMapUpdateConfig &config = StereoMapUpdateConfig{});

        /*
         * Update() 只在当前帧成功估计 Pose 后调用。
         *
         * 没有可创建的 MapPoint 不等于错误，
         * 因此正常执行但新增 0 个点，也返回 true。
         */
        bool Update(const Frame::Ptr &frame, const std::vector<cv::DMatch> &stereo_matches,
                    const StereoCamera &stereo_camera, Map &map);

        const StereoMapUpdateStats &stats() const;
        const std::string &lastError() const;

    private:
        StereoMapUpdateConfig config_;
        StereoMapUpdateStats stats_;
        std::string last_error_;

        /*
         * 下一次新建 MapPoint 使用的 ID。
         *
         * 第一次 Update 时从已有 Map ID 初始化。
         *
         * 一个 Updater 实例只服务于同一张 Map；
         * 不允许多个独立 Updater 并发分配 ID。
         */
        unsigned long next_map_point_id_ = 0;
        bool id_initialized_ = false;
    };

} // namespace mini_vo
