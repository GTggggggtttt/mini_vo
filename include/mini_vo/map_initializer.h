#pragma once

#include <iostream>
#include <string>

#include <opencv2/core.hpp>

#include "mini_vo/frame.h"
#include "mini_vo/frontend.h"
#include "mini_vo/map.h"
#include "mini_vo/stereo_camera.h"

namespace mini_vo
{
    /*
     * 第一帧地图初始化参数。
     *
     * 深度上下限会转换成 disparity 上下限。
     */
    struct MapInitializerConfig
    {
        double min_depth = 0.5;
        double max_depth = 50.0;

        float ratio_threshold = 0.75f;
        float max_vertical_error = 2.0f;

        std::size_t min_map_points = 20;
    };

    // 地图初始化统计
    struct MapInitializationStats
    {
        std::size_t stereo_matches = 0;
        std::size_t duplicate_right = 0;
        std::size_t invalid_depth = 0;
        std::size_t created_map_points = 0;
    };

    /*
     * ================================================================
     * MapInitializer
     * ================================================================
     *
     * 职责：
     *
     *   第一帧 Stereo Feature -> MapPoint
     *  双目特征去形成地图点
     * 输入必须是校正后的双目图像。
     *
     * 输出：
     *
     *   Frame0 持有 Feature
     *   Feature 关联 Frame / MapPoint  weak_ptr
     *   Map 持有 MapPoint
     *
     * 坐标约定：
     *
     *   第一帧左相机 = 世界坐标系
     *
     *   T_cw(0) = I
     *
     *   Pw = Pc
     *
     * 不负责后续时间匹配或 PnP。
     */
    class MapInitializer
    {
    public:
        // 构造函数 不能进行隐饰转换
        explicit MapInitializer(const MapInitializerConfig &config = MapInitializerConfig{});

        // 检测当前帧是否完成初始化的函数声明
        bool Initialize(const Frame::Ptr &frame, const cv::Mat &left_rectified, const cv::Mat &right_rectified,
                        const StereoCamera &stereo_camera, Frontend &frontend, Map &map);

        const MapInitializationStats &stats() const;
        const std::string &lastError() const;

    private:
        MapInitializerConfig config_;
        MapInitializationStats stats_;
        std::string last_error_;
    };

} // namespace mini_vo
