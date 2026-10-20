#include "mini_vo/map_initializer.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "mini_vo/feature.h"
#include "mini_vo/mappoint.h"

namespace mini_vo
{
    MapInitializer::MapInitializer(const MapInitializerConfig &config) : config_(config)
    {
        if (!std::isfinite(config_.min_depth) || !std::isfinite(config_.max_depth) ||
            config_.min_depth <= 0 || config_.max_depth <= config_.min_depth || !std::isfinite(config_.ratio_threshold) || config_.ratio_threshold <= 0.0f || config_.ratio_threshold >= 1.0f || !std::isfinite(config_.max_vertical_error) || config_.max_vertical_error < 0.0f || config_.min_map_points == 0)
        {
            throw std::invalid_argument("MapInitializer: invalid config.");
        }
    }

    // Initialize
    bool MapInitializer::Initialize(const Frame::Ptr &frame, const cv::Mat &left_rectified, const cv::Mat &right_rectified,
                                    const StereoCamera &stereo_camera, Frontend &frontend, Map &map)
    {
        stats_ = MapInitializationStats{};
        last_error_.clear();

        /*
         * V1 只允许从空地图初始化。
         *
         * 不允许在已有 Map 上重新从 ID=0 插入，
         * 否则会造成 MapPoint ID 冲突。
         */
        if (!frame || left_rectified.empty() || right_rectified.empty() || !stereo_camera.initialized() || !map.empty())
        {
            last_error_ = "Invalid initialization state.";
            return false;
        }

        // 先配置双目匹配参数 fb
        const double fb = stereo_camera.fx() * stereo_camera.rectifiedBaseline();

        if (!std::isfinite(fb) || fb <= 0.0)
        {
            last_error_ = "Invalid stereo focal-baseline product.";
            return false;
        }

        frontend.SetStereoMatchingParameters(config_.ratio_threshold, config_.max_vertical_error,
                                             static_cast<float>(fb / config_.max_depth), static_cast<float>(fb / config_.min_depth));

        // 定义第一帧世界坐标系
        frame->setPose(Eigen::Matrix4d::Identity());

        // Stereo ORB + Matching V2
        if (!frontend.ProcessStereo(left_rectified, right_rectified))
        {
            last_error_ = "Stereo matching failed.";
            return false;
        }

        /*
         * Frame 拥有 Feature。
         *
         * 这里复制 shared_ptr，而非 Feature 对象。
         */
        frame->setLeftFeatures(frontend.leftFeatures());
        frame->setRightFeatures(frontend.rightFeatures());

        // Feature 通过 weak_ptr 关联 Frame
        for (const auto &feature : frame->leftFeatures())
        {
            if (feature)
            {
                feature->setFrame(frame);
            }
        }

        for (const auto &feature : frame->rightFeatures())
        {
            if (feature)
            {
                feature->setFrame(frame);
            }
        }

        const auto &left_features = frame->leftFeatures();
        const auto &right_features = frame->rightFeatures();
        const auto &stereo_matches = frontend.stereoMatches();
        stats_.stereo_matches = stereo_matches.size();

        /*
         * 一对一双目约束：
         *
         *   一个 Right Feature
         *   最多只能初始化一个 MapPoint。
         */
        std::vector<unsigned char> right_used(right_features.size(), 0);
        unsigned long next_map_point_id = 0;

        /*
         * ============================================================
         * Stereo Match -> MapPoint
         * ============================================================
         */
        for (const auto &match : stereo_matches)
        {
            if (match.queryIdx < 0 || match.trainIdx < 0)
            {
                continue;
            }

            const std::size_t li = static_cast<std::size_t>(match.queryIdx);
            const std::size_t ri = static_cast<std::size_t>(match.trainIdx);

            if (li >= left_features.size() || ri >= right_features.size())
            {
                continue;
            }

            if (right_used[ri])
            {
                ++stats_.duplicate_right;
                continue;
            }

            const auto &feature_left = left_features[li];
            const auto &feature_right = right_features[ri];

            if (!feature_left || !feature_right)
            {
                continue;
            }

            if (feature_left->hasMapPoint() || feature_right->hasMapPoint())
            {
                continue;
            }

            const cv::Point2f &pl = feature_left->keypoint().pt;
            const cv::Point2f &pr = feature_right->keypoint().pt;

            // 双目视差
            const double disparity = static_cast<double>(pl.x) - static_cast<double>(pr.x);
            if (!std::isfinite(disparity) || disparity <= 0.0)
            {
                ++stats_.invalid_depth;
                continue;
            }

            // 双目恢复深度
            const double depth = fb / disparity;
            if (!std::isfinite(depth) || depth < config_.min_depth || depth > config_.max_depth)
            {
                ++stats_.invalid_depth;
                continue;
            }

            /*
             * Rectified Pixel -> Camera 3D。
             */
            const Eigen::Vector2d pixel(pl.x, pl.y);
            const Eigen::Vector3d Pc = stereo_camera.PixelToCamera(pixel, depth);

            if (!Pc.allFinite() || Pc.z() <= 0.0)
            {
                ++stats_.invalid_depth;
                continue;
            }
            /*
             * 因为：
             *
             *   Frame0 T_cw = I
             *
             * 所以：
             *
             *   Pw = Pc
             */
            const Eigen::Vector3d Pw = Pc;
            auto map_point = std::make_shared<MapPoint>(next_map_point_id, Pw);

            // map 从此拥有mappoint
            if (!map.InsertMapPoint(map_point))
            {
                last_error_ = "MapPoint insertion failed.";
                return false;
            }
            /*
             * 左右观测关联同一个 3D Landmark。
             *
             * 内部使用 weak_ptr，
             * 不增加 MapPoint 强引用计数。
             */
            feature_left->setMapPoint(map_point);
            feature_right->setMapPoint(map_point);

            right_used[ri] = 1;

            ++next_map_point_id;
            ++stats_.created_map_points;
        }

        if(stats_.created_map_points < config_.min_map_points)
        {
            last_error_="Too few initial MapPoints.";
            return false;
        }
        return true;
    }

    // getter 函数
    const MapInitializationStats& MapInitializer::stats() const
    {
        return stats_;
    }

    const std::string&MapInitializer::lastError() const
    {
        return last_error_;
    }

} // namespace mini_vo
