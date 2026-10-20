#include "mini_vo/stereo_map_updater.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "mini_vo/feature.h"
#include "mini_vo/mappoint.h"

namespace mini_vo
{
    /*
     * ================================================================
     * Constructor
     * ================================================================
     */
    StereoMapUpdater::StereoMapUpdater(const StereoMapUpdateConfig &config) : config_(config)
    {
        if (!std::isfinite(config_.min_depth) ||
            !std::isfinite(config_.max_depth) ||
            config_.min_depth <= 0.0 ||
            config_.max_depth <= config_.min_depth ||

            !std::isfinite(
                config_.max_right_reprojection_error_px) ||
            config_.max_right_reprojection_error_px < 0.0 ||

            !std::isfinite(
                config_.stereo_ratio_threshold) ||
            config_.stereo_ratio_threshold <= 0.0f ||
            config_.stereo_ratio_threshold >= 1.0f ||

            !std::isfinite(config_.max_vertical_error) ||
            config_.max_vertical_error < 0.0f)
        {
            throw std::invalid_argument(
                "StereoMapUpdater: invalid config.");
        }
    }

    // Update
    bool StereoMapUpdater::Update(const Frame::Ptr &frame,
                                  const std::vector<cv::DMatch> &stereo_matches,
                                  const StereoCamera &stereo_camera, Map &map)
    {
        // 1.Reser Statistics
        stats_ = StereoMapUpdateStats{};
        last_error_.clear();
        stats_.stereo_matches = stereo_matches.size();

        /*
         * ============================================================
         * 2. Basic Validation
         * ============================================================
         */
        if (!frame)
        {
            last_error_ = "Null Frame.";
            return false;
        }

        if (!stereo_camera.initialized())
        {
            last_error_ = "StereoCamera not initialized.";
            return false;
        }

        if (!frame->pose().allFinite())
        {
            last_error_ = "Invalid Frame Pose.";
            return false;
        }

        /*
         * 注意：
         *
         * 这里只验证矩阵数值有效。
         *
         * Frame Pose 是否真的通过 PnP 验证，
         * 必须由上层 MultiFrameTracker 保证。
         */
        const auto &left_features = frame->leftFeatures();
        const auto &right_features = frame->rightFeatures();

        /*
         * 即使没有 Stereo Matches，
         * 也不应判为整个跟踪失败。
         */
        if (stereo_matches.empty() || config_.max_new_points_per_frame == 0)
        {
            return true;
        }

        // 3. Stereo Geomatry
        const double fx = stereo_camera.fx();
        const double fy = stereo_camera.fy();
        const double cx = stereo_camera.cx();
        const double cy = stereo_camera.cy();
        const double baseline = stereo_camera.rectifiedBaseline();
        const double fb = fx * baseline;

        if (!std::isfinite(fb) || fb <= 0.0 || !std::isfinite(fy) || fy <= 0.0 || !std::isfinite(cx) || !std::isfinite(cy))
        {
            last_error_ = "Invalid rectified stereo geometry.";
            return false;
        }

        /*
         * ============================================================
         * 4. Prepare MapPoint ID
         * ============================================================
         *
         * MapInitializer 已经创建：
         *
         *   ID 0 ... N-1
         *
         * 所以 StereoMapUpdater 第一次使用时，
         * 需要从 Map 中计算下一个可用 ID。
         *
         * 不允许从 0 重新开始。
         */
        if (!id_initialized_)
        {
            for (const auto &item : map.mapPoints())
            {
                const unsigned long id = item.first;
                if (id == std::numeric_limits<unsigned long>::max())
                {
                    last_error_ = "MapPoint ID overflow.";
                    return false;
                }

                next_map_point_id_ = std::max(next_map_point_id_, id + 1);
            }
            id_initialized_ = true;
        }

        /*
         * ============================================================
         * 5. T_cw -> T_wc
         * ============================================================
         *
         * Frame Pose:
         *
         *   Pc = T_cw * Pw
         *
         *
         * 当前双目得到的是 Pc。
         *
         * 所以新地图点：
         *
         *   Pw = T_wc * Pc
         *
         *   T_wc = inverse(T_cw)
         */
        const Eigen::Matrix4d T_wc = frame->pose().inverse();
        if (!T_wc.allFinite())
        {
            last_error_ = "Cannot invert Frame Pose";
            return false;
        }

        /*
         * ============================================================
         * 6. Sort Stereo Matches
         * ============================================================
         *
         * 先选择 Hamming Distance 更小的匹配。
         *
         * 如果多个 Left Feature 指向同一个 Right Feature，
         * 优先处理 descriptor 距离较小的匹配。
         *
         * 这是贪心策略，不是全局最优匹配。
         */
        std::vector<cv::DMatch> sorted_matches = stereo_matches;

        std::sort(sorted_matches.begin(), sorted_matches.end(),
                  [](const cv::DMatch &a, const cv::DMatch &b)
                  {
                      if (a.distance != b.distance)
                      {
                          return a.distance < b.distance;
                      }
                      if (a.queryIdx != b.queryIdx)
                      {
                          return a.queryIdx < b.queryIdx;
                      }
                      return a.trainIdx < b.trainIdx;
                  });

        // 7.Right Feature Uniqueness
        std::vector<unsigned char> right_used(right_features.size(), 0);

        // 8.Stereo Match -> New MapPoint
        for (const auto &match : sorted_matches)
        {
            // 限制每帧点的数量
            if (stats_.new_map_points >= config_.max_new_points_per_frame)
            {
                break;
            }

            // 8.1 Validata Match Index
            if (match.queryIdx < 0 || match.trainIdx < 0)
            {
                ++stats_.invalid_index;
                continue;
            }

            const std::size_t li = static_cast<std::size_t>(match.queryIdx);
            const std::size_t ri = static_cast<std::size_t>(match.trainIdx);

            if (li >= left_features.size() || ri >= right_features.size())
            {
                ++stats_.invalid_index;
                continue;
            }

            if (right_used[ri])
            {
                ++stats_.duplicate_right;
                continue;
            }

            const auto &left_feature = left_features[li];
            const auto &right_feature = right_features[ri];

            if (!left_feature || !right_feature)
            {
                ++stats_.invalid_index;
                continue;
            }

            /*
             * --------------------------------------------------------
             * 8.2 Existing MapPoint Association
             * --------------------------------------------------------
             *
             * 当前 Frame 的左目 Feature
             * 已经有一部分经过 PnP RANSAC：
             *
             *   Left Feature -> Existing MapPoint
             *
             * 这些绝对不能再创建新地图点。
             *
             *
             * Right Feature 如果已经关联 MapPoint，
             * 也不能再覆盖。
             */
            if (left_feature->hasMapPoint() || right_feature->hasMapPoint())
            {
                ++stats_.already_associated;
                continue;
            }

            /*
             * --------------------------------------------------------
             * 8.3 Pixel
             * --------------------------------------------------------
             */
            const cv::Point2f &pl = left_feature->keypoint().pt;
            const cv::Point2f &pr = right_feature->keypoint().pt;

            if (!std::isfinite(pl.x) || !std::isfinite(pl.y) || !std::isfinite(pr.x) || !std::isfinite(pr.y))
            {
                ++stats_.invalid_geometry;
                continue;
            }

            /*
             * --------------------------------------------------------
             * 8.4 Disparity
             * --------------------------------------------------------
             *
             * 双目校正后：
             *
             *   d = uL - uR
             */
            const double disparity = static_cast<double>(pl.x) - static_cast<double>(pr.x);
            if (!std::isfinite(disparity) || disparity <= 0.0)
            {
                ++stats_.invalid_geometry;
                continue;
            }

            /*
             * --------------------------------------------------------
             * 8.5 Stereo Depth
             * --------------------------------------------------------
             *
             *   Z = f*b/d
             */
            const double depth = fb / disparity;
            if (!std::isfinite(depth) || depth < config_.min_depth || depth > config_.max_depth)
            {
                ++stats_.invalid_geometry;
                continue;
            }

            /*
             * --------------------------------------------------------
             * 8.6 Rectified Pixel -> Camera Point
             * --------------------------------------------------------
             */
            const Eigen::Vector2d pixel_left(pl.x, pl.y);
            const Eigen::Vector3d Pc = stereo_camera.PixelToCamera(pixel_left, depth);

            if (!Pc.allFinite() || Pc.z() <= 0.0)
            {
                ++stats_.invalid_geometry;
                continue;
            }

            /*
             * --------------------------------------------------------
             * 8.7 Right Reprojection Check
             * --------------------------------------------------------
             *
             * 对已校正的标准水平双目：
             *
             *   uR = fx*(X-baseline)/Z + cx
             *
             *   vR = fy*Y/Z + cy
             *
             * 与 Right Feature 的观测比较。
             */
            const double predicted_right_u =
                fx * (Pc.x() - baseline) / Pc.z() + cx;

            const double predicted_right_v =
                fy * Pc.y() / Pc.z() + cy;

            const double right_error =
                std::hypot(
                    predicted_right_u - pr.x,
                    predicted_right_v - pr.y);

            if (!std::isfinite(right_error) ||
                right_error >
                    config_.max_right_reprojection_error_px)
            {
                ++stats_.invalid_geometry;

                continue;
            }

            /*
             * 注意：
             *
             * 因为 depth 本身由 disparity 得到，
             * 这里的水平重投影误差理论上接近 0。
             *
             * 因此该检查主要再次验证垂直极线误差，
             * 不是独立的双目深度准确性验证。
             */

            /*
             * --------------------------------------------------------
             * 8.8 Camera Point -> World Point
             * --------------------------------------------------------
             *
             *   Pw = T_wc * Pc
             *
             * 这一转换非常重要。
             *
             * 不能在 Frame1 以后继续写：
             *
             *   Pw = Pc
             */
            Eigen::Vector4d Pc_h;
            Pc_h << Pc, 1.0;
            const Eigen::Vector4d Pw_h = T_wc * Pc_h;

            if (!Pw_h.allFinite() ||
                std::abs(Pw_h.w()) < 1e-12)
            {
                ++stats_.invalid_geometry;

                continue;
            }

            const Eigen::Vector3d Pw =
                Pw_h.head<3>() / Pw_h.w();

            if (!Pw.allFinite())
            {
                ++stats_.invalid_geometry;

                continue;
            }

            /*
             * --------------------------------------------------------
             * 8.9 Allocate MapPoint ID
             * --------------------------------------------------------
             */
            if (next_map_point_id_ ==
                std::numeric_limits<unsigned long>::max())
            {
                last_error_ =
                    "MapPoint ID overflow.";

                return false;
            }

            /*
             * 防御性检查。
             *
             * 当前设计要求：
             *   同一张 Map 由一个 Updater 顺序补点。
             */
            if (map.GetMapPoint(next_map_point_id_))
            {
                last_error_ =
                    "Duplicate generated MapPoint ID.";

                return false;
            }

            /*
             * --------------------------------------------------------
             * 8.10 Create MapPoint
             * --------------------------------------------------------
             */
            auto map_point =
                std::make_shared<MapPoint>(
                    next_map_point_id_,
                    Pw);

            /*
             * --------------------------------------------------------
             * 8.11 Insert Map
             * --------------------------------------------------------
             *
             * 先确认 Map 插入成功，
             * 再修改 Feature 关联。
             *
             * 避免出现：
             *
             *   Feature -> MapPoint
             *
             * 但是 Map 中根本不存在该点。
             */
            if (!map.InsertMapPoint(map_point))
            {
                last_error_ =
                    "MapPoint insertion failed.";

                return false;
            }

            /*
             * --------------------------------------------------------
             * 8.12 Feature -> MapPoint
             * --------------------------------------------------------
             *
             * 两个 Feature 观测同一个三维地图点。
             *
             * 内部使用 weak_ptr。
             */
            left_feature->setMapPoint(
                map_point);

            right_feature->setMapPoint(
                map_point);

            right_used[ri] = 1;

            /*
             * --------------------------------------------------------
             * 8.13 Statistics
             * --------------------------------------------------------
             */
            ++next_map_point_id_;

            ++stats_.new_map_points;
        }

        return true;
    }

    /*
     * ================================================================
     * Getters
     * ================================================================
     */
    const StereoMapUpdateStats&  StereoMapUpdater::stats() const
    {
        return stats_;
    }

    const std::string& StereoMapUpdater::lastError() const
    {
        return last_error_;
    }

} // namespace mini_vo
