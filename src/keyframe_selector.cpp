#include "mini_vo/keyframe_selector.h"

#include <cmath>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace mini_vo
{
    KeyFrameSelector::KeyFrameSelector(const KeyFrameSelectionConfig &config) : config_(config)
    {
        if (config_.min_frame_gap == 0 || config_.max_frame_gap < config_.min_frame_gap || !std::isfinite(config_.min_translation_m) || config_.min_translation_m < 0.0 || !std::isfinite(config_.min_rotation_rad) || config_.min_rotation_rad < 0.0 || config_.min_rotation_rad > 3.14159)
        {
            throw std::invalid_argument("KeyFrameSelector: invalid config");
        }
    }

    /*
     * ================================================================
     * ShouldInsert
     * ================================================================
     */
    bool KeyFrameSelector::ShouldInsert(const KeyFrame &last_keyframe, const Frame &current,
                                        const PnPTrackingStats &tracking_stats) const
    {
        /*
         * 1. PnP 必须成功。
         *
         * 不能把 Tracking Lost 的 Frame
         * 直接作为新 KeyFrame。
         */
        if (!tracking_stats.success)
        {
            return false;
        }

        // 2.Fram ID 必须向前推进
        if (current.id() <= last_keyframe.frameId())
        {
            return false;
        }

        const unsigned long frame_gap = current.id() - last_keyframe.frameId();

        /*
         * 3. 最小插入间隔。
         *
         * 防止相邻每帧都被选为 KeyFrame。
         */
        if (frame_gap < config_.min_frame_gap)
        {
            return false;
        }

        // 4. 最大插入间隔
        if (frame_gap >= config_.max_frame_gap)
        {
            return true;
        }
        /*
                5.跟踪点较少 这是一个提前触发的条件
                但不保证插入 keyframe后 跟踪点数量一定会增加 */
        if (tracking_stats.final_inliers < config_.low_inlier_threshold)
        {
            return true;
        }

        /*
         * 6. 获得上一关键帧与当前帧的
         *    Camera -> World 变换。
         *
         * Frame::pose() 保存 T_cw，
         * 因此这里需要 inverse()。
         */
        const Eigen::Matrix4d T_wc_last = last_keyframe.frame()->pose().inverse();
        const Eigen::Matrix4d T_wc_current = current.pose().inverse();

        if (!T_wc_current.allFinite() || !T_wc_last.allFinite())
        {
            return false;
        }

        // 7.计算相机中心的平移距离
        const Eigen::Vector3d C_last = T_wc_last.block<3,1>(0,3);
        const Eigen::Vector3d C_current = T_wc_current.block<3,1>(0,3);
        const double translation = (C_current - C_last).norm();
        if(translation >= config_.min_translation_m)
        {
            return true;
        }

        // 8.计算相对旋转角 R_relative = R_last * R_current
        const Eigen::Matrix3d R_last = T_wc_last.block<3,3>(0,0);
        const Eigen::Matrix3d R_current = T_wc_current.block<3,3>(0,0);
        const Eigen::Matrix3d R_relative = R_last.transpose()* R_current;
        const Eigen::AngleAxisd angle_axis(R_relative);
        const double rotation_angle = angle_axis.angle();
        if(std::isfinite(rotation_angle)&& rotation_angle>= config_.min_rotation_rad)
        {
            return true;
        }

        // 所有插入条件都没有达到
        return false;

    }

}//namespace mini_vo
