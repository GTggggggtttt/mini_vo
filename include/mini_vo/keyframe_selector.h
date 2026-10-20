#pragma once
#include "mini_vo/frame.h"
#include "mini_vo/keyframe.h"
#include "mini_vo/pnp_tracker.h"

namespace mini_vo
{
    /*
     * ================================================================
     * KeyFrameSelectionConfig
     * ================================================================
     *
     * 以下是教学阶段的经验参数。
     *
     * 后续可以通过 EuRoC 实验调整。
     */
    struct KeyFrameSelectionConfig
    {
        /*
         * 至少间隔多少个 Frame ID，
         * 才允许插入新的 KeyFrame。
         */
        unsigned long min_frame_gap = 3;

        /*
         * 最多间隔多少个 Frame ID。
         *
         * 到达此间隔后强制选择 KeyFrame。
         */
        unsigned long max_frame_gap = 8;

        // 相对上一Keyframe 的相机中心位移。单位：m
        double min_translation_m = 0.10;

        // 相对上一KeyFrame 的旋转角度 单位：rad
        double min_rotation_rad = 0.15;

        // 当PnP Inlier 数量将至此值以下时，可以提前插入KeyFrame
        std::size_t low_inlier_threshold = 100;
    };

    /*
     * ================================================================
     * KeyFrameSelector
     * ================================================================
     *
     * 只做决策：
     *
     *    ShouldInsert() -> true / false
     *
     * 不拥有 Frame。
     * 不修改 Frame。
     * 不向 Map 插入 KeyFrame。
     *
     * 这是职责分离的关键。
     */
    class KeyFrameSelector
    {
    public:
        explicit KeyFrameSelector(const KeyFrameSelectionConfig &config = KeyFrameSelectionConfig{});

        bool ShouldInsert(const KeyFrame &last_keyframe, const Frame &current, const PnPTrackingStats &tracking_stats) const;

    private:
        KeyFrameSelectionConfig config_;
    };

} // namespace mini_vo
