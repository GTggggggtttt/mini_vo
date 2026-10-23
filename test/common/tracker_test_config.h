#pragma once
#include "mini_vo/multi_frame_tracker.h"
namespace mini_vo::test
{

    /*
     * 统一的 V2 基线参数。
     *
     * 各测试可以在返回的 config 上继续修改。
     */
    inline MultiFrameTrackingConfig MakeV2Config()
    {
        MultiFrameTrackingConfig config;

        config.num_features = 1000;
        config.temporal_ratio = 0.75f;

        config.map_init.min_depth = 0.5;
        config.map_init.max_depth = 50.0;
        config.map_init.ratio_threshold = 0.75f;
        config.map_init.max_vertical_error = 2.0f;
        config.map_init.min_map_points = 20;

        config.pnp.min_correspondences = 6;
        config.pnp.min_inliers = 15;
        config.pnp.ransac_iterations = 200;
        config.pnp.reprojection_error_px = 3.0;
        config.pnp.confidence = 0.99;
        config.pnp.min_inlier_ratio = 0.5;

        config.enable_map_replenishment = true;

        config.map_update.min_depth = 0.5;
        config.map_update.max_depth = 50.0;
        config.map_update.max_new_points_per_frame = 150;
        config.map_update.stereo_ratio_threshold = 0.75f;
        config.map_update.max_vertical_error = 2.0f;
        config.map_update.max_right_reprojection_error_px = 2.0;

        return config;
    }

    /*
     * 在 V2 上启用 KeyFrame V1。
     *
     * 不重复维护另一整套 PnP / Stereo 参数。
     */
inline MultiFrameTrackingConfig MakeKeyFrameConfig()
{
    auto config = MakeV2Config();
    config.enable_keyframes =true;

    config.keyframe_selection.min_frame_gap=3;
    config.keyframe_selection.max_frame_gap =8;
    config.keyframe_selection.min_translation_m =0.10;
    config.keyframe_selection.min_rotation_rad=0.15;
    config.keyframe_selection.low_inlier_threshold=100;

    return config;
    }
}