#pragma once

#include <memory>
#include <cstddef>
#include <string>

#include "mini_vo/frame.h"
#include "mini_vo/frontend.h"
#include "mini_vo/map.h"
#include "mini_vo/map_initializer.h"
#include "mini_vo/pnp_tracker.h"
#include "mini_vo/stereo_camera.h"
#include "mini_vo/stereo_map_updater.h"
#include "mini_vo/temporal_matcher.h"
#include "mini_vo/keyframe.h"
#include "mini_vo/keyframe_selector.h"

namespace mini_vo
{
    /*
     * ================================================================
     * MultiFrameTrackingConfig
     * ================================================================
     */
    struct MultiFrameTrackingConfig
    {
        // 每张图像的 ORB 特征数量上限，以及时间匹配的 Ratio Test 阈值。
        int num_features = 1000;
        float temporal_ratio = 0.75f;

        MapInitializerConfig map_init;
        PnPConfig pnp;

        // false 使用 V1 固定地图；true 使用 V2 双目补图。
        bool enable_map_replenishment = false;
        StereoMapUpdateConfig map_update;

        /*
         * 是否启用 KeyFrame V1。
         *
         * false：
         *   原来的 V1/V2 测试行为保持不变。
         *
         * true：
         *   Tracking 成功后，
         *   根据 KeyFrameSelector 决定是否保存关键帧。
         */
        bool enable_keyframes = false;
        KeyFrameSelectionConfig keyframe_selection;
    };

    /*
     * ================================================================
     * MultiFrameTracker V3
     * ================================================================
     *
     * 负责调度：
     *
     *   Map Initialization
     *   Stereo Feature Extraction
     *   Stereo Matching
     *   Temporal Matching
     *   PnP Tracking
     *   MapPoint Replenishment
     *   KeyFrame
     *
     * Ownership:
     *
     *   Map                 owned
     *   MapInitializer      owned
     *   Frontend            owned
     *   TemporalMatcher     owned
     *   PnPTracker          owned
     *   StereoMapUpdater    owned
     *   LastFrame           shared_ptr
     *
     *   StereoCamera        const reference
     *
     *   相机由调用者持有，其生命周期必须覆盖整个 Tracker 的使用期。
     *
     *
     * 当前只支持单线程顺序处理 Frame。
     */
    class MultiFrameTracker
    {
    public:
        // 相机必须在 Initialize() 之前完成双目校正参数初始化。
        explicit MultiFrameTracker(const StereoCamera &stereo_camera,
                                   const MultiFrameTrackingConfig &config = MultiFrameTrackingConfig{});

        /*
         * 第一帧初始化。
         *
         * 成功后：
         *
         *   map_ 具有 MapPoints
         *   last_frame_ = frame0
         *   initialized_ = true
         *   tracked_frames_ = 1
         *
         * 世界坐标系采用第一帧校正后的左相机坐标系。
         * 一个 Tracker 实例只允许成功初始化一次。
         */
        bool Initialize(const Frame::Ptr &first_frame);

        /*
         * 连续帧跟踪。
         *
         * V1:
         *   ORB -> Temporal -> PnP
         *
         * V2:
         *   Stereo ORB
         *     -> Temporal -> PnP
         *     -> Stereo Map Update
         *
         *
         * 返回 true 表示 Pose Tracking 成功。
         *
         * Map Update 是一个独立的后处理阶段，
         * 其状态通过 mapUpdateSucceeded() 查询。
         * 即使补图失败，成功估计位姿的当前帧仍会成为 last_frame_。
         * 位姿跟踪失败时不推进 last_frame_ 和帧计数。
         */
        bool Track(const Frame::Ptr &current_frame);

        bool initialized() const;

        // 成功处理的帧数，包含第一帧；不是最近一帧的 ID。
        std::size_t trackedFrames() const;

        const Map &map() const;

        Frame::Ptr lastFrame() const;

        const MapInitializationStats &initializationStats() const;

        const TemporalMatchStats &temporalStats() const;

        const PnPTrackingStats &pnpStats() const;

        // 最近一次实际调用 StereoMapUpdater::Update() 的统计。
        const StereoMapUpdateStats &mapUpdateStats() const;

        // 在 Track() 返回 true 后查询；true 表示本轮没有补图错误。
        // 未启用补图，或补图正常执行但新增 0 个点时，同样返回 true。
        bool mapUpdateSucceeded() const;

        const std::string &lastMapUpdateError() const;

        // 初始化或位姿跟踪的错误；补图错误由 lastMapUpdateError() 提供。
        const std::string &lastError() const;

        /*
         * 获取最近插入的 KeyFrame。
         *
         * 返回 shared_ptr；
         * 调用者可以安全地暂时持有它。
         */
        KeyFrame::Ptr lastKeyFrame() const;

        // 关键帧更新是否正常与pnp_tracking 成功分开记录
        bool keyFrameUpdateSucceeded() const;
        const std::string &lastKeyFrameError() const;

    private:
        // 声明顺序决定初始化顺序；相机引用不拥有相机。
        const StereoCamera &stereo_camera_;
        MultiFrameTrackingConfig config_;

        /*
         * Map 的生命周期贯穿整个跟踪过程。
         *
         * 否则 Feature 的 weak_ptr<MapPoint>
         * 可能全部失效。
         */
        Map map_;

        MapInitializer map_initializer_;
        Frontend frontend_;
        TemporalMatcher temporal_matcher_;
        PnPTracker pnp_tracker_;
        StereoMapUpdater map_updater_;

        /*
         * 持有上一帧。
         *
         * 成功跟踪新帧以后，
         * 上一帧才可以被替换。
         */
        Frame::Ptr last_frame_;
        std::size_t tracked_frames_ = 0;
        bool initialized_ = false;
        std::string last_error_;

        // 地图更新与位姿跟踪分开统计
        bool map_update_success_ = true;
        std::string last_map_update_error_;

        // KeyFrameSelector只负责决策
        KeyFrameSelector keyframe_selector_;

        /*
         * 最近的 KeyFrame。
         *
         * 为什么是 weak_ptr？
         *
         * 因为 KeyFrame 的长期所有者已经是 Map。
         *
         * MultiFrameTracker 这里只需要引用它，
         * 没有必要增加第二条长期所有权链。
         */
        std::weak_ptr<KeyFrame> last_keyframe_;

        /*
         * 独立于 Frame ID 的 KeyFrame ID 分配器。
         *
         * KeyFrame ID：
         *   0, 1, 2, ...
         *
         * Frame ID：
         *   0, 5, 10, ...
         *
         * 未来删除关键帧时也不能复用旧 ID。
         */
        unsigned long next_keyfrmae_id_ = 0;

        // 与map update 一样 keyframe update独立维护状态。
        bool keyframe_update_success_ = true;
        std::string last_keyframe_error_;
    };

} // namespace mini_vo
