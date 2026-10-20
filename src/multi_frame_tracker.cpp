#include "mini_vo/multi_frame_tracker.h"

#include <cmath>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace mini_vo
{
    /*
     * ================================================================
     * Constructor
     * ================================================================
     */
    MultiFrameTracker::MultiFrameTracker(
        const StereoCamera &stereo_camera,
        const MultiFrameTrackingConfig &config)
        : stereo_camera_(stereo_camera),
          config_(config),
          map_initializer_(config.map_init),
          frontend_(config.num_features),
          temporal_matcher_(config.temporal_ratio),
          pnp_tracker_(config.pnp),
          map_updater_(config.map_update),
          keyframe_selector_(config.keyframe_selection)
    {
        if (config_.num_features <= 0)
        {
            throw std::invalid_argument("MultiFrameTracker: invalid num_features.");
        }
    }

    /*
     * ================================================================
     * Initialize
     * ================================================================
     *
     * 只负责 Frame0。
     */
    bool MultiFrameTracker::Initialize(const Frame::Ptr &first_frame)
    {
        last_error_.clear();

        if (initialized_)
        {
            last_error_ = "Tracker has already been initialized.";
            return false;
        }

        if (!first_frame)
        {
            last_error_ = "Null first Frame.";
            return false;
        }

        if (!stereo_camera_.initialized())
        {
            last_error_ = "StereoCamera not initialized.";
            return false;
        }

        // 1. Stereo Rectification
        cv::Mat left_rectified;
        cv::Mat right_rectified;

        if (!stereo_camera_.Rectify(first_frame->leftImage(), first_frame->rightImage(),
                                    left_rectified, right_rectified))
        {
            last_error_ = "First Frame rectification failed.";
            return false;
        }

        /*
         * ============================================================
         * 2. Map Initialization
         * ============================================================
         *
         * 使用局部 Frontend 初始化地图。
         *
         * 初始化完成后：
         *
         *   Frame0 拥有 Feature
         *   Feature 拥有独立 Descriptor
         *   Map 拥有 MapPoint
         *
         * 所以局部 Frontend 可以正常销毁。
         */
        Frontend stereo_frontend(config_.num_features);

        if (!map_initializer_.Initialize(first_frame, left_rectified, right_rectified, stereo_camera_, stereo_frontend, map_))
        {
            last_error_ = map_initializer_.lastError();
            return false;
        }

        /*
         * ============================================================
         * 3. V2 Stereo Matching Parameters
         * ============================================================
         *
         * 后续连续帧 Frontend 使用 map_update 中的
         * 双目匹配配置。
         *
         * 特别注意：
         *
         *   MapInitializer 中配置的是局部 stereo_frontend。
         *
         *   这里配置的是成员 frontend_。
         *
         * 二者不是同一个对象。
         */
        if (config_.enable_map_replenishment)
        {
            const double fb = stereo_camera_.fx() * stereo_camera_.rectifiedBaseline();

            if (!std::isfinite(fb) || fb <= 0.0)
            {
                last_error_ = " Invalid stereo focal-baseline product.";
                return false;
            }

            frontend_.SetStereoMatchingParameters(config_.map_update.stereo_ratio_threshold, config_.map_update.max_vertical_error,
                                                  static_cast<float>(fb / config_.map_update.max_depth), static_cast<float>(fb / config_.map_update.min_depth));
        }

        /*
         * ================================================================
         * KeyFrame V1: Initial KeyFrame
         * ================================================================
         *
         * Frame0 已经完成：
         *
         *   Stereo Matching
         *   Initial MapPoint Creation
         *   Feature -> MapPoint Association
         *
         * 因此可以将其保存为 KeyFrame0。
         */
        if (config_.enable_keyframes)
        {
            /*
             * 构造关键帧。
             *
             * 这里只复制 Frame::Ptr，
             * 不复制 Frame 内部的图像和 Feature。
             */
            auto keyframe0 = std::make_shared<KeyFrame>(next_keyfrmae_id_, first_frame);

            // Map 开始拥有keyframe0
            if (!map_.InsertKeyFrame(keyframe0))
            {
                last_error_ = "Failed to insert initial keyframe.";
                return false;
            }

            /*
             * last_keyframe_ 是 weak_ptr。
             *
             * 它不会增加 KeyFrame 的强引用计数。
             */
            last_keyframe_ = keyframe0;
            ++next_keyfrmae_id_;
        }

        /*
         * ============================================================
         * 4. 保存 LastFrame commit Initialization
         * ============================================================
         *
         * 只有地图初始化成功，
         * 才正式进入 TRACKING 状态。
         */
        last_frame_ = first_frame;
        tracked_frames_ = 1;
        initialized_ = true;
        return true;
    }

    /*
     * ================================================================
     * Track
     * ================================================================
     *
     * 每调用一次，处理一个新的 Frame。
     *
     * 不读取 Dataset，也不控制循环。
     *
     * 所以这个类以后可以支持：
     *
     *   EuRoC
     *   KITTI
     *   ROS Camera Topic
     *
     * 只要调用者提供 Frame 即可。
     */
    bool MultiFrameTracker::Track(const Frame::Ptr &current_frame)
    {
        // 清空错误变量
        last_error_.clear();
        last_map_update_error_.clear();
        last_keyframe_error_.clear();

        map_update_success_ = true;
        keyframe_update_success_ = true;

        /*
         * ============================================================
         * 1. State Check
         * ============================================================
         */
        if (!initialized_ || !last_frame_)
        {
            last_error_ = "Tracker is not initialized.";
            return false;
        }

        if (!current_frame)
        {
            last_error_ = " Null current Frame.";
            return false;
        }

        /*
         * Frame ID 必须向前推进。
         *
         * 防止：
         *   重复处理同一帧
         *   将旧帧作为新帧
         */
        if (current_frame->id() <= last_frame_->id())
        {
            last_error_ = "Current Frame ID is not newer.";
            return false;
        }

        // V1只接受尚未处理的新 Feame
        if (current_frame->leftFeatureCount() != 0 || current_frame->rightFeatureCount() != 0)
        {
            last_error_ = "Current Frame already contains Features.";
            return false;
        }

        /*
         * ============================================================
         * 2. Stereo Rectification
         * ============================================================
         *
         * 虽然当前 PnP 只使用左图，
         * 但 StereoCamera::Rectify() 接口
         * 需要输入左右图像。
         *
         * 后续可以增加专用 RectifyLeft()，
         * 避免不必要的右图 remap。
         */
        cv::Mat left_rectified;
        cv::Mat right_rectified;

        if (!stereo_camera_.Rectify(current_frame->leftImage(), current_frame->rightImage(),
                                    left_rectified, right_rectified))
        {
            last_error_ = "Current Frame Stereo Retification failed.";
            return false;
        }

        /*
         * ============================================================
         * 3. Feature Extraction
         * ============================================================
         *
         * V1:
         *
         *   只提取左图 ORB。
         *
         *
         * V2:
         *
         *   提取左右图 ORB，
         *   并匹配同一帧的左右 Feature。
         *
         * 必须让 Temporal Matching 和 Stereo Matching
         * 共享同一批 Left Feature。
         */

        if (config_.enable_map_replenishment)
        {
            /*
             * 左右图同时提取。
             *
             * 即使右图未找到特征，
             * 只要左图特征有效，
             * 仍然可以尝试 Temporal Tracking。
             */
            const bool stereo_extracted = frontend_.ExtractStereoFeatures(left_rectified, right_rectified);

            if (frontend_.leftFeatures().empty())
            {
                last_error_ = "Current Frame left ORB extraction failed.";
                return false;
            }

            /*
             * Stereo Matching 失败或者匹配为 0，
             * 不应该直接导致 PnP 跟踪失败。
             *
             * 因为 PnP 依然可以依赖旧 MapPoint。
             */
            if (stereo_extracted)
            {
                frontend_.MatchStereo();
            }

            current_frame->setLeftFeatures(frontend_.leftFeatures());
            current_frame->setRightFeatures(frontend_.rightFeatures());
        }
        else
        {
            // 保留v1行为
            const int feature_count = frontend_.ExtractFeatures(left_rectified);
            if (feature_count <= 0)
            {
                last_error_ = "Current Frame ORB extraction failed.";
                return false;
            }
            current_frame->setLeftFeatures(frontend_.features());
        }

        /*
         * ============================================================
         * 4. Frame 保存 Feature
         * ============================================================
         * 建立 Feature -> Frame。
         * Feature 内部使用 weak_ptr。
         */
        for (const auto &feature : current_frame->leftFeatures())
        {
            if (feature)
            {
                feature->setFrame(current_frame);
            }
        }

        for (const auto &feature : current_frame->rightFeatures())
        {
            if (feature)
            {
                feature->setFrame(current_frame);
            }
        }

        /*
         * ============================================================
         * 5. Temporal Matching
         * ============================================================
         *
         * LastFrame：
         *
         *   有一部分 Feature 关联 MapPoint。
         *
         * CurrentFrame：
         *
         *   只有新提取的 2D Feature。
         */
        const int match_count = temporal_matcher_.Match(*last_frame_, *current_frame);

        if (match_count < config_.pnp.min_correspondences)
        {
            last_error_ = "Too few temporal matches.";
            return false;
        }

        /*
         * ============================================================
         * 6. PnP Tracking
         * ============================================================
         *
         * PnPTracker 会：
         *
         *   1. 构造 Pw ↔ pixel
         *   2. solvePnPRansac()
         *   3. 验证 Inlier
         *   4. 更新 Frame Pose
         *   5. 关联 Inlier Feature -> MapPoint
         *
         * 失败时不会修改 Frame Pose 或 MapPoint 关联。
         */
        const bool pnp_success = pnp_tracker_.Track(*last_frame_, *current_frame,
                                                    temporal_matcher_.matches(), stereo_camera_);

        if (!pnp_success)
        {
            last_error_ = pnp_tracker_.lastError();
            return false;
        }

        /*
         * 到这里：
         *
         *   Frame::pose() 已经得到 T_cw
         *
         *   当前帧 Inlier Feature 已关联旧 MapPoint
         *
         *   Outlier / 未匹配 Feature 仍然没有 MapPoint
         */
        /*
         * ============================================================
         * 7. V2: Stereo MapPoint Replenishment
         * ============================================================
         *
         * 当前已经知道：
         *
         *   T_cw(k)
         *
         * 因此可以将当前双目恢复的 Pc
         * 转换为世界坐标：
         *
         *   Pw = inverse(T_cw) * Pc
         *
         *
         * 最重要的是：
         *
         *   不重新提取 Feature！
         *
         * 使用前面已有的：
         *
         *   frontend_.stereoMatches()
         *
         * 它的索引直接对应：
         *
         *   current_frame->leftFeatures()
         *   current_frame->rightFeatures()
         */

        if (config_.enable_map_replenishment)
        {
            map_update_success_ = map_updater_.Update(current_frame, frontend_.stereoMatches(), stereo_camera_, map_);

            if (!map_update_success_)
            {
                last_map_update_error_ = map_updater_.lastError();
            }
        }

        /*
         * ================================================================
         * KeyFrame V1: KeyFrame Selection
         * ================================================================
         *
         * 调用前必须完成：
         *
         *   1. Temporal Matching
         *   2. PnP Tracking
         *   3. Current Frame Pose T_cw
         *
         * 当前 V2 还允许继续完成 Stereo Map Update。
         *
         * KeyFrameSelection 放在其后，
         * 保留当前帧已经建立的地图关联。
         */
        if (config_.enable_keyframes && map_update_success_)
        {
            /*
             * 从 weak_ptr 临时取得 KeyFrame。
             *
             * 如果 Map 已经被清空，
             * lock() 可能返回 nullptr。
             */
            auto previous_keyframe = last_keyframe_.lock();

            if (!previous_keyframe)
            {
                keyframe_update_success_ = false;
                last_keyframe_error_ = "Previous KeyFrame expired.";
            }
            else
            {
                /*
                 * KeyFrameSelector 只做决策。
                 *
                 * 它不会修改 Frame 或 Map。
                 */
                const bool need_new_keyframe = keyframe_selector_.ShouldInsert(*previous_keyframe, *current_frame, pnp_tracker_.stats());

                if (need_new_keyframe)
                {
                    /*
                     * 新关键帧保存当前已经完成 PnP
                     * 和可选地图更新的 Frame。
                     */
                    auto new_keyframe = std::make_shared<KeyFrame>(next_keyfrmae_id_, current_frame);

                    // Map 获得长期所有权
                    if (!map_.InsertKeyFrame(new_keyframe))
                    {
                        keyframe_update_success_ = false;
                        last_keyframe_error_ = "Failed to insert new KeyFrame.";
                    }
                    else
                    {
                        // 更新最近关键帧的引用
                        last_keyframe_ = new_keyframe;
                        ++next_keyfrmae_id_;
                    }
                }
            }
        }

        /*
         * ============================================================
         * 8. Commit Current Frame
         * ============================================================
         *
         * PnP 已成功。
         *
         * 因此即使可选的 Map Update 出现问题，
         * 当前 Pose 仍然可以成立。
         *
         * 这里 Track() 的 true 专门表示：
         *
         *   Pose Tracking 成功。
         *
         * 地图更新是否成功由另一个状态记录。
         *
         * 如果地图更新发生硬错误，
         * 上层应记录并决定是否停止运行。
         */
        last_frame_ = current_frame;

        ++tracked_frames_;
        return true;
    }

    /*
     * ================================================================
     * Getters
     * ================================================================
     */
    bool MultiFrameTracker::initialized() const
    {
        return initialized_;
    }

    std::size_t MultiFrameTracker::trackedFrames() const
    {
        return tracked_frames_;
    }

    const Map &MultiFrameTracker::map() const
    {
        return map_;
    }

    Frame::Ptr MultiFrameTracker::lastFrame() const
    {
        return last_frame_;
    }

    const MapInitializationStats &MultiFrameTracker::initializationStats() const
    {
        return map_initializer_.stats();
    }

    const TemporalMatchStats &MultiFrameTracker::temporalStats() const
    {
        return temporal_matcher_.stats();
    }

    const PnPTrackingStats &MultiFrameTracker::pnpStats() const
    {
        return pnp_tracker_.stats();
    }

    const std::string &MultiFrameTracker::lastError() const
    {
        return last_error_;
    }

    const StereoMapUpdateStats &MultiFrameTracker::mapUpdateStats() const
    {
        return map_updater_.stats();
    }

    bool MultiFrameTracker::mapUpdateSucceeded() const
    {
        return map_update_success_;
    }

    const std::string &MultiFrameTracker::lastMapUpdateError() const
    {
        return last_map_update_error_;
    }

    KeyFrame::Ptr MultiFrameTracker::lastKeyFrame() const
    {
        return last_keyframe_.lock();
    }

    bool MultiFrameTracker::keyFrameUpdateSucceeded() const{
        return keyframe_update_success_;
    }

    const std::string& MultiFrameTracker::lastKeyFrameError() const
    {
        return last_keyframe_error_;
    }



} // namespace mini_vo
