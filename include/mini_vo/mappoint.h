#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>

#include <Eigen/Core>


namespace mini_vo {

// 前向声明：避免头文件之间不必要的相互包含。
class KeyFrame;
class Map;


/*
 * ================================================================
 * MapPoint
 * ================================================================
 *
 * 表示世界坐标系中的一个三维地图点。
 *
 * 当前职责：
 *
 * 1. 保存地图点 ID 和三维位置。
 * 2. 记录该地图点被哪些 KeyFrame 观测过。
 * 3. 记录该地图点的创建时间与成功跟踪历史。
 *
 * 注意：
 *
 * Map 负责长期持有 MapPoint。
 *
 * Observation 使用 weak_ptr 引用 KeyFrame，
 * 避免形成循环强引用。
 */
class MapPoint
{
public:

    // ============================================================
    // 基础类型定义
    // ============================================================

    using Ptr = std::shared_ptr<MapPoint>;

    using ConstPtr = std::shared_ptr<const MapPoint>;


    /*
     * ============================================================
     * Observation V1：关键帧观测记录
     * ============================================================
     *
     * 表示某个 KeyFrame 的左目 Feature
     * 观测到了当前 MapPoint。
     */
    struct Observation
    {
        /*
         * 非拥有引用。
         *
         * Map 负责持有 KeyFrame，
         * MapPoint 不应该反过来拥有 KeyFrame。
         */
        std::weak_ptr<KeyFrame> keyframe;


        /*
         * Feature 在 KeyFrame 对应 Frame 的
         * leftFeatures() 中的索引。
         *
         * 后续可以通过：
         *
         * keyframe->frame()->leftFeatures()[index]
         *
         * 恢复对应的二维观测。
         */
        std::size_t left_feature_index = 0;
    };


    /*
     * 使用 KeyFrame ID 作为键。
     *
     * 每个 KeyFrame 对当前 MapPoint
     * 最多记录一条左目 Observation。
     */
    using ObservationContainer =
        std::unordered_map<
            unsigned long,
            Observation
        >;


    // ============================================================
    // 构造函数
    // ============================================================

    /*
     * [新增：Quality V1]
     *
     * 增加 created_frame_id 参数。
     *
     * 默认值 0 保持旧代码兼容：
     *
     * MapPoint(id, position)
     *
     * 新的建图代码应传入真实创建帧：
     *
     * MapPoint(id, position, frame->id())
     */
    explicit MapPoint(
        unsigned long id,
        const Eigen::Vector3d& position,
        unsigned long created_frame_id = 0
    );


    // ============================================================
    // 基础属性接口
    // ============================================================

    unsigned long id() const noexcept;


    const Eigen::Vector3d& position() const noexcept;


    void setPosition(
        const Eigen::Vector3d& position
    );


    // ============================================================
    // Observation V1：观测关系接口
    // ============================================================

    /*
     * 添加一个 KeyFrame 对当前 MapPoint 的观测。
     *
     * 前提：
     *
     * 1. KeyFrame 有效。
     * 2. Feature 索引有效。
     * 3. Feature 属于 KeyFrame 对应的 Frame。
     * 4. Feature 已经关联当前 MapPoint。
     * 5. 该 KeyFrame 尚未登记此 MapPoint。
     *
     * 返回 true 表示添加成功。
     */
    bool AddObservation(
        const std::shared_ptr<KeyFrame>& keyframe,
        std::size_t left_feature_index
    );


    /*
     * 删除指定 KeyFrame 的 Observation。
     *
     * 只删除观测记录，
     * 不销毁 KeyFrame 或 MapPoint。
     */
    bool RemoveObservation(
        unsigned long keyframe_id
    );


    /*
     * 当前 MapPoint 是否登记过指定 KeyFrame。
     */
    bool HasObservation(
        unsigned long keyframe_id
    ) const;


    /*
     * Observation 容器中的记录数量。
     *
     * 注意：
     * 该数量包含 weak_ptr 已失效的记录。
     */
    std::size_t observationCount() const noexcept;


    /*
     * KeyFrame 仍然存活的 Observation 数量。
     */
    std::size_t activeObservationCount() const noexcept;


    /*
     * 只读访问全部 Observation。
     *
     * 不允许调用者通过返回的容器
     * 直接修改观测关系。
     */
    const ObservationContainer&
    observations() const noexcept;


    // ============================================================
    // [新增：Quality V1] 地图点质量统计接口
    // ============================================================

    /*
     * 创建该 MapPoint 时的 Frame ID。
     */
    unsigned long createdFrameId() const noexcept;


    /*
     * 最近一次成功参与 PnP 跟踪的 Frame ID。
     *
     * std::nullopt 表示：
     * 该地图点还没有被记录过成功跟踪。
     */
    std::optional<unsigned long>
    lastTrackedFrameId() const noexcept;


    /*
     * 累计成功跟踪次数。
     *
     * 只统计最终被 PnP 验证为 Inlier 的观测。
     */
    std::size_t successfulTrackCount() const noexcept;


    /*
     * 记录该 MapPoint 在指定 Frame 中
     * 成功参与了一次位姿跟踪。
     *
     * 返回 false 的情况：
     *
     * 1. Frame ID 早于创建帧。
     * 2. Frame ID 不大于上一次记录的 Frame ID。
     *
     * 这样可以防止同一帧重复累计。
     */
    bool RecordSuccessfulTrack(
        unsigned long frame_id
    );


private:

    // ============================================================
    // 原有基础数据
    // ============================================================

    /*
     * MapPoint 的唯一 ID。
     */
    unsigned long id_ = 0;


    /*
     * MapPoint 在世界坐标系中的三维坐标 Pw。
     */
    Eigen::Vector3d position_ =
        Eigen::Vector3d::Zero();


    // ============================================================
    // Observation V1：观测关系容器
    // ============================================================

    /*
     * KeyFrame ID -> Observation。
     *
     * MapPoint 不拥有 KeyFrame，
     * Observation 内部通过 weak_ptr 引用 KeyFrame。
     */
    ObservationContainer observations_;


    // ============================================================
    // [新增：Quality V1] 地图点生命周期与跟踪历史
    // ============================================================

    /*
     * 创建该 MapPoint 时所在的 Frame ID。
     *
     * 例如：
     *
     * Frame0 创建：
     * created_frame_id_ = 0
     *
     * Frame20 创建：
     * created_frame_id_ = 20
     */
    unsigned long created_frame_id_ = 0;


    /*
     * 最近一次成功跟踪的 Frame ID。
     *
     * 尚未被成功跟踪时保持 std::nullopt。
     */
    std::optional<unsigned long>
        last_tracked_frame_id_;


    /*
     * 累计成功跟踪次数。
     *
     * 仅在 RecordSuccessfulTrack()
     * 成功执行时递增。
     */
    std::size_t successful_track_count_ = 0;


    // ============================================================
    // 内部维护接口
    // ============================================================

    /*
     * 仅允许 Map 在清理地图时调用。
     *
     * 防止普通调用者随意破坏
     * MapPoint 与 KeyFrame 的观测关系。
     */
    friend class Map;


    /*
     * 清空当前 MapPoint 的全部 Observation。
     *
     * 由 Map::Clear() 使用。
     */
    void ClearObservations();
};


} // namespace mini_vo