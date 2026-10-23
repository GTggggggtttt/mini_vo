#include "mini_vo/mappoint.h"

#include "mini_vo/feature.h"
#include "mini_vo/frame.h"
#include "mini_vo/keyframe.h"


namespace mini_vo {


// ================================================================
// 构造函数
// ================================================================

MapPoint::MapPoint(
    unsigned long id,
    const Eigen::Vector3d& position,
    unsigned long created_frame_id
)
    : id_(id),
      position_(position),

      /*
       * [新增：Quality V1]
       *
       * 保存真实创建帧。
       *
       * observations_ 会自动默认构造为空容器；
       * last_tracked_frame_id_ 默认是 std::nullopt；
       * successful_track_count_ 默认是 0。
       */
      created_frame_id_(created_frame_id)
{
}


// ================================================================
// 基础属性接口
// ================================================================

unsigned long MapPoint::id() const noexcept
{
    return id_;
}


const Eigen::Vector3d&
MapPoint::position() const noexcept
{
    return position_;
}


void MapPoint::setPosition(
    const Eigen::Vector3d& position
)
{
    position_ = position;
}


// ================================================================
// Observation V1：添加观测
// ================================================================

bool MapPoint::AddObservation(
    const std::shared_ptr<KeyFrame>& keyframe,
    std::size_t left_feature_index
)
{
    /*
     * ------------------------------------------------------------
     * 1. 检查 KeyFrame 是否有效
     * ------------------------------------------------------------
     */
    if (!keyframe)
    {
        return false;
    }


    /*
     * ------------------------------------------------------------
     * 2. 获取 KeyFrame 对应的 Frame
     * ------------------------------------------------------------
     *
     * KeyFrame 保存 Frame 的强引用，
     * 因此正常情况下 Frame 应该仍然有效。
     */
    auto frame = keyframe->frame();

    if (!frame)
    {
        return false;
    }


    /*
     * ------------------------------------------------------------
     * 3. 检查 Feature 索引
     * ------------------------------------------------------------
     *
     * 防止访问 leftFeatures() 时越界。
     */
    const auto& features =
        frame->leftFeatures();

    if (left_feature_index >= features.size())
    {
        return false;
    }


    const auto& feature =
        features[left_feature_index];

    if (!feature)
    {
        return false;
    }


    /*
     * ------------------------------------------------------------
     * 4. 验证 Feature 确实属于当前 Frame
     * ------------------------------------------------------------
     *
     * 避免把其他 Frame 中的 Feature
     * 错误地登记到当前 KeyFrame。
     */
    auto feature_frame = feature->frame();

    if (!feature_frame ||
        feature_frame.get() != frame.get())
    {
        return false;
    }


    /*
     * ------------------------------------------------------------
     * 5. 验证 Feature 已关联当前 MapPoint
     * ------------------------------------------------------------
     *
     * feature->mapPoint() 返回关联的 shared_ptr。
     *
     * 比较对象地址，而不是只比较 ID：
     * 可以防止不同对象意外使用相同 ID 时
     * 被当成同一个对象。
     */
    auto associated_map_point =
        feature->mapPoint();

    if (associated_map_point.get() != this)
    {
        return false;
    }


    /*
     * ------------------------------------------------------------
     * 6. 防止重复登记
     * ------------------------------------------------------------
     *
     * 同一个 KeyFrame 对同一个 MapPoint
     * 只能有一条左目 Observation。
     */
    if (HasObservation(keyframe->id()))
    {
        return false;
    }


    /*
     * ------------------------------------------------------------
     * 7. 创建并插入 Observation
     * ------------------------------------------------------------
     */
    Observation observation;

    observation.keyframe = keyframe;

    observation.left_feature_index =
        left_feature_index;


    const auto result =
        observations_.emplace(
            keyframe->id(),
            observation
        );


    /*
     * unordered_map::emplace()
     *
     * result.second == true：
     * 插入成功。
     *
     * result.second == false：
     * 对应的 KeyFrame ID 已经存在。
     */
    return result.second;
}


// ================================================================
// Observation V1：删除观测
// ================================================================

bool MapPoint::RemoveObservation(
    unsigned long keyframe_id
)
{
    /*
     * erase() 返回被删除元素的数量。
     *
     * 0：没有找到对应记录。
     * 1：成功删除。
     */
    return observations_.erase(keyframe_id) != 0;
}


// ================================================================
// Observation V1：检查观测
// ================================================================

bool MapPoint::HasObservation(
    unsigned long keyframe_id
) const
{
    return observations_.find(keyframe_id)
        != observations_.end();
}


// ================================================================
// Observation V1：观测数量
// ================================================================

std::size_t
MapPoint::observationCount() const noexcept
{
    return observations_.size();
}


// ================================================================
// Observation V1：有效观测数量
// ================================================================

std::size_t
MapPoint::activeObservationCount() const noexcept
{
    std::size_t count = 0;

    for (const auto& item : observations_)
    {
        /*
         * weak_ptr::expired()
         *
         * true：
         * KeyFrame 已经销毁，
         * 或不存在可用的强引用。
         *
         * false：
         * 仍然可以通过 lock() 获取 KeyFrame。
         */
        if (!item.second.keyframe.expired())
        {
            ++count;
        }
    }

    return count;
}


// ================================================================
// Observation V1：只读访问
// ================================================================

const MapPoint::ObservationContainer&
MapPoint::observations() const noexcept
{
    return observations_;
}


// ================================================================
// Observation V1：清理全部观测
// ================================================================

void MapPoint::ClearObservations()
{
    /*
     * 由 Map::Clear() 调用。
     *
     * 这里只清空 Observation 容器，
     * 不销毁 KeyFrame 或 MapPoint。
     */
    observations_.clear();
}


// ================================================================
// [新增：Quality V1] 获取创建帧 ID
// ================================================================

unsigned long
MapPoint::createdFrameId() const noexcept
{
    return created_frame_id_;
}


// ================================================================
// [新增：Quality V1] 获取最近成功跟踪帧 ID
// ================================================================

std::optional<unsigned long>
MapPoint::lastTrackedFrameId() const noexcept
{
    /*
     * 返回 optional 的副本。
     *
     * 调用者不能通过返回值修改
     * MapPoint 内部的跟踪历史。
     */
    return last_tracked_frame_id_;
}


// ================================================================
// [新增：Quality V1] 获取累计成功跟踪次数
// ================================================================

std::size_t
MapPoint::successfulTrackCount() const noexcept
{
    return successful_track_count_;
}


// ================================================================
// [新增：Quality V1] 记录一次成功跟踪
// ================================================================

bool MapPoint::RecordSuccessfulTrack(
    unsigned long frame_id
)
{
    /*
     * ------------------------------------------------------------
     * 1. 检查跟踪帧不能早于创建帧
     * ------------------------------------------------------------
     *
     * 例如：
     *
     * created_frame_id_ = 10
     *
     * RecordSuccessfulTrack(9)
     *
     * 属于无效的时间顺序。
     */
    if (frame_id < created_frame_id_)
    {
        return false;
    }


    /*
     * ------------------------------------------------------------
     * 2. 检查 Frame ID 必须严格递增
     * ------------------------------------------------------------
     *
     * 防止同一帧重复累计。
     *
     * 例如：
     *
     * RecordSuccessfulTrack(11) -> true
     * RecordSuccessfulTrack(11) -> false
     * RecordSuccessfulTrack(12) -> true
     */
    if (last_tracked_frame_id_.has_value())
    {
        if (frame_id <= *last_tracked_frame_id_)
        {
            return false;
        }
    }


    /*
     * ------------------------------------------------------------
     * 3. 提交跟踪历史
     * ------------------------------------------------------------
     */
    last_tracked_frame_id_ = frame_id;

    ++successful_track_count_;

    return true;
}


} // namespace mini_vo