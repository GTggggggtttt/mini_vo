#include "mini_vo/map.h"

namespace mini_vo
{
    bool Map::InsertMapPoint(const MapPoint::Ptr &map_point)
    {
        if (!map_point)
        {
            return false;
        }

        // 获取mappoint ID
        const unsigned long id = map_point->id();

        const auto iterator = map_points_.find(id);

        if (iterator != map_points_.end())
        {
            // 存在相同id
            return false;
            // v1版本不允许静默覆盖
        }

        /*
         * ============================================================
         * 插入 MapPoint
         * ============================================================
         *
         * emplace：
         *
         *      key = id
         *      value = shared_ptr<MapPoint>
         *
         *
         * shared_ptr 被复制进 unordered_map 后，
         * 引用计数增加。
         *
         * 因此即使调用者自己的 shared_ptr
         * 离开作用域：
         *
         *      MapPoint 仍然活着
         *
         * 因为 Map 还拥有它。
         */
        map_points_.emplace(id, map_point);
        return true;
    }

    /*
     * ================================================================
     * GetMapPoint
     * ================================================================
     */
    MapPoint::Ptr Map::GetMapPoint(unsigned long id) const
    {
        /*
         * 在 unordered_map 中查找 ID。
         */
        const auto iterator = map_points_.find(id);

        if (iterator == map_points_.end())
        {
            return nullptr;
        }

        return iterator->second;
    }

    /*
     * ================================================================
     * InsertKeyFrame
     * ================================================================
     */
    bool Map::InsertKeyFrame(const KeyFrame::Ptr &keyframe)
    {
        if (!keyframe)
        {
            return false;
        }
        /*
         * unordered_map::emplace()
         *
         * 若 ID 已存在，不覆盖原来的 KeyFrame。
         *
         * second:
         *   true  插入成功
         *   false ID 冲突
         */
        const auto result = key_frames_.emplace(keyframe->id(), keyframe);
        return result.second;
    }

    /*
     * ================================================================
     * GetKeyFrame
     * ================================================================
     */
    KeyFrame::Ptr Map::GetKeyFrame(unsigned long id) const
    {
        const auto it = key_frames_.find(id);
        if (it == key_frames_.end())
        {
            return nullptr;
        }
        return it->second;
    }

    /*
     * ================================================================
     * keyFrames
     * ================================================================
     */
    const Map::KeyFrameContainer &Map::keyFrames() const
    {
        return key_frames_;
    }

    /*
     * ================================================================
     * keyFramecount
     * ================================================================
     */

    std::size_t Map::keyFrameCount() const
    {
        return key_frames_.size();
    }

    const Map::MapPointContainer &Map::mapPoints() const { return map_points_; }

    std::size_t Map::size() const { return map_points_.size(); }

    bool Map::empty() const { return map_points_.empty(); }

    void Map::Clear()
    {
        /*
         * Map 不再持有关键帧。
         *
         * 如果外部还有 shared_ptr，
         * KeyFrame 仍然可能存活。
         */
        key_frames_.clear();

        // Map不再持有地图点
        map_points_.clear();
    }
} // namespace mini_vo
