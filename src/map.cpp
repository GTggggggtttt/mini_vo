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

        if(iterator == map_points_.end())
        {
            return nullptr;
        }

        return iterator->second;

    }

    const Map::MapPointContainer& Map::mapPoints() const{return map_points_;}

    std::size_t Map::size() const{return map_points_.size();}

    bool Map::empty() const {return map_points_.empty();}

    void Map::Clear() { map_points_.clear();}
}//namespace mini_vo