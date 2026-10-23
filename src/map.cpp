#include "mini_vo/map.h"
#include <unordered_set>
#include <utility>
#include <vector>

#include "mini_vo/feature.h"
#include "mini_vo/keyframe.h"
#include "mini_vo/mappoint.h"

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
        /*
         * ============================================================
         * 1. Basic Validation
         * ============================================================
         */
        if (!keyframe || !keyframe->frame())
        {
            return false;
        }
        /*
         * 不允许重复 KeyFrame ID。
         */
        if (key_frames_.find(keyframe->id()) != key_frames_.end())
        {
            return false;
        }

        auto frame = keyframe->frame();
        const auto &features = frame->leftFeatures();

        /*
         * ============================================================
         * 2. 收集该 KeyFrame 的有效地图关联
         * ============================================================
         *
         * 暂时不修改 MapPoint Observation。
         *
         * 先把所有候选检查完成，
         * 避免中途验证失败造成部分登记。
         */
        using Candidate = std::pair<MapPoint::Ptr, std::size_t>;
        std::vector<Candidate> candidates;
        candidates.reserve(features.size());

        /*
         * 防止同一个 MapPoint 被同一 KeyFrame
         * 的多个左目 Feature 重复登记。
         */
        std::unordered_set<unsigned long> used_mappoint_ids;
        for (std::size_t i = 0; i < features.size(); ++i)
        {
            const auto &feature = features[i];
            if (!feature)
            {
                continue;
            }
            auto map_point = feature->mapPoint();

            /*
             * 尚未关联 MapPoint 的 Feature
             * 不需要登记 Observation。
             */
            if (!map_point)
            {
                continue;
            }

            /*
             * Feature 必须属于当前 KeyFrame 的 Frame。
             */
            if (feature->frame().get() != frame.get())
            {
                return false;
            }

            /*
             * 这个 MapPoint 必须由当前 Map 管理。
             *
             * 防止把别的 Map 中的 MapPoint
             * 错误地登记进来。
             */
            const auto map_it = map_points_.find(map_point->id());

            if (map_it == map_points_.end() || map_it->second.get() != map_point.get())
            {
                return false;
            }

            /*
             * 每个 KeyFrame 对同一个 MapPoint
             * 最多一条左目观测。
             */
            if (!used_mappoint_ids.insert(map_point->id()).second)
            {
                return false;
            }

            // 不允许重复登记该keyframe
            if (map_point->HasObservation(keyframe->id()))
            {
                return false;
            }

            candidates.emplace_back(map_point, i);
        }

        /*
         * ============================================================
         * 3. Map 开始持有 KeyFrame
         * ============================================================
         */
        const auto inserted = key_frames_.emplace(keyframe->id(), keyframe);

        if (!inserted.second)
        {
            return false;
        }

        /*
         * ============================================================
         * 4. 登记 Observation
         * ============================================================
         */
        std::size_t registered = 0;
        for (const auto &candidate : candidates)
        {
            const auto &map_point = candidate.first;
            const std::size_t feature_index = candidate.second;
            if (!map_point->AddObservation(keyframe, feature_index))
            {
                /*
                 * 回滚此前已经登记的 Observation。
                 *
                 * 当前实现针对普通逻辑失败回滚；
                 * 尚未实现多线程事务或异常回滚。
                 */
                for (std::size_t j = 0; j < registered; ++j)
                {
                    candidates[j].first->RemoveObservation(keyframe->id());
                }
                /*
                 * 取消 Map 对该 KeyFrame 的所有权。
                 */
                key_frames_.erase(keyframe->id());
                return false;
            }
            ++registered;
        }
        return true;
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
         * 先删除 MapPoint 的 Observation 记录。
         *
         * 即使外部还持有某个 MapPoint::Ptr，
         * 它也不应该保留已清空地图中的观测索引。
         */
        for (auto &item : map_points_)
        {
            if (item.second)
            {
                item.second->ClearObservations();
            }
        }
        /*
         * 释放 Map 的 KeyFrame 所有权。
         */
        key_frames_.clear();


        // Map不再持有地图点
        map_points_.clear();
    }
} // namespace mini_vo
