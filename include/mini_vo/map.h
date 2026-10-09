#pragma once
#include <cstddef>
#include <memory>
#include <unordered_map>
#include "mini_vo/mappoint.h"

namespace mini_vo
{
    /*
     * ================================================================
     * Map
     * ================================================================
     *
     * Map 表示当前 VO / SLAM 系统维护的地图。
     *
     * 当前 V1 只负责：
     *
     *      管理 MapPoint
     *
     *
     * 当前拥有的数据：
     *
     *      id -> shared_ptr<MapPoint>
     *
     *
     * 为什么使用 unordered_map？
     *
     * 因为：
     *
     *      MapPoint 本身具有唯一 id
     *
     * 后面经常会出现：
     *
     *      根据 id 查找 MapPoint
     *
     * unordered_map 平均查找复杂度：
     *
     *      O(1)
     *
     *
     * 当前暂时不负责：
     *
     *      KeyFrame
     *      Local Map
     *      Active MapPoint
     *      Backend
     *      Map cleanup
     *      Thread safety
     *
     * 这些后面逐渐加入。
     * ================================================================
     */
    class Map
    {
    public:
        using ptr = std::shared_ptr<Map>;

        /*
         * MapPoint 容器类型。
         *
         * key：
         *
         *      MapPoint ID
         *
         * value：
         *
         *      shared_ptr<MapPoint>
         *
         *
         * Map 通过 shared_ptr 参与 MapPoint 生命周期管理。
         */

        using MapPointContainer = std::unordered_map<unsigned long, MapPoint::Ptr>;

        Map() = default;

        /*
         * ============================================================
         * InsertMapPoint
         * ============================================================
         *
         * 向地图中插入一个 MapPoint。
         *
         *
         * 返回：
         *
         *      true
         *          插入成功
         *
         *      false
         *          nullptr
         *          或 ID 已经存在
         *
         *
         * 为什么不直接覆盖已有 MapPoint？
         *
         * 因为 MapPoint ID 应该具有唯一性。
         *
         * 如果相同 ID 被重复创建，
         * 更可能意味着上层逻辑有问题。
         *
         * 所以第一版选择：
         *
         *      拒绝重复插入
         *
         * 而不是静默覆盖。
         */
        bool InsertMapPoint(const MapPoint::Ptr &map_point);

        /*
         * ============================================================
         * GetMapPoint
         * ============================================================
         *
         * 根据 ID 查找地图点。
         *
         *
         * 找到：
         *
         *      返回 shared_ptr<MapPoint>
         *
         * 没找到：
         *
         *      返回 nullptr
         *
         *
         * 返回 shared_ptr 是因为：
         *
         * 调用者可能需要在离开 Map 查询接口以后
         * 继续安全地使用这个 MapPoint。
         *
         * 返回的 shared_ptr 会暂时增加引用计数。
         */
        MapPoint::Ptr GetMapPoint(unsigned long id) const;

        /*
         * ============================================================
         * mapPoints()
         * ============================================================
         *
         * 获取全部 MapPoint。
         *
         * 返回 const reference：
         *
         *      不复制整个 unordered_map
         *
         * 同时外部不能：
         *
         *      insert
         *      erase
         *      clear
         *
         * 修改 Map 自己的容器结构。
         */
        const MapPointContainer& mapPoints() const;

        std::size_t size() const;

        bool empty() const;

        void Clear();

    private:
        MapPointContainer map_points_;
    };

}