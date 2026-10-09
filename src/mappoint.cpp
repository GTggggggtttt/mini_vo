#include "mini_vo/mappoint.h"

namespace mini_vo
{
    /*
     * ================================================================
     * MapPoint Constructor
     * ================================================================
     *
     * 使用成员初始化列表：
     *
     *      id_(id)
     *      position_(position)
     *
     * 而不是：
     *
     *      id_ = id;
     *      position_ = position;
     *
     * 因为对象成员应该尽可能在构造阶段直接完成初始化。
     */
    MapPoint::MapPoint(unsigned long id, const Eigen::Vector3d &position) : id_(id), position_(position)
    {
    }

    unsigned long MapPoint::id() const { return id_; }

    const Eigen::Vector3d &MapPoint::position() const { return position_; }

    void MapPoint::setPosition(const Eigen::Vector3d&position){position_= position;}

}//namespcae mini_vo