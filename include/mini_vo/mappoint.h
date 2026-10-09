#pragma once
#include <memory>
#include <Eigen/Core>

namespace mini_vo
{
class MapPoint
    {
    public:
        using Ptr = std::shared_ptr<MapPoint>;

        MapPoint() = default;

        MapPoint(unsigned long id, const Eigen::Vector3d &position);

        unsigned long id() const;

        const Eigen::Vector3d &position() const;

        void setPosition(const Eigen::Vector3d &position);

    private:

        unsigned long id_ =0;
        Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
    };

} // namespace mini_vo
