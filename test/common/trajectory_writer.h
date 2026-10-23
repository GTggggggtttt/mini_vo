#pragma once

#include <iomanip>
#include <ostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mini_vo/frame.h"


namespace mini_vo::test {


/*
 * ================================================================
 * WriteTumPose
 * ================================================================
 *
 * TUM trajectory:
 *
 * timestamp tx ty tz qx qy qz qw
 *
 * Frame::pose() 是 T_cw。
 *
 * TUM 文件写的是相机在世界中的位姿 T_wc。
 */
inline void WriteTumPose(
    std::ostream& output,
    const Frame& frame
)
{
    const Eigen::Matrix4d T_wc =
        frame.pose().inverse();

    const Eigen::Vector3d t =
        T_wc.block<3, 1>(0, 3);

    const Eigen::Matrix3d R =
        T_wc.block<3, 3>(0, 0);

    Eigen::Quaterniond q(R);

    q.normalize();

    output
        << std::fixed
        << std::setprecision(9)
        << frame.timestamp() << " "
        << t.x() << " "
        << t.y() << " "
        << t.z() << " "
        << q.x() << " "
        << q.y() << " "
        << q.z() << " "
        << q.w() << '\n';
}


} // namespace mini_vo::test