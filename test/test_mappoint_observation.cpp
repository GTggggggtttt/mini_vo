#include <iostream>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "mini_vo/feature.h"
#include "mini_vo/frame.h"
#include "mini_vo/keyframe.h"
#include "mini_vo/map.h"
#include "mini_vo/mappoint.h"

void Check(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

/*
 * 创建一个只包含单个 Feature 的测试 Frame。
 *
 * 不需要真实图像，
 * 因为这个单元测试不涉及 ORB 和 PnP。
 */
mini_vo::Frame::Ptr MakeObserveFrame(unsigned long id,
                                     const mini_vo::MapPoint::Ptr &map_point)
{
    auto frame = std::make_shared<mini_vo::Frame>(id, static_cast<double>(id), cv::Mat{}, cv::Mat{});

    auto feature = std::make_shared<mini_vo::Feature>(cv::KeyPoint(100.0f, 120.0f, 8.0f));

    feature->setFrame(frame);
    feature->setMapPoint(map_point);
    frame->addLeftFeature(feature);
    return frame;
}

int main()
{
    mini_vo::Map map;
    /*
     * ============================================================
     * 1. 创建地图点
     * ============================================================
     */
    auto map_point = std::make_shared<mini_vo::MapPoint>(42, Eigen::Vector3d(1.0, 2.0, 3.0));
    Check(map.InsertMapPoint(map_point), "InsertMapPoint failed.");

    // 尚无任何KeyFrame Observation.
    Check(map_point->observationCount() == 0, "Unexpexted initial observation.");

    // 2.创建KeyFrame0
    auto frame0 = MakeObserveFrame(0, map_point);
    auto keyframe0 = std::make_shared<mini_vo::KeyFrame>(0, frame0);

    /*
     * Map 插入 KeyFrame 时，
     * 自动登记其左目 Feature 的 Observation。
     */
    Check(map.InsertKeyFrame(keyframe0), "KeyFrame insertion failed.");
    Check(map_point->observationCount() == 1, "KeyFrame0 observation missing.");
    Check(map_point->HasObservation(0), "MapPoint does not know KeyFrame0.");

    // 验证记录指向正确KeyFrame 和 Feature
    const auto &observation0 = map_point->observations().at(0);
    Check(observation0.keyframe.lock() == keyframe0, "Incorrect KeyFrame pointer.");
    Check(observation0.left_feature_index == 0, "Incorrect Feature index.");

    /*
     * 不允许重复登记。
     */
    Check(!map_point->AddObservation(keyframe0, 0), "Duplicate obseration was accepted.");

    /*
     * ============================================================
     * 3. 创建 KeyFrame1
     * ============================================================
     *
     * 同一个 MapPoint 在另一个时刻
     * 被另一个 Frame 的 Feature 观测到。
     */
    auto frame1 = MakeObserveFrame(5, map_point);
    auto keyframe1 = std::make_shared<mini_vo::KeyFrame>(1, frame1);
    Check(map.InsertKeyFrame(keyframe1), "KeyFrame1 insertion failed.");
    Check(map_point->observationCount() == 2, "Expected two KeyFrame observations.");
    Check(map.keyFrameCount() == 2, "Incorrect Map KeyFrame count.");

    std::cout
        << "Multiple KeyFrame observations passed.\n";

    /*
     * ============================================================
     * 4. 删除、恢复 Observation
     * ============================================================
     */
    Check(
        map_point->RemoveObservation(1),
        "RemoveObservation failed.");

    Check(
        map_point->observationCount() == 1,
        "Observation was not removed.");

    Check(
        map_point->AddObservation(keyframe1, 0),
        "Re-register observation failed.");

    Check(
        map_point->observationCount() == 2,
        "Observation restoration failed.");

    /*
     * ============================================================
     * 5. Map 清理
     * ============================================================
     */
    map.Clear();

    Check(
        map.keyFrameCount() == 0,
        "KeyFrames were not cleared.");

    Check(
        map.size() == 0,
        "MapPoints were not cleared.");

    /*
     * 此处局部 shared_ptr<MapPoint>
     * 仍然使 map_point 存活。
     *
     * 但它的 Observation 已被 Map::Clear() 清理。
     */
    Check(
        map_point->observationCount() == 0,
        "MapPoint observations were not cleared.");

    std::cout
        << "MapPoint observation cleanup passed.\n";

    std::cout
        << "\nAll MapPoint Observation V1 tests passed.\n";

    return 0;
}