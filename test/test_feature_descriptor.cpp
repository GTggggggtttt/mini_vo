#include <iostream>
#include <opencv2/core.hpp>

#include "mini_vo/dataset.h"
#include "mini_vo/frontend.h"
#include "mini_vo/stereo_camera.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <euroc_dataset_path>" << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * Dataset
     * ============================================================
     */
    mini_vo::Dataset dataset(argv[1]);

    if (!dataset.Init())
    {
        std::cerr << "Dataset initialization failed." << std::endl;
        return 1;
    }

    auto left_camera = dataset.leftCamera();

    auto right_camera = dataset.rightCamera();

    if (!left_camera || !right_camera)
    {
        return 1;
    }

    mini_vo::StereoCamera stereo_camera(left_camera,right_camera);

    /*
     * ============================================================
     * Frame 0
     * ============================================================
     */
    auto frame0 = dataset.NextFrame();

    if (!frame0)
    {
        return 1;
    }

    if (!stereo_camera.InitRectification(frame0->leftImage().size()))
    {
        return 1;
    }

    cv::Mat left0;
    cv::Mat right0;

    if (!stereo_camera.Rectify( frame0->leftImage(),frame0->rightImage(),left0,right0))
    {
        return 1;
    }

    mini_vo::Frontend frontend(1000);

    /*
     * Frame0 ORB。
     */
    if (!frontend.ExtractStereoFeatures(left0,right0))
    {
        return 1;
    }

    const auto &frame0_features =frontend.leftFeatures();

    if (frame0_features.empty())
    {
        std::cerr<< "Frame0 has no feature."<< std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 保存 Frame0 第一个 Feature
     * ============================================================
     *
     * shared_ptr 会让这个 Feature
     * 在 Frontend 处理下一帧后仍然活着。
     */
    auto feature0 =frame0_features.front();

    if (!feature0 ||!feature0->hasDescriptor())
    {
        std::cerr<< "Frame0 feature has no descriptor."<< std::endl;
        return 1;
    }

    /*
     * 保存一份 descriptor 内容，
     * 后面做对比。
     */
    const cv::Mat descriptor_before =
        feature0
            ->descriptor()
            .clone();

    /*
     * 保存 Feature descriptor 的 buffer 地址。
     */
    const unsigned char *
        feature_buffer_before =
            feature0
                ->descriptor()
                .data;

    /*
     * Frontend 当前大 descriptor matrix
     * 的 buffer 地址。
     */
    const unsigned char *
        frontend_buffer_before =
            frontend
                .leftDescriptors()
                .data;

    std::cout
        << "Frame0 feature descriptor: "
        << feature0->descriptor().rows
        << " x "
        << feature0->descriptor().cols
        << '\n';

    std::cout
        << "Type = "
        << feature0->descriptor().type()
        << '\n';

    /*
     * ============================================================
     * 重要检查：
     *
     * Feature descriptor 不应该和 Frontend
     * descriptor matrix 共用同一个 buffer 起点。
     *
     * 注意：
     * row(i) 的地址也可能在大 matrix 内部，
     * clone 后则拥有独立 allocation。
     * ============================================================
     */
    std::cout
        << "Feature buffer  : "
        << static_cast<const void *>(
               feature_buffer_before)
        << '\n';

    std::cout
        << "Frontend buffer : "
        << static_cast<const void *>(
               frontend_buffer_before)
        << '\n';

    /*
     * ============================================================
     * Frame 1
     * ============================================================
     */
    auto frame1 =
        dataset.NextFrame();

    if (!frame1)
    {
        return 1;
    }

    cv::Mat left1;

    cv::Mat right1;

    if (!stereo_camera.Rectify(
            frame1->leftImage(),
            frame1->rightImage(),
            left1,
            right1))
    {
        return 1;
    }

    /*
     * ============================================================
     * Frontend 处理下一帧
     * ============================================================
     *
     * 内部：
     *
     *      descriptors_left_.release()
     *
     * 然后重新生成 Frame1 descriptors。
     *
     * 如果 Feature0 只是共享旧 buffer，
     * 生命周期设计就有问题。
     */
    if (!frontend.ExtractStereoFeatures(
            left1,
            right1))
    {
        return 1;
    }

    /*
     * ============================================================
     * 验证 Frame0 Feature descriptor 仍然有效
     * ============================================================
     */
    if (!feature0->hasDescriptor())
    {
        std::cerr
            << "Feature descriptor disappeared."
            << std::endl;

        return 1;
    }

    const double descriptor_difference =
        cv::norm(
            descriptor_before,
            feature0->descriptor(),
            cv::NORM_HAMMING);

    std::cout
        << "\nAfter processing Frame1:\n";

    std::cout
        << "Frame0 Feature still has descriptor: "
        << feature0->hasDescriptor()
        << '\n';

    std::cout
        << "Descriptor difference = "
        << descriptor_difference
        << '\n';

    /*
     * descriptor 内容不应该改变。
     */
    if (descriptor_difference != 0.0)
    {
        std::cerr
            << "Feature descriptor changed unexpectedly."
            << std::endl;

        return 1;
    }

    /*
     * ORB descriptor 应该是 1x32。
     */
    if (feature0->descriptor().rows != 1 ||
        feature0->descriptor().cols != 32)
    {
        std::cerr
            << "Unexpected ORB descriptor size."
            << std::endl;

        return 1;
    }

    std::cout
        << "\nFeature V3 descriptor test passed."
        << std::endl;

    return 0;
}