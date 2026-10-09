#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mini_vo/dataset.h"
#include "mini_vo/stereo_camera.h"

namespace
{
    /*
     * ================================================================
     * 创建双目可视化图
     * ================================================================
     *
     * 输入：
     *
     *      left
     *      right
     *
     * 输出：
     *
     *      left | right
     *
     * 并每隔固定像素画一条水平线。
     *
     *
     * 为什么画线？
     *
     * 双目校正成功以后：
     *
     *      左图中的某个特征
     *
     * 和：
     *
     *      右图中的对应特征
     *
     * 应该基本位于同一条水平线上。
     */
    cv::Mat MakeStereoVisualization(const cv::Mat &left, const cv::Mat &right)
    {
        cv::Mat left_color;
        cv::Mat right_color;

        /*
         * EuRoC 当前读取的是灰度图。
         *
         * 为了画彩色水平线，
         * 转换成 BGR。
         */
        if (left.channels() == 1)
        {
            cv::cvtColor(left, left_color, cv::COLOR_GRAY2BGR);
        }
        else
        {
            left.copyTo(left_color);
        }

        if (right.channels() == 1)
        {
            cv::cvtColor(right, right_color, cv::COLOR_GRAY2BGR);
        }
        else
        {
            right.copyTo(right_color);
        }

        /*
         * 左右图横向拼接。
         */
        cv::Mat stereo;
        cv::hconcat(left_color, right_color, stereo); // 将左右彩色图像拼接在一起

        /*
         * 每隔 40 像素画一条水平线。
         *
         * 后面观察：
         *
         * 同一个空间结构在左右两张图中，
         * 是否落在相同水平线附近。
         */

        constexpr int line_step = 40;
        for (int y = line_step; y < stereo.rows; y += line_step)
        {
            cv::line(stereo, cv::Point(0, y), cv::Point(stereo.cols - 1, y), cv::Scalar(0, 255, 0), 1);
        }
        // 中间画一条竖线
        cv::line(stereo, cv::Point(left.cols, 0), cv::Point(left.cols, stereo.rows - 1), cv::Scalar(0, 0, 255), 1);
        return stereo;
    }
} // anonymous namespace

int main(int argc, char **argv)
{
    /*
     * ============================================================
     * 1. 检查命令行参数
     * ============================================================
     *
     * 使用：
     *
     * ./test_stereo_rectification \
     *     /data/EuRoc/machine_hall/MH_01_easy
     */

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << "<euroc_dataset_path>" << std::endl;
        return 1;
    }

    const std::string dataset_path = argv[1];
    /*
     * ============================================================
     * 2. 初始化 Dataset
     * ============================================================
     */
    mini_vo::Dataset dataset(dataset_path);
    if (!dataset.Init())
    {
        std::cerr << "Dataset initialization failed." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 3. 获取 Camera
     * ============================================================
     */
    mini_vo::Camera::Ptr left_camera = dataset.leftCamera();
    mini_vo::Camera::Ptr right_camera = dataset.rightCamera();

    if (!left_camera || !right_camera)
    {
        std::cerr << "Camera calibration is not available." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 4. 创建 StereoCamera
     * ============================================================
     *
     * 构造时会计算：
     *
     *      T_C1_C0
     */
    mini_vo::StereoCamera stereo_camera(left_camera, right_camera);

    /*
     * ============================================================
     * 5. 验证 stereo extrinsic
     * ============================================================
     */
    std::cout << "\n=====================Stereo Camera===========================\n";

    std::cout << "T_C1_C0:\n"
              << stereo_camera.TRightLeft() << "\n\n";
    std::cout << "Stereo baseline = " << stereo_camera.baseline() << "m\n";

    /*
     * ============================================================
     * 6. 读取第一帧
     * ============================================================
     *
     * 我们需要第一帧来获得真实 image size。
     */

    mini_vo::Frame::Ptr frame = dataset.NextFrame();

    if (!frame)
    {
        std::cerr << "Cannot load first frame." << std::endl;
        return 1;
    }
    /*
     * ============================================================
     * 7. 初始化 Rectification
     * ============================================================
     *
     * 只执行一次。
     */
    if (!stereo_camera.InitRectification(frame->leftImage().size()))
    {
        std::cerr << "stereo rectification initialization failed. " << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 8. 打印校正后的 projection matrix
     * ============================================================
     *
     * 这两个矩阵后面非常重要。
     *
     * 后面 disparity / triangulation 使用的相机几何，
     * 应该基于校正后的模型。
     */
    std::cout << "\nP_left:\n"
              << stereo_camera.PLeft() << "\n";
    std::cout
        << "\nP_right:\n"
        << stereo_camera.PRight()
        << "\n";

    std::cout
        << "\nQ:\n"
        << stereo_camera.Q()
        << "\n";

    std::cout
        << "\nControls:\n"
        << "ESC / q : quit\n"
        << "SPACE   : pause\n\n";

    bool paused = false;

    /*
     * ============================================================
     * 9. 连续显示数据集
     * ============================================================
     */
    while (frame)
    {
        cv::Mat left_rectified;
        cv::Mat right_rectified;

        /*
         * Raw
         *
         *      ↓
         *
         * Rectified
         */
        if (!stereo_camera.Rectify(frame->leftImage(), frame->rightImage(),
                                   left_rectified, right_rectified))
        {
            std::cerr << "Rectification failed at frame " << frame->id() << std::endl;
            return 1;
        }

        /*
         * ========================================================
         * Raw stereo visualization
         * ========================================================
         *
         * 原始图上对应点通常不会严格位于水平极线上。
         */
        cv::Mat raw_stereo = MakeStereoVisualization(frame->leftImage(), frame->rightImage());

        /*
         * ========================================================
         * Rectified stereo visualization
         * ========================================================
         *
         * 校正后观察：
         *
         * 左右图中的同一物体特征
         * 应该基本沿绿色水平线对应。
         */
        cv::Mat rectified_stereo = MakeStereoVisualization(left_rectified, right_rectified);

        /*
         * 显示当前 Frame ID。
         */
        const std::string text = "Frame: " + std::to_string(frame->id());
        cv::putText(raw_stereo, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::putText(rectified_stereo, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Raw Stereo ", raw_stereo);
        cv::imshow("Rectified Stereo ", rectified_stereo);

        /*
         * ========================================================
         * Keyboard
         * ========================================================
         */
        const int key = cv::waitKey(paused ? 0:20);
        if(key ==27 || key=='q')
        {
            break;
        }
        if(key==' ')
        {
            paused =!paused;
        }
        if(paused)
        {
            continue;;
        }
        frame = dataset.NextFrame();
    }
    cv::destroyAllWindows();
    return 0;
}
