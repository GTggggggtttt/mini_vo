#include <iostream>
#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "mini_vo/dataset.h"
#include "mini_vo/frontend.h"
#include "mini_vo/stereo_camera.h"

int main(int argc, char **argv)
{
    /*
     * ============================================================
     * 1. 检查数据集路径
     * ============================================================
     *
     * 使用：
     *
     * ./test_feature \
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
        std::cerr << "数据集初始化失败" << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 3. 创建 StereoCamera
     * ============================================================
     */
    auto left_camera = dataset.leftCamera();
    auto right_camera = dataset.rightCamera();

    if (!left_camera || !right_camera)
    {
        std::cerr << "Camera calibration is unavailable." << std::endl;
        return 1;
    }

    mini_vo::StereoCamera stereo_camera(left_camera, right_camera);

    /*
     * ============================================================
     * 4. 获取第一帧
     * ============================================================
     *
     * 首先需要从真实图像获得 resolution，
     * 然后初始化 Stereo Rectification。
     */
    auto frame = dataset.NextFrame();
    if (!frame)
    {
        std::cerr << "cannot read first frame." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 5. 初始化 Stereo Rectification
     * ============================================================
     *
     * 只执行一次。
     */
    if (!stereo_camera.InitRectification(frame->leftImage().size()))
    {
        std::cerr << "stereo rectification initialization failed." << std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 6. 输出校正后的相机参数
     * ============================================================
     */

    std::cout
        << "\n============ Rectified Camera ============\n";

    std::cout
        << "fx = "
        << stereo_camera.fx()
        << '\n';

    std::cout
        << "fy = "
        << stereo_camera.fy()
        << '\n';

    std::cout
        << "cx = "
        << stereo_camera.cx()
        << '\n';

    std::cout
        << "cy = "
        << stereo_camera.cy()
        << '\n';

    std::cout
        << "baseline = "
        << stereo_camera.baseline()
        << " m\n\n";

    /*
     * ============================================================
     * 7. 创建 Frontend
     * ============================================================
     *
     * 最大 ORB 数量暂定：
     *
     *      1000
     */
    mini_vo::Frontend frontend(1000);

    std::cout << "Controls:\n"
              << "ESC / q : quit\n"
              << "SPACE : pause\n\n";

    bool paused = false;

    /*
     * ============================================================
     * 8. 连续处理 EuRoC
     * ============================================================
     */

    while (frame)
    {
        /*
         * --------------------------------------------------------
         * 8.1 Stereo Rectification
         * --------------------------------------------------------
         */

        cv::Mat left_rectified;
        cv::Mat right_rectified;
        if (!stereo_camera.Rectify(
                frame->leftImage(),
                frame->rightImage(),
                left_rectified,
                right_rectified))
        {
            std::cerr << "Rectification failed at frame " << frame->id() << std::endl;
            return 1;
        }

        /*
         * --------------------------------------------------------
         * 8.2 ORB Feature Extraction
         * --------------------------------------------------------
         *
         * 注意：
         *
         * 特征提取是在：
         *
         *      rectified image
         *
         * 上完成的。
         *
         * 后续 stereo matching 和 triangulation
         * 使用同一个校正后的几何模型。
         */

        const int feature_count = frontend.ExtractFeatures(left_rectified);
        std::cout << "Frame " << frame->id()
                  << " : ORB features = " << feature_count << std::endl;

        /*
         * --------------------------------------------------------
         * 8.3 可视化 ORB
         * --------------------------------------------------------
         */

         cv::Mat feature_image;
         cv::drawKeypoints(left_rectified,frontend.keypoints(),feature_image,
        cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

           /*
         * --------------------------------------------------------
         * 8.4 显示信息
         * --------------------------------------------------------
         */
        const std::string text = "Frame: "+ std::to_string(frame->id())
        + " Features: "+std::to_string(feature_count);

        cv::putText(feature_image,text,cv::Point(20,30),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(0,255,0),2);
          /*
         * 同时显示：
         *
         * ① 校正后的原图
         * ② ORB 特征
         */

         cv::imshow("Rectified Left",left_rectified);
         cv::imshow("ORB Features",feature_image);

             /*
         * --------------------------------------------------------
         * 8.5 键盘控制
         * --------------------------------------------------------
         */
        const int key = cv::waitKey(paused ? 0:20);

        if(key==27||key=='q')
        {
            break;
        }

        if(key==' ')
        {
            paused = !paused;
        }

        //暂停时保持当前帧
        if(paused)
        {
            continue;
        }

        //8.6 进入下一帧
        frame = dataset.NextFrame();
    }

    // 序列处理结束或用户退出后，再关闭窗口并结束程序。
    cv::destroyAllWindows();
    return 0;
}
