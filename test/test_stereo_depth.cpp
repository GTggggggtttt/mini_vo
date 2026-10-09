#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mini_vo/dataset.h"
#include "mini_vo/frontend.h"
#include "mini_vo/stereo_camera.h"


/*
 * ================================================================
 * StereoPoint
 * ================================================================
 *
 * 当前仅用于测试：
 *
 *      stereo pixel
 *          ↓
 *      disparity
 *          ↓
 *      depth
 *          ↓
 *      camera 3D point
 *
 *
 * 这还不是 MapPoint。
 *
 * 因为：
 *
 *      point_camera
 *
 * 当前是在：
 *
 *      当前左相机坐标系 C0
 *
 * 中，而不是世界坐标系。
 */
struct StereoPoint
{
    cv::Point2f left_pixel;

    cv::Point2f right_pixel;


    double disparity = 0.0;

    double depth = 0.0;


    Eigen::Vector3d point_camera =
        Eigen::Vector3d::Zero();


    /*
     * Pixel -> Camera -> Pixel
     *
     * 的重投影误差。
     */
    double reprojection_error = 0.0;


    /*
     * 与 OpenCV triangulatePoints()
     * 得到结果的 3D 距离。
     */
    double triangulation_error = 0.0;
};


int main(
    int argc,
    char** argv
)
{
    /*
     * ============================================================
     * 1. Dataset path
     * ============================================================
     */
    if (argc != 2)
    {
        std::cerr
            << "Usage: "
            << argv[0]
            << " <euroc_dataset_path>"
            << std::endl;

        return 1;
    }


    const std::string dataset_path =
        argv[1];


    /*
     * ============================================================
     * 2. Dataset
     * ============================================================
     */
    mini_vo::Dataset dataset(
        dataset_path
    );


    if (!dataset.Init())
    {
        std::cerr
            << "Dataset initialization failed."
            << std::endl;

        return 1;
    }


    /*
     * ============================================================
     * 3. 获取左右原始 Camera
     * ============================================================
     */
    auto left_camera =
        dataset.leftCamera();


    auto right_camera =
        dataset.rightCamera();


    if (!left_camera ||
        !right_camera)
    {
        std::cerr
            << "Camera calibration unavailable."
            << std::endl;

        return 1;
    }


    /*
     * ============================================================
     * 4. Stereo Camera
     * ============================================================
     */
    mini_vo::StereoCamera stereo_camera(
        left_camera,
        right_camera
    );


    /*
     * ============================================================
     * 5. 第一帧
     * ============================================================
     */
    auto frame =
        dataset.NextFrame();


    if (!frame)
    {
        std::cerr
            << "Cannot read first frame."
            << std::endl;

        return 1;
    }


    /*
     * ============================================================
     * 6. 初始化双目校正
     * ============================================================
     */
    if (!stereo_camera.InitRectification(
            frame->leftImage().size()))
    {
        std::cerr
            << "Stereo rectification initialization failed."
            << std::endl;

        return 1;
    }


    /*
     * ============================================================
     * 7. 打印校正后的双目参数
     * ============================================================
     */
    std::cout
        << "\n========== Rectified Stereo Camera ==========\n";

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
        << "physical baseline = "
        << stereo_camera.baseline()
        << " m\n";


    std::cout
        << "rectified baseline = "
        << stereo_camera.rectifiedBaseline()
        << " m\n";


    /*
     * ============================================================
     * 8. Frontend
     * ============================================================
     */
    mini_vo::Frontend frontend(
        1000
    );


    /*
     * ============================================================
     * 几何过滤参数
     * ============================================================
     *
     * disparity 太小时：
     *
     *      depth = fb/d
     *
     * 会非常大，而且深度误差会迅速放大。
     */
    constexpr double min_disparity =
        1.0;


    /*
     * EuRoC 是室内场景。
     *
     * 第一版暂时过滤特别遥远的点。
     *
     * 这是工程阈值，不是数学定理。
     */
    constexpr double max_depth =
        50.0;


    /*
     * 当前测试前多少帧。
     *
     * 如果只想看一帧，
     *
     * 改成：
     *
     *      1
     */
    constexpr std::size_t max_test_frames =
        100;


    std::cout
        << "\nPress SPACE to pause."
        << "\nPress q / ESC to quit."
        << "\n\n";


    bool paused = false;

    std::size_t tested_frames = 0;


    /*
     * ============================================================
     * 9. Main Loop
     * ============================================================
     */
    while (frame &&
           tested_frames < max_test_frames)
    {
        /*
         * ========================================================
         * 9.1 Rectification
         * ========================================================
         */
        cv::Mat left_rectified;

        cv::Mat right_rectified;


        if (!stereo_camera.Rectify(
                frame->leftImage(),
                frame->rightImage(),
                left_rectified,
                right_rectified))
        {
            std::cerr
                << "Rectification failed at frame "
                << frame->id()
                << std::endl;

            return 1;
        }


        /*
         * ========================================================
         * 9.2 Stereo Matching
         * ========================================================
         */
        if (!frontend.ProcessStereo(
                left_rectified,
                right_rectified))
        {
            std::cerr
                << "Stereo matching failed at frame "
                << frame->id()
                << std::endl;

            frame =
                dataset.NextFrame();

            ++tested_frames;

            continue;
        }


        const auto& left_keypoints =
            frontend.leftKeypoints();


        const auto& right_keypoints =
            frontend.rightKeypoints();


        const auto& matches =
            frontend.stereoMatches();


        /*
         * ========================================================
         * 9.3 Stereo 2D -> Depth -> 3D
         * ========================================================
         */
        std::vector<StereoPoint>
            stereo_points;


        stereo_points.reserve(
            matches.size()
        );


        /*
         * 同时准备 OpenCV triangulation 的像素数组。
         *
         * 这里先保存所有通过：
         *
         *      disparity
         *      depth
         *
         * 过滤的匹配。
         */
        std::vector<cv::Point2f>
            points_left;


        std::vector<cv::Point2f>
            points_right;


        for (const auto& match : matches)
        {
            const cv::Point2f left_pixel =
                left_keypoints[
                    match.queryIdx
                ].pt;


            const cv::Point2f right_pixel =
                right_keypoints[
                    match.trainIdx
                ].pt;


            /*
             * ----------------------------------------------------
             * disparity
             * ----------------------------------------------------
             *
             * 水平校正双目：
             *
             *      d = u_left - u_right
             */
            const double disparity =
                static_cast<double>(
                    left_pixel.x
                    - right_pixel.x
                );


            /*
             * 太小的 disparity 不可靠。
             */
            if (disparity <
                min_disparity)
            {
                continue;
            }


            /*
             * ----------------------------------------------------
             * depth
             * ----------------------------------------------------
             *
             *             fx * baseline
             *      Z = -------------------
             *                disparity
             */
            const double depth =
                stereo_camera.fx()
                * stereo_camera.rectifiedBaseline()
                / disparity;


            /*
             * 深度检查。
             */
            if (!std::isfinite(depth) ||
                depth <= 0.0 ||
                depth > max_depth)
            {
                continue;
            }


            /*
             * ----------------------------------------------------
             * left pixel -> camera 3D
             * ----------------------------------------------------
             */
            const Eigen::Vector2d pixel(
                static_cast<double>(
                    left_pixel.x
                ),

                static_cast<double>(
                    left_pixel.y
                )
            );


            const Eigen::Vector3d point_camera =
                stereo_camera.PixelToCamera(
                    pixel,
                    depth
                );


            /*
             * ----------------------------------------------------
             * Reprojection Test
             * ----------------------------------------------------
             *
             * pixel
             *    ↓
             *  Pc
             *    ↓
             * pixel_reprojected
             *
             *
             * 理论上应该回到原像素。
             */
            const Eigen::Vector2d
                pixel_reprojected =
                    stereo_camera.CameraToPixel(
                        point_camera
                    );


            const double
                reprojection_error =
                    (
                        pixel
                        - pixel_reprojected
                    ).norm();


            /*
             * 保存测试点。
             */
            StereoPoint point;

            point.left_pixel =
                left_pixel;

            point.right_pixel =
                right_pixel;

            point.disparity =
                disparity;

            point.depth =
                depth;

            point.point_camera =
                point_camera;

            point.reprojection_error =
                reprojection_error;


            stereo_points.push_back(
                point
            );


            points_left.push_back(
                left_pixel
            );


            points_right.push_back(
                right_pixel
            );
        }


        /*
         * ========================================================
         * 9.4 OpenCV triangulatePoints()
         * ========================================================
         *
         * 用一个完全不同的方式计算 3D，
         * 和我们自己的：
         *
         *      Z = fb/d
         *
         * 进行交叉验证。
         */
        if (!points_left.empty())
        {
            /*
             * OpenCV triangulatePoints 在 float 数据上
             * 使用非常方便。
             *
             * 把 3x4 projection matrix 转成 CV_32F。
             */
            cv::Mat P_left_float;

            cv::Mat P_right_float;


            stereo_camera
                .PLeft()
                .convertTo(
                    P_left_float,
                    CV_32F
                );


            stereo_camera
                .PRight()
                .convertTo(
                    P_right_float,
                    CV_32F
                );


            /*
             * 输出：
             *
             *      4 × N
             *
             * 每一列：
             *
             *      [X,Y,Z,W]^T
             *
             * 是齐次坐标。
             */
            cv::Mat points_4d;


            cv::triangulatePoints(
                P_left_float,
                P_right_float,
                points_left,
                points_right,
                points_4d
            );


            /*
             * 数量理论上应该一致。
             */
            const int num_points =
                std::min(
                    points_4d.cols,
                    static_cast<int>(
                        stereo_points.size()
                    )
                );


            for (int i = 0;
                 i < num_points;
                 ++i)
            {
                /*
                 * Homogeneous：
                 *
                 *      [X,Y,Z,W]
                 *
                 * 转普通 3D：
                 *
                 *      [X/W,Y/W,Z/W]
                 */
                const float w =
                    points_4d.at<float>(
                        3,
                        i
                    );


                /*
                 * 避免除零。
                 */
                if (std::abs(w) <
                    1e-8f)
                {
                    stereo_points[i]
                        .triangulation_error =
                            std::numeric_limits<double>::infinity();

                    continue;
                }


                const Eigen::Vector3d
                    point_cv(
                        points_4d.at<float>(
                            0,
                            i
                        ) / w,

                        points_4d.at<float>(
                            1,
                            i
                        ) / w,

                        points_4d.at<float>(
                            2,
                            i
                        ) / w
                    );


                /*
                 * 我们自己的公式结果：
                 *
                 *      point_camera
                 *
                 * OpenCV：
                 *
                 *      point_cv
                 *
                 * 两者应该比较接近。
                 */
                stereo_points[i]
                    .triangulation_error =
                        (
                            stereo_points[i]
                                .point_camera
                            - point_cv
                        ).norm();
            }
        }


        /*
         * ========================================================
         * 9.5 统计
         * ========================================================
         */
        double depth_sum = 0.0;

        double min_depth =
            std::numeric_limits<double>::max();

        double max_valid_depth = 0.0;


        double reprojection_error_sum =
            0.0;


        double triangulation_error_sum =
            0.0;


        std::size_t triangulation_count =
            0;


        for (const auto& point :
             stereo_points)
        {
            depth_sum +=
                point.depth;


            min_depth =
                std::min(
                    min_depth,
                    point.depth
                );


            max_valid_depth =
                std::max(
                    max_valid_depth,
                    point.depth
                );


            reprojection_error_sum +=
                point.reprojection_error;


            if (std::isfinite(
                    point.triangulation_error))
            {
                triangulation_error_sum +=
                    point.triangulation_error;

                ++triangulation_count;
            }
        }


        double mean_depth = 0.0;

        double mean_reprojection_error =
            0.0;

        double mean_triangulation_error =
            0.0;


        if (!stereo_points.empty())
        {
            mean_depth =
                depth_sum
                / stereo_points.size();


            mean_reprojection_error =
                reprojection_error_sum
                / stereo_points.size();
        }


        if (triangulation_count > 0)
        {
            mean_triangulation_error =
                triangulation_error_sum
                / triangulation_count;
        }


        /*
         * ========================================================
         * 9.6 输出当前帧统计
         * ========================================================
         */
        std::cout
            << "Frame "
            << frame->id()

            << " | matches: "
            << matches.size()

            << " | valid 3D: "
            << stereo_points.size()

            << " | mean depth: "
            << std::fixed
            << std::setprecision(3)
            << mean_depth
            << " m"

            << " | reproj: "
            << mean_reprojection_error
            << " px"

            << " | triangulation diff: "
            << mean_triangulation_error
            << " m"

            << std::endl;


        /*
         * ========================================================
         * 9.7 第一帧打印部分 3D 点
         * ========================================================
         *
         * 不要每帧全部打印，
         * 否则终端信息会爆炸。
         */
        if (tested_frames == 0)
        {
            std::cout
                << "\n========== First Stereo Points ==========\n";


            const std::size_t print_count =
                std::min<std::size_t>(
                    10,
                    stereo_points.size()
                );


            for (std::size_t i = 0;
                 i < print_count;
                 ++i)
            {
                const auto& point =
                    stereo_points[i];


                std::cout
                    << "[" << i << "] "

                    << "L=("
                    << point.left_pixel.x
                    << ", "
                    << point.left_pixel.y
                    << ") "

                    << "R=("
                    << point.right_pixel.x
                    << ", "
                    << point.right_pixel.y
                    << ") "

                    << "d="
                    << point.disparity
                    << " px "

                    << "Z="
                    << point.depth
                    << " m "

                    << "Pc=["
                    << point.point_camera
                           .transpose()
                    << "] "

                    << "reproj="
                    << point.reprojection_error

                    << " tri_diff="
                    << point.triangulation_error

                    << '\n';
            }


            std::cout
                << '\n';
        }


        /*
         * ========================================================
         * 9.8 可视化深度
         * ========================================================
         *
         * 这里不画真正的 dense depth map。
         *
         * 只把当前有效 stereo feature
         * 按 depth 显示到左图。
         */
        cv::Mat visualization;


        cv::cvtColor(
            left_rectified,
            visualization,
            cv::COLOR_GRAY2BGR
        );


        for (const auto& point :
             stereo_points)
        {
            /*
             * 半径只用于视觉效果。
             */
            cv::circle(
                visualization,

                point.left_pixel,

                3,

                cv::Scalar(
                    0,
                    255,
                    0
                ),

                -1
            );


            /*
             * 只给少量点显示 depth，
             * 避免画面太乱。
             */
            if (point.depth < 10.0)
            {
                const std::string
                    depth_text =
                        std::to_string(
                            static_cast<int>(
                                point.depth
                            )
                        )
                        + "m";


                cv::putText(
                    visualization,

                    depth_text,

                    point.left_pixel
                    + cv::Point2f(
                        4.0f,
                        -4.0f
                    ),

                    cv::FONT_HERSHEY_SIMPLEX,

                    0.3,

                    cv::Scalar(
                        0,
                        0,
                        255
                    ),

                    1
                );
            }
        }


        cv::imshow(
            "Stereo Depth Points",
            visualization
        );


        /*
         * ========================================================
         * 9.9 Keyboard
         * ========================================================
         */
        const int key =
            cv::waitKey(
                paused ? 0 : 30
            );


        if (key == 27 ||
            key == 'q')
        {
            break;
        }


        if (key == ' ')
        {
            paused = !paused;
        }


        if (paused)
        {
            continue;
        }


        /*
         * ========================================================
         * 9.10 Next Frame
         * ========================================================
         */
        frame =
            dataset.NextFrame();


        ++tested_frames;
    }


    cv::destroyAllWindows();


    return 0;
}