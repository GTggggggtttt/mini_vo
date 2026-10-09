#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mini_vo/dataset.h"
#include "mini_vo/frontend.h"
#include "mini_vo/stereo_camera.h"

int main(
    int argc,
    char **argv)
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

    const std::string dataset_path = argv[1];

    /*
     * ============================================================
     * 2. Dataset
     * ============================================================
     */
    mini_vo::Dataset dataset(dataset_path);

    if (!dataset.Init())
    {
        std::cerr
            << "Dataset initialization failed."
            << std::endl;

        return 1;
    }

    /*
     * ============================================================
     * 3. Camera
     * ============================================================
     */
    auto left_camera = dataset.leftCamera();
    auto right_camera = dataset.rightCamera();

    if (!left_camera || !right_camera)
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
        right_camera);

    /*
     * ============================================================
     * 5. First Frame
     * ============================================================
     */
    auto frame = dataset.NextFrame();

    if (!frame)
    {
        std::cerr
            << "Cannot read first frame."
            << std::endl;

        return 1;
    }

    /*
     * ============================================================
     * 6. Stereo Rectification Initialization
     * ============================================================
     */
    if (!stereo_camera.InitRectification(frame->leftImage().size()))
    {
        std::cerr
            << "Stereo rectification initialization failed."
            << std::endl;

        return 1;
    }

    /*
     * ============================================================
     * 7. 根据实际相机几何计算 disparity range
     * ============================================================
     *
     * 第一版希望处理：
     *
     *      0.5m ~ 50m
     */
    constexpr double min_depth =0.5;

    constexpr double max_depth =50.0;

    const double fb =
        stereo_camera.fx() * stereo_camera.rectifiedBaseline();

    /*
     * Z = fb/d
     *
     * 所以：
     *
     * d = fb/Z
     */
    const float min_disparity =
        static_cast<float>(
            fb / max_depth);

    const float max_disparity =
        static_cast<float>(
            fb / min_depth);

    /*
     * ============================================================
     * 8. Frontend
     * ============================================================
     */
    mini_vo::Frontend frontend(
        1000);

    /*
     * Ratio Test：
     *
     *      0.75
     *
     * Epipolar error：
     *
     *      2 px
     *
     * disparity：
     *
     *      根据真实相机几何得到。
     */
    frontend.SetStereoMatchingParameters(
        0.75f,
        2.0f,
        min_disparity,
        max_disparity);

    std::cout
        << "\n========== Stereo Matching V2 ==========\n";

    std::cout
        << "fx = "
        << stereo_camera.fx()
        << '\n';

    std::cout
        << "baseline = "
        << stereo_camera.rectifiedBaseline()
        << " m\n";

    std::cout
        << "f*b = "
        << fb
        << '\n';

    std::cout
        << "Depth range = ["
        << min_depth
        << ", "
        << max_depth
        << "] m\n";

    std::cout
        << "Disparity range = ["
        << min_disparity
        << ", "
        << max_disparity
        << "] px\n\n";

    std::cout
        << "Controls:\n"
        << "SPACE   : pause/resume\n"
        << "ESC / q : quit\n\n";

    bool paused = false;

    /*
     * ============================================================
     * 9. Main Loop
     * ============================================================
     */
    while (frame)
    {
        /*
         * --------------------------------------------------------
         * 9.1 Stereo Rectification
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
            std::cerr
                << "Rectification failed at frame "
                << frame->id()
                << std::endl;

            return 1;
        }

        /*
         * --------------------------------------------------------
         * 9.2 Stereo Matching V2
         * --------------------------------------------------------
         */
        frontend.ProcessStereo(
            left_rectified,
            right_rectified);

        const auto &left_keypoints =
            frontend.leftKeypoints();

        const auto &right_keypoints =
            frontend.rightKeypoints();

        const auto &matches =
            frontend.stereoMatches();

        const auto &stats =
            frontend.stereoMatchStats();

        /*
         * ========================================================
         * 9.3 对最终匹配计算质量统计
         * ========================================================
         */
        double vertical_error_sum =
            0.0;

        double disparity_sum =
            0.0;

        double hamming_sum =
            0.0;

        double min_disp_observed =
            std::numeric_limits<double>::max();

        double max_disp_observed =
            0.0;

        for (const auto &match :
             matches)
        {
            const cv::Point2f &left_point =
                left_keypoints[match.queryIdx].pt;

            const cv::Point2f &right_point =
                right_keypoints[match.trainIdx].pt;

            const double vertical_error =
                std::abs(
                    left_point.y - right_point.y);

            const double disparity =
                left_point.x - right_point.x;

            vertical_error_sum +=
                vertical_error;

            disparity_sum +=
                disparity;

            hamming_sum +=
                match.distance;

            min_disp_observed =
                std::min(
                    min_disp_observed,
                    disparity);

            max_disp_observed =
                std::max(
                    max_disp_observed,
                    disparity);
        }

        double mean_vertical_error =
            0.0;

        double mean_disparity =
            0.0;

        double mean_hamming =
            0.0;

        if (!matches.empty())
        {
            const double count =
                static_cast<double>(
                    matches.size());

            mean_vertical_error =
                vertical_error_sum / count;

            mean_disparity =
                disparity_sum / count;

            mean_hamming =
                hamming_sum / count;
        }
        else
        {
            min_disp_observed =
                0.0;
        }

        /*
         * ========================================================
         * 9.4 打印完整 Pipeline
         * ========================================================
         *
         * 这里故意每帧换行。
         *
         * 之前使用 '\r' 时，
         * 当新字符串比旧字符串短，
         * 终端容易残留字符。
         */
        std::cout
            << "Frame "
            << std::setw(4)
            << frame->id()

            << " | L/R "
            << stats.left_features
            << "/"
            << stats.right_features

            << " | KNN "
            << stats.knn_candidates

            << " -> Ratio "
            << stats.ratio_passed

            << " -> Epipolar "
            << stats.epipolar_passed

            << " -> Disp "
            << stats.disparity_passed

            << " | dy "
            << std::fixed
            << std::setprecision(3)
            << mean_vertical_error

            << " | disp "
            << mean_disparity

            << " | Hamming "
            << mean_hamming

            << '\n';

        /*
         * ========================================================
         * 9.5 Visualize
         * ========================================================
         */
        cv::Mat match_image;

        cv::drawMatches(
            left_rectified,
            left_keypoints,

            right_rectified,
            right_keypoints,

            matches,

            match_image,

            cv::Scalar::all(-1),

            cv::Scalar::all(-1),

            std::vector<char>(),

            cv::DrawMatchesFlags::
                NOT_DRAW_SINGLE_POINTS);

        /*
         * 显示当前统计。
         */
        const std::string text =
            "Frame " + std::to_string(frame->id()) + "  Matches " + std::to_string(matches.size());

        cv::putText(
            match_image,

            text,

            cv::Point(
                20,
                30),

            cv::FONT_HERSHEY_SIMPLEX,

            0.6,

            cv::Scalar(
                0,
                255,
                0),

            2);

        cv::imshow(
            "Stereo Matching V2",
            match_image);

        /*
         * ========================================================
         * 9.6 Keyboard
         * ========================================================
         */
        const int key =
            cv::waitKey(
                paused ? 0 : 20);

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
         * 9.7 Next Frame
         * ========================================================
         */
        frame =
            dataset.NextFrame();
    }

    cv::destroyAllWindows();

    return 0;
}