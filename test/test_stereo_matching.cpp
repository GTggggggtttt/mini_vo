#include <cmath>
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
     * 1. Dataset path
     * ============================================================
     */
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <euroc_dataset_path>" << std::endl;
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
        std::cerr<< "Dataset initialization failed."<< std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 3. Camera
     * ============================================================
     */
    auto left_camera = dataset.leftCamera();
    auto right_camera = dataset.rightCamera();

    if (!left_camera ||!right_camera)
    {
        std::cerr<< "Camera calibration unavailable."<< std::endl;
        return 1;
    }

    /*
     * ============================================================
     * 4. StereoCamera
     * ============================================================
     */
    mini_vo::StereoCamera stereo_camera(left_camera,right_camera);

    /*
     * ============================================================
     * 5. First Frame
     * ============================================================
     */
    auto frame = dataset.NextFrame();

    if (!frame)
    {
        std::cerr<< "Cannot read first frame."<< std::endl;
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
            << "Stereo rectification initialization failed." << std::endl;
            return 1;
    }

    /*
     * ============================================================
     * 7. Frontend
     * ============================================================
     */
    mini_vo::Frontend frontend(1000);

    std::cout<< "\n=========== Stereo Matching Test ===========\n";

    std::cout<< "fx = "<< stereo_camera.fx() << '\n';

    std::cout<< "baseline = "<< stereo_camera.baseline()<< " m\n\n";

    std::cout<< "Controls:\n"<< "ESC / q : quit\n"<< "SPACE   : pause\n\n";

    bool paused = false;

    /*
     * ============================================================
     * 8. Main Loop
     * ============================================================
     */
    while (frame)
    {
        /*
         * --------------------------------------------------------
         * 8.1 Rectification
         * --------------------------------------------------------
         */
        cv::Mat left_rectified;

        cv::Mat right_rectified;

        if (!stereo_camera.Rectify(frame->leftImage(),frame->rightImage(),left_rectified,right_rectified))
        {

            std::cerr
                << "Rectification failed at frame "
                << frame->id()
                << std::endl;

            return 1;
        }

        /*
         * --------------------------------------------------------
         * 8.2 Stereo Frontend
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

        /*
         * ========================================================
         * 8.3 计算一些验证统计量
         * ========================================================
         */
        double total_vertical_error = 0.0;

        double total_disparity = 0.0;

        for (const auto &match : matches)
        {
            const auto &left_point =
                left_keypoints[match.queryIdx].pt;

            const auto &right_point =
                right_keypoints[match.trainIdx].pt;

            total_vertical_error +=
                std::abs(
                    left_point.y - right_point.y);

            total_disparity +=
                left_point.x - right_point.x;
        }

        double mean_vertical_error = 0.0;

        double mean_disparity = 0.0;

        if (!matches.empty())
        {
            mean_vertical_error =
                total_vertical_error / matches.size();

            mean_disparity =
                total_disparity / matches.size();
        }

        /*
         * ========================================================
         * 8.4 Terminal Output
         * ========================================================
         */
        std::cout
            << "Frame "
            << frame->id()

            << " | left: "
            << left_keypoints.size()

            << " | right: "
            << right_keypoints.size()

            << " | valid matches: "
            << matches.size()

            << " | mean |dy|: "
            << mean_vertical_error

            << " px"

            << " | mean disparity: "
            << mean_disparity

            << " px\r"
            << std::flush;

        /*
         * ========================================================
         * 8.5 drawMatches
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

            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        /*
         * 当前 Frame 信息。
         */
        const std::string text =
            "Frame " + std::to_string(frame->id()) + "  Matches: " + std::to_string(matches.size());

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
            "Stereo Matches",
            match_image);

        /*
         * ========================================================
         * 8.6 Keyboard
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
         * 8.7 Next Frame
         * ========================================================
         */
        frame =
            dataset.NextFrame();
    }

    std::cout << '\n';

    cv::destroyAllWindows();

    return 0;
}