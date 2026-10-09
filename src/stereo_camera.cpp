#include "mini_vo/stereo_camera.h"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <cmath>

#include <Eigen/LU> // 当前源文件调用 inverse()，需要包含其模板实现。

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace mini_vo
{

    StereoCamera::StereoCamera(
        Camera::Ptr left,
        Camera::Ptr right)
        : left_(std::move(left)),
          right_(std::move(right))
    {
        if (!left_ || !right_)
        {
            throw std::invalid_argument(
                "StereoCamera received nullptr Camera.");
        }

        /*
         * 已知：
         *
         * P_B = T_BS0 * P_C0
         * P_B = T_BS1 * P_C1
         *
         * 所以：
         *
         * P_C1 =
         *      inverse(T_BS1)
         *      * T_BS0
         *      * P_C0
         */
        T_right_left_ =
            right_->T_BS().inverse() * left_->T_BS();
    }

    double StereoCamera::baseline() const
    {
        const Eigen::Vector3d translation =
            T_right_left_.block<3, 1>(
                0,
                3);

        return translation.norm();
    }

    /*
     * ================================================================
     * 校正后的 fx
     * ================================================================
     *
     * P_left：
     *
     *      [fx  0  cx  0]
     *      [0  fy  cy  0]
     *      [0   0   1  0]
     */
    double StereoCamera::fx() const
    {
        if (!initialized_)
        {
            throw std::logic_error(
                "StereoCamera::fx(): rectification is not initialized.");
        }

        return P_left_.at<double>(0, 0);
    }

    double StereoCamera::fy() const
    {
        if (!initialized_)
        {
            throw std::logic_error(
                "StereoCamera::fy(): rectification is not initialized.");
        }

        return P_left_.at<double>(1, 1);
    }

    double StereoCamera::cx() const
    {
        if (!initialized_)
        {
            throw std::logic_error(
                "StereoCamera::cx(): rectification is not initialized.");
        }

        return P_left_.at<double>(0, 2);
    }

    double StereoCamera::cy() const
    {
        if (!initialized_)
        {
            throw std::logic_error(
                "StereoCamera::cy(): rectification is not initialized.");
        }

        return P_left_.at<double>(1, 2);
    }

    cv::Mat StereoCamera::MakeCameraMatrix(const Camera &camera)
    {
        return (
            cv::Mat_<double>(3, 3)
                << camera.fx(),
            0.0, camera.cx(),
            0.0, camera.fy(), camera.cy(),
            0.0, 0.0, 1.0);
    }

    cv::Mat StereoCamera::MakeDistortionMatrix(const Camera &camera)
    {
        const Eigen::Vector4d &d =
            camera.distortion();

        return (
            cv::Mat_<double>(1, 4)
                << d[0],
            d[1],
            d[2],
            d[3]);
    }

    cv::Mat StereoCamera::EigenRotationToCv(const Eigen::Matrix3d &rotation)
    {
        cv::Mat result(
            3,
            3,
            CV_64F);

        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                result.at<double>(row, col) =
                    rotation(row, col);
            }
        }

        return result;
    }

    cv::Mat StereoCamera::EigenTranslationToCv(const Eigen::Vector3d &translation)
    {
        return (
            cv::Mat_<double>(3, 1)
                << translation.x(),
            translation.y(),
            translation.z());
    }

    bool StereoCamera::InitRectification(const cv::Size &image_size)
    {
        if (image_size.width <= 0 ||
            image_size.height <= 0)
        {
            std::cerr
                << "Invalid image size."
                << std::endl;

            return false;
        }

        image_size_ = image_size;

        const cv::Mat K_left = MakeCameraMatrix(*left_);

        const cv::Mat K_right = MakeCameraMatrix(*right_);

        const cv::Mat D_left = MakeDistortionMatrix(*left_);

        const cv::Mat D_right = MakeDistortionMatrix(*right_);

        /*
         * T_C1_C0 =
         *
         *      [R t]
         *      [0 1]
         */
        const Eigen::Matrix3d R_eigen =
            T_right_left_.block<3, 3>(
                0,
                0);

        const Eigen::Vector3d t_eigen =
            T_right_left_.block<3, 1>(
                0,
                3);

        const cv::Mat R =
            EigenRotationToCv(
                R_eigen);

        const cv::Mat t =
            EigenTranslationToCv(
                t_eigen);

        try
        {
            cv::stereoRectify(
                K_left,
                D_left,

                K_right,
                D_right,

                image_size_,

                R,
                t,

                R_left_,
                R_right_,

                P_left_,
                P_right_,

                Q_,

                cv::CALIB_ZERO_DISPARITY,

                0.0,

                image_size_);

            /*
             * 预计算左图 remap。
             */
            cv::initUndistortRectifyMap(
                K_left,
                D_left,
                R_left_,
                P_left_,
                image_size_,
                CV_32FC1,
                map_left_x_,
                map_left_y_);

            /*
             * 预计算右图 remap。
             */
            cv::initUndistortRectifyMap(
                K_right,
                D_right,
                R_right_,
                P_right_,
                image_size_,
                CV_32FC1,
                map_right_x_,
                map_right_y_);
        }
        catch (const cv::Exception &e)
        {
            std::cerr
                << "Stereo rectification failed:\n"
                << e.what()
                << std::endl;

            initialized_ = false;

            return false;
        }

        initialized_ = true;

        return true;
    }

    bool StereoCamera::Rectify(const cv::Mat &left_raw, const cv::Mat &right_raw, cv::Mat &left_rectified, cv::Mat &right_rectified) const
    {
        if (!initialized_)
        {
            std::cerr
                << "StereoCamera is not initialized."
                << std::endl;

            return false;
        }

        if (left_raw.empty() ||
            right_raw.empty())
        {
            std::cerr
                << "Rectify received empty image."
                << std::endl;

            return false;
        }

        if (left_raw.size() != image_size_ ||
            right_raw.size() != image_size_)
        {
            std::cerr
                << "Image size does not match rectification map."
                << std::endl;

            return false;
        }

        cv::remap(
            left_raw,
            left_rectified,
            map_left_x_,
            map_left_y_,
            cv::INTER_LINEAR);

        cv::remap(
            right_raw,
            right_rectified,
            map_right_x_,
            map_right_y_,
            cv::INTER_LINEAR);

        return true;
    }

    /*
     * ================================================================
     * Rectified Pixel -> Camera Coordinate
     * ================================================================
     *
     * 已知：
     *
     *      pixel = [u, v]
     *
     *      depth = Z
     *
     *
     * 校正后的针孔模型：
     *
     *              X
     *      u = fx ---- + cx
     *              Z
     *
     *              Y
     *      v = fy ---- + cy
     *
     *
     * 所以：
     *
     *          (u-cx)Z
     *      X = --------
     *             fx
     *
     *          (v-cy)Z
     *      Y = --------
     *             fy
     */
    Eigen::Vector3d StereoCamera::PixelToCamera(const Eigen::Vector2d &pixel, double depth) const
    {
        if (!initialized_)
        {
            throw std::logic_error("StereoCamera::PixelToCamera():"
                                   "rectification is not initialized.");
        }

        const double x = (pixel.x() - cx()) * depth / fx();
        const double y = (pixel.y() - cy()) * depth / fy();
        return Eigen::Vector3d(x, y, depth);
    }

    /*
     * ================================================================
     * Camera Coordinate -> Rectified Pixel
     * ================================================================
     *
     * Pc = [X,Y,Z]^T
     *
     *      ↓
     *
     * pixel = [u,v]^T
     */
    Eigen::Vector2d StereoCamera::CameraToPixel(const Eigen::Vector3d &point) const
    {
        if (!initialized_)
        {
            throw std::logic_error("StereoCamera::CameraToPixel(): "
                                   "rectification is not initialized.");
        }

        if (point.z() <= 0.0)
        {
            throw std::invalid_argument("StereoCamera::CameraToPixel(): "
                                        "point must have positive Z");
        }

        const double u = fx() * point.x() / point.z() + cx();
        const double v = fy() * point.y() / point.z() + cy();

        return Eigen::Vector2d(u, v);

        /*
         * ================================================================
         * Rectified Baseline
         * ================================================================
         *
         * 你当前 P_right：
         *
         * [436.234..., 0, 364.441..., -48.01976...]
         *
         *
         * 根据：
         *
         *      P_right(0,3) = -fx * baseline
         *
         * 所以：
         *
         *                     P_right(0,3)
         *      baseline = | ---------------- |
         *                     P_right(0,0)
         *
         */
    }

    double StereoCamera::rectifiedBaseline() const
    {
        if (!initialized_)
        {
            throw std::logic_error("StereoCamera::rectifiedBaselint(): "
                                   "rectification is not initialized.");
        }

        const double fx_rect = P_right_.at<double>(0, 0);
        const double tx = P_right_.at<double>(0, 3);
        return std::abs(tx/fx_rect);
    }
} // namespace mini_vo
