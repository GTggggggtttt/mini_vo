#pragma once

#include <memory>

#include <Eigen/Core>

#include <opencv2/core.hpp>

#include "mini_vo/camera.h"

namespace mini_vo
{

    class StereoCamera
    {
    public:
        using Ptr = std::shared_ptr<StereoCamera>;

        StereoCamera(
            Camera::Ptr left,
            Camera::Ptr right);

        /*
         * 根据原始左右相机：
         *
         * K0 D0
         * K1 D1
         * T_C1_C0
         *
         * 计算双目校正参数。
         *
         * 整个数据集通常只执行一次。
         */
        bool InitRectification(
            const cv::Size &image_size);

        /*
         * 对一对原始左右图像执行：
         *
         * undistortion
         * +
         * stereo rectification
         */
        bool Rectify(
            const cv::Mat &left_raw,
            const cv::Mat &right_raw,
            cv::Mat &left_rectified,
            cv::Mat &right_rectified) const;

        /*
         * cam0 -> cam1
         *
         * P_C1 = T_C1_C0 * P_C0
         */
        const Eigen::Matrix4d &TRightLeft() const
        {
            return T_right_left_;
        }

        /*
         * 双目基线：
         *
         * baseline = ||t_C1_C0||
         */
        double baseline() const;

        bool initialized() const
        {
            return initialized_;
        }

        /*
         * ============================================================
         * 校正后的相机内参
         * ============================================================
         *
         * 注意：
         *
         * 这里不是原始 Camera::K() 中的：
         *
         *      fx fy cx cy
         *
         * 而是 stereoRectify() 生成的虚拟校正相机参数。
         *
         * 后面在 rectified image 上：
         *
         * ORB
         * disparity
         * triangulation
         *
         * 都应该使用这里的几何模型。
         */
        double fx() const;

        double fy() const;

        double cx() const;

        double cy() const;

        /*
         * 校正后的左/右投影矩阵。
         */
        const cv::Mat &PLeft() const { return P_left_; }

        const cv::Mat &PRight() const { return P_right_; }

        /*
         * disparity -> 3D 重投影矩阵。
         */
        const cv::Mat &Q() const { return Q_; }

        /*
         * ================================================================
         * 校正后像素 + 深度 -> 左相机坐标
         * ================================================================
         *
         * 注意：
         *
         * pixel 位于 rectified left image。
         *
         * 因此这里使用的是：
         *
         *      StereoCamera::fx()
         *      StereoCamera::fy()
         *      StereoCamera::cx()
         *      StereoCamera::cy()
         *
         * 而不是原始 cam0 的 K。
         */
        Eigen::Vector3d PixelToCamera(const Eigen::Vector2d &pixel, double depth) const;

        /*
         * ================================================================
         * 左相机坐标 -> 校正后左图像素
         * ================================================================
         *
         * 用于后面的 reprojection test。
         */
        Eigen::Vector2d CameraToPixel(const Eigen::Vector3d &point) const;

        /*
         * ================================================================
         * 校正后的水平双目 baseline
         * ================================================================
         *
         * P_right 通常形式：
         *
         *      [fx 0 cx -fx*b]
         *      [0 fy cy     0]
         *      [0  0  1     0]
         *
         * 所以：
         *
         *      b = |-P_right(0,3) / P_right(0,0)|
         *
         * 对你当前 EuRoC：
         *
         *      ≈ 0.110078 m
         */
        double rectifiedBaseline() const;

    private:
        static cv::Mat MakeCameraMatrix(
            const Camera &camera);

        static cv::Mat MakeDistortionMatrix(
            const Camera &camera);

        static cv::Mat EigenRotationToCv(
            const Eigen::Matrix3d &rotation);

        static cv::Mat EigenTranslationToCv(
            const Eigen::Vector3d &translation);

    private:
        Camera::Ptr left_;

        Camera::Ptr right_;

        /*
         * cam0 -> cam1
         */
        Eigen::Matrix4d T_right_left_ =
            Eigen::Matrix4d::Identity();

        cv::Size image_size_;

        /*
         * Rectification rotations
         */
        cv::Mat R_left_;

        cv::Mat R_right_;

        /*
         * Rectified projection matrices
         */
        cv::Mat P_left_;

        cv::Mat P_right_;

        cv::Mat Q_;

        /*
         * remap 查找表
         */
        cv::Mat map_left_x_;

        cv::Mat map_left_y_;

        cv::Mat map_right_x_;

        cv::Mat map_right_y_;

        bool initialized_ = false;
    };

} // namespace mini_vo