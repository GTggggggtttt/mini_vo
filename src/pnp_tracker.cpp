#include "mini_vo/pnp_tracker.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "mini_vo/feature.h"
#include "mini_vo/mappoint.h"

namespace mini_vo
{
    namespace
    {
        /*
         * ================================================================
         * Correspondence
         * ================================================================
         *
         * 一组 PnP 候选：
         *
         *   MapPoint Pw
         *         ↕
         *   Current Frame Pixel
         *
         * 同时保存：
         *
         *   match
         *   map_point
         *
         * 以便 RANSAC 成功后恢复原始 Feature 关联。
         *
         * map_point 使用 shared_ptr：
         *
         *   在本次 Track() 执行期间，
         *   保证被读取的地图点仍然有效。
         */
        struct Correspondence
        {
            cv::Point3d world_point;
            cv::Point2d image_point;
            cv::DMatch match;
            MapPoint::Ptr map_point;
            /* data */
        };
    } // anonymous namespace

    /*
     * ================================================================
     * Constructor
     * ================================================================
     */
    PnPTracker::PnPTracker(const PnPConfig &config) : config_(config)
    {
        // 检查数据
        if (config_.min_correspondences < 4 ||
            config_.min_inliers < 4 ||
            config_.ransac_iterations <= 0 ||
            !std::isfinite(config_.reprojection_error_px) ||
            config_.reprojection_error_px <= 0 ||
            !std::isfinite(config_.confidence) ||
            config_.confidence <= 0 ||
            config_.confidence >= 1.0 ||
            !std::isfinite(config_.min_inlier_ratio) ||
            config_.min_inlier_ratio <= 0.0 ||
            config_.min_inlier_ratio > 1.0)
        {
            throw std::invalid_argument(" pnpTracker: invalid PnPConfig.");
        }
    }

    /*
     * ================================================================
     * Track
     * ================================================================
     */
    bool PnPTracker::Track(const Frame &reference, Frame &current,
                           const std::vector<cv::DMatch> &temporal_matches,
                           const StereoCamera &stereo_camera)
    {
        // 1.重置上一轮结果
        stats_ = PnPTrackingStats{};

        inlier_matches_.clear();

        last_error_.clear();

        stats_.temporal_matches = temporal_matches.size();

        // 不允许把同一个 Frame 当成前后两帧
        if (&reference == &current)
        {
            last_error_ = "Reference and current Frame are the same.";
            return false;
        }

        if (!stereo_camera.initialized())
        {
            last_error_ = "StereoCamera is not rectified.";
            return false;
        }

        const auto &reference_features = reference.leftFeatures();
        const auto &current_features = current.leftFeatures();

        /*
         * ============================================================
         * 2. 构造 3D-2D Correspondences
         * ============================================================
         *
         * 注意：
         *
         *   queryIdx -> Frame0.leftFeatures()
         *
         *   trainIdx -> Frame1.leftFeatures()
         *
         * TemporalMatcher 已经恢复了原始索引。
         */
        std::vector<Correspondence> correspondences;
        correspondences.reserve(temporal_matches.size());

        /*
         * 防止重复使用某个 Reference Feature
         * 或 Current Feature。
         */
        std::vector<unsigned char> reference_used(reference_features.size(), 0);
        std::vector<unsigned char> current_used(current_features.size(), 0);

        /*
         * 同一个 MapPoint 不应在本次 PnP 中
         * 作为多个独立三维点重复计入。
         */
        std::unordered_set<unsigned long> used_mappoint_ids;

        for (const auto &match : temporal_matches)
        {
            if (match.queryIdx < 0 || match.trainIdx < 0)
            {
                continue;
            }

            const std::size_t reference_index = static_cast<std::size_t>(match.queryIdx);
            const std::size_t current_index = static_cast<std::size_t>(match.trainIdx);

            if (reference_index >= reference_features.size() || current_index >= current_features.size())
            {
                continue;
            }

            // 检查是否用过
            if (reference_used[reference_index] || current_used[current_index])
            {
                continue;
            }

            // 检查提取特征
            const auto &reference_feature = reference_features[reference_index];
            const auto &current_feature = current_features[current_index];

            if (!reference_feature || !current_feature)
            {
                continue;
            }

            /*
             * V1 要求 Current Frame 是尚未建立
             * MapPoint 关联的新帧。
             *
             * 避免覆盖已有地图关联。
             * 如果current_frame有地图点 退出
             */
            if (current_feature->hasMapPoint())
            {
                continue;
            }

            /*
             * 从参考帧 上一帧 Feature 获取地图点。
             *
             * 内部 weak_ptr::lock()。
             */
            auto map_point = reference_feature->mapPoint();
            if (!map_point)
            {
                continue;
            }

            const Eigen::Vector3d &Pw = map_point->position();
            if (!Pw.allFinite())
            {
                continue;
            }

            const cv::Point2f &pixel = current_feature->keypoint().pt; // 当前帧的像素坐标

            if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y))
            {
                continue;
            }

            if (used_mappoint_ids.count(map_point->id()) != 0)
            {
                continue;
            }

            /*
             * 保存一组真正用于 PnP 的数据。
             *
             * world_point：
             *     Pw
             *
             * image_point：
             *     Frame1 rectified pixel
             */
            Correspondence correspondence;

            correspondence.world_point = cv::Point3d(Pw.x(), Pw.y(), Pw.z());
            correspondence.image_point = cv::Point2d(pixel.x, pixel.y);

            correspondence.match = match;
            correspondence.map_point = map_point;
            correspondences.push_back(correspondence);

            reference_used[reference_index] = 1;
            current_used[current_index] = 1;
            used_mappoint_ids.insert(map_point->id());
        }

        stats_.valid_correspondences = correspondences.size();
        if (correspondences.size() < static_cast<std::size_t>(config_.min_correspondences))
        {
            last_error_ = "Insufficient 3D-2D correspondences.";
            return false;
        }

        /*
         * ============================================================
         * 3. 生成 OpenCV PnP 输入
         * ============================================================
         */
        std::vector<cv::Point3d> object_points;
        std::vector<cv::Point2d> image_points;
        object_points.reserve(correspondences.size());
        image_points.reserve(correspondences.size());

        for (const auto &correspondence : correspondences)
        {
            object_points.push_back(correspondence.world_point);
            image_points.push_back(correspondence.image_point);
        }

        /*
         * ============================================================
         * 4. 构造 Rectified Camera Matrix
         * ============================================================
         *
         * 当前使用的像素来自 rectified left image。
         *
         * 所以必须使用：
         *
         *     K_rect
         *
         * 而不是原始 Camera::K()。
         */
        const double fx = stereo_camera.fx();
        const double fy = stereo_camera.fy();
        const double cx = stereo_camera.cx();
        const double cy = stereo_camera.cy();

        if (!std::isfinite(fx) || !std::isfinite(fy) ||
            !std::isfinite(cx) || !std::isfinite(cy) ||
            fx <= 0.0 || fy <= 0.0)
        {
            last_error_ = "Invalid rectified camera intrinsics.";
            return false;
        }

        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

        // 图像去畸变 以及校正： distortion = 0
        cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);

        /*
         * ============================================================
         * 5. solvePnPRansac
         * ============================================================
         *
         * 输入：
         *
         *   Pw
         *   rectified pixel
         *   K_rect
         *
         * 输出：
         *
         *   rvec
         *   tvec
         *   inlier_indices
         *
         *
         * OpenCV 使用：
         *
         *   Pc = R * Pw + t
         *
         * 与我们的 T_cw 约定一致。
         */
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat inlier_indices;

        try
        {
            const bool solved = cv::solvePnPRansac(object_points, image_points, K, distortion,
                                                   rvec, tvec, false, config_.ransac_iterations, static_cast<float>(config_.reprojection_error_px),
                                                   config_.confidence, inlier_indices, cv::SOLVEPNP_EPNP);

            if (!solved)
            {
                last_error_ = "solvePnPRansac returned false. ";
                return false;
            }
        }
        catch (const cv::Exception &e)
        {
            last_error_ = std::string("OpenCV PnP error.") + e.what(); // 上一个solvepnpransan函数检查是否抛出错误
            return false;
        }

        stats_.ransac_inliers = inlier_indices.total();
        if (stats_.ransac_inliers < static_cast<std::size_t>(config_.min_inliers))
        {
            last_error_ = " Too few RANSAC inliers.";
            return false;
        }

        /*
         * ============================================================
         * 6. rvec -> Rotation Matrix
         * ============================================================
         *  向量变旋转矩阵
         * Rodrigues:
         *
         *   rvec -> R_cw
         */
        cv::Mat R_cv;

        try
        {
            cv::Rodrigues(rvec, R_cv);
        }
        catch (const cv::Exception &e)
        {
            last_error_ = std::string("Rodrigues error:") + e.what();
            return false;
        }
        /*
         * ============================================================
         * 7. OpenCV Pose -> Eigen::Matrix4d
         * ============================================================
         *
         *      [ R_cw  t_cw ]
         * Tcw = [            ]
         *      [ 0 0 0   1  ]
         */
        Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                T_cw(r, c) = R_cv.at<double>(r, c);
            }
            T_cw(r, 3) = tvec.at<double>(r, 0);
        }
        if (!T_cw.allFinite())
        {
            last_error_ = "PnP produced a non-finite pose.";
            return false;
        }
        /*
         * ============================================================
         * 8. Reprojection
         * ============================================================
         *
         * 使用求出的相机 Pose 将所有候选 Pw
         * 重新投影到 Frame1。
         *
         * 此处虽然投影全部点，
         * 但最终只保留 RANSAC 指定、
         * 且误差再次通过阈值的 Inlier。
         */
        std::vector<cv::Point2d> projected_points;
        try
        {
            cv::projectPoints(object_points, rvec, tvec, K, distortion, projected_points);
        }
        catch (const cv::Exception &e)
        {
            last_error_ = std::string("Reprojection error: ") + e.what();
            return false;
        }

        /*
         * 先收集最终内点的 correspondence 索引。
         *
         * 暂时不修改 Frame1。
         */
        std::vector<std::size_t> accepted_indices;
        accepted_indices.reserve(inlier_indices.total());
        std::vector<unsigned char> inlier_used(correspondences.size(), 0);

        double error_sum = 0.0;
        double max_error = 0.0;

        for (int i = 0; i < inlier_indices.rows; ++i)
        {
            const int index = inlier_indices.at<int>(i, 0);
            if (index < 0 || static_cast<std::size_t>(index) >= correspondences.size())
            {
                continue;
            }

            const std::size_t j = static_cast<std::size_t>(index);
            if (inlier_used[j])
            {
                continue;
            }
            inlier_used[j] = 1;
            const cv::Point3d &Pw = correspondences[j].world_point;

            /*
             * 先检查点在当前相机中的深度。
             *
             * Pc = R_cw * Pw + t_cw
             */
            const double camera_z =
                T_cw(2, 0) * Pw.x +
                T_cw(2, 1) * Pw.y +
                T_cw(2, 2) * Pw.z +
                T_cw(2, 3);

            if(!std::isfinite(camera_z) || camera_z <=1e-6)
            {
                continue;
            }

            const cv::Point2d &observed = image_points[j];      // 当前观测点
            const cv::Point2d &projected = projected_points[j]; // 投影点

            /*
             * Reprojection Error:
             *
             *   || observed - projected ||_2
             */
            const double error = std::hypot(observed.x - projected.x,
                                            observed.y - projected.y);

            if (!std::isfinite(error) || error > config_.reprojection_error_px)
            {
                continue;
            }

            accepted_indices.push_back(j);
            error_sum += error;

            if (error > max_error)
            {
                max_error = error;
            }
        }

        /*
         * ============================================================
         * 9. Final Inlier Validation
         * ============================================================
         */
        stats_.final_inliers = accepted_indices.size();
        stats_.inlier_ratio = static_cast<double>(stats_.final_inliers) / static_cast<double>(stats_.valid_correspondences);
        if (stats_.final_inliers > 0)
        {
            stats_.mean_reprojection_error_px = error_sum / static_cast<double>(stats_.final_inliers);
            stats_.max_reprojection_error_px = max_error;
        }

        if(stats_.inlier_ratio < config_.min_inlier_ratio)
        {
            last_error_ = "pnp inlier ratio is too low";
            return false;
        }

        if (stats_.final_inliers < static_cast<std::size_t>(config_.min_inliers))
        {
            last_error_ = "Too few verified PnP inliers.";
            return false;
        }

        /*
         * ============================================================
         * 10. 提交 Tracking 结果
         * ============================================================
         *
         * 只有执行到这里：
         *
         *   Pose 有效
         *   Inlier 数量足够
         *   Inlier Ratio 合理
         *   重投影误差通过验证
         *
         * 才修改 Current Frame。
         */
        inlier_matches_.reserve(accepted_indices.size());

        for (const std::size_t index : accepted_indices)
        {
            inlier_matches_.push_back(correspondences[index].match);
        }

        // 保存当前Frame的世界到相机变换 也就是当前帧的pose解算出
        current.setPose(T_cw);

        /*
         * ============================================================
         * 11. Inlier Feature -> MapPoint
         * ============================================================
         * 在当前帧用内点去关联 地图点
         * 注意：
         *
         * 这里只关联最终 Inlier。
         *
         * Outlier 的 Feature 不建立地图关联。
         */
        for (const std::size_t index : accepted_indices)
        {
            const auto &correspondence = correspondences[index];
            const std::size_t current_index = static_cast<std::size_t>(correspondence.match.trainIdx);
            const auto &current_feature = current_features[current_index];
            current_feature->setMapPoint(correspondence.map_point);
        }

        // 记录当前状态
        stats_.success = true;
        return true;
    }

    /*
     * ================================================================
     * Getters
     * ================================================================
     */
    const PnPTrackingStats& PnPTracker::stats() const
    {
        return stats_;
    }

    const std::vector<cv::DMatch>& PnPTracker::inlierMatches() const
    {
        return inlier_matches_;
    }

    const std::string& PnPTracker::lastError() const
    {
        return last_error_;
    }

}
