#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include "mini_vo/frame.h"
#include "mini_vo/stereo_camera.h"

namespace mini_vo
{
    /*
     * ================================================================
     * PnPConfig
     * ================================================================
     *
     * PnP + RANSAC 的配置。
     *
     * 这些值是当前 V1 的经验初值，
     * 并不是对所有 EuRoC 序列都最优。
     */
    struct PnPConfig
    {
        // 至少需要多少 3D-2D 对应才能尝试Pnp
        int min_correspondences = 6;

        // 成功跟踪要求的最小Inlier数量
        int min_inliers = 15;

        // RANSAC 最大迭代次数。
        int ransac_iterations = 200;

        // 重投影误差阈值，单位px 像素
        double reprojection_error_px = 3.0;

        // RANSAC 置信度
        double confidence = 0.99;

        // 成功跟踪要求的最小Inlier比例
        double min_inlier_ratio = 0.5;
    };

    /*
     * ================================================================
     * PnPTrackingStats
     * ================================================================
     *
     * 记录当前帧 Tracking 的几何验证结果。
     */
    struct PnPTrackingStats
    {
        std::size_t temporal_matches = 0;
        std::size_t valid_correspondences = 0;
        std::size_t ransac_inliers = 0;
        std::size_t final_inliers = 0;

        double inlier_ratio = 0.0;
        double mean_reprojection_error_px = 0.0;
        double max_reprojection_error_px = 0.0;

        bool success = false;
    };

    /*
     * ================================================================
     * PnPTracker V1
     * ================================================================
     *
     * 输入：
     *
     *   Reference Frame
     *       已经关联 MapPoint Pw
     *
     *   Current Frame
     *       已提取 ORB Feature
     *
     *   Temporal Matches
     *       Reference Feature <-> Current Feature
     *
     *   StereoCamera
     *       提供校正后左相机内参
     *
     *
     * 输出：
     *
     *   1. 当前帧 Pose T_cw
     *
     *   2. RANSAC Inlier Matches
     *
     *   3. Inlier 的 Feature -> MapPoint 关联
     *
     *   4. PnPTrackingStats
     *
     *
     * 生命周期：
     *
     *   PnPTracker 不拥有 Frame、Map 或 MapPoint。
     *
     *   只在 Track() 中临时读取对应对象。
     */
    class PnPTracker
    {
    public:
        //构造函数声明
        //explicit 禁止该构造函数进行隐式类型转换
        //pnpconfig() 默认参数：创建一个默认配置对象
        explicit PnPTracker(const PnPConfig& config = PnPConfig());

        /*
         * ============================================================
         * Track
         * ============================================================
         *
         * 返回 true：
         *
         *   PnP 成功，几何验证通过，
         *   已更新 current 的 Pose 和 Inlier 关联。
         *
         * 返回 false：
         *
         *   没有获得可信 Pose。
         *   不修改 current Pose / Feature 关联。
         *
         *
         * DMatch 索引约定：
         *
         *   queryIdx -> reference.leftFeatures()
         *
         *   trainIdx -> current.leftFeatures()
         */
        bool Track(const Frame &reference, Frame &current,
                   const std::vector<cv::DMatch> &temporal_matches,
                   const StereoCamera &stereo_camera);

        const PnPTrackingStats& stats() const;

        /*
         * 返回通过几何验证的时间匹配。
         *
         * 索引仍然对应两个 Frame 的原始 Feature。
         */
        const std::vector<cv::DMatch> &inlierMatches() const;

        // 最近一次失败的原因
        const std::string &lastError() const;

    private:
        PnPConfig config_;

        PnPTrackingStats stats_;

        std::vector<cv::DMatch> inlier_matches_;

        std::string last_error_;
    };

}//namespace mini_vo
