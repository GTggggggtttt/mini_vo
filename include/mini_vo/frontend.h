#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "mini_vo/feature.h"

namespace mini_vo
{
    /*
     * ================================================================
     * StereoMatchStats
     * ================================================================
     *
     * 记录 Stereo Matching pipeline 每一级剩余多少匹配。
     *
     * 目的不是算法必须使用这些数据，
     * 而是为了 Debug：
     *
     *      到底是哪一级过滤太严格？
     */
    struct StereoMatchStats
    {
        std::size_t left_features = 0;
        std::size_t right_features = 0;
        // knn中真正拥有两个候选的query数量
        std::size_t knn_candidates = 0;

        // 通过Ratio Test
        std::size_t ratio_passed = 0;

        /*
         * 通过：
         *
         *      |vL - vR| <= max_vertical_error
         */
        std::size_t epipolar_passed = 0;

        /*
         * 通过 disparity range。
         *
         * 也是最终 stereo match 数量。
         */
        std::size_t disparity_passed = 0;
    };

    /*
     * ================================================================
     * Frontend
     * ================================================================
     *
     * 当前职责：
     *
     * 1. ORB Feature Extraction
     *
     * 2. Stereo Descriptor Matching
     *
     * 3. Ratio Test
     *
     * 4. Epipolar Constraint
     *
     * 5. Disparity Constraint
     *
     *
     * 当前还不负责：
     *
     *      depth
     *      MapPoint
     *      temporal tracking
     *      PnP
     *      pose optimization
     */
    
    class Frontend
    {
    public:
        using Ptr = std::shared_ptr<Frontend>;
        /*
         * num_features：
         *
         * 在一个画面中ORB 希望提取的最大特征数量。
         *
         * 默认先设 1000。
         */
        explicit Frontend(int num_features = 1000);

        /*
         * ============================================================
         * 特征提取
         * ============================================================
         *
         * 输入：
         *
         *      一张图像
         *
         * 输出：
         *
         *      返回实际检测到的 Feature 数量。
         *
         * 同时更新：
         *
         *      keypoints_
         *      descriptors_
         *      features_
         */

        int ExtractFeatures(const cv::Mat &image);

        /*
         * ============================================================
         * 双目特征提取
         * ============================================================
         *
         * 分别在：
         *
         *      rectified left
         *      rectified right
         *
         * 上提取 ORB。
         *
         * 返回：
         *
         *      左右两张图是否都成功提取出特征。
         */

        bool ExtractStereoFeatures(const cv::Mat &left, const cv::Mat &right);

        /*
         * ============================================================
         * Stereo Matching V2
         * ============================================================
         *
         * knnMatch(k=2)
         *      ↓
         * Ratio Test
         *      ↓
         * Epipolar Constraint
         *      ↓
         * Disparity Range
         */
        int MatchStereo();

        /*
         * ============================================================
         * 完整双目前端入口
         * ============================================================
         *
         * 负责组织：
         *
         * ExtractStereoFeatures()
         *          ↓
         * MatchStereo()
         */
        bool ProcessStereo(const cv::Mat &left, const cv::Mat &right);

        /*
         * ============================================================
         * 设置 Stereo Matching 参数
         * ============================================================
         *
         * ratio_threshold:
         *
         *      best.distance
         *      -------------
         *      second.distance
         *
         * 必须小于该阈值。
         *
         *
         * max_vertical_error:
         *
         *      |vL - vR|
         *
         *
         * min/max disparity:
         *
         *      min_disp <= uL-uR <= max_disp
         *
         *
         * 推荐由实际相机几何推导 disparity，
         * 而不是永久硬编码。
         */
        void SetStereoMatchingParameters(float ratio_threshold, float max_vertical_error,
                                         float min_disparity, float max_disparity);

        /*
         * 单目 getter
         * Feature 对象。
         */
        const std::vector<Feature::Ptr> &features() const { return features_; }
        /*
         * 原始 OpenCV KeyPoint。
         *
         * test_feature 使用 drawKeypoints() 时很方便。
         */
        const std::vector<cv::KeyPoint> &keypoints() const { return keypoints_; }
        /*
         * ORB descriptors。
         *
         * 对 ORB 来说通常：
         *
         * rows = keypoints 数量
         *
         * 每一行 descriptor[i]
         *
         * 对应：
         *
         * keypoints[i]
         */
        const cv::Mat &descriptors() const { return descriptors_; }

        // ============================================================
        // Stereo getters
        // ============================================================
        const std::vector<cv::KeyPoint> &leftKeypoints() const { return keypoints_left_; }
        const std::vector<cv::KeyPoint> &rightKeypoints() const { return keypoints_right_; }

        const cv::Mat &leftDescriptors() const { return descriptors_left_; }
        const cv::Mat &rightDescriptors() const { return descriptors_right_; }

        const std::vector<Feature::Ptr> &leftFeatures() const { return features_left_; }
        const std::vector<Feature::Ptr> &rightFeatures() const { return features_right_; }

        /*
         * 最终经过几何筛选后的左右匹配。
         */
        const std::vector<cv::DMatch> &stereoMatches() const { return stereo_matches_; }

        // 获取最近一帧 Stereo Matching 的逐级统计。
        const StereoMatchStats &stereoMatchStats() const { return stereo_match_stats_; }

    private:
        /*
         * ============================================================
         * 内部通用特征提取函数
         * ============================================================
         *
         * 单目和双目都使用同一套实现，
         * 避免复制两份 ORB 代码。
         */
        int ExtractOne(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints,
                       cv::Mat &descriptors, std::vector<Feature::Ptr> &features);

    private:
        /*
         * ORB detector + descriptor extractor。
         */
        cv::Ptr<cv::ORB> orb_;

        // ============================================================
        // 单图模式
        // ============================================================
        std::vector<cv::KeyPoint> keypoints_;
        cv::Mat descriptors_;
        std::vector<Feature::Ptr> features_;
        // ============================================================
        // Stereo 模式
        // ============================================================
        std::vector<cv::KeyPoint> keypoints_left_;
        std::vector<cv::KeyPoint> keypoints_right_;
        cv::Mat descriptors_left_;
        cv::Mat descriptors_right_;
        std::vector<Feature::Ptr> features_left_;
        std::vector<Feature::Ptr> features_right_;
        /*
         * 最终通过所有筛选条件的匹配。
         */
        std::vector<cv::DMatch> stereo_matches_;

        /*最近一帧匹配过程的统计 */
        StereoMatchStats stereo_match_stats_;

        /*
         * ============================================================
         * Stereo matching 参数
         * ============================================================
         *
         * 校正后左右对应点纵坐标理论上应该相同。
         *
         * 实际由于：
         *
         *      特征定位误差
         *      calibration error
         *      image interpolation
         *
         * 允许少量误差。
         */
        float ratio_threshold_ = 0.75f;

        /*
         * 双目校正后：
         *
         *      vL ≈ vR
         */
        float max_vertical_error_ = 2.0f;

        /*
         * disparity range。
         *
         * 后面由相机几何设置。
         */
        float min_disparity_ = 1.0f;
        float max_disparity_ = 100.0f;
    };
} // namespace mini_vo