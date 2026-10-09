#include "mini_vo/frontend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace mini_vo
{
    Frontend::Frontend(int num_features)
    {
        if (num_features <= 0)
        {
            throw std::invalid_argument("Frontend: num_features must be positive.");
        }
        /*
         * ============================================================
         * 创建 ORB
         * ============================================================
         *
         * 当前只指定：
         *
         *      最大特征数量
         *
         * 其它参数先使用 OpenCV 默认值。
         *
         * 后面我们再研究：
         *
         * scaleFactor
         * nlevels
         * edgeThreshold
         * FAST threshold
         */
        orb_ = cv::ORB::create(num_features);
    }
    /*
     * ================================================================
     * SetStereoMatchingParameters
     * ================================================================
     */
    void Frontend::SetStereoMatchingParameters(float ratio_threshold,
                                               float max_vertical_error, float min_disparity, float max_disparity)
    {
        /*
         * Ratio Test 一般应该位于：
         *
         *      0 < ratio < 1
         */
        if (ratio_threshold <= 0.0f || ratio_threshold >= 1.0f)
        {
            throw std::invalid_argument("ratio_threshold must be between 0 and 1.");
        }

        if (max_vertical_error < 0.0f)
        {
            throw std::invalid_argument("max_vertical_error must be non-negative.");
        }

        if (min_disparity <= 0.0f)
        {
            throw std::invalid_argument("min_disparity must be positive.");
        }

        if (max_disparity <= min_disparity)
        {
            throw std::invalid_argument("max_disparity must be greater than min_disparity.");
        }

        ratio_threshold_ = ratio_threshold;
        max_vertical_error_ = max_vertical_error;
        min_disparity_ = min_disparity;
        max_disparity_ = max_disparity;
    }

    /*
     * ================================================================
     * ExtractOne
     * ================================================================
     *
     * 所有单张图像 ORB 提取都经过这里。
     */
    int Frontend::ExtractOne(
        const cv::Mat &image,
        std::vector<cv::KeyPoint> &keypoints,
        cv::Mat &descriptors,
        std::vector<Feature::Ptr> &features)
    {
        /*
         * ============================================================
         * 1. 清除上一帧结果
         * ============================================================
         *
         * Frontend 对象会被连续复用。
         *
         * 所以每处理新的一帧，
         * 必须先清理上一帧的数据。
         */
        keypoints.clear();
        descriptors.release();
        features.clear();

        /*
         * ============================================================
         * 2. 检查输入图像
         * ============================================================
         */

        if (image.empty())
        {
            std::cerr << "Frontend::ExtractOne():empty image." << std::endl;
            return 0;
        }
        /*
         * ============================================================
         * 3. 确保使用灰度图
         * ============================================================
         *
         * EuRoC 本身就是灰度图，
         * 因此正常情况下不会发生拷贝。
         *
         * cv::Mat gray = image;
         *
         * 是浅拷贝：
         *
         * gray 与 image 共享 buffer。
         */
        cv::Mat gray;
        if (image.channels() == 1)
        {
            gray = image;
        }
        else if (image.channels() == 3)
        {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        else if (image.channels() == 4)
        {
            cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        }
        else
        {
            std::cerr << "Unsupported image channels: " << image.channels() << std::endl;
            return 0;
        }
        /*
         * ============================================================
         * 4. ORB 特征提取
         * ============================================================
         *
         * detectAndCompute：
         *
         *      image
         *        ↓
         *
         *      detect
         *        ↓
         *      KeyPoint
         *
         *        +
         *
         *      compute
         *        ↓
         *      Descriptor
         *
         *
         * mask 使用：
         *
         *      cv::noArray()
         *
         * 表示整张图都允许提取特征。
         */
        orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

        //检查ORB 数据一致性
        if(keypoints.empty()|| descriptors.empty())
        {
            return 0;
        }
        if(descriptors.rows != static_cast<int>(keypoints.size()))
        {
            std::cerr<< "Frontend::ExtractOne(): "<< "KeyPoint / descriptor count mismatch."<< std::endl;
            keypoints.clear();
            descriptors.release();
            features.clear();
            return 0;
        }



        /*
         * ============================================================
         * 5. KeyPoint -> Feature
         * ============================================================
         *
         * 顺序保持完全一致：
         *
         * keypoints[i]
         *
         *      ↕
         *
         * descriptors.row(i)
         *
         *      ↕
         *
         * features[i]
         *
         *
         * 后面做 Stereo Matching 时，
         * 这个索引关系非常重要。
         */
        features.reserve(keypoints.size());

        for (std::size_t i = 0; i < keypoints.size(); ++i)
        {
            const int descriptor_index = static_cast<int>(i);
            const cv::Mat descriptor_row = descriptors.row(descriptor_index);
            features.push_back(std::make_shared<Feature>(keypoints[i],descriptor_row));
        }

      
        /*
         * ORB 正常情况下：
         *
         * descriptors.rows
         *
         * 应该等于：
         *
         * keypoints.size()
         */
        if (!descriptors.empty() && descriptors.rows != static_cast<int>(keypoints.size()))
        {
            std::cerr << "Warnning: 描述子 与 特征点 数量不匹配" << std::endl;
        }
        return static_cast<int>(features.size());
    }

    /*
     * ================================================================
     * 原来的单图接口
     * ================================================================
     */
    int Frontend::ExtractFeatures(const cv::Mat &image)
    {
        return ExtractOne(image, keypoints_, descriptors_, features_);
    }

    /*
     * ================================================================
     * Stereo feature extraction
     * ================================================================
     */

    bool Frontend::ExtractStereoFeatures(const cv::Mat &left, const cv::Mat &right)
    {
        /*
         * 每次处理新的一对图像，
         * 清空上一帧 stereo matching。
         */
        stereo_matches_.clear();
        const int left_count = ExtractOne(left, keypoints_left_, descriptors_left_, features_left_);
        const int right_count = ExtractOne(right, keypoints_right_, descriptors_right_, features_right_);

        // 更新统计
        stereo_match_stats_ = StereoMatchStats{};
        stereo_match_stats_.left_features = keypoints_left_.size();
        stereo_match_stats_.right_features = keypoints_right_.size();

        if (left_count == 0 || right_count == 0)
        {
            /* std::cerr << "Stereo feature extraction failed.\n"
                      << "left = " << left_count
                      << ", right: " << right_count << std::endl; */
            return false;
        }
        return true;
    }

    /*
     * ================================================================
     * Stereo Matching-----------V2
     * ================================================================
     */

    int Frontend::MatchStereo()
    {
        // 每次匹配重新清空结果
        stereo_matches_.clear();

        // 保留左右特征数量，清空其余统计
        const std::size_t left_count = keypoints_left_.size();
        const std::size_t right_count = keypoints_right_.size();

        stereo_match_stats_ = StereoMatchStats{};
        stereo_match_stats_.left_features = left_count;
        stereo_match_stats_.right_features = right_count;

        /*
         * ============================================================
         * 1. 检查 descriptor
         * ============================================================
         */
        if (descriptors_left_.empty() || descriptors_right_.empty())
        {
            std::cerr << "Frontend::MatchStereo(): empty descriptors." << std::endl;
            return 0;
        }
        /*
         * ============================================================
         * 2. BFMatcher
         * ============================================================
         *
         * ORB descriptor：
         *
         *      binary descriptor
         *
         * 因此使用：
         *
         *      NORM_HAMMING
         *
         * 而不是：
         *
         *      NORM_L2
         *
         *
         * crossCheck = true：
         *
         * 如果：
         *
         *      left A 最佳匹配 right B
         *
         * 同时：
         *
         *      right B 最佳匹配 left A
         *
         * 才保留。
         *
         * 这能减少一些错误匹配。
         */

        cv::BFMatcher matcher(cv::NORM_HAMMING, false);

        /*
         * ============================================================
         * 3. KNN Match
         * ============================================================
         *
         * 对每一个左图 descriptor：
         *
         * 找右图中距离最近的两个候选：
         *
         *      best
         *      second
         *
         *
         * 数据结构：
         *
         * knn_matches[i]
         *
         *      ↓
         *
         * 左图第 i 个 descriptor 的候选集合。
         */
        // std::vector<cv::DMatch> raw_matches;
        // matcher.match(descriptors_left_, descriptors_right_, raw_matches);

        std::vector<std::vector<cv::DMatch>> knn_matches;

        matcher.knnMatch(descriptors_left_, descriptors_right_, knn_matches, 2);

        /*
         * 最终数量最多不会超过
         * left feature 数量。
         */
        stereo_matches_.reserve(knn_matches.size());

        /*
         * ============================================================
         * 4. Pipeline Filtering
         * ============================================================
         */
        for (const auto &candidates : knn_matches)
        {
            /*
             * Ratio Test 必须有：
             *
             * best
             * second-best
             *
             * 两个候选。
             */
            if (candidates.size() < 2)
            {
                continue;
            }
            ++stereo_match_stats_.knn_candidates;
            /*
             * 因为 knnMatch：
             *
             * candidates[0]
             *
             * 是最近邻。
             *
             * candidates[1]
             *
             * 是第二近邻。
             */
            const cv::DMatch &best = candidates[0];
            const cv::DMatch &second = candidates[1];

            /*
             * ========================================================
             * 4.1 Ratio Test
             * ========================================================
             *
             * 如果：
             *
             *      best 和 second 差不多
             *
             * 说明：
             *
             *      当前 descriptor 很模糊，
             *      有多个候选都长得很像。
             *
             *
             * 我们要求：
             *
             * best.distance
             * ---------------- < ratio
             * second.distance
             */
            if (best.distance >= ratio_threshold_ * second.distance)
            {
                continue;
            }
            ++stereo_match_stats_.ratio_passed;

            /*
             * ========================================================
             * queryIdx / trainIdx
             * ========================================================
             *
             * knnMatch(
             *
             *      descriptors_left_,
             *      descriptors_right_
             *
             * )
             *
             *
             * queryIdx：
             *
             *      left
             *
             * trainIdx：
             *
             *      right
             */
            const cv::Point2f &left_point = keypoints_left_[best.queryIdx].pt;
            const cv::Point2f &right_point = keypoints_right_[best.trainIdx].pt;

            /*
             * ========================================================
             * 4.2 Epipolar Constraint
             * ========================================================
             *
             * 已经 Stereo Rectification：
             *
             *      vL ≈ vR
             */
            const float vertical_error = std::abs(left_point.y - right_point.y);
            if (vertical_error > max_vertical_error_)
            {
                continue;
                ;
            }
            ++stereo_match_stats_.epipolar_passed;

            /*
             * ========================================================
             * 4.3 Disparity
             * ========================================================
             *
             * 水平双目：
             *
             *      d = uL - uR
             */
            const float disparity = left_point.x - right_point.x;
            /*
             * 不再只是：
             *
             *      disparity > 0
             *
             * 而是要求：
             *
             *      min <= disparity <= max
             *
             * 这个范围应该根据实际深度范围推导。
             */
            if (disparity < min_disparity_ || disparity > max_disparity_)
            {
                continue;
            }
            ++stereo_match_stats_.disparity_passed;

            // 所有条件通过
            stereo_matches_.push_back(best);
        }
        return static_cast<int>(stereo_matches_.size());
    }
    /*
     * ================================================================
     * ProcessStereo
     * ================================================================
     */
    bool Frontend::ProcessStereo(const cv::Mat &left, const cv::Mat &right)
    {
        if (!ExtractStereoFeatures(left, right))
        {
            return false;
        }

        const int valid_matches = MatchStereo();
        return valid_matches > 0;
    }

} // namespace mini_vo
