#include "mini_vo/temporal_matcher.h"
#include "mini_vo/mappoint.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace mini_vo
{

    namespace
    {
        /*
         * ================================================================
         * IsValidOrbDescriptor
         * ================================================================
         *
         * ORB 默认 descriptor：
         *
         *     1 row
         *     32 columns
         *     CV_8UC1
         *
         * 只有格式正确的 descriptor
         * 才能加入 BFMatcher。
         *
         * 该函数只在本 .cpp 中使用，
         * 所以放入匿名命名空间。
         */

        bool IsValidOrbDescriptor(const Feature::Ptr &feature)
        {
            if (!feature || !feature->hasDescriptor())
            {
                return false;
            }
            const cv::Mat &descriptor = feature->descriptor();
            return descriptor.rows == 1 && descriptor.cols == 32 && descriptor.type() == CV_8UC1;
        }
    } // anonymous namespace

    /*
     * ================================================================
     * Constructor
     * ================================================================
     */
    TemporalMatcher::TemporalMatcher(float ratio_threshold)
    {
        /*
         * std::isfinite 防止 NaN / Inf 绕过普通比较。
         */
        if (!std::isfinite(ratio_threshold) || ratio_threshold <= 0.0f || ratio_threshold >= 1.0f)
        {
            throw std::invalid_argument("TemporalMatcher:: invalid ratio threshold");
        }

        ratio_threshold_ = ratio_threshold;
    }

    /*
     * ================================================================
     * Match
     * ================================================================
     */
    int TemporalMatcher::Match(const Frame &reference, const Frame &current)
    {
        /*
         * ============================================================
         * 1. 清空上一轮结果
         * ============================================================
         *
         * TemporalMatcher 可以重复使用。
         *
         * 每次 Match() 不能保留上一次的匹配。
         */
        matches_.clear();
        stats_ = TemporalMatchStats{};

        const auto &reference_features = reference.leftFeatures();
        const auto &current_features = current.leftFeatures();

        stats_.reference_features = reference_features.size();
        stats_.current_features = current_features.size();
        /*
         * ============================================================
         * 2. 收集 Reference Descriptor
         * ============================================================
         *
         * 不是所有 Frame0 Feature 都能参加 PnP。
         *
         * 只有：
         *
         *      Feature -> 有效 MapPoint
         *
         * 的 Feature，才具有对应的 Pw。
         *
         *
         * 因此只收集这部分 Feature。
         */
        cv::Mat reference_descriptors;

        /*
         * 保存 BFMatcher 内部索引
         * 到原始 Frame Feature 索引的映射。
         *
         * 例如：
         *
         *   reference_descriptors.row(0)
         *       来自 reference_features[17]
         *
         *   reference_descriptors.row(1)
         *       来自 reference_features[45]
         *
         * 则：
         *
         *   reference_indices = {17, 45, ...}
         */
        std::vector<int> reference_indices;

        reference_indices.reserve(reference_features.size());

        for (std::size_t i = 0; i < reference_features.size(); ++i)
        {
            const auto &feature = reference_features[i];
            if (!feature)
            {
                continue;
            }
            /*
             * Feature::mapPoint() 内部通过
             * weak_ptr::lock() 获取 MapPoint。
             *
             * 只有地图点仍然存活，才允许参与匹配。
             */
            auto map_point = feature->mapPoint();
            if (!map_point)
            {
                continue;
            }

            ++stats_.reference_with_mappoint;

            if (!map_point->position().allFinite())
            {
                continue;
            }

            if (!IsValidOrbDescriptor(feature))
            {
                continue;
            }

            reference_descriptors.push_back(feature->descriptor());
            reference_indices.push_back(static_cast<int>(i));
        }
        stats_.reference_descriptors = reference_indices.size();
        /*
         * ============================================================
         * 3. 收集 Current Descriptor
         * ============================================================
         *
         * Current Frame 的 Feature
         * 还没有 MapPoint 关联。
         *
         * 所以这里只需要：
         *
         *      有效 ORB Descriptor
         */
        cv::Mat current_descriptors;
        std::vector<int> current_indices;

        current_indices.reserve(current_features.size());

        for (std::size_t i = 0; i < current_features.size(); ++i)
        {
            const auto &feature = current_features[i];
            if (!IsValidOrbDescriptor(feature))
            {
                continue;
            }
            current_descriptors.push_back(feature->descriptor());
            current_indices.push_back(static_cast<int>(i));
        }
        stats_.current_descriptors = current_indices.size();

        /*
         * KNN(k=2) 至少需要两个 train descriptors。
         */
        if (reference_descriptors.empty() || current_descriptors.rows < 2)
        {
            return 0;
        }
        /*
         * ============================================================
         * 4. BFMatcher
         * ============================================================
         *
         * 当前 ORB 使用默认 WTA_K = 2，
         * 因此采用 NORM_HAMMING。
         *
         * crossCheck = false，
         * 因为后面要使用 knnMatch(k=2)。
         */
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(reference_descriptors, current_descriptors, knn_matches, 2);

        /*
         * ============================================================
         * 5. Ratio Test
         * ============================================================
         *
         * 暂存通过 Ratio Test 的候选。
         *
         * 此时 DMatch 仍然使用 BFMatcher 的
         * 临时 descriptor matrix 索引。
         */
        std::vector<cv::DMatch> ratio_matches;
        ratio_matches.reserve(knn_matches.size());
        for (const auto &candidates : knn_matches)
        {
            if (candidates.size() < 2)
            {
                continue;
            }
            ++stats_.knn_candidates;

            const cv::DMatch &best = candidates[0];
            const cv::DMatch &second = candidates[1];

            /*
             * Ratio Test：
             *
             *   best.distance
             *   -------------
             *   second.distance
             *
             * 必须 < ratio_threshold_。
             */
            if (best.distance >= ratio_threshold_ * second.distance)
            {
                continue;
            }

            ++stats_.ratio_passed;
            ratio_matches.push_back(best);
        }
        /*
         * ============================================================
         * 6. 一对一匹配过滤
         * ============================================================
         *
         * forward knnMatch 可能得到：
         *
         *   Reference A -> Current X
         *   Reference B -> Current X
         *
         * 但当前一个 Current Feature
         * 不应该同时对应两个不同 MapPoint。
         *
         *
         * 简单 V1 策略：
         *
         *   先按 Hamming Distance 从小到大排序；
         *
         *   同一个 Current Descriptor
         *   只保留最先出现的匹配。
         *
         * 这是贪心唯一性过滤，
         * 并不等于完整的双向一致性匹配。
         */
        std::sort(
            ratio_matches.begin(), ratio_matches.end(),
            [](const cv::DMatch &a, const cv::DMatch &b)
            {
                if (a.distance != b.distance)
                {
                    return a.distance < b.distance;
                }

                if (a.queryIdx != b.queryIdx)
                {
                    return a.queryIdx < b.queryIdx;
                }

                return a.trainIdx < b.trainIdx;
            });

        std::vector<unsigned char> current_used(current_indices.size(), 0);

        matches_.reserve(ratio_matches.size());

        for (const auto &match : ratio_matches)
        {
            const std::size_t train_index = static_cast<std::size_t>(match.trainIdx);

            if (current_used[train_index])
            {
                continue;
            }

            current_used[train_index] = 1;

            /*
             * ========================================================
             * 7. 恢复原始 Frame Feature 索引
             * ========================================================
             *
             * 这是本模块特别重要的一步。
             *
             * 因为 reference_descriptors
             * 只包含有 MapPoint 的部分 Feature。
             *
             * 所以：
             *
             *   match.queryIdx
             *
             * 不等于：
             *
             *   reference.leftFeatures() 的原始索引。
             *
             * 必须通过 reference_indices 做映射。
             */
            const int original_reference_index = reference_indices[static_cast<std::size_t>(match.queryIdx)];

            const int original_current_index = current_indices[train_index];
            matches_.emplace_back(original_reference_index, original_current_index, match.distance);

            ++stats_.unique_passed;
        }
        return static_cast<int>(matches_.size());
    }

    /*
     * ================================================================
     * matches()
     * ================================================================
     */
    const std::vector<cv::DMatch> &TemporalMatcher::matches() const
    {
        return matches_;
    }

    /*
     * ================================================================
     * stats()
     * ================================================================
     */
    const TemporalMatchStats &
    TemporalMatcher::stats() const
    {
        return stats_;
    }

} // namespace mini_vo
