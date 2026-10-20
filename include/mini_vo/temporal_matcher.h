#pragma once
#include <cstddef>
#include <vector>

#include <opencv2/features2d.hpp>
#include "mini_vo/frame.h"

namespace mini_vo
{
    /*
     * ================================================================
     * TemporalMatchStats
     * ================================================================
     *
     * 记录时间匹配每个阶段的数据量。
     *
     * 目的是方便以后分析：
     *
     *   1. Frame0 有多少有效 MapPoint？
     *   2. 有多少 descriptor 可以参与匹配？
     *   3. Ratio Test 过滤了多少？
     *   4. 一对一约束过滤了多少？
     */
    struct TemporalMatchStats
    {
        std::size_t reference_features = 0;
        std::size_t reference_with_mappoint = 0;
        std::size_t reference_descriptors = 0;
        std::size_t current_features = 0;
        std::size_t current_descriptors = 0;
        std::size_t knn_candidates = 0;
        std::size_t ratio_passed = 0;
        std::size_t unique_passed = 0;
        /* data */
    };

    /*
     * ================================================================
     * TemporalMatcher
     * ================================================================
     *
     * 职责：
     *
     *   在两个不同时间的 Frame 之间匹配左图 ORB Feature。
     *
     *
     * 输入：
     *
     *   Reference Frame（已有 MapPoint）
     *
     *   Current Frame（待估计 Pose）
     *
     *
     * 输出：
     *
     *   std::vector<cv::DMatch>
     *
     *
     * 索引约定非常重要：
     *
     *   match.queryIdx
     *
     *       对应 reference.leftFeatures() 的索引
     *
     *
     *   match.trainIdx
     *
     *       对应 current.leftFeatures() 的索引
     *
     *
     * 本类暂时不负责：
     *
     *   Stereo Matching
     *   Triangulation
     *   PnP
     *   Pose Update
     *   MapPoint Association Update
     *
     *
     * 本类不拥有 Frame，只在 Match() 执行期间读取 Frame。
     */
    class TemporalMatcher
    {
    public:
        explicit TemporalMatcher(float ratio_threshold = 0.75f);
        /*
         * ============================================================
         * Match
         * ============================================================
         *
         * reference:
         *     参考帧，通常为已经初始化地图的 Frame0
         *
         * current:
         *     当前帧，通常为 Frame1
         *
         * 返回：
         *     最终时间匹配数量
         */
        int Match(const Frame &reference, const Frame &current);

        /*
         * 匹配结果。
         *
         * 返回引用，不复制 vector。
         *
         * 注意：
         * 下一次 Match() 会覆盖内部结果。
         */
        const std::vector<cv::DMatch> &matches() const;

        /*
         * 最近一次匹配的统计。
         */
        const TemporalMatchStats &stats() const;

    private:
        float ratio_threshold_ = 0.75f;
        /*
         * 最终匹配。
         *
         * 每个 DMatch 中保存的是原始
         * Frame::leftFeatures() 容器索引。
         */
        std::vector<cv::DMatch> matches_;
        TemporalMatchStats stats_;
    };

} // namespace mini_vo
