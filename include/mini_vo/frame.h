// ----------------- FRAME v3 --------------------------------------
#pragma once
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "mini_vo/feature.h"

namespace mini_vo
{

    class Frame
    {
    public:
        using Ptr = std::shared_ptr<Frame>;

        /*
         * Feature 容器类型。
         *
         * 每一个 shared_ptr 表示：
         *
         *      当前 Frame 拥有这个 Feature。
         */
        using FeatureContainer = std::vector<Feature::Ptr>;

        Frame() = default;
        Frame(unsigned long id, double timestamp, const cv::Mat &left_img, const cv::Mat &right_img);

        /*
         * ============================================================
         * Basic Information
         * ============================================================
         */
        unsigned long id() const;
        double timestamp() const;
        const cv::Mat &leftImage() const;
        const cv::Mat &rightImage() const;

        const Eigen::Matrix4d &pose() const;
        void setPose(const Eigen::Matrix4d &pose);

        /*
         * ============================================================
         * Left Features
         * ============================================================
         *
         * 向当前 Frame 添加一个左目 Feature。
         *
         * 返回 false：
         *
         *      feature == nullptr
         *
         * 返回 true：
         *
         *      成功加入容器。
         */
        bool addLeftFeature(const Feature::Ptr& feature);
        bool addRightFeature(const Feature::Ptr& feature);

        void setLeftFeatures(const FeatureContainer& features);
        void setRightFeatures(const FeatureContainer& features);

        // 获取左右Feature
        const FeatureContainer& leftFeatures()const;
        const FeatureContainer& rightFeatures()const;

        // 当前左右帧 Feature数量
        std::size_t leftFeatureCount() const;
        std::size_t rightFeatureCount() const;

        // 清空feature
        void clearFeatures();

    private:

        unsigned long id_ = 0;
        double timestamp_ = 0.0;
        cv::Mat left_img_;
        cv::Mat right_img_;
        Eigen::Matrix4d pose_ = Eigen::Matrix4d::Identity();

        // Feature ownership
        FeatureContainer left_features_;
        FeatureContainer right_features_;
    };

} // namespace mini_vo