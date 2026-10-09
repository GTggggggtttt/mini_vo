// Feature V3

#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace mini_vo
{
    /*
     * ================================================================
     * Forward Declaration
     * ================================================================
     */
    class Frame;
    class MapPoint;

    class Feature
    {
    public:
        using Ptr = std::shared_ptr<Feature>;

        Feature() = default;
        explicit Feature(const cv::KeyPoint &keypoint);
        Feature(const cv::KeyPoint &keypoint, const std::shared_ptr<Frame> &frame);
        Feature(const cv::KeyPoint &keypoint, const cv::Mat &descriptor);
        Feature(const cv::KeyPoint &keypoint, const cv::Mat &descriptor, const std::shared_ptr<Frame> &frame);

        const cv::KeyPoint &keypoint() const;
        const cv::Mat &descriptor() const;

        void setDescriptor(const cv::Mat&descriptor);

        bool hasDescriptor() const;

        void setFrame(const std::shared_ptr<Frame> &frame);

        std::shared_ptr<Frame> frame() const;

        bool hasFrame() const;

        void setMapPoint(const std::shared_ptr<MapPoint> &map_point);

        std::shared_ptr<MapPoint> mapPoint() const;

        bool hasMapPoint() const;

        void resetMapPoint();

    public:
        // 当前特征的像素位置
        cv::KeyPoint keypoint_;

        cv::Mat descriptor_;

        std::weak_ptr<Frame> frame_;

        std::weak_ptr<MapPoint> map_point_;
    };
} // namespace mini_vo
