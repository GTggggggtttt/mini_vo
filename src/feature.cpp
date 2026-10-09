#include "mini_vo/feature.h"
#include <stdexcept>
#include "mini_vo/frame.h"
#include "mini_vo/mappoint.h"

namespace mini_vo
{
    Feature::Feature(const cv::KeyPoint &keypoint) : keypoint_(keypoint) {}
    Feature::Feature(const cv::KeyPoint &keypoint, const std::shared_ptr<Frame> &frame)
        : keypoint_(keypoint), frame_(frame) {}
    Feature::Feature(const cv::KeyPoint &keypoint, const cv::Mat &descriptor)
        : keypoint_(keypoint), descriptor_(descriptor) {}
    Feature::Feature(const cv::KeyPoint &keypoint, const cv::Mat &descriptor, const std::shared_ptr<Frame> &frame)
        : keypoint_(keypoint), descriptor_(descriptor), frame_(frame) {}


    const cv::KeyPoint &Feature::keypoint() const
    {
        return keypoint_;
    }

    const cv::Mat& Feature::descriptor()const
    {
        return descriptor_;
    }

    void Feature::setDescriptor(const cv::Mat&descriptor)
    {
        if(descriptor.empty())
        {
            descriptor_.release();
            return;
        }
        if(descriptor.rows !=1)
        {
            throw std::invalid_argument("Feature::setDescriptor():""descriptor must contain exactly one now");
        }
        //clone 描述子
        descriptor_ = descriptor.clone();
    }

    bool Feature::hasDescriptor() const
    {
        return !descriptor_.empty();   
    }

    void Feature::setFrame(const std::shared_ptr<Frame> &frame)
    {
        frame_ = frame;
    }

    std::shared_ptr<Frame> Feature::frame() const
    {
        return frame_.lock();
    }

    bool Feature::hasFrame() const
    {
        return !frame_.expired();
    }

    void Feature::setMapPoint(const std::shared_ptr<MapPoint> &map_point)
    {
        map_point_ = map_point;
    }

    std::shared_ptr<MapPoint> Feature::mapPoint() const
    {
        return map_point_.lock();
    }

    bool Feature::hasMapPoint() const
    {
        return !map_point_.expired();
    }

    void Feature::resetMapPoint()
    {
        map_point_.reset();
    }

} // namespace mini_vo
