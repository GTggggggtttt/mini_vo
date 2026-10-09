#include "mini_vo/frame.h"

namespace mini_vo
{

    Frame::Frame(unsigned long id, double timestamp, const cv::Mat &left_img, const cv::Mat &right_img)
        : id_(id), timestamp_(timestamp), left_img_(left_img), right_img_(right_img) {}

    /*
     * ================================================================
     * Basic Information
     * ================================================================
     */
    unsigned long Frame::id() const
    {
        return id_;
    }

    double Frame::timestamp() const
    {
        return timestamp_;
    }

    const cv::Mat &Frame::leftImage() const
    {
        return left_img_;
    }

    const cv::Mat &Frame::rightImage() const
    {
        return right_img_;
    }

    /*
     * ================================================================
     * Pose
     * ================================================================
     */
    const Eigen::Matrix4d &Frame::pose() const
    {
        return pose_;
    }

    void Frame::setPose(const Eigen::Matrix4d &pose)
    {
        pose_ = pose;
    }

    /*
     * ================================================================
     * addLeftFeature
     * ================================================================
     */
    bool Frame::addLeftFeature(const Feature::Ptr &feature)
    {
        // nullptr 没有意义。
        if (!feature)
        {
            return false;
        }
        left_features_.push_back(feature);
        return true;
    }

    bool Frame::addRightFeature(const Feature::Ptr &feature)
    {
        // nullptr 没有意义。
        if (!feature)
        {
            return false;
        }
        right_features_.push_back(feature);
        return true;
    }

    /*
     * ================================================================
     * setLeftFeatures
     * ================================================================
     */
    void Frame::setLeftFeatures(const FeatureContainer &features)
    {
        left_features_ = features;
    }

    void Frame::setRightFeatures(const FeatureContainer &features)
    {
        right_features_ = features;
    }

    /*
     * ================================================================
     * leftFeatures
     * ================================================================
     */
    const Frame::FeatureContainer &Frame::leftFeatures() const
    {
        return left_features_;
    }

    const Frame::FeatureContainer &Frame::rightFeatures() const
    {
        return right_features_;
    }

    /*
     * ================================================================
     * Feature Count
     * ================================================================
     */
    std::size_t Frame::leftFeatureCount() const
    {
        return left_features_.size();
    }

    std::size_t Frame::rightFeatureCount() const
    {
        return right_features_.size();
    }

    /*
     * ================================================================
     * clearFeatures
     * ================================================================
     */
    void Frame::clearFeatures()
    {
        left_features_.clear();
        right_features_.clear();
    }
} // namespace mini_vo