#include<stdexcept>
#include"mini_vo/keyframe.h"

namespace mini_vo
{
    KeyFrame::KeyFrame(unsigned long keyframe_id,const Frame::Ptr& frame): id_(keyframe_id),frame_(frame)
    {
        //keyframe 必须保存一个有效Frame
        if(!frame)
        {
            throw std::invalid_argument("KeyFrame: null Frame.");
        }
    }

    unsigned long KeyFrame::id() const
    {
        return id_;
    }

    unsigned long KeyFrame::frameId() const
    {
        return frame_->id();
    }

    double KeyFrame::timestamp() const
    {
        return frame_->timestamp();
    }

    Frame::Ptr KeyFrame::frame() const
    {
        return frame_;
    }

} // namespace mini_vo
