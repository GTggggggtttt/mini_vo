#pragma once

#include <memory>
#include "mini_vo/frame.h"

namespace mini_vo
{
    /*
     * ================================================================
     * KeyFrame V1
     * ================================================================
     *
     * KeyFrame 是经过筛选、需要长期保留的 Frame。
     *
     * 本阶段不复制 Frame 内部的图像、Feature。
     *
     * Ownership:
     *
     *   Map
     *     └── shared_ptr<KeyFrame>
     *             └── shared_ptr<Frame>
     *                     └── shared_ptr<Feature>
     *
     * KeyFrame 通过 shared_ptr 延长 Frame 生命周期。
     */
    class KeyFrame
    {
    public:
        using Ptr = std::shared_ptr<KeyFrame>;

        /*
         * keyframe_id：
         *   KeyFrame 自己的 ID。
         *
         * frame：
         *   被选中的原始 Frame。
         */
        KeyFrame(unsigned long keyframe_id, const Frame::Ptr &frame);
        unsigned long id() const;

        /*
         * 被保存的原始 Frame ID。
         *
         * KeyFrame ID 和 Frame ID 是两个概念。
         */
        unsigned long frameId() const;
        double timestamp() const;

        /*
         * 返回共享所有权。
         *
         * 调用者也可以暂时持有这个 Frame。
         */
        Frame::Ptr frame() const;

    private:
        unsigned long id_ = 0;

        /*
         * KeyFrame 拥有 Frame 的共享所有权。
         *
         * 不能用 weak_ptr，
         * 否则原始 Frame 被释放后，
         * KeyFrame 可能无法访问自己的观测。
         */
        Frame::Ptr frame_;

    };

} // namespace mini_vo
