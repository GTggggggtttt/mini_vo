#pragma once

#include <memory>
#include <string>
#include <utility>

#include "mini_vo/dataset.h"
#include "mini_vo/frame.h"
#include "mini_vo/stereo_camera.h"

namespace mini_vo::test
{
    /*
     * ================================================================
     * EurocTestContext
     * ================================================================
     *
     * 只用于测试。
     *
     * 一次性准备：
     *   Dataset
     *   Camera Calibration
     *   StereoCamera
     *   Frame0
     *   Rectification
     *
     * 不负责：
     *   ORB
     *   PnP
     *   Map Initialization
     */
    struct EurocTestContext
    {
        explicit EurocTestContext(const std::string &dataset_path) : dataset(dataset_path) {}
        bool Initialize()
        {
            error.clear();
            if (!dataset.Init())
            {
                error = "Dataset initialization failed.";
                return false;
            }

            auto left_camera = dataset.leftCamera();
            auto right_camera = dataset.rightCamera();

            if (!left_camera || !right_camera)
            {
                error = "Camera calibration unavailable.";
                return false;
            }

            /*
             * Dataset 已初始化，
             * 现在才能使用其相机标定创建 StereoCamera。
             *
             * unique_ptr 表示明确的独占所有权。
             */

             stereo_camera=std::make_unique<StereoCamera>(left_camera,right_camera);
             frame0 = dataset.NextFrame();

             if(!frame0)
             {
                error = "Cannot read Frame0.";
                return false;
             }

             if(!stereo_camera->InitRectification(frame0->leftImage().size()))
             {
                error = "Stereo rectification failed.";
                return false;
             }
             return true;
        }

        /*
         * 声明顺序也决定成员销毁顺序：
         * 析构时按声明顺序的逆序销毁。
         */
        Dataset dataset;
        std::unique_ptr<StereoCamera> stereo_camera;
        Frame::Ptr frame0;
        std::string error;
    };

}