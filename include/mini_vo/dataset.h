#pragma once

#include<memory>
#include<string>
#include<vector>

#include"mini_vo/frame.h"
#include"mini_vo/camera.h"

namespace  mini_vo
{
    class Dataset
    {
    public:
        using Ptr = std::shared_ptr<Dataset>;
        explicit Dataset(const std::string&dataset_path);

        //初始化EuRoc数据集
        bool Init();
        Frame::Ptr NextFrame();

        // 获取左右Camera 标定参数 
        Camera::Ptr leftCamera()const {return camera_left_;}
        Camera::Ptr rightCamera() const {return camera_right_;}

        // 【修正 2】NextFrame 返回 nullptr 后，用它区分正常结束和读取失败。
        bool IsFinished() const;
    
    private:
        bool LoadCameraData(const std::string &csv_path,
        std::vector<std::string>&image_files,
        std::vector<long long>&timestamps_ns);

        // 去读标定参数 sensor.yaml
        bool LoadCameraCalibration(const std::string &yaml_path, Camera::Ptr& camera);

    private:
        //EuRoc序列根目录，例如MH_01_easy
        std::string dataset_path_;
        // 【修正 2】只有完整初始化成功后，数据才可读取。
        bool initialized_ = false;
        //当前读取到第几帧
        std::size_t current_image_index_ =0;
        // cam0图像文件名
        std::vector<std::string>left_image_files_;
        // cam1图像文件名
        std::vector<std::string>right_image_files_;
        // 左相机时间戳，单位ns
        std::vector<long long >left_timestamps_ns_;
        // 右相机时间戳，单位ns
        std::vector<long long >right_timestamps_ns_;

        //shared_ptr 允许以后Frontend 继续共享camera 
        Camera::Ptr camera_left_;
        Camera::Ptr camera_right_;
    };
}//namespace mini_vo
