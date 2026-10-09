#include "mini_vo/dataset.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

namespace mini_vo
{
    Dataset::Dataset(const std::string &dataset_path) : dataset_path_(dataset_path)
    {
        // 【修正 3】统一补上目录分隔符，传入路径末尾有没有 '/' 都能正确拼接。
        if (!dataset_path_.empty() && dataset_path_.back() != '/')
        {
            dataset_path_ += '/';
        }
    }

    bool Dataset::LoadCameraData(const std::string &csv_path, std::vector<std::string> &image_files, std::vector<long long> &timestamps_ns)
    {
        // 1.打开 CSV文件
        std::ifstream file(csv_path);
        if (!file.is_open())
        {
            std::cerr << "无法打开csv文件,请检查文件地址:"
                      << csv_path << std::endl;
            return false;
        }

        // 2.每次读取一行
        std::string line;
        while (std::getline(file, line))
        {
            // 【修正 1】getline 去掉 '\n' 后可能留下 '\r'，避免它混入图片文件名。
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }

            if (line.empty() || line[0] == '#')
            {
                continue; // 跳过空行或者是注释行
            }

            // 将当前这一行放入stringsstream
            std::stringstream ss(line);
            std::string timestamp_string;
            std::string filename;

            // 读取逗号左边时间戳timestamp
            if (!std::getline(ss, timestamp_string, ','))
            {
                continue; // ss放入了一整行内容，读取了逗号左边 截至条件放入了其中一个string变量
            }

            // 再读取逗号右边
            if (!std::getline(ss, filename))
            {
                continue;
            }

            // 将字符串时间戳内容转换成long long
            const long long timestamp_ns = std::stoll(timestamp_string);
            // 将两者放入各自的容器中
            timestamps_ns.push_back(timestamp_ns);
            image_files.push_back(filename);
        }
        return !image_files.empty();
    }

    bool Dataset::LoadCameraCalibration(const std::string &yaml_path, Camera::Ptr& camera)
    {
        // 1.打开YAML
        cv::FileStorage fs(yaml_path,cv::FileStorage::READ);

        // 检查文件是否打开
        if (!fs.isOpened())
        {
            std::cerr<<"打不开相机标定参数文件"<<yaml_path<<std::endl;
            return false;
        }

        // 2.读取内参    
        std::vector<double> intrinsics;
        fs["intrinsics"]>> intrinsics;

        if(intrinsics.size() !=4)
        {
            std::cerr<<"Invalidd intrinsics in: "<<yaml_path<<std::endl;
            return false;
        }

        const double fx = intrinsics[0];
        const double fy = intrinsics[1];
        const double cx = intrinsics[2];
        const double cy = intrinsics[3];

        // 3.读取畸变参数   
        std::vector<double> distortion_data;
        fs["distortion_coefficients"]>> distortion_data;

        if(distortion_data.size()!=4)
        {   
            std::cerr<<"Invalid distortion in: "<<yaml_path<<std::endl;
            return false;
        }

        Eigen::Vector4d distortion;
        distortion<<distortion_data[0],distortion_data[1],distortion_data[2],distortion_data[3];

        // 4.读取T_BS
        const cv::FileNode T_node = fs["T_BS"];
        
        if(T_node.empty())
        {
            std::cerr<<"不能找到t_bs的数据in"<<yaml_path<<std::endl;
            return false;
        }

        // 4.1 读取行列数
        const int rows = static_cast<int>(T_node["rows"]);
        const int cols = static_cast<int>(T_node["cols"]);
        if(rows!=4 || cols!=4)
        {
            std::cerr<<"T_BS is not 4*4 in:"<<yaml_path<<std::endl;
            return false;
        }

        std::vector<double>T_data;
        T_node["data"]>>T_data;

        if(T_data.size()!= 16)
        {
            std::cerr<<"Invalid T_BS data in: "<<yaml_path<<std::endl;
            return false;
        }

        // 5.将vector容器里面的数据换成matrix4d
        Eigen::Matrix4d T_BS;

        for (int row = 0; row < 4; ++row)
        {
            for(int col = 0;col<4;++col)
            {
                T_BS(row,col)=T_data[row*4+col];
            }
        }

        // 6.创建camera
        camera = std::make_shared<Camera>(fx,fy,cx,cy,distortion,T_BS);
        
        return true;
    }


    bool Dataset::Init()
    {
        // 【修正 2】初始化失败时，不允许读取半成品数据或报告正常结束。
        initialized_ = false;

        // 【修正 4】重新初始化时清空旧数据，避免 CSV 内容重复追加。
        current_image_index_ = 0;
        left_image_files_.clear();
        right_image_files_.clear();
        left_timestamps_ns_.clear();
        right_timestamps_ns_.clear();

        const std::string cam0_csv = dataset_path_ + "mav0/cam0/data.csv";
        const std::string cam1_csv = dataset_path_ + "mav0/cam1/data.csv";
        

        // 读取左相机
        if (!LoadCameraData(cam0_csv, left_image_files_, left_timestamps_ns_))
        {
            std::cerr << "无法读取cam0的数据" << std::endl;
            return false;
        }

        // 读取右相机数据
        if (!LoadCameraData(cam1_csv, right_image_files_, right_timestamps_ns_))
        {
            std::cerr << "无法读取cam1的数据" << std::endl;
            return false;
        }

        // 检查双目应该拥有相同数量的图像
        if (left_image_files_.size() != right_image_files_.size())
        {
            std::cerr << "双目数量图像不一致.\n"
                      << "cam0 : " << left_image_files_.size()
                      << '\n'
                      << "cam1: " << right_image_files_.size() << std::endl;
            return false;
        }

        // 读取标定参数
        const std::string cam0_yaml =dataset_path_+"mav0/cam0/sensor.yaml";
        const std::string cam1_yaml =dataset_path_+"mav0/cam1/sensor.yaml";
        
        if(!LoadCameraCalibration(cam0_yaml,camera_left_))
        {
            std::cerr<<"Failed to load cam0 calibration"<<std::endl;
            return false;
        }

        if(!LoadCameraCalibration(cam1_yaml,camera_right_))
        {
            std::cerr<<"Failed to load cam1 calibration"<<std::endl;
            return false;
        }

        initialized_ = true;

        // 假设双目图像数量一致，打印相关信息
        std::cout << "EuRoC数据集初始化成功.\n"
                  << "双目的帧数如下："
                  << left_image_files_.size() << std::endl;

        return true;
    }

    // 【修正 2】只有初始化成功且所有帧都已读取，才算正常结束。
    bool Dataset::IsFinished() const
    {
        return initialized_ && current_image_index_ >= left_image_files_.size();
    }

    Frame::Ptr Dataset::NextFrame()
    {
        // 【修正 2】防止在未初始化或初始化失败后访问不完整的左右目数据。
        if (!initialized_)
        {
            std::cerr << "Dataset is not initialized." << std::endl;
            return nullptr;
        }

        // 1.首先判断整个csv文件数据已经读完
        if (current_image_index_ >= left_image_files_.size())
        {
            return nullptr;
        }

        // 检查左右目 timestamp 理论上两边是同步的
        const long long left_timestamp = left_timestamps_ns_[current_image_index_];
        const long long right_timestamp = right_timestamps_ns_[current_image_index_];

        if (left_timestamp != right_timestamp)
        {
            std::cerr << "Stereo timestamp mismatch at index " << current_image_index_ << '\n';
            return nullptr;
        }

        // 拼接左右目图像路径
        const std::string left_path = dataset_path_ + "mav0/cam0/data/" + left_image_files_[current_image_index_];
        const std::string right_path = dataset_path_ + "mav0/cam1/data/" + right_image_files_[current_image_index_];

        // 读取灰度图
        cv::Mat left_image = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_image = cv::imread(right_path, cv::IMREAD_GRAYSCALE);

        // 检查图像
        if (left_image.empty())
        {
            std::cerr << "Failed to load image:" << left_path << std::endl;
            return nullptr;
        }

        if (right_image.empty())
        {
            std::cerr << "Failed to load image: " << right_path << std::endl;
            return nullptr;
        }

        // 将timestamp ns -> seconds
        const double timestamp = static_cast<double>(left_timestamp * 1e-9);

        // 7.创建Frame
        Frame::Ptr frame = std::make_shared<Frame>(current_image_index_,
                                                   timestamp, left_image, right_image);

        // 当前帧构造成功后 才进入下一帧
        ++current_image_index_;
        return frame;
    } // namespace mini_vo
}
