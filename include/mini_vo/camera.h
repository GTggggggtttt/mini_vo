#pragma once

#include<Eigen/Core>
#include<memory>
//定义camera这个类别
namespace mini_vo
{
class Camera    
    {
public:
    //智能指针类型别名
    using Ptr=std::shared_ptr<Camera>;

    //默认构造函数
    Camera()=default;

    //根据相机参数构造函数 这个函数只给test_camera使用 方便继续测试最基础的投影模型
    Camera(double fx,double fy,double cx,double cy);

    //函数重构 完整标定构造函数
    Camera(double fx,double fy,double cx,double cy,const Eigen::Vector4d &distortion,
    const Eigen::Matrix4d&T_BS);

    // 返回内参矩阵相机内参矩阵k
    Eigen::Matrix3d K() const;

    //相机像素坐标 + 深度 ->相机坐标 这是理想针孔模型
    Eigen::Vector3d Pixel2Camera(const Eigen::Vector2d&pixel,double depth)const;

    // 相机坐标->像素坐标
    Eigen::Vector2d Camera2Pixel(const Eigen::Vector3d &point)const;

    //getter提供接口访问内参
    double fx()const{return fx_;}
    double fy()const{return fy_;}
    double cx()const{return cx_;}
    double cy()const{return cy_;}

    //返回畸变参数
    const Eigen::Vector4d &distortion()const{return distortion_;}
    const Eigen::Matrix4d &T_BS()const{return T_BS_;}

private:
    //相机内参
    double fx_=0.0;
    double fy_=0.0;
    double cx_=0.0;
    double cy_=0.0;

    Eigen::Vector4d distortion_=Eigen::Vector4d::Zero();
    Eigen::Matrix4d T_BS_ = Eigen::Matrix4d::Identity();
    };
}//namespcae mini_vo