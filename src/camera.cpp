#include"mini_vo/camera.h"

namespace mini_vo
{
    //使用相机参数初始化Camera对象 //用于没有标定外参的简单模型
    Camera::Camera(double fx,double fy,double cx,double cy):fx_(fx),fy_(fy),cx_(cx),cy_(cy){}

    //EuRoc完整camear构造函数
    Camera::Camera(double fx,double fy,double cx,double cy,const Eigen::Vector4d &distortion,
    const Eigen::Matrix4d& T_BS):fx_(fx),fy_(fy),cx_(cx),cy_(cy),distortion_(distortion),T_BS_(T_BS){}

    //返回相机内参矩阵：
    Eigen::Matrix3d Camera::K() const
    {
        Eigen::Matrix3d K;
        K<< fx_,0.0,cx_,
            0.0,fy_,cy_,
            0.0,0.0,1.0;
        return K;   
    }

    //简单Pixel -> Camera
    Eigen::Vector3d Camera::Pixel2Camera(const Eigen::Vector2d&pixel,double depth)const
    {
        const double x=(pixel.x()-cx_)*depth/fx_;
        const double y =(pixel.y()-cy_)*depth/fy_;
        return Eigen::Vector3d(x,y,depth);
    }

    //同样是理想的针孔摄像头
    Eigen::Vector2d Camera::Camera2Pixel(const Eigen::Vector3d&point)const
    {       
        const double u = fx_ * point.x()/point.z()+cx_;
        const double v = fy_ * point.y()/point.z()+cy_;
        return Eigen::Vector2d(u,v);
    }

    



}//namespcae mini_vo



