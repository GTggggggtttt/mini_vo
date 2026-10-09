#include<iostream>
#include<cmath>

#include"mini_vo/camera.h"


int main()
{
    //1.构造一个相机类别
     mini_vo::Camera camera(
    718.856,
    718.856,
    607.1928,
    185.2157
    );

    std::cout<<"Camera intrinsic matrix K:\n"<<camera.K()<<"\n\n";

    Eigen::Vector3d point_camera(1.0,0.5,5.0);
    std::cout<<" Point in camera coordinate: \n"<<point_camera.transpose()<<"\n\n";


   // ------------------------------------------------------------
    // 4. 三维点投影到图像
    //
    // Pc -> uv
    // ------------------------------------------------------------ 
    Eigen::Vector2d pixel = camera.Camera2Pixel(point_camera);
    std::cout<<"Projected pixel:\n"<<pixel.transpose()<<"\n\n";

    // ------------------------------------------------------------
    // 5. 使用像素 + 原来的深度反投影
    //
    // uv + Z -> Pc
    // ------------------------------------------------------------
    Eigen::Vector3d point_back = camera.Pixel2Camera(pixel,point_camera.z());
    std::cout<<"反投影点云： \n"<<point_back.transpose()<<"\n\n";

    
    // ------------------------------------------------------------
    // 6. 检查误差
    //
    // 如果 Camera2Pixel 和 Pixel2Camera 正确，
    //
    // point_camera 和 point_back 应该基本一致。
    // ------------------------------------------------------------
     double error=(point_camera-point_back).norm();
     std::cout<<"Error = "<<error<<std::endl;

     if(error<1e-9)
     {
        std::cout<<"23纬投影关系通过测试"<<std::endl;
        return 0;
     }
    //  return 0;
     std::cerr<<"不通过"<<std::endl;
     return 1;




}