#include<iostream>
#include<memory>
#include<Eigen/Core>
#include<opencv2/features2d.hpp>
#include<mini_vo/feature.h>
#include<mini_vo/frame.h>
#include<mini_vo/mappoint.h>

int main()
{
    //1. 创建frame
    auto frame = std::make_shared<mini_vo::Frame>();

    //2. 创建Feature
    cv::KeyPoint keypoint(100.0f,200.0f,10.0f);
    auto feature = std::make_shared<mini_vo::Feature>(keypoint,frame);

    //3. Feature -> Frame  
    std::cout<<"has frame:"<<feature->hasFrame()<<'\n';
    auto feature_frame = feature->frame();

    if(feature_frame)
    {
        std::cout<<"Frame exists . "<<'\n';
    }
    feature_frame.reset();

    //4 .创建mappoint
    auto map_point = std::make_shared<mini_vo::MapPoint>(1,Eigen::Vector3d(1.0,2.0,3.0));
    feature->setMapPoint(map_point);

    std::cout<<"has map point: "<<feature->hasMapPoint()<<'\n';

    // 5.销毁Frame strong owner 
    frame.reset();
    std::cout<<"after frame.reset: "<<feature->hasFrame()<<'\n';

    // 6.销毁mappoint strong owner
    map_point.reset();

    std::cout<<"after map_point.reset(): "<<feature->hasMapPoint()<<'\n';

    return 0;





}