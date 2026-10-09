#include <iostream>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "mini_vo/frame.h"

int main()
{

    cv::Mat left_image =
        cv::Mat::zeros(
            480,
            640,
            CV_8UC1);

    cv::Mat right_image =
        cv::Mat::zeros(
            480,
            640,
            CV_8UC1);


    mini_vo::Frame frame(
        1,
        10.5,
        left_image,
        right_image);


    std::cout
        << "Frame ID: "
        << frame.id()
        << '\n';

    std::cout
        << "Timestamp: "
        << frame.timestamp()
        << '\n';

    std::cout<<"\nInitial_T_cw:\n"<<frame.pose()<<'\n';
    Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
     T_cw(0, 3) = 1.0;
     T_cw(1, 3) = 2.0;
     T_cw(2, 3) = 3.0;

     frame.setPose(T_cw);

    std::cout<<"\nUpdated T_cw:\n"<<frame.pose()<<'/n';

    //检查pose
    const double error = (frame.pose()  - T_cw).norm();

    std::cout<<"\nPose error = "<<error<<'\n';

    if(error > 1e-12)
    {
        std::cerr<<"Frame pose test failed."<<std::endl;
        return 1;
    }

    std::cout
        << "\nFrame test passed."
        << std::endl;

    return 0;

    // std::cout
    //     << "Left image size: "
    //     << frame.leftImage().cols
    //     << " x "
    //     << frame.leftImage().rows
    //     << '\n';

    // std::cout
    //     << "Right image size: "
    //     << frame.rightImage().cols
    //     << " x "
    //     << frame.rightImage().rows
    //     << '\n';

    // bool passed = true;

    // if (frame.id() != 0)
    // {
    //     std::cerr
    //         << "Frame ID test failed.\n";

    //     passed = false;
    // }

    // constexpr double epsilon = 1e-9;

    // if (std::abs(frame.timestamp() - 0.123) > epsilon)
    // {
    //     std::cerr
    //         << "Timestamp test failed.\n";

    //     passed = false;
    // }

    // if (
    //     frame.leftImage().cols != 640 ||
    //     frame.leftImage().rows != 480)
    // {
    //     std::cerr
    //         << "Left image size test failed.\n";

    //     passed = false;
    // }

    // if (
    //     frame.rightImage().cols != 640 ||
    //     frame.rightImage().rows != 480)
    // {
    //     std::cerr
    //         << "Right image size test failed.\n";

    //     passed = false;
    // }

    // if (passed)
    // {
    //     std::cout
    //         << "Frame test passed."
    //         << std::endl;

    //     return 0;
    // }

    // std::cerr
    //     << "Frame test failed."
    //     << std::endl;

    // return 1;
}