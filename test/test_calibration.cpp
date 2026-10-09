#include <cmath>
#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/LU>  // 提供 Matrix4d::inverse() 的实现。

#include "mini_vo/dataset.h"

int main(int argc, char **argv)
{
    // 1.检查dataset path
    if (argc != 2)
    {
        std::cerr << "Usage : " << argv[0] << "< euroc_dataset_path" << std::endl;
        return 1;
    }

    mini_vo::Dataset dataset(argv[1]);

    // 2. Init()
    if (!dataset.Init())
    {
        std::cerr << "数据集初始化失败" << std::endl;
        return 1;
    }

    // 3.获取相机参数
    auto cam0 = dataset.leftCamera();
    auto cam1 = dataset.rightCamera();
    if (!cam0 || !cam1)
    {
        std::cerr << "相机是空指针" << std::endl;
        return 1;
    }

    // 4. 打印相机内参
    std::cout << "\n==================== Camera 0 =====================\n";
    std::cout << "K0:\n"
              << cam0->K() << '\n';
    std::cout << "Distortion0:\n"
              << cam0->distortion().transpose() << '\n';
    std::cout << "T_BS:\n"
              << cam0->T_BS() << '\n';

    std::cout << "\n==================== Camera 1 =====================\n";
    std::cout << "K1:\n"
              << cam1->K() << '\n';
    std::cout << "Distortion1:\n"
              << cam1->distortion().transpose() << '\n';
    std::cout << "T_BS:\n"
              << cam1->T_BS() << '\n';

    // 5.计算 cam0 -> cam1
    const Eigen::Matrix4d T_C1_C0 = cam1->T_BS().inverse() * cam0->T_BS();
    std::cout << "\n============== Stereo Transform =============== \n";
    std::cout<<"T_C1_CO: \n"<<T_C1_C0<<'\n';

    //6. 提取translation
    const Eigen::Vector3d translation = T_C1_C0.block<3,1>(0,3);
    std::cout<<"\nTranslation C1 <- C0:\n"<<translation.transpose()<<'\n';

    //7. 计算baseline
    const double baseline = translation.norm();
    std::cout<<"\nStereo baseline = "<<baseline<<"m\n";

    constexpr double expected_baseline=0.11008;
    constexpr double tolerance = 1e-3;
    if(std::abs(baseline - expected_baseline)<tolerance)
    {
        std::cout<<"\nCalibration test passed."<<std::endl;
        return 0;

    }
    std::cerr<<"\nCalibration test failed. "<<std::endl;
    return 1;
}
