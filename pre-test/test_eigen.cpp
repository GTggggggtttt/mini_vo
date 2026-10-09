// ===========编译指令==========================
//  g++ -std=c++17 -Wall -Wextra -Wpedantic     -I/usr/include/eigen3     test_eigen.cpp -o build/test_eigen


#include <iostream>
#include<Eigen/Dense>
#include<Eigen/Geometry>
#include<cmath>
using namespace std;


//0.测试
// int main()
// {
//     //固定尺寸矩阵声明
//     Eigen::Matrix3d fixedMatrix;
//     //动态矩阵声明
//     Eigen::MatrixXd dynamicMatrix(2,3);
//     std::cout<<"Fixed matrix rows: "<<fixedMatrix.rows()
//     <<",cols: "<<fixedMatrix.cols()<<std::endl;
//     std::cout<<"Dynamic matrix rows: "<<dynamicMatrix.rows()
//     <<", cols: "<<dynamicMatrix.cols()<<std::endl;
//     return 0;
// }


// //1.创建与初始化方法
// int main()
// {
//     Eigen::Matrix3f m;
//     m<<1,2,3,4,5,6,7,8,9;
//     std::cout<<"Matrix initialized by comma:\n"<<m<<std::endl;
//     //特殊矩阵生成函数
//     Eigen::Matrix3d zeroMatrix = Eigen::Matrix3d::Zero();
//     Eigen::Matrix3d identityMatrix =Eigen::Matrix3d::Identity();
//     Eigen::Matrix3d randomMatrix = Eigen::Matrix3d::Random();

//     cout<<"Zero Matrix: \n"<<zeroMatrix<<endl;
//     cout<<"Identity Matrix: \n"<<identityMatrix<<endl;
//     cout<<"Random matrix: "<<randomMatrix<<endl;
//     //块操作与切片技巧用于提取矩阵中的部分元素
//     Eigen::Matrix4f mat;
//     mat<<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
//     Eigen::Matrix2f subMat=mat.block(1,1,2,2);
//     cout<<"sub Matrix:\n"<<subMat<<endl;
//     return 0;
// }


//2.算术运算与线性代数
// int main()
// {
//     Eigen::Matrix2d a,b;
//     a<<1,2,3,4;
//     b<<2,3,1,4;
//     cout<<"Matrix addition:\n"<<a+b<<endl;
//     cout<<"Matrix subtraction:\n"<<a-b<<endl;
//     cout<<"Element-wise multiplication:\n"<<a.array()*b.array();
//     cout<<"Matrix multiplication:\n"<<a*b<<endl;

//     //逆矩阵、行列式和迹的运算
//     Eigen::Matrix2d mat;
//     mat<<1,2,3,4;
//     cout<<"Inverse Matrix:\n"<<mat.inverse()<<endl;
//     cout<<"Determinant: "<<mat.determinant()<<endl;
//     cout<<"Trace: "<<mat.trace()<<endl;
//     return 0;

// }

//3.矩阵分解与求解器
// int main()
// {
// Eigen::Matrix3d A;
// A<<1,2,3,4,5,6,7,8,10;

// Eigen::Vector3d b(3,3,4);
// Eigen::Vector3d  x = A.lu().solve(b);
// cout<<"b is: "<<b<<endl;


// cout<<"Solution of the linear system:\n"<<x<<endl;


// }


//4.高级特性与优化实践
int main()
{
    /*
     * ============================================================
     * 1. 使用欧拉角构造旋转
     * ============================================================
     *
     * 这里定义三个基本旋转：
     *
     * roll  : 绕 X 轴旋转
     * pitch : 绕 Y 轴旋转
     * yaw   : 绕 Z 轴旋转
     *
     * 注意：
     * Eigen::AngleAxisd 的角度单位是弧度 rad，不是 degree。
     *
     * M_PI / 4 = 45°
     * M_PI / 3 = 60°
     * M_PI / 6 = 30°
     */

    // 绕 X 轴旋转 45°，即 roll
    Eigen::AngleAxisd rollAngle(
        M_PI / 4,
        Eigen::Vector3d::UnitX()
    );

    // 绕 Y 轴旋转 60°，即 pitch
    Eigen::AngleAxisd pitchAngle(
        M_PI / 3,
        Eigen::Vector3d::UnitY()
    );

    // 绕 Z 轴旋转 30°，即 yaw
    Eigen::AngleAxisd yawAngle(
        M_PI / 6,
        Eigen::Vector3d::UnitZ()
    );


    /*
     * ============================================================
     * 2. 将三个基本旋转组合成一个旋转
     * ============================================================
     *
     * 这里采用：
     *
     *      R = Rz * Ry * Rx
     *
     * 即：
     *
     *      yaw * pitch * roll
     *
     * 对一个列向量 p：
     *
     *      p' = Rz * Ry * Rx * p
     *
     * 矩阵运算从右往左执行，所以：
     *
     *      ① 先绕 X 轴 roll
     *      ② 再绕 Y 轴 pitch
     *      ③ 最后绕 Z 轴 yaw
     *
     * 这通常称为 ZYX 欧拉角形式。
     *
     * AngleAxis 相乘后可以转换成 Quaternion。
     */

    Eigen::Quaterniond q =
        yawAngle * pitchAngle * rollAngle;


    /*
     * ============================================================
     * 3. 四元数 -> 旋转矩阵
     * ============================================================
     *
     * 一个三维旋转可以有多种表示：
     *
     * 欧拉角
     *    ↓
     * AngleAxis
     *    ↓
     * Quaternion
     *    ↓
     * Rotation Matrix
     *
     * Quaternion：
     *
     *      q = [w, x, y, z]
     *
     * Rotation Matrix：
     *
     *      R ∈ SO(3)
     *
     * 是一个 3×3 矩阵。
     */

    Eigen::Matrix3d rotationMatrix = q.toRotationMatrix();

    // 也可以写：
    // Eigen::Matrix3d rotationMatrix = q.matrix();


    /*
     * ============================================================
     * 4. 旋转矩阵 -> 四元数
     * ============================================================
     *
     * Eigen 可以直接使用一个合法的 3×3 旋转矩阵
     * 构造 Quaternion。
     */

    Eigen::Quaterniond q2(rotationMatrix);


    /*
     * ============================================================
     * 5. 四元数 / 旋转矩阵 -> 欧拉角
     * ============================================================
     *
     * Eigen 中 eulerAngles(a0, a1, a2)
     * 表示按照指定轴顺序提取欧拉角。
     *
     * 轴编号：
     *
     *      0 -> X
     *      1 -> Y
     *      2 -> Z
     *
     * 所以：
     *
     *      eulerAngles(2, 1, 0)
     *
     * 表示：
     *
     *      Z -> Y -> X
     *
     * 对应：
     *
     *      yaw
     *      pitch
     *      roll
     *
     * 返回值：
     *
     *      eulerAngles[0] = yaw
     *      eulerAngles[1] = pitch
     *      eulerAngles[2] = roll
     *
     * 注意：
     * eulerAngles() 是旋转矩阵上的操作，因此先将 Quaternion
     * 转换成 Rotation Matrix。
     */

    Eigen::Vector3d eulerAngles =
        q2.toRotationMatrix().eulerAngles(2, 1, 0);


    /*
     * ============================================================
     * 6. 输出旋转矩阵
     * ============================================================
     */

    std::cout
        << "Rotation matrix:\n"
        << rotationMatrix
        << std::endl;


    /*
     * ============================================================
     * 7. 输出四元数
     * ============================================================
     *
     * Eigen::Quaterniond 内部可以通过：
     *
     *      q.w()
     *      q.x()
     *      q.y()
     *      q.z()
     *
     * 获取四元数各个分量。
     *
     * 注意：
     *
     * 有些库使用：
     *
     *      [x, y, z, w]
     *
     * 有些使用：
     *
     *      [w, x, y, z]
     *
     * SLAM 中经常因为这里的顺序出错。
     */

    std::cout
        << "Quaternion: "
        << q2.w() << " "
        << q2.x() << " "
        << q2.y() << " "
        << q2.z()
        << std::endl;


    /*
     * ============================================================
     * 8. 输出欧拉角
     * ============================================================
     *
     * transpose() 只是为了让 Vector3d 横着输出。
     *
     * 原来：
     *
     * yaw
     * pitch
     * roll
     *
     * transpose 后：
     *
     * yaw pitch roll
     *
     * 注意输出仍然是弧度。
     */

    std::cout
        << "Euler angles (yaw, pitch, roll): "
        << eulerAngles.transpose()
        << std::endl;


    /*
     * 如果希望转换成角度：
     */

    Eigen::Vector3d eulerDegrees =
        eulerAngles * 180.0 / M_PI;

    std::cout
        << "Euler angles in degrees: "
        << eulerDegrees.transpose()
        << std::endl;

    return 0;
}