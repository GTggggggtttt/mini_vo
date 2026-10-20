# 项目进度：2026-10-20

当前阶段是可以在 EuRoC 短序列上运行的双目视觉里程计。已加入关键帧管理，后端优化和回环尚未实现。

## 已实现

| 模块 | 当前能力 |
| --- | --- |
| `Dataset` | 读取 EuRoC 标定、同步双目图像及时间戳，转换为秒 |
| `Camera` / `StereoCamera` | 投影、反投影、双目校正和校正后相机模型 |
| `Frame` / `Feature` | 图像、位姿、ORB 描述子和地图点关联 |
| `MapInitializer` | 双目匹配、视差深度、初始世界地图点 |
| `TemporalMatcher` | 参考帧到当前帧的 ORB 匹配、Ratio Test、一对一筛选和原始索引恢复 |
| `PnPTracker` | 3D–2D 对应、PnP RANSAC、正深度与重投影验证、内点关联 |
| `StereoMapUpdater` | 位姿估计成功后创建世界坐标地图点，限制每帧新增数量 |
| `KeyFrame` / `KeyFrameSelector` | 根据帧间隔、平移、旋转及内点数量选择关键帧，由 Map 长期保存 |
| `MultiFrameTracker` | 连续跟踪，独立报告位姿、补图和关键帧更新状态 |
| 集成测试 | 逐帧统计、地图增长核对、关键帧计数与有效性验证、TUM 轨迹输出 |

位姿约定为 `Pc = T_cw * Pw`。世界坐标系采用第一帧校正后的左相机坐标系；轨迹输出使用 `T_wc = inverse(T_cw)`，每行是 `timestamp tx ty tz qx qy qz qw`。

## 构建与运行

依赖 C++17、Eigen3 和 OpenCV。本次验证环境为 GCC 9.4.0、Eigen 3.3.7、OpenCV 4.2.0。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

./build/test_camera
./build/test_frame
./build/test_feature_relation
./build/test_keyframe
./build/test_feature_descriptor /data/EuRoc/machine_hall/MH_01_easy/
./build/test_multi_frame_tracking /data/EuRoc/machine_hall/MH_01_easy/ 2
./build/test_multi_frame_tracking_v2 /data/EuRoc/machine_hall/MH_01_easy/ 30
./build/test_keyframe_tracking /data/EuRoc/machine_hall/MH_01_easy/ 30
```

数据集路径按本机位置调整。轨迹文件写入运行时的当前目录。

`test_multi_frame_tracking_v2` 已开启关键帧并验证 `KF +`、`KF Total`、关键帧持有的 Frame 及其 ID。当前 `test_keyframe_tracking` 入口沿用双目补图配置，未开启关键帧；它在本次作为双目跟踪回归运行。

## 本次验证结果

- 从独立的空构建目录完成 Release 全量构建。
- 上述 8 项运行检查全部通过，涵盖相机、帧、Feature 关系、描述子生命周期、关键帧所有权与选择，以及连续跟踪。
- EuRoC `MH_01_easy` 前 30 帧：成功处理 30 帧，初始地图点 465 个，新增 3564 个，最终地图点 4029 个，关键帧 8 个。
- 逐帧关键帧新增数与累计数一致，同时覆盖插入和不插入两种情况；保存的关键帧均通过 Frame 指针和 ID 检查。
- 输出轨迹有 30 行，每行 8 个有限数值；时间戳递增，四元数归一化。
- 带交互窗口的可视化测试未纳入本次自动运行；尚未完成全序列及真值误差评估。

## 后续工作

1. 增加 PnP 后的位姿优化，再逐步实现局部 BA 和局部地图管理。
2. 处理 EuRoC 相机与机体外参、轨迹坐标系对齐，进行真值误差评估。
3. 扩展序列长度，检查跟踪丢失、地图增长和资源占用。
4. 实现正式的程序入口与 `VisualOdometry` 封装；当前 `app/run_kitti.cpp`、`visual_odometry` 和 `backend` 文件仍为空占位。
5. 后续学习 KITTI 数据适配、回环检测和全局优化。

`README.md` 下半部分保留最初的 KITTI 实现计划；本文件记录当前实际完成情况。`beiwang.cpp` 是学习备忘，不参与 CMake 构建。
