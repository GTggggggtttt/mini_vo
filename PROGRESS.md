# 项目进度：2026-10-23

当前阶段是可以在 EuRoC 短序列上运行的双目视觉里程计。已加入关键帧管理、地图点观测关系及跟踪历史接口，后端优化和回环尚未实现。

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
| `MapPoint` / `Map` 观测关系 | 插入关键帧时验证并登记左目观测，支持删除、重复检查和地图清理；观测通过 weak_ptr 引用关键帧 |
| `MapPoint` 跟踪历史 | 保存创建帧、最近成功跟踪帧和成功次数，拒绝重复或倒序登记；尚未接入建图与 PnP 调用流程 |
| `MultiFrameTracker` | 连续跟踪，独立报告位姿、补图和关键帧更新状态 |
| `test/common` | 复用 EuRoC 初始化、跟踪参数、跟踪循环及 TUM 轨迹输出 |
| 集成测试 | 逐帧统计、地图增长核对、关键帧计数与有效性验证、MapPoint → KeyFrame → Feature 关系验证 |

位姿约定为 `Pc = T_cw * Pw`。世界坐标系采用第一帧校正后的左相机坐标系；轨迹输出使用 `T_wc = inverse(T_cw)`，每行是 `timestamp tx ty tz qx qy qz qw`。

## 构建与运行

依赖 C++17、Eigen3 和 OpenCV。验证环境为 GCC 9.4.0、Eigen 3.3.7、OpenCV 4.2.0。以下是本次执行的构建和回归命令；未指定构建类型时沿用现有 build 配置。

```bash
cmake -S . -B build
cmake --build build -j4

./build/test_keyframe
./build/test_mappoint_observation
./build/test_observation_tracking /data/EuRoc/machine_hall/MH_01_easy/ 30
./build/test_multi_frame_tracking_v2 /data/EuRoc/machine_hall/MH_01_easy/ 30
```

数据集路径按本机位置调整。轨迹文件写入运行时的当前目录。

`test_multi_frame_tracking_v2` 已开启关键帧并验证 `KF +`、`KF Total`、关键帧持有的 Frame 及其 ID。当前 `test_keyframe_tracking` 入口沿用双目补图配置，未开启关键帧；它属于双目跟踪回归入口。

## 本次验证结果

- CMake 配置及现有构建目录中的全部目标编译、链接通过；已修复测试目标重复定义及缺失 `RemoveObservation()` 实现的问题。
- 上述 4 项运行检查全部通过，覆盖关键帧所有权与选择、观测添加/删除/恢复/清理、30 帧观测关系检查和 V2 连续跟踪。
- EuRoC `MH_01_easy` 前 30 帧：成功处理 30 帧，初始地图点 465 个，新增 3564 个，最终地图点 4029 个，关键帧 8 个。
- 观测集成测试最终有 2274 个地图点拥有关键帧观测，共登记 2748 条左目观测；逐帧验证观测对应的关键帧、Feature 索引和地图点对象一致。
- 带交互窗口的可视化测试未纳入本次自动运行；尚未完成全序列及真值误差评估。

2026-10-20 已完成独立空目录的 Release 全量构建和 8 项运行检查，涵盖相机、帧、Feature 关系、描述子生命周期及跟踪；当时也检查了 30 行轨迹的有限值、时间戳顺序与四元数归一化。

## 尚未集成的质量统计

`MapPoint` 已实现质量统计所需的基础接口，但建图代码尚未传入真实创建帧，PnP 也尚未调用 `RecordSuccessfulTrack()`。`mappoint_quality.h/.cpp`、`test_mappoint_quality.cpp`、`test_map_quality_tracking.cpp` 目前为空占位，未加入 CMake；尚未完成质量评估、地图点剔除及对应专项测试。

## 后续工作

1. 将真实创建帧和 PnP 内点跟踪记录接入 MapPoint，完成质量评估、专项测试及地图点维护。
2. 增加 PnP 后的位姿优化，再逐步实现局部 BA 和局部地图管理。
3. 处理 EuRoC 相机与机体外参、轨迹坐标系对齐，进行真值误差评估。
4. 扩展序列长度，检查跟踪丢失、地图增长和资源占用。
5. 实现正式的程序入口与 `VisualOdometry` 封装；当前 `app/run_kitti.cpp`、`visual_odometry` 和 `backend` 文件仍为空占位。
6. 后续学习 KITTI 数据适配、回环检测和全局优化。

`README.md` 下半部分保留最初的 KITTI 实现计划；本文件记录当前实际完成情况。`beiwang.cpp` 是学习备忘，不参与 CMake 构建。
