---
name: project-status
description: 项目各模块的完成状态和待处理事项
metadata: 
  node_type: memory
  type: project
  originSessionId: 58539c75-72f9-41a4-8593-495d11dfec21
---

## 状态总览（从 Windows 迁移时的状态）

| 模块 | 状态 | 备注 |
|------|------|------|
| `core/config.py` | ✅ 基本完成 | 需要根据 macOS 路径调整默认值 |
| `core/constants.py` | ✅ 完成 | 关节名、16维布局、delta mask |
| `core/hdf5_utils.py` | ✅ 基本完成 | 各函数已验证 |
| `core/video_utils.py` | ✅ 基本完成 | 已通过测试 |
| `rename/engine.py` | ✅ 完成 | Windows 上已验证通过 |
| `quality/detector.py` | ✅ 完成 | 异常评分算法 |
| `quality/hdf5_checker.py` | ✅ 基本完成 | 已通过测试 |
| `quality/lerobot_checker.py` | ❓ 未知 | 需要确认与 hdf5_checker 的关系 |
| `render/engine.py` | ⚠️ 有 BUG | cursor line, 关节图表, canvas 计算有问题 |
| `render/batch_gui.py` | ⚠️ 有 BUG | 进度条复用、中止后删除失败、重复转换计数 |
| `label/` | ⬜️ 空目录 | 未开始 |
| `preview/` | ⬜️ 空目录 | 未开始 |

## Windows 上已确认的知识点

### HDF5 数据布局（30维）
```
[0:7]   left_end_effector_pose   (训练时丢弃)
[7:14]  right_end_effector_pose  (训练时丢弃)
[14:21] left_arm_joints          (保留到16维)
[21:28] right_arm_joints         (保留到16维)
[28]    left_gripper             (保留)
[29]    right_gripper            (保留)
```

### 16维训练空间
```
[0:7]   left_joints
[7]     left_gripper
[8:15]  right_joints
[15]    right_gripper
```
末端位姿可以被关节角度推导出来，所以在训练中丢掉。

## 待办（迁移到 macOS 后）

1. ✅ 搭建 conda 环境（已完成）
2. ✅ 安装依赖 + ffmpeg（已完成）
3. ✅ 创建测试 HDF5 文件（已完成）
4. 🔄 修复 render 模块 BUG
5. ⬜ 完成 label 模块
6. ⬜ 完成 preview 模块
7. ⬜ 完善 README
8. ⬜ 配置 config.json 中的 macOS 路径

**References:** [[project-overview]], [[user-profile]]
