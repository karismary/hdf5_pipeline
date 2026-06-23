---
name: project-overview
description: hdf5_pipeline — 机器人遥操作 HDF5 数据流水线（从原始数据到 MP4 视频）
metadata: 
  node_type: memory
  type: project
  originSessionId: 58539c75-72f9-41a4-8593-495d11dfec21
---

## 项目定位

机器人数据流水线，处理从遥操作采集的 HDF5 原始数据，经过重命名→质量检测→视频渲染→打标归档四个步骤，最终产出可用于训练的数据集。

原项目在 Windows 上从零散脚本重构为规范包结构（`hdf5_pipeline/`），现在迁移到 macOS 继续开发。

## 架构

```
hdf5_pipeline/           ← 规范 Python 包 (pip install -e .)
├── __init__.py          版本号
├── __main__.py          python -m 入口
├── core/                核心工具层
│   ├── config.py        统一配置（config.json 读写）
│   ├── constants.py     关节名、维度映射、delta mask、预设
│   ├── hdf5_utils.py    HDF5 读写（加载图像、动作、关节、30→16维投影）
│   └── video_utils.py   视频工具（提取首末帧、获取视频信息）
├── rename/              Step 1: HDF5 文件重命名
│   └── engine.py        重命名引擎
├── quality/             Step 2: 异常帧检测
│   ├── detector.py      异常评分算法
│   ├── hdf5_checker.py  读取 HDF5 → 投影到16维 → 调detector评分
│   └── lerobot_checker.py  处理 Lerobot 格式 HDF5
├── render/              Step 3: HDF5 → MP4 渲染
│   ├── engine.py        渲染引擎（matplotlib 绘图 + 合成视频）
│   └── batch_gui.py     批量渲染 GUI（并发处理，Tkinter）
├── label/               Step 4: 打标（TODO 空目录）
└── preview/             预览工具（TODO 空目录）
```

## 关键配置

- `config.json` — 路径配置 + 自定义打标属性，运行时由 UI 生成
- `pyproject.toml` — 项目元数据 + 依赖声明

**References:** [[user-profile]], [[project-status]]
