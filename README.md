<div align="center">

# hdf5_pipeline

机器人数据流水线工具 —— **重命名 → 质检 → 渲染 → 打标归档** 一站式完成。

基于 **Streamlit** 的图形界面 + 完整命令行入口，支持 **HDF5** 与 **LeRobot Parquet** 两种数据格式。

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![HDF5](https://img.shields.io/badge/HDF5-1B1B1B?logo=hdf5&logoColor=white)

</div>

---

## 功能一览

| 模块 | 说明 |
|---|---|
| 文件重命名 | 递归扫描 HDF5 原始数据，自然排序统一命名为 `episode_NNNNNN` |
| 质量检测 | DeltaAction 差分 + 分位数打分检测异常帧，导出 CSV / JSON 报告 |
| 视频渲染 | HDF5 逐帧渲染为 MP4，多进程并发，支持中断与断点续传 |
| 视频打标 | 首末帧 + 视频预览，一键标记 good / bad / pending / unlabeled 并自动归档 |
| 数据总览 | SQL 风格自定义查询、批量修改、分页浏览与页码跳转 |
| Spirit 支持 | 千寻原始 Parquet → 标准 LeRobot v2.1 数据集转换、质检与校验 |

## 快速开始

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate hdf5_pipeline
pip install -e .

# 2. 图形界面（完整打标系统）
streamlit run hdf5_pipeline/label/app.py

# 3. 或命令行一键流水线
hdf5-pipeline pipeline <src> <out> <report> <mp4>
```

### 命令行（CLI）

安装 `pip install -e .` 后即可使用 `hdf5-pipeline` 命令：

| 命令 | 功能 |
|---|---|
| `hdf5-pipeline rename <src> <out>` | 重命名数据文件到统一目录 |
| `hdf5-pipeline check <dir> [--strictness] [--format]` | 异常帧检测，导出 CSV / JSON 报告 |
| `hdf5-pipeline pipeline <src> <out> <report> <mp4>` | 一键流水线：重命名 → 质检 → 渲染 |
| `hdf5-pipeline render` | 启动独立视频渲染页面 |
| `hdf5-pipeline label` | 启动完整打标系统 |
| `hdf5-pipeline spirit convert / check / validate` | Spirit Parquet 转换 / 质检 / 校验 |

## 🗂️ 目录结构

```
hdf5_pipeline/
├── hdf5_pipeline/
│   ├── core/        # 配置、HDF5/Parquet 读写、校验、常量
│   ├── label/       # 打标系统（Streamlit 主界面 + SQLite + 状态管理）
│   ├── quality/     # 异常帧检测
│   ├── render/      # 视频渲染引擎
│   ├── rename/      # 数据重命名
│   ├── ui/          # Streamlit 页面组件
│   └── parquet_ui_redo/   # Spirit parquet 专用模块
├── config.json      # 运行配置（路径 + 自定义属性）
└── tests/           # pytest 测试
```

## 📚 文档

- [使用文档](./使用文档.md) — 环境搭建、启动方式、各功能模块操作指南
- [接口与开发文档](./接口与开发文档.md) — 模块接口、数据库 / 状态层、GUI 架构与开发规范
