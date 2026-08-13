# HDF5 Pipeline — 机器人数据流水线

将原始机器人遥操作数据文件，经过**重命名 → 异常检测 → 视频渲染 → 打标归档**四个步骤，变成可用于训练的结构化数据集。支持 HDF5 与 LeRobot Parquet 两种数据格式。

---

## 安装

```bash
# 方式一：pip 安装
pip install -e .

# 方式二：conda 环境（推荐）
conda env create -f environment.yml
conda activate hdf5_pipeline
```

---

## 项目结构

```
hdf5_pipeline/
├── core/                       # 共享工具层（所有模块依赖）
│   ├── __init__.py             # 统一导出公共 API
│   ├── constants.py            # 关节名称、维度映射、严格度预设
│   ├── config.py               # config.json 读写 + 默认值
│   ├── hdf5_utils.py           # HDF5 数据读取：图像、动作、关节
│   ├── video_utils.py          # MP4 解析：元信息、帧提取
│   └── utils.py                # 跨平台文件夹选择对话框
│
├── rename/                     # Step 1: 文件重命名
│   └── engine.py               # 统一命名为 episode_NNNNNN 格式
│
├── quality/                    # Step 2: 异常帧检测
│   ├── detector.py             # 核心算法：DeltaActions + 分位数评分
│   ├── hdf5_checker.py         # HDF5 格式数据入口
│   └── lerobot_checker.py      # LeRobot Parquet 格式数据入口
│
├── render/                     # Step 3: 视频渲染
│   └── engine.py               # HDF5 → MP4 多面板合成渲染
│
├── label/                      # Step 4: 打标归档
│   ├── database.py             # SQLite 数据库操作（init/add/get/scan）
│   ├── app.py                  # Streamlit 统一操作界面（6 标签页）
│   └── style.css               # 自定义样式
│
├── ui/                         # 各步骤 Streamlit 界面模块（可独立启动）
│   ├── module_app.py           # 通用入口：streamlit run module_app.py -- --module X
│   ├── rename_tab.py           # 📁 文件重命名标签页
│   ├── quality_tab.py          # 🧹 质量检测标签页
│   └── render_tab.py           # 🎬 视频渲染标签页
│
├── parquet_ui_redo/            # spirit（千寻 moz1）parquet 专项（复现版，见 README.md）
│   ├── constants.py            # 22 维列名 / CAMERAS / mask 契约（唯一依据）
│   ├── convert.py              # 原始 parquet → 标准 LeRobot v2.1 数据集
│   ├── quality.py              # spirit 数据单独质检（22 维 mask）
│   ├── validator.py            # 结构 + event_log 一致性校验
│   └── app.py                  # 独立 Streamlit UI（转换/质检/校验）

config.json                     # 运行时路径 + 自定义打标属性定义
pyproject.toml                  # 包元数据与依赖
environment.yml                 # Conda 环境配置（ffmpeg、SDL 等）
todo/                           # 待优化计划清单
test_data/                      # 测试数据
```

---

## 数据流

```
原始数据 (采集器输出的 HDF5 / Parquet)
    │
    ▼
┌─── rename ──────────────────────────────────────┐
│  收集子目录中的 .hdf5 / .parquet                  │
│  → 统一命名为 episode_NNNNNN                     │
│  输入: 含子目录的文件夹                            │
│  输出: 重命名后的统一格式文件                       │
└──────────────────────────────────────────────────┘
    │
    ▼
┌─── quality (异常检测) ───────────────────────────┐
│  读取 action + state                              │
│  HDF5: 30 维 → 16 维投影（丢弃末端位姿）            │
│  Parquet: 直接读取 16 维数据                        │
│  DeltaActions → 分位数归一化 → 异常帧评分           │
│  输出: outlier_frames.csv + outlier_summary.json   │
└──────────────────────────────────────────────────┘
    │
    ▼
┌─── render (仅 HDF5) ─────────────────────────────┐
│  读 HDF5 中的图像 + 动作 + 关节数据                 │
│  曲线图用 matplotlib 预渲染一次 → 逐帧 OpenCV 合成   │
│  GUI: 多进程并发、断点续传、ETA 预估                 │
└──────────────────────────────────────────────────┘
    │
    ▼
┌─── label (打标归档) ─────────────────────────────┐
│  Streamlit 打标系统                                │
│  ① 文件重命名  — 批量重命名 HDF5                    │
│  ② 质量检测    — 运行异常帧检测                      │
│  ③ 视频渲染    — HDF5 → MP4 渲染                   │
│  ④ 视频打标    — MP4 预览 → good/bad 分类          │
│                  → HDF5 自动归档到对应目录            │
│  ⑤ 数据总览    — SQL 查询 + 批量修改质量/属性         │
│  ⑥ 配置        — 自定义打标属性定义                   │
│                                                    │
│  SQLite 数据库管理 + calattr 自定义属性              │
└──────────────────────────────────────────────────┘
```

---

## 各模块函数速查

### core (共享层)

| 函数 | 作用 |
|---|---|
| `load_images_from_hdf5(path)` | 读所有摄像头帧 (支持 NCHW→NHWC 转置) |
| `load_actions_from_hdf5(path, n)` | 读动作数据 (自动识别 action/actions 键) |
| `load_joints_from_hdf5(path, n)` | 读左右臂 7 维关节数据 |
| `load_raw_30dim(path)` | 读完整 30 维 action + state |
| `project_30_to_16(x30)` | 丢弃末端位姿，保留关节+夹爪 |
| `normalize_image_array(arr)` | 统一图像格式 float→uint8, NCHW→NHWC |
| `get_hdf5_files(folder)` | 递归搜集 .h5/.hdf5 |
| `get_sorted_files(folder, ext)` | 通用递归文件搜集 (自然排序) |
| `extract_first_last_frames(path)` | 视频首末帧提取 |
| `get_video_info(path)` | 视频总帧数 + fps |
| `get_frame(path, n)` | 随机读取指定帧 |
| `pick_folder()` | 跨平台原生文件夹选择对话框 |
| `load_config()` / `save_config()` | config.json 读写 |

### rename (重命名)

| 函数 | 作用 |
|---|---|
| `collect_hdf5_files(data_dir)` | 递归搜集 .h5/.hdf5，去重+自然排序 |
| `rename_files(files, output_dir)` | 复制并重命名为 `episode_NNNNNN.hdf5` |

### quality (异常检测)

| 函数 | 作用 |
|---|---|
| `parse_mask(mask_str)` | "1,1,0,..." → bool 数组 |
| `apply_delta(action, state, mask)` | 计算 DeltaActions |
| `fit_quantiles(all_delta)` | 每维度的 1%/99% 分位数 |
| `compute_outliers(episodes, ...)` | 完整检测链路：Delta → 归一化 → 评分 → 截断 |
| `export_results(rows, summary, csv, json)` | 导出 CSV + JSON |
| `run_hdf5_check(data_glob, ...)` | HDF5 专用入口 |
| `run_lerobot_check(data_glob, ...)` | LeRobot Parquet 专用入口 |

严格度预设:

| 预设 | 最低得分 | 每轮最多帧 | 全局最多帧 |
|---|---|---|---|
| loose | 15 | 40 | 5000 |
| medium | 40 | 20 | 2500 |
| strict | 80 | 10 | 1000 |

### render (渲染)

| 函数/类 | 作用 |
|---|---|
| `render_mp4(hdf5_path, out_mp4, ...)` | 单个 HDF5 → MP4 渲染 |
| `BatchApp` (Tkinter 类) | 批量渲染 GUI：多进程、断点续传、ETA 预估 |

渲染特点：
- 多面板合成：图像行（多摄像头拼接）+ 动作曲线行 + 关节曲线行
- 曲线图用 matplotlib 预渲染一次，逐帧只叠加红色光标线（避免逐帧重绘）
- 编码 avc1，15fps，支持 `ProcessPoolExecutor` 并行渲染

### label (打标系统)

| 函数 | 作用 |
|---|---|
| `init_db(db_path)` | 创建 label 表 |
| `scan_pairs(db_path, mp4_dir, raw_dir)` | 扫描 MP4 + 数据文件配对入库 |
| `add_label(db_path, name, path, quality, attr)` | 更新打标结果 |
| `get_list(db_path)` | 获取全部记录 |
| `get_records(db_path, mp4_name)` | 查询单条记录 |
| `query_records(db_path, where)` | 自定义 SQL 查询 |
| `translate_where(condition, attr_map)` | `@属性="值"` → SQLite json_extract |

数据库表结构:

```sql
CREATE TABLE label (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hdf5_name TEXT,
    hdf5_path TEXT,
    mp4_name TEXT,
    mp4_path TEXT,
    quality TEXT DEFAULT 'unlabeled',   -- good / bad / unlabeled
    attr TEXT DEFAULT '{}',             -- 自定义属性 JSON
    created_at DATETIME,
    labeled_at TIMESTAMP
);
```

---

## 数据维度

### 30 维 → 16 维映射

```
HDF5 原始 30 维:                           训练 16 维:
┌─────────────────────────┐               ┌──────────────────┐
│ 0-6   左端位姿 (丢弃)    │               │ 0-6  左关节      │
│ 7-13  右端位姿 (丢弃)    │  project_30   │ 7    左夹爪      │
│ 14-20 左关节            │  ─────────→  │ 8-14 右关节      │
│ 21-27 右关节            │               │ 15   右夹爪      │
│ 28    左夹爪            │               └──────────────────┘
│ 29    右夹爪            │  末端位姿可从关节角度推导，丢弃
└─────────────────────────┘
```

### 关节名称

`JOINT_NAMES`: `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow_pitch`, `wrist_yaw`, `wrist_pitch`, `wrist_roll`

### DeltaActions 掩码

`DEFAULT_DELTA_MASK_16`: 关节维度 True（delta = action − state），夹爪维度 False（保留 raw action）

---

## 配置

`config.json` 管理运行时路径与自定义打标属性：

```json
{
  "paths": {
    "db_dir": "test_data/db/label.db",        // SQLite 数据库
    "raw_dir": "test_data/raw/",               // HDF5 原始数据
    "mp4_dir": "test_data/mp4/",               // MP4 视频
    "good_dir": "test_data/good_quality/",     // 好质量 HDF5 归档
    "bad_dir": "test_data/bad_quality/"        // 差质量 HDF5 归档
  },
  "custom_cols": {
    "attr_background": { "label": "背景", "option": ["阳台", "书房", "厨房"], "type": "text" },
    "attr_weather":    { "label": "天气", "option": ["下雨天", "晴天", "雾霾天"], "type": "text" }
  }
}
```

---

## 启动方式

### Streamlit 打标系统（主要入口）

```bash
streamlit run hdf5_pipeline/label/app.py
```

### 独立模块页面

```bash
# 单独启动某个界面模块（默认 render）
streamlit run hdf5_pipeline/ui/module_app.py -- --module quality
# 或通过 CLI
hdf5-pipeline ui --module quality
```

### spirit（千寻 moz1）parquet 专项

```bash
# 独立 Streamlit UI（转换/质检/校验）
streamlit run hdf5_pipeline/parquet_ui_redo/app.py
# 或通过 CLI：转换 / 质检 / 校验 / 启动 UI
python hdf5_pipeline/cli.py spirit convert <raw_dir> <out_dir>
python hdf5_pipeline/cli.py spirit check <raw_dir> <out_csv> <out_json>
python hdf5_pipeline/cli.py spirit validate <raw_dir>
python hdf5_pipeline/cli.py spirit ui
```

spirit 专项把千寻 moz1 采集的**原始** LeRobot parquet（命令/状态分散在 per-part 列，
无 `action` / `observation.state`）转换为标准 LeRobot v2.1（22 维 = 左臂7 + 左爪1 + 右臂7 + 右爪1 + 腰6），
并单独质检与校验。详见 [parquet_ui_redo/README.md](hdf5_pipeline/parquet_ui_redo/README.md)。

### 编程调用

```python
from hdf5_pipeline import (
    collect_hdf5_files, rename_files,
    run_hdf5_check, render_mp4,
    init_db, scan_pairs, add_label,
)

# Step 1: 重命名
files = collect_hdf5_files("raw_data/")
n = rename_files(files, "renamed/")

# Step 2: 异常检测
summary = run_hdf5_check("renamed/*.hdf5", "out.csv", "out.json", strictness="medium")

# Step 3: 渲染
ok, msg, name = render_mp4("renamed/episode_000000.hdf5", "videos/episode_000000.mp4")

# Step 4: 打标
init_db("label.db")
scan_pairs("label.db", "videos/", "renamed/")
add_label("label.db", "episode_000000.mp4", quality="good", attr='{"背景": "阳台"}')
```

---

## 开发状态

| 模块 | 状态 | 备注 |
|---|---|---|
| core/ | ✅ 完成 | 工具函数 + config 读写 + 文件选择器 |
| rename/engine.py | ✅ 完成 | 统一命名，当前仅 HDF5 |
| quality/ | ✅ 完成 | 双格式支持（HDF5 / LeRobot Parquet） |
| render/engine.py | ✅ 完成 | 多面板合成 + 多进程同步 |
| label/database.py | ✅ 完成 | SQLite 封装 |
| label/app.py | ✅ 完成 | 6 标签页 Streamlit 界面 |
| ui/module_app.py | ✅ 完成 | 独立模块入口（--module 切换） |
| ui/rename_tab.py | ✅ 完成 | |
| ui/quality_tab.py | ✅ 完成 | |
| ui/render_tab.py | ✅ 完成 | 多进程并发 + @fragment 日志 + 超时保护 |
| cli.py | ✅ 完成 | rename / check / pipeline / ui / render / label + spirit 子命令 |
| parquet_ui_redo/ | ✅ 完成 | spirit 专项（复现版）：convert / quality / validator / app，CLI 入口 `spirit` |
| 单元测试 | ⏳ 未开始 | pytest 已配置 |

---

## 待优化项

参见 [todo/](todo/README.md)，包含 3 个专项计划：

- [x] [session_state 统一管理](todo/01-session_state-统一管理.md)
- [x] [数据库大数据量优化](todo/02-database-大数据量优化.md)
- [x] [Parquet 格式支持](todo/03-补充parquet格式支持.md) — 已完成：`hdf5_pipeline/parquet_ui_redo/`（+ `parquet_ui/` 正式版）
