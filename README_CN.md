# HDF5 Pipeline

将机器人遥操作采集的原始 HDF5 和 LeRobot Parquet 数据，经过重命名、异常检测、视频渲染、人工打标四个步骤，处理为可用于训练的结构化数据集。

---

## 安装

```bash
conda env create -f environment.yml
conda activate hdf5_pipeline
pip install -e .
```

---

## 启动

```bash
streamlit run hdf5_pipeline/label/app.py
```

浏览器打开 `http://localhost:8501`，界面包含 6 个功能页面。

### spirit（千寻 moz1）parquet 专项

```bash
streamlit run hdf5_pipeline/parquet_ui_redo/app.py
# 或通过 CLI：转换 / 质检 / 校验 / 启动 UI
python hdf5_pipeline/cli.py spirit convert <raw_dir> <out_dir>
python hdf5_pipeline/cli.py spirit check <raw_dir> <out_csv> <out_json>
python hdf5_pipeline/cli.py spirit validate <raw_dir>
python hdf5_pipeline/cli.py spirit ui
```

spirit 专项把千寻 moz1 采集的原始 LeRobot parquet 转换为标准 LeRobot v2.1 数据集（22 维），
并单独质检与校验。UI 含转换 / 质检 / 校验三个 tab。

---

## 功能页面

### 文件重命名
扫描源目录及其子目录下的所有 HDF5 文件，按自然排序后统一复制并重命名为 `episode_NNNNNN` 格式到输出目录。完成后可锁定界面防止误操作。

### 质量检测
对指定目录下的 HDF5 或 LeRobot Parquet 文件进行异常帧检测。算法基于 DeltaActions 分位数评分，提供 loose / medium / strict 三级严格度预设。检测结果以 CSV 和 JSON 格式导出，界面中显示异常帧总数、异常最多的维度和文件。

### 视频渲染
将 HDF5 文件渲染为 MP4 视频。渲染面板由相机图像、16 维动作曲线、左右臂 7 维关节曲线三部分组成。支持多进程并发渲染、跳过已生成文件、实时进度条和日志监控。使用 @st.fragment 实现日志区域自动刷新。

### 视频打标
从数据库加载文件列表，播放 MP4 预览视频并进行 good / bad 质量分类。打标后 HDF5 文件自动移动到对应目录，数据库记录同步更新。支持自定义属性标注，属性值通过 JSON 字段存储。

### 数据总览
以翻页列表展示数据库中的全部记录，支持按质量标签筛选和按 SQL WHERE 条件自定义查询。查询结果可直接进行批量质量修改和属性赋值。

### 配置
管理自定义打标属性（新增、编辑、删除），支持将属性配置导出为 JSON 文件或从 JSON 文件导入。

---

## 数据格式

原始 HDF5 包含 30 维数据：左端位姿（7 维）、右端位姿（7 维）、左关节（7 维）、右关节（7 维）、左夹爪（1 维）、右夹爪（1 维）。训练时投影为 16 维，丢弃末端位姿（可从关节角度推导），只保留关节和夹爪。

LeRobot Parquet 格式直接读取 16 维训练空间数据。

---

## 技术说明

- 视频渲染使用 OpenCV 进行帧合成，曲线图使用 matplotlib 预渲染，逐帧只叠加光标线而非重绘
- 异常检测算法：action 与 state 做差 → 按 1%/99% 分位数归一化 → 映射到 [0,1] 区间 → 按严格度截断
- 多进程渲染使用 ProcessPoolExecutor，通过 Manager().Event() 实现跨进程终止信号
- 属性值以 JSON 字符串存储在 SQLite 的 attr 字段中，查询时使用 json_extract 函数
- 配置文件 config.json 管理运行时路径和自定义属性定义

## 跨平台

macOS、Windows、Linux 均可运行。文件夹选择器调用系统原生对话框（osascript / PowerShell / zenity）。
