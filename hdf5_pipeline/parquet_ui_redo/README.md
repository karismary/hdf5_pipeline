# parquet_ui_redo — spirit（千寻 moz1）parquet 数据链路

API 签名与契约对齐，实现细节独立（不含 torch / lerobot，只依赖标准库 + numpy + pyarrow + streamlit）。

---

## 1. 数据契约

### 1.1 源格式

```
{任务名}/{时间戳}/{hash}/
├── data/chunk-000/episode_{i:06d}.parquet    # 每集一个，约 1082 行，30fps
├── videos/chunk-000/{cam}/episode_{i:06d}.mp4   # cam ∈ {cam_high, cam_left_wrist, cam_right_wrist}
├── meta/{info.json, episodes.jsonl, tasks.jsonl}
└── event_log.jsonl
```

parquet 物理列（26 列，命令/状态列为 `list<float>`）：

| 类型 | 列名 | 长度 |
|---|---|---|
| 双臂+腰 命令/状态 | `leftarm_cmd_joint_pos` / `leftarm_state_joint_pos` | 7 |
| 夹爪 | `leftarm_gripper_cmd_pos` / `leftarm_gripper_state_pos` | 1 |
| 双臂+腰 命令/状态 | `rightarm_cmd_joint_pos` / `rightarm_state_joint_pos` | 7 |
| 夹爪 | `rightarm_gripper_cmd_pos` / `rightarm_gripper_state_pos` | 1 |
| 腰部 | `torso_cmd_joint_pos` / `torso_state_joint_pos` | 6 |
| 底盘（忽略） | `base_cmd_speed` / `base_state_speed` | 3 |
| 索引标量 | `frame_index`(int64) `episode_index`(int64) `timestamp`(double) `task_index`(int64) | — |

- 相机像素不在 parquet：`meta/info.json` 将相机 feature 声明为
  `"dtype": "video"`、`"shape": [480, 640, 3]`、`"format": "rgb"`，真实像素在
  `videos/` 下对应 mp4。
- `meta/episodes.jsonl` 每行 `{"episode_index": 0, "length": 1082, "tasks": ["离线_normal"]}`；
  `meta/tasks.jsonl` 每行 `{"task_index": 0, "task": "离线_normal"}`。
- `event_log.jsonl` 每行一个事件，payload 含 `episode_idx`、`is_mistake`(bool)、
  `episode_info.tasks`（该集任务列表）、`episode_stats.stats`；`is_mistake=true` 的集转换/质检时剔除。

### 1.2 目标格式（标准 LeRobot v2.1）

```
{repo_id}/
├── data/chunk-000/episode_{i:06d}.parquet
├── videos/chunk-000/{cam}/episode_{i:06d}.mp4
└── meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl}
```

feature 布局（22 维 = 左臂7 + 左爪1 + 右臂7 + 右爪1 + 腰6，顺序即 `constants.MOTORS`）：

- `observation.state`：`list<float32>`(22) = 左臂 state(7) + 左爪 state(1) +
  右臂 state(7) + 右爪 state(1) + 腰 state(6)
- `action`：`list<float32>`(22)，同上但取 `*_cmd_*` 命令列
- `observation.images.{cam}`：`dtype: video`（像素不入 parquet）
- `frame_index` / `timestamp` / `episode_index` / `task_index`：int64 / float32 / int64 / int64

---

## 2. 模块结构

```
hdf5_pipeline/parquet_ui_redo/
├── __init__.py
├── constants.py      # 常量（列名 / 22 维布局 / CAMERAS / mask）——唯一依据，勿改
├── convert.py        # 格式转换
├── quality.py        # spirit 数据单独质检
├── validator.py      # 结构校验
└── app.py            # 独立 Streamlit UI（三个 tab）
```

### constants.py 关键常量

- `MOTORS`（22 项，即 22 维顺序）
- `CAMERAS` = `["cam_high", "cam_left_wrist", "cam_right_wrist"]`
- `SPIRIT_ACTION_COLS`（5 列命令列）
- `SPIRIT_STATE_COLS` = 命令列把 `_cmd_` 换成 `_state_`
- `DEFAULT_DELTA_MASK_22`：左臂 7×True、左爪 False、右臂 7×True、右爪 False、腰 6×True

---

## 3. 使用方式

### 3.1 CLI（主入口）

```bash
# 转换：raw_dir 下所有含 event_log.jsonl 的目录视为实例；out_dir 存在则先清空
python hdf5_pipeline/cli.py spirit convert <raw_dir> <out_dir> [--copy-videos]

# 质检：复用 hdf5_pipeline.quality.detector，mask 默认 DEFAULT_DELTA_MASK_22
python hdf5_pipeline/cli.py spirit check <raw_dir> <out_csv> <out_json> [--strictness strict]

# 校验：结构 + event_log 一致性；发现问题时退出码非 0
python hdf5_pipeline/cli.py spirit validate <raw_dir>

# 启动独立 Streamlit UI
python hdf5_pipeline/cli.py spirit ui
```

### 3.2 Python API

```python
from hdf5_pipeline.parquet_ui_redo.convert import convert_spirit
from hdf5_pipeline.parquet_ui_redo.quality import run_spirit_quality
from hdf5_pipeline.parquet_ui_redo.validator import validate_spirit_dataset

stats = convert_spirit("path/to/raw", "path/to/out", link_videos=True)
# -> {"converted": n, "skipped": m, "videos": k}

summary = run_spirit_quality("path/to/raw", "o.csv", "o.json", strictness="loose")
# -> summary（含 num_outliers / num_files / num_frames / top_dims / top_episodes）

ok, errors = validate_spirit_dataset("path/to/raw")
```

关键函数：

| 函数 | 说明 |
|---|---|
| `convert.assemble_episode(path)` | 单集 → `(action, state, frame_index, timestamp)`，action/state 为 `(T, 22)` float32 |
| `convert.parse_episodes(event_log_path)` | `event_log.jsonl` → `{episode_idx: {is_mistake, tasks, stats}}` |
| `convert.convert_one_instance(dir, out_dir, out_offset, link_videos=True)` | 单实例转换，返回 `(converted, skipped, video_count)` |
| `convert.convert_spirit(raw_dir, out_dir, link_videos=True)` | 批量转换 + 写 meta 四件套 |
| `quality.run_spirit_quality(raw_dir, out_csv, out_json, mask=..., strictness=...)` | spirit 质检入口 |
| `validator.validate_spirit_file(parquet_path)` | 单文件：可读 / 列齐全 / 行数 ≥ 1 |
| `validator.validate_spirit_dataset(raw_dir)` | 整目录 + event_log 一致性 |

### 3.3 Streamlit UI

```bash
streamlit run hdf5_pipeline/parquet_ui_redo/app.py
```

三个 tab（共用顶部 raw_dir，跨 tab 用版本号广播刷新缓存）：

- **转换**：选 raw_dir / out_dir，跑 `convert_spirit`，显示 converted / skipped / videos
- **质检**：选 raw_dir，跑 `run_spirit_quality`，展示 top 异常帧
- **校验**：选 raw_dir，跑 `validate_spirit_dataset`，列出错误

---

## 4. 验证

1. **单集**：`assemble_episode` 输出 shape `(T, 22)`，与源列手工
   `np.concatenate([np.stack(t[c].to_numpy()) ...])` 数值一致。
2. **整实例**：`convert_spirit` 后读回 parquet —— 列名 6 列、
   `observation.state` / `action` 类型 `list<float32>`、每行 `len == 22`、
   `episode_index` 从 0 连续、视频 symlink 目标存在。
3. **is_mistake 剔除**：`is_mistake=true` 的集不产出（parquet 与视频一并跳过）。
4. **质检**：真实数据跑 `spirit check`，产出 CSV/JSON（无异常帧时按设计不生成），`num_outliers >= 0` 不崩。
5. **校验**：真实数据跑 `spirit validate`，坏文件（如 0 字节 parquet）被显式列出、退出码非 0。