# 测试（pytest）— 完整链路回归

对整条流水线做回归：HDF5 四步（重命名 → 异常检测 → 视频渲染 → 打标归档）+ spirit（千寻 moz1）parquet 专项。
配置见 `pyproject.toml`（`testpaths = ["tests"]`，pytest >= 7）。

## 运行

```bash
./conda_env/bin/python -m pytest          # 全部
./conda_env/bin/python -m pytest tests/test_spirit.py -v   # 单个模块
./conda_env/bin/python -m pytest -k rename -v              # 按名字过滤
```

> 从项目根运行。`python -m` 会把当前目录加入 `sys.path`，保证 `hdf5_pipeline` 可导入。

## 数据来源

| 链路 | 数据 | 说明 |
|---|---|---|
| HDF5 四步 | 项目内 `test_data` 下的 HDF5 样本（相对路径定位） | 真实样本（30 帧 / 双相机），复制到 `tmp_path`，不污染源数据 |
| spirit 专项 | 环境变量 `SPIRIT_RAW_DIR` 指定的本地实例 | 真实实例，**只读**；未设置时相关测试 skip |


## 覆盖范围

| 文件 | 覆盖 |
|---|---|
| `conftest.py` | 共享 fixtures：`sample_hdf5`、`raw_dir`（嵌套目录乱序）、`spirit_raw` |
| `test_rename.py` | 递归收集 + 自然排序、排除 `rename/` 目录、`episode_NNNNNN` 编号、copy/move、非空输出目录建时间戳子目录 |
| `test_quality.py` | HDF5 异常检测摘要、`export_results` 有/无异常帧时行为、`parse_mask` |
| `test_render.py` | HDF5 → MP4 成功（文件存在 + 帧数 > 0）、输入缺失时安全失败 |
| `test_label.py` | `scan_pairs` 配对（忽略无配对的孤儿文件）、`add_label`、`add_labels` 批量与归档模式、`query_records`/`count_list` |
| `test_spirit.py` | `parse_episodes`、`convert_spirit`（输出 6 列 / 22 维 / meta 四件套 / 视频 symlink 与 copy）、`run_spirit_quality`、`validate_spirit_dataset` |
| `test_pipeline.py` | 集成：rename → quality → render → label 全链路数据贯通 |

## 已验证的真实行为（避免踩坑）

- `add_labels(if_qualify=True)` 只更新 DB 的 `hdf5_path`，**不移动文件**（移动由 UI 层负责）——模块职责单一。
- `export_results` 在异常帧列表为空时不写 CSV/JSON（by-design）。
- spirit `convert` 的 `videos` 数 = `converted × 3`（每有效集 3 相机）。
- `collect_hdf5_files` 显式排除名字含 `rename` / `__pycache__` 的目录。