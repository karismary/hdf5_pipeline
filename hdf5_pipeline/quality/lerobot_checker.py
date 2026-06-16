"""LeRobot Parquet 格式异常帧检测。

读取 Parquet → 调 detector 评分 → 导出 CSV/JSON。
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hdf5_pipeline.quality.detector import (
    parse_mask, compute_outliers, export_results,
)
from hdf5_pipeline.core.constants import DEFAULT_DELTA_MASK_16


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_parquet_episodes(
    data_glob: str,
    action_key: str = "action",
    state_key: str = "observation.state",
    frame_key: str = "frame_index",
) -> list:
    """批量读取 LeRobot parquet 文件。

    Args:
        data_glob: 通配符路径，如 "./data/chunk-000/*.parquet"
        action_key: parquet 中 action 列名
        state_key: parquet 中 state 列名
        frame_key: parquet 中帧序号列名

    Returns:
        [(path, action, state, frame_index), ...]
    """

    p_data = Path(data_glob)
    parquet_files = sorted(p_data.parent.glob(data_glob.name))
    if not parquet_files:
        raise FileNotFoundError(f"No files matched: {data_glob}")

    arrays = []
    for p in parquet_files:
        t = pq.read_table(p, columns=[action_key, state_key, frame_key])
        a = np.stack(t[action_key].to_numpy())
        s = np.stack(t[state_key].to_numpy())
        fidx = np.asarray(t[frame_key])
        arrays.append((p, a, s, fidx))

    return arrays


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_lerobot_check(
    data_glob: str,
    out_csv: str = "outlier_frames.csv",
    out_json: str = "outlier_summary.json",
    mask: str = None,
    strictness: str = "strict",
    min_score: float = None,
    top_k_per_episode: int = None,
    top_k_global: int = None,
    min_denom: float = None,
) -> dict:
    """扫描 LeRobot parquet 文件，导出异常帧列表。

    Args:
        data_glob: 通配符路径
        out_csv / out_json: 输出路径
        mask: 逗号分隔掩码字符串，None 用默认
        strictness: "loose" / "medium" / "strict"
        其余: 手动覆盖预设

    Returns:
        统计摘要 dict
    """
    if mask is None:
        mask = ",".join("1" if v else "0" for v in DEFAULT_DELTA_MASK_16)

    mask_arr = parse_mask(mask)
    episodes = load_parquet_episodes(data_glob)
    rows, summary = compute_outliers(
        episodes, mask_arr, strictness,
        min_score, top_k_per_episode, top_k_global, min_denom,
    )
    export_results(rows, summary, out_csv, out_json)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote JSON: {out_json}")
    print(f"Outliers kept: {len(rows)}")

    return summary