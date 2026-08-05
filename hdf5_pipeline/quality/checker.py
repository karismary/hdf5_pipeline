"""通用异常帧检测 —— 兼容 HDF5 与 LeRobot Parquet 两种格式。

用法示例::

    # HDF5 格式
    run_quality_check("./data", format="hdf5")

    # LeRobot Parquet 格式
    run_quality_check("./data/chunk-000", format="lerobot",
                      action_key="action", state_key="observation.state")
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pyarrow.parquet as pq

from hdf5_pipeline.quality.detector import (
    parse_mask, compute_outliers, export_results,
)
from hdf5_pipeline.core.constants import DEFAULT_DELTA_MASK_16
from hdf5_pipeline.core.hdf5_utils import load_raw_30dim, project_30_to_16, natural_sort_key


# 格式 -> (扩展名通配)
_FORMAT_GLOB = {
    "hdf5": "*.hdf5",
    "lerobot": "*.parquet",
}
_FORMAT_LABELS: dict[str, str] = {
    "hdf5": "HDF5",
    "lerobot": "LeRobot Parquet",
}

# ---------------------------------------------------------------------------
# Loader —— 按格式分发
# ---------------------------------------------------------------------------

def load_episodes(
    data_dir: str,
    format: Literal["hdf5", "lerobot"] = "hdf5",
    *,
    action_key: str = "action",
    state_key: str = "observation.state",
    frame_key: str = "frame_index",
) -> list:
    """批量读取目录下的数据文件，返回统一的 episodes 列表。

    Args:
        data_dir: 存放数据文件的目录路径。
        format: ``"hdf5"`` — 读取 ``*.hdf5``，自动从 30 维投影到 16 维；
                ``"lerobot"`` — 读取 ``*.parquet``。
        action_key / state_key / frame_key: 仅 ``format="lerobot"`` 时生效。

    Returns:
        [(path, action_array, state_array, frame_index_array), ...]
    """
    glob_expr = _FORMAT_GLOB[format]
    data_path = Path(data_dir)
    files = sorted(
        data_path.glob(glob_expr),
        key=lambda f: natural_sort_key(f.name),
    )
    if not files:
        raise FileNotFoundError(
            f"No {_FORMAT_LABELS.get(format, format)} files found in {data_dir!r} "
            f"(expected {glob_expr})"
        )

    if format == "hdf5":
        return _load_hdf5_episodes(files)
    else:
        return _load_parquet_episodes(files, action_key, state_key, frame_key)


def _load_hdf5_episodes(files: list[Path]) -> list:
    episodes = []
    for path_obj in files:
        path_str = str(path_obj)
        action_30, state_30 = load_raw_30dim(path_str)
        action_16 = project_30_to_16(action_30)
        state_16 = project_30_to_16(state_30)
        frame_index = np.arange(action_16.shape[0], dtype=np.int64)
        episodes.append((path_str, action_16, state_16, frame_index))
    return episodes


def _load_parquet_episodes(
    files: list[Path],
    action_key: str,
    state_key: str,
    frame_key: str,
) -> list:
    episodes = []
    for p in files:
        t = pq.read_table(p, columns=[action_key, state_key, frame_key])
        a = np.stack(t[action_key].to_numpy())
        s = np.stack(t[state_key].to_numpy())
        fidx = np.asarray(t[frame_key])
        episodes.append((str(p), a, s, fidx))
    return episodes


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_quality_check(
    data_dir: str,
    format: Literal["hdf5", "lerobot"],
    out_csv: str = "outlier_frames.csv",
    out_json: str = "outlier_summary.json",
    *,
    mask: str | None = None,
    strictness: str = "strict",
    min_score: float | None = None,
    top_k_per_episode: int | None = None,
    top_k_global: int | None = None,
    min_denom: float | None = None,
    # --- LeRobot 专用（传递给 load_episodes） ---
    action_key: str = "action",
    state_key: str = "observation.state",
    frame_key: str = "frame_index",
    # --- 自定义 loader（优先级最高） ---
    loader: Callable[[str], list] | None = None,
) -> dict:
    """通用异常帧检测入口。

    Args:
        data_dir: 数据文件所在目录。
        format: ``"hdf5"`` 或 ``"lerobot"``，决定扫描何种扩展名的文件。
        out_csv / out_json: 输出路径。
        mask: 逗号分隔掩码字符串，None 用默认 16 维掩码。
        strictness: ``"loose"`` / ``"medium"`` / ``"strict"``。
        其余: 手动覆盖预设。
        action_key / state_key / frame_key: 仅 ``format="lerobot"`` 有效。
        loader: 自定义加载函数 ``(data_dir: str) -> list``。
                传入后 ``format`` 仅用于日志，不再决定加载逻辑。

    Returns:
        统计摘要 dict。
    """
    # --- 默认掩码 ---
    if mask is None:
        mask = ",".join("1" if v else "0" for v in DEFAULT_DELTA_MASK_16)
    mask_arr = parse_mask(mask)

    # --- 加载 ---
    if loader is not None:
        episodes = loader(data_dir)
    else:
        episodes = load_episodes(
            data_dir, format,
            action_key=action_key,
            state_key=state_key,
            frame_key=frame_key,
        )

    # --- 检测 & 输出 ---
    rows, summary = compute_outliers(
        episodes, mask_arr, strictness,
        min_score, top_k_per_episode, top_k_global, min_denom,
    )
    export_results(rows, summary, out_csv, out_json)

    return summary