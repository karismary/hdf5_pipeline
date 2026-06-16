"""HDF5 原始文件异常帧检测。

读取原始 30 维 HDF5 → 投影到 16 维训练空间 → 调 detector 评分 → 导出。
"""

from pathlib import Path

import numpy as np

from hdf5_pipeline.quality.detector import (
    parse_mask, compute_outliers, export_results,
)
from hdf5_pipeline.core.constants import DEFAULT_DELTA_MASK_16
from hdf5_pipeline.core.hdf5_utils import load_raw_30dim, project_30_to_16, natural_sort_key


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_hdf5_episodes(data_glob: str) -> list:
    """批量读取原始 30 维 HDF5，投影到 16 维训练空间。

    Args:
        data_glob: 通配符路径，如 "./data/*.hdf5"

    Returns:
        [(path, action_16, state_16, frame_index), ...]
    """
    p_glob = Path(data_glob)

    hdf5_files = sorted(
        p_glob.parent.glob(p_glob.name), 
        key=lambda f: natural_sort_key(f.name)
    )
    
    if not hdf5_files:
        raise FileNotFoundError(f"No files matched: {data_glob}")

    episodes = []
    for path_obj in hdf5_files:
        path_str = str(path_obj)
        action_30, state_30 = load_raw_30dim(path_str)
        action_16 = project_30_to_16(action_30)
        state_16 = project_30_to_16(state_30)
        frame_index = np.arange(action_16.shape[0], dtype=np.int64)
        
        episodes.append((path_str, action_16, state_16, frame_index))

    return episodes


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_hdf5_check(
    data_glob: str,
    out_csv: str = "hdf5_outlier_frames.csv",
    out_json: str = "hdf5_outlier_summary.json",
    mask: str = None,
    strictness: str = "strict",
    min_score: float = None,
    top_k_per_episode: int = None,
    top_k_global: int = None,
    min_denom: float = None,
) -> dict:
    """扫描原始 HDF5 文件，导出异常帧列表。

    Args:
        data_glob: 通配符路径
        out_csv / out_json: 输出路径
        mask: 逗号分隔掩码，None 用默认
        strictness: "loose" / "medium" / "strict"
        其余: 手动覆盖预设

    Returns:
        统计摘要 dict
    """
    if mask is None:
        mask = ",".join("1" if v else "0" for v in DEFAULT_DELTA_MASK_16)

    mask_arr = parse_mask(mask)
    episodes = load_hdf5_episodes(data_glob)
    rows, summary = compute_outliers(
        episodes, mask_arr, strictness,
        min_score, top_k_per_episode, top_k_global, min_denom,
    )
    export_results(rows, summary, out_csv, out_json)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote JSON: {out_json}")
    print(f"Outliers kept: {len(rows)}")

    return summary