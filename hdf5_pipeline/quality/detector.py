"""异常检测核心算法：掩码解析、DeltaActions、分位数拟合、评分、导出。"""

import csv
import json
import re
from pathlib import Path

import numpy as np

from hdf5_pipeline.core.constants import STRICTNESS_PRESETS


# ==================== 基础工具 ====================

def parse_mask(mask_str: str) -> np.ndarray:
    """将逗号分隔的 1/0 字符串转为布尔数组。

    Args:
        mask_str: 如 "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0"

    Returns:
        布尔数组，True 表示该维度参与 action-state 差分。
    """
    vals = [x.strip() for x in mask_str.split(',') if x.strip()]
    out = []
    for v in vals:
        if v in {'1', 'true', 'True', 'T', 't'}:
            out.append(True)
        elif v in {'0', 'false', 'False', 'F', 'f'}:
            out.append(False)
        else:
            raise ValueError(f'Invalid mask value: {v}')
    return np.array(out, dtype=bool)


def apply_delta(action: np.ndarray, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """计算 DeltaActions。

    mask=1 的维度: delta = action - state
    mask=0 的维度: delta = action（保留原值）

    Args:
        action: 动作数据, shape (T, D)
        state:  状态数据, shape (T, D)
        mask:   布尔掩码, shape (D,)

    Returns:
        delta 数据, shape (T, D)
    """
    out = action.copy()
    dims = min(action.shape[1], state.shape[1], len(mask))
    out[:, :dims] = out[:, :dims] - np.where(mask[:dims], state[:, :dims], 0.0)
    return out


def fit_quantiles(all_data: np.ndarray):
    """在全量 delta 数据上按维度拟合 1% 和 99% 分位数。

    Args:
        all_data: shape (N, D)，所有帧的 delta 拼在一起。

    Returns:
        (q01, q99) — 各维度的 1% 和 99% 分位数, shape 均为 (D,)
    """
    q01 = np.percentile(all_data, 1, axis=0)
    q99 = np.percentile(all_data, 99, axis=0)
    return q01, q99


def episode_id_from_path(path: str) -> int:
    """从文件名提取 episode 编号。

    支持 'episode_000123.parquet' / '.hdf5' / '.h5'

    Args:
        path: 文件路径字符串

    Returns:
        episode 编号，匹配失败返回 -1
    """
    m = re.search(r"episode_(\d+)\.(parquet|hdf5|h5)$", path)
    return int(m.group(1)) if m else -1


# ==================== 核心评分 ====================

def compute_outliers(
    episodes: list,
    mask: np.ndarray,
    strictness: str = "strict",
    min_score: float = None,
    top_k_per_episode: int = None,
    top_k_global: int = None,
    min_denom: float = None,
) -> tuple:
    """对所有 episode 计算异常帧。

    算法链路: 全量 DeltaActions → 分位数拟合 → 逐帧打分 → 排序截断

    Args:
        episodes: [(path, action, state, frame_index), ...]
        mask: 布尔掩码, shape (D,)
        strictness: "loose" / "medium" / "strict"
        min_score / top_k_per_episode / top_k_global / min_denom: 手动覆盖预设

    Returns:
        (outlier_rows, summary) — 异常帧列表和统计摘要
    """
    # 1. 合并预设
    p = STRICTNESS_PRESETS[strictness]
    if min_score is None:
        min_score = p["min_score"]
    if top_k_per_episode is None:
        top_k_per_episode = p["top_k_per_episode"]
    if top_k_global is None:
        top_k_global = p["top_k_global"]
    if min_denom is None:
        min_denom = p["min_denom"]

    # 2. 全量 DeltaActions → 分位数
    all_delta = np.concatenate(
        [apply_delta(a, s, mask) for _, a, s, _ in episodes], axis=0
    )
    q01, q99 = fit_quantiles(all_delta)
    denom = q99 - q01

    # 3. 逐 episode 扫描
    outlier_rows = []
    for path, action, state, frame_idx in episodes:
        ep_id = episode_id_from_path(path)
        delta = apply_delta(action, state, mask)
        norm_abs = np.abs((delta - q01) / (denom + 1e-6) * 2.0 - 1.0)
        frame_max = norm_abs.max(axis=1)

        candidate_idx = np.where(frame_max >= min_score)[0]
        if candidate_idx.size == 0:
            continue

        if candidate_idx.size > top_k_per_episode:
            pick = np.argpartition(frame_max[candidate_idx], -top_k_per_episode)[-top_k_per_episode:]
            candidate_idx = candidate_idx[pick]

        for i in candidate_idx:
            d = int(np.argmax(norm_abs[i]))
            if min_denom is not None and not (denom[d] <= min_denom):
                continue

            outlier_rows.append({
                "episode": int(ep_id),
                "frame": int(frame_idx[i]),
                "dim": d,
                "score": float(norm_abs[i, d]),
                "frame_max": float(frame_max[i]),
                "delta_value": float(delta[i, d]),
                "q01": float(q01[d]),
                "q99": float(q99[d]),
                "denom": float(denom[d]),
                "file": str(Path(path).name),
            })

    # 4. 全局截断
    outlier_rows.sort(key=lambda x: x["score"], reverse=True)
    outlier_rows = outlier_rows[:top_k_global]

    # 5. 摘要
    by_dim, by_episode = {}, {}
    for r in outlier_rows:
        by_dim[r["dim"]] = by_dim.get(r["dim"], 0) + 1
        by_episode[r["episode"]] = by_episode.get(r["episode"], 0) + 1

    summary = {
        "num_files": len(episodes),
        "num_frames": int(sum(len(r[1]) for r in episodes)),
        "num_outliers": len(outlier_rows),
        "strictness": strictness,
        "min_score": min_score,
        "top_k_per_episode": top_k_per_episode,
        "top_k_global": top_k_global,
        "min_denom": min_denom,
        "smallest_denoms": [
            {"dim": int(i), "denom": float(denom[i]), "q01": float(q01[i]), "q99": float(q99[i])}
            for i in np.argsort(denom)[:8]
        ],
        "top_dims": sorted(
            [{"dim": int(k), "count": int(v)} for k, v in by_dim.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "top_episodes": sorted(
            [{"episode": int(k), "count": int(v)} for k, v in by_episode.items()],
            key=lambda x: x["count"], reverse=True,
        )[:20],
    }

    return outlier_rows, summary


# ==================== 导出 ====================

def export_results(outlier_rows: list, summary: dict, out_csv: str, out_json: str) -> None:
    """将异常帧列表写入 CSV，统计摘要写入 JSON。

    Args:
        outlier_rows: compute_outliers 返回的第一项
        summary: compute_outliers 返回的第二项
        out_csv: CSV 输出路径
        out_json: JSON 输出路径
    """
    if outlier_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(outlier_rows[0].keys()))
            writer.writeheader()
            writer.writerows(outlier_rows)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)