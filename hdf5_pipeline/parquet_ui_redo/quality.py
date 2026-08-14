"""（千寻智能 moz1）原始 LeRobot parquet 数据的独立质检链路。

复用 ``hdf5_pipeline.quality.detector`` 的 compute_outliers / export_results，

22 维掩码用 ``constants.DEFAULT_DELTA_MASK_22``：mask=True 的维度 delta = cmd - state，
mask=False（夹爪）保留 cmd 原值。
"""

from pathlib import Path
from typing import Callable, Optional, cast

import numpy as np

from hdf5_pipeline.quality.detector import (
    STRICTNESS_PRESETS,
    compute_outliers,
    export_results,
)
from hdf5_pipeline.parquet_ui_redo.constants import DEFAULT_DELTA_MASK_22
from hdf5_pipeline.parquet_ui_redo.convert import assemble_episode, episode_number, parse_episodes

def _find_instance_dirs(raw_dir: str | Path) -> list[Path]:
    """raw_dir 下所有含 event_log.jsonl 的目录；raw_dir 本身是实例也兼容。"""
    raw_dir = Path(raw_dir)
    if (raw_dir / "event_log.jsonl").exists():
        return [raw_dir]
    instance_dirs = sorted({p.parent for p in raw_dir.rglob("event_log.jsonl")})
    if not instance_dirs:
        raise FileNotFoundError(f"{raw_dir} 下未找到 event_log.jsonl（无有效实例目录）")
    return instance_dirs


def _load_spirit_parquet_episodes(raw_dir: str | Path) -> list:
    """批量读取原始 parquet，返回 compute_outliers 的输入 episodes 列表。

    Args:
        raw_dir: 原始数据根目录（或单个实例目录）。

    Returns:
        [(path, action, state, frame_index), ...]
        每项 path 为源 episode_*.parquet 绝对路径；
        action / state 为 (T, 22) float32，frame_index 为 (T,) int64。
        标记 is_mistake=True（或不可读）的集被跳过。

    Raises:
        FileNotFoundError: 没有可读的 episode。
    """
    raw_dir_list = _find_instance_dirs(raw_dir)
    if not raw_dir_list:
        raise FileNotFoundError("目录下没有可读的 episode。")
    else:
        episode_list = []
        for raw in raw_dir_list:
            episodes_info = parse_episodes(Path(raw) / "event_log.jsonl")
            parquet_list = (Path(raw) / "data" / "chunk-000").glob("episode_*.parquet")
            for episode in parquet_list:
                episode_path = str(episode.resolve())
                src_id = episode_number(episode_path)
                episode_info = episodes_info.get(src_id, {})
                if episode_info.get("is_mistake", True):
                    continue
                else:
                    try:
                        action, state, frame_index, _ = assemble_episode(episode_path)
                        episode_list.append((episode_path, action, state, frame_index))
                    except Exception:
                        continue
    return episode_list

def run_spirit_quality(
    raw_dir: str,
    out_csv: str,
    out_json: str,
    mask=None,
    strictness: str = "strict",
    min_score: Optional[float] = None,
    top_k_per_episode: Optional[int] = None,
    top_k_global: Optional[int] = None,
    min_denom: Optional[float] = None,
    loader: Optional[Callable[[str], list]] = None,
) -> dict:
    """原始数据异常帧检测入口。

    Args:
        raw_dir: 原始数据根目录（或单个实例目录）。
        out_csv / out_json: 输出路径（目录或文件）。
        mask: 布尔掩码数组，默认 DEFAULT_DELTA_MASK_22。
        strictness: "loose" / "medium" / "strict"。
        min_score / top_k_per_episode / top_k_global / min_denom: 手动覆盖预设；
            None 时取 STRICTNESS_PRESETS[strictness] 的默认值。
        loader: 自定义加载函数 ``(raw_dir) -> [(path, action, state, frame_index)]``。

    Returns:
        统计摘要 dict（含 num_outliers 等）。
    """
    if mask is None:
        mask = np.asarray(DEFAULT_DELTA_MASK_22, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    if loader is not None:
        episodes = loader(raw_dir)
    else:
        episodes = _load_spirit_parquet_episodes(raw_dir)

    preset = STRICTNESS_PRESETS[strictness]
    rows, summary = compute_outliers(
        episodes,
        mask,
        strictness,
        cast(float, preset["min_score"] if min_score is None else min_score),
        cast(int, preset["top_k_per_episode"] if top_k_per_episode is None else top_k_per_episode),
        cast(int, preset["top_k_global"] if top_k_global is None else top_k_global),
        cast(float, preset["min_denom"] if min_denom is None else min_denom),
    )
    export_results(rows, summary, out_csv, out_json)
    return summary


__all__ = [
    "run_spirit_quality",
    "_load_spirit_parquet_episodes",
]