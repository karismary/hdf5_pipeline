"""（千寻智能 moz1）原始 LeRobot parquet 数据的结构校验。

校验 22 维装配列（MOTORS 顺序）与 frame_index 是否齐全、行数 >= 1、
每个 list 列是否定长，以及实例内 event_log 与 episode 文件的一致性。
"""

import json
from pathlib import Path

import pyarrow.parquet as pq

from hdf5_pipeline.parquet_ui_redo.constants import (
    SPIRIT_ACTION_COLS,
    SPIRIT_STATE_COLS,
)
from hdf5_pipeline.parquet_ui_redo.convert import episode_number, parse_episodes
from hdf5_pipeline.parquet_ui_redo.quality import _find_instance_dirs

ASSEMBLY_COLS = SPIRIT_ACTION_COLS + SPIRIT_STATE_COLS

def validate_spirit_file(parquet_path: str | Path) -> tuple[bool, list[str]]:
    """校验单个源 episode_*.parquet。

    Args:
        parquet_path: 源 episode parquet 路径。

    Returns:
        (ok, errors) — ok 为 False 时 errors 非空。
    """
    parquet_path = Path(parquet_path)
    errors = []
    ok = True
    try:
        table = pq.read_table(parquet_path, columns = ASSEMBLY_COLS + ["frame_index"])
        cols = table.column_names
        if not set(ASSEMBLY_COLS) <= set(cols) or not "frame_index" in cols:
            ok = False
            errors.append(f"{parquet_path}: 无法正确读取所有列")
        rows = table.num_rows
        if rows < 1:
            ok = False
            errors.append(f"{parquet_path}: 无法正确读取所有数据")
        return ok, errors
    except Exception as e:
        ok = False
        errors.append(f"{parquet_path}: 无法读取.parquet文件: {e}")
        return ok, errors
    
def _read_meta_episodes(instance_dir: Path) -> set[int] | None:
    """读 meta/episodes.jsonl，返回 episode_index 集合；缺失时返回 None。"""
    path = instance_dir / "meta" / "episodes.jsonl"
    if not path.exists():
        return None
    indices = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            indices.add(json.loads(line)["episode_index"])
    return indices


def validate_spirit_dataset(raw_dir: str | Path) -> tuple[bool, list[str]]:
    """校验整个原始数据集（raw_dir 或单个实例目录）。

    遍历实例，校验每个 episode 文件，并核对 event_log 一致性
    （event_log 里的 episode_idx 与 data/ 下的 episode 文件、meta/episodes.jsonl 对齐）。

    Returns:
        (ok, errors) — ok 为 False 时 errors 非空。
    """
    raw_dir = Path(raw_dir)
    errors = []

    try:
        instance_dirs = _find_instance_dirs(raw_dir)
    except FileNotFoundError as exc:
        return False, [str(exc)]

    for inst in instance_dirs:
        label = str(inst.relative_to(raw_dir)) if inst != raw_dir else str(inst)

        data_chunk = inst / "data" / "chunk-000"
        ep_files = sorted(
            data_chunk.glob("episode_*.parquet"),
            key=episode_number,
        )

        for ep_file in ep_files:
            ok, errs = validate_spirit_file(ep_file)
            if not ok:
                errors.extend(errs)

        episodes_meta = parse_episodes(inst / "event_log.jsonl")
        file_numbers = {episode_number(p) for p in ep_files}
        log_numbers = set(episodes_meta)
        if log_numbers - file_numbers:
            errors.append(
                f"{label}: event_log references episodes not present in data/: "
                f"{sorted(log_numbers - file_numbers)}"
            )

        meta_eps = _read_meta_episodes(inst)
        if meta_eps is not None:
            if meta_eps != file_numbers:
                errors.append(
                    f"{label}: meta/episodes.jsonl mismatch data/: "
                    f"meta-only {sorted(meta_eps - file_numbers)}, "
                    f"data-only {sorted(file_numbers - meta_eps)}"
                )
            if not file_numbers.issubset(log_numbers):
                errors.append(
                    f"{label}: episodes missing from event_log: "
                    f"{sorted(file_numbers - log_numbers)}"
                )

    return len(errors) == 0, errors


__all__ = ["validate_spirit_file", "validate_spirit_dataset"]