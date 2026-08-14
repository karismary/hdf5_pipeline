"""（千寻智能 moz1）原始 LeRobot parquet 数据的结构校验。

校验 22 维装配列（MOTORS 顺序）与 frame_index 是否齐全、行数 >= 1、
每个 list 列是否定长，以及实例内 event_log 与 episode 文件的一致性。
"""

import csv
import json
import re
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
        pf = pq.ParquetFile(parquet_path)
        missing = [c for c in ASSEMBLY_COLS + ["frame_index"] if c not in pf.schema_arrow.names]
        if missing:
            ok = False
            errors.append(f"{parquet_path.name}: 缺少必需列：{', '.join(missing)}")
        rows = pf.metadata.num_rows
        if rows < 1:
            ok = False
            errors.append(f"{parquet_path.name}: 数据为空（0 行）")
        return ok, errors
    except Exception as e:
        ok = False
        errors.append(f"{parquet_path.name}: 无法读取 parquet 文件：{e}")
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
        return False, [f"未找到数据：{exc}"]

    for inst in instance_dirs:
        rel_dir = str(inst.relative_to(raw_dir)) if inst != raw_dir else ""
        label = rel_dir or inst.name

        data_chunk = inst / "data" / "chunk-000"
        ep_files = sorted(
            data_chunk.glob("episode_*.parquet"),
            key=episode_number,
        )

        for ep_file in ep_files:
            ok, errs = validate_spirit_file(ep_file)
            if not ok:
                rel_path = str(ep_file.parent.relative_to(inst))
                rel_prefix = f"{rel_dir}/{rel_path}" if rel_dir else rel_path
                errors.extend(f"{rel_prefix}/{e}" for e in errs)

        episodes_meta = parse_episodes(inst / "event_log.jsonl")
        file_numbers = {episode_number(p) for p in ep_files}
        log_numbers = set(episodes_meta)
        if log_numbers - file_numbers:
            errors.append(
                f"{label}: event_log 引用了 data/ 下不存在的集："
                f"{_fmt_eps(log_numbers - file_numbers)}"
            )

        meta_eps = _read_meta_episodes(inst)
        if meta_eps is not None:
            if meta_eps != file_numbers:
                errors.append(
                    f"{label}: meta/episodes.jsonl 与 data/ 不一致："
                    f"仅 meta 有 {_fmt_eps(meta_eps - file_numbers)}，"
                    f"仅 data 有 {_fmt_eps(file_numbers - meta_eps)}"
                )
            if not file_numbers.issubset(log_numbers):
                errors.append(
                    f"{label}: 以下集未出现在 event_log 中："
                    f"{_fmt_eps(file_numbers - log_numbers)}"
                )

    return len(errors) == 0, errors


def _fmt_eps(numbers: set[int]) -> str:
    """编号集合 → 可读且可解析的 episode 列表（episode_000001, episode_000002）。"""
    return ", ".join(f"episode_{n:06d}" for n in sorted(numbers))


_FILE_PATH_RE = re.compile(r"([\w./\\-]+\.parquet)\s*:")


def read_skip_episodes(
    raw_dir: str | Path,
    csv_path: str | Path | None,
    log_path: str | Path | None,
) -> set[Path]:
    """从检测文档提取要跳过的源 parquet 文件（绝对路径）集合。

    episode 编号按实例内编号，跨实例会重复（实例 A、B 都有 episode_000003），
    因此按**具体文件**定位而不是裸编号，避免把一个实例的问题集套到所有实例：

    - 来源一：质检导出的 CSV（``outlier_frames.csv``），取 ``path`` 列（异常帧
      所在的完整源文件路径）。
    - 来源二：校验报告（.txt/.log），抓取 ``instanceX/data/chunk-000/episode_0000xx.parquet``
      形式的实例相对路径，再基于 ``raw_dir`` 解析为绝对路径。

    文件不存在或路径字段缺失时静默跳过该来源。

    Args:
        raw_dir: 原始数据根目录（与生成检测文档时一致，用于解析报告里的相对路径）。
        csv_path: 质检异常帧 CSV 路径；可为空/不存在。
        log_path: 校验报告文本路径；可为空/不存在。

    Returns:
        需要跳过的源 parquet 文件绝对路径集合。
    """
    raw_dir = Path(raw_dir)
    skip: set[Path] = set()

    if csv_path and Path(csv_path).is_file():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = str(row.get("path") or "").strip()
                if p and Path(p).suffix == ".parquet":
                    skip.add(Path(p).resolve())

    if log_path and Path(log_path).is_file():
        with open(log_path, encoding="utf-8") as f:
            for m in _FILE_PATH_RE.finditer(f.read()):
                skip.add((raw_dir / m.group(1)).resolve())

    return skip


__all__ = ["validate_spirit_file", "validate_spirit_dataset", "read_skip_episodes"]