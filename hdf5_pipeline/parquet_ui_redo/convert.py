"""Spirit（千寻智能 moz1）原始 LeRobot parquet → 标准 LeRobot v2.1 格式转换。

只依赖标准库 + numpy + pyarrow，不引入 torch / lerobot。
列名与 22 维布局一律以 ``parquet_ui_redo.constants`` 为唯一依据，禁止硬编码数字。
"""

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hdf5_pipeline.parquet_ui_redo.constants import (
    CAMERAS,
    MOTORS,
    SPIRIT_ACTION_COLS,
    SPIRIT_STATE_COLS,
)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
FPS = 30
DEFAULT_TASK = "perform the task"

_EPISODE_RE = re.compile(r"episode_(\d+)")


def episode_number(path: str | Path) -> int:
    """从文件名提取 episode 编号（episode_000003.parquet -> 3），失败返回 -1。"""
    m = _EPISODE_RE.search(Path(path).stem)
    return int(m.group(1)) if m else -1


def _camera_shape(instance_dir: Path) -> tuple[int, int]:
    """从源 meta/info.json 读相机 (height, width)，读不到用默认值。"""
    info_path = instance_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        for cam in CAMERAS:
            feat = info.get("features", {}).get(cam)
            shape = feat.get("shape") if feat else None
            if shape:
                return int(shape[0]), int(shape[1])
    return IMAGE_HEIGHT, IMAGE_WIDTH


def parse_episodes(event_log_path: str | Path) -> dict[int, dict]:
    """解析 event_log.jsonl，返回 {episode_idx: {is_mistake, tasks, stats}}。

    只记录 payload 带 ``episode_idx`` 的行；缺省 is_mistake 视为 False。
    tasks 取 episode_info.tasks（每集任务列表），stats 取 episode_stats.stats。
    """
    episodes = {}
    with open(event_log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line).get("payload", {})
            if "episode_idx" not in payload:
                continue
            idx = payload["episode_idx"]
            info = payload.get("episode_info", {})
            tasks = [str(t).strip() for t in info.get("tasks", []) if str(t).strip()]
            episodes[idx] = {
                "is_mistake": bool(payload.get("is_mistake", False)),
                "tasks": tasks,
                "stats": payload.get("episode_stats", {}).get("stats", {}) or {},
            }
    return episodes


def _load_instructions(instance_dir: Path) -> list[str]:
    """全局指令回退：读 meta/tasks.jsonl 的所有 task，无则用默认单条。"""
    tasks_path = instance_dir / "meta" / "tasks.jsonl"
    instructions = []
    if tasks_path.exists():
        with open(tasks_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                task = str(json.loads(line).get("task", "")).strip()
                if task:
                    instructions.append(task)
    return instructions or [DEFAULT_TASK]


def _choose_instruction(ep_tasks: list[str], fallback: list[str]) -> str:
    """从该集任务里随机选一条；该集无任务时从全局指令里随机选。"""
    choices = [t for t in ep_tasks if t] or fallback
    return str(np.random.choice(choices))


def assemble_episode(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取单集源 parquet，组装成标准 22 维 action / state。

    Args:
        path: 源 ``episode_*.parquet`` 路径。

    Returns:
        (action, state, frame_index, timestamp)
        action / state: float32, shape (T, 22)，按 MOTORS 顺序
        frame_index: int64, shape (T,)
        timestamp: float32, shape (T,)
    """
    t = pq.read_table(
        path,
        columns=SPIRIT_ACTION_COLS + SPIRIT_STATE_COLS + ["frame_index", "timestamp"],
    )
    action = np.concatenate(
        [np.stack(t[c].to_numpy()) for c in SPIRIT_ACTION_COLS], axis=1
    )
    state = np.concatenate(
        [np.stack(t[c].to_numpy()) for c in SPIRIT_STATE_COLS], axis=1
    )
    frame_index = np.asarray(t["frame_index"], dtype=np.int64)
    timestamp = np.asarray(t["timestamp"], dtype=np.float32)
    return action.astype(np.float32), state.astype(np.float32), frame_index, timestamp

def _link_episode_videos(
    instance_dir: Path, src_idx: int, out_dir: Path, out_idx: int, link_videos: bool
) -> int:
    """复制/链接单集三个相机的视频，返回成功数。"""
    count = 0
    for cam in CAMERAS:
        src = instance_dir / "videos" / "chunk-000" / cam / f"episode_{src_idx:06d}.mp4"
        if not src.exists():
            continue
        dst = out_dir / "videos" / "chunk-000" / cam / f"episode_{out_idx:06d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if link_videos:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        count += 1
    return count

def convert_one_instance(
    instance_dir: str | Path,
    out_dir: str | Path,
    out_offset: int = 0,
    link_videos: bool = True,
    task_index_map: dict | None = None,
    out_records: list | None = None,
) -> tuple[int, int, int]:
    """转换单个实例目录（含 event_log.jsonl 的数据目录）。

    Args:
        instance_dir: 实例目录。
        out_dir: 输出数据集根目录。
        out_offset: 本实例首个 episode 的输出编号（跨实例续接）。
        link_videos: True 用 Path.symlink_to，False 用 shutil.copy2。
        task_index_map: {task: task_index} 共享注册表（跨实例累积）。
        out_records: 接收每集 {"episode_index", "length", "task", "stats"} 记录的列表。

    Returns:
        (converted, skipped, video_count)
    """
    instance_dir = Path(instance_dir)
    out_dir = Path(out_dir)
    if task_index_map is None:
        task_index_map = {}
    if out_records is None:
        out_records = []

    episodes_meta = parse_episodes(instance_dir / "event_log.jsonl")
    fallback = _load_instructions(instance_dir)

    data_chunk = instance_dir / "data" / "chunk-000"
    episode_files = sorted(
        data_chunk.glob("episode_*.parquet"),
        key=lambda p: episode_number(p),
    )

    out_parquet_dir = out_dir / "data" / "chunk-000"
    out_parquet_dir.mkdir(parents=True, exist_ok=True)

    converted = skipped = video_count = 0
    out_idx = out_offset
    for ep_file in episode_files:
        src_idx = episode_number(ep_file)
        meta = episodes_meta.get(src_idx)
        if meta and meta["is_mistake"]:
            skipped += 1
            continue

        try:
            action, state, frame_index, timestamp = assemble_episode(ep_file)
        except Exception as exc:
            print(
                f"[parquet_ui] skip unreadable episode {ep_file.name}: {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        instruction = _choose_instruction(meta["tasks"] if meta else [], fallback)
        if instruction not in task_index_map:
            task_index_map[instruction] = len(task_index_map)
        task_index = task_index_map[instruction]

        n = len(frame_index)

        table = pa.table(
            {
                "observation.state" : pa.array(state.tolist(), type = pa.list_(pa.float32())),
                "action" : pa.array(action.tolist(), type = pa.list_(pa.float32())),
                "frame_index" : pa.array(frame_index, type = pa.int64()),
                "timestamp" : pa.array(timestamp, type = pa.float32()),
                "episode_index" : pa.array(np.full(n, out_idx, dtype = np.int64), type = pa.int64()),
                "task_index" : pa.array(np.full(n, task_index, dtype = np.int64), type = pa.int64())
            }
        )
        pq.write_table(table, out_parquet_dir / f"episode_{out_idx:06d}.parquet")

        video_count += _link_episode_videos(
            instance_dir, src_idx, out_dir, out_idx, link_videos
        )

        out_records.append(
            {
                "episode_index": out_idx,
                "length": int(n),
                "task": instruction,
                "stats": meta["stats"] if meta else {},
            }
        )
        converted += 1
        out_idx += 1

    return converted, skipped, video_count


def _write_meta(
    out_dir: Path,
    task_index_map: dict[str, int],
    out_records: list[dict],
    video_count: int,
    image_height: int,
    image_width: int,
) -> None:
    """写 meta 四件套：info.json / episodes.jsonl / tasks.jsonl / episodes_stats.jsonl。

    Args:
        out_dir: 输出数据集根目录。
        task_index_map: {task: task_index} 全局注册表。
        out_records: 每集 {"episode_index", "length", "task", "stats"} 记录。
        video_count: 视频总数（写入 info.json）。
        image_height / image_width: 相机分辨率（写入视频 feature）。
    """
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(task_index_map.items(), key=lambda kv: kv[1])
    total_episodes = len(out_records)
    total_frames = int(sum(r["length"] for r in out_records))

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(MOTORS)],
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": [len(MOTORS)],
            "names": [MOTORS],
        },
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(FPS),
                "video.height": image_height,
                "video.width": image_width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "moz1",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_index_map),
        "total_videos": video_count,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(
                json.dumps(
                    {
                        "episode_index": r["episode_index"],
                        "length": r["length"],
                        "tasks": [r["task"]],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task, idx in tasks:
            f.write(
                json.dumps({"task_index": idx, "task": task}, ensure_ascii=False)
                + "\n"
            )

    with open(meta_dir / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(
                json.dumps(
                    {"episode_index": r["episode_index"], "stats": r["stats"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def convert_spirit(raw_dir: str | Path, out_dir: str | Path, link_videos: bool = True) -> dict:
    """批量转换 raw_dir 下所有 spirit 实例，输出标准 LeRobot v2.1 数据集。

    raw_dir 下所有含 event_log.jsonl 的目录视为实例（rglob 递归发现）；
    若 raw_dir 本身就是实例目录也兼容。episode 编号跨实例顺序续接。

    Args:
        raw_dir: 原始数据根目录。
        out_dir: 输出数据集根目录（存在则先 rmtree 清理，不静默覆盖）。
        link_videos: True 用 Path.symlink_to，False 用 shutil.copy2。

    Returns:
        {"converted": n, "skipped": m, "videos": k}
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir not found or not a directory: {raw_dir}")

    if (raw_dir / "event_log.jsonl").exists():
        instance_dirs = [raw_dir]
    else:
        instance_dirs = sorted({p.parent for p in raw_dir.rglob("event_log.jsonl")})
    if not instance_dirs:
        raise FileNotFoundError(f"No event_log.jsonl found under {raw_dir}")

    image_height, image_width = _camera_shape(instance_dirs[0])

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_index_map = {}
    out_records = []
    converted = skipped = video_count = 0
    out_offset = 0
    for inst in instance_dirs:
        c, s, v = convert_one_instance(
            inst,
            out_dir,
            out_offset,
            link_videos,
            task_index_map=task_index_map,
            out_records=out_records,
        )
        converted += c
        skipped += s
        video_count += v
        out_offset += c

    if not out_records:
        raise RuntimeError(
            "No episode converted: all episodes are mistakes or no data found."
        )

    _write_meta(out_dir, task_index_map, out_records, video_count, image_height, image_width)

    return {"converted": converted, "skipped": skipped, "videos": video_count}