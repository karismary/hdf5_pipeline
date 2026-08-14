"""spirit 专项：原始 LeRobot parquet → 标准 v2.1 转换 + 质检 + 校验。

数据：本地 spirit 真实实例（环境变量 SPIRIT_RAW_DIR 指定，只读）。
所有输出写入 tmp_path，源数据只读。
"""

import csv
from pathlib import Path

import pyarrow.parquet as pq

from hdf5_pipeline.parquet_ui_redo.convert import convert_spirit, parse_episodes
from hdf5_pipeline.parquet_ui_redo.quality import run_spirit_quality
from hdf5_pipeline.parquet_ui_redo.validator import read_skip_episodes, validate_spirit_dataset


def test_parse_episodes(spirit_raw):
    eps = parse_episodes(spirit_raw / "event_log.jsonl")

    assert len(eps) >= 9
    mistakes = [i for i, e in eps.items() if e["is_mistake"]]
    assert len(mistakes) >= 4


def test_convert_spirit_full(spirit_raw, tmp_path):
    out = tmp_path / "dataset"
    stats = convert_spirit(spirit_raw, out, link_videos=True)
    dataset = Path(stats["out_dir"])

    assert stats["converted"] >= 9
    assert stats["skipped"] >= 4
    # 每有效集 3 个相机视频
    assert stats["videos"] == stats["converted"] * 3

    # meta 四件套
    meta_files = ["info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"]
    for m in meta_files:
        assert (dataset / "meta" / m).exists()

    # data 目录：episode 数量与统计一致
    data = dataset / "data" / "chunk-000"
    eps = sorted(data.glob("episode_*.parquet"))
    assert len(eps) == stats["converted"]

    # 每集：6 列齐全，action / observation.state 为 22 维 list<float32>
    for ep in eps:
        t = pq.read_table(ep)
        cols = set(t.column_names)
        assert {
            "action", "observation.state", "frame_index",
            "timestamp", "episode_index", "task_index",
        } <= cols
        action = t["action"].to_numpy()
        state = t["observation.state"].to_numpy()
        assert len(action) == len(state) > 0
        assert all(len(row) == 22 for row in action)
        assert all(len(row) == 22 for row in state)

    # 视频：3 个相机目录、各含 converted 个 mp4 软链接
    vid = dataset / "videos" / "chunk-000"
    cams = sorted(p.name for p in vid.iterdir())
    assert cams == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    total = sum(len(list((vid / c).glob("episode_*.mp4"))) for c in cams)
    assert total == stats["videos"]
    sample_link = vid / "cam_high" / "episode_000000.mp4"
    assert sample_link.is_symlink()
    assert sample_link.resolve().exists()


def test_convert_spirit_copy_videos(spirit_raw, tmp_path):
    out = tmp_path / "dataset_copy"
    stats = convert_spirit(spirit_raw, out, link_videos=False)
    dataset = Path(stats["out_dir"])

    assert stats["converted"] >= 9
    sample = dataset / "videos" / "chunk-000" / "cam_high" / "episode_000000.mp4"
    assert sample.is_file()
    assert not sample.is_symlink()


def test_convert_spirit_preserves_out_dir(spirit_raw, tmp_path):
    out = tmp_path / "dataset"
    keep = out / "keep_me.txt"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("do not delete", encoding="utf-8")

    stats = convert_spirit(spirit_raw, out, link_videos=True)
    dataset = Path(stats["out_dir"])

    assert keep.exists()  # 既有文件不被清空
    assert dataset != out  # 结果写入新建的时间戳子目录
    assert (dataset / "meta" / "info.json").exists()


def test_run_spirit_quality(spirit_raw, tmp_path):
    csv = tmp_path / "spirit_outliers.csv"
    js = tmp_path / "spirit_summary.json"

    summary = run_spirit_quality(str(spirit_raw), str(csv), str(js))

    assert summary["num_files"] >= 9
    assert summary["num_frames"] > 0
    assert summary["num_outliers"] >= 0


def test_validate_spirit_dataset(spirit_raw):
    ok, errors = validate_spirit_dataset(spirit_raw)

    assert ok, errors


def test_read_skip_episodes_scoped_per_instance(tmp_path):
    """跳过按具体文件定位，不因同编号 episode 跨实例误伤。"""
    raw = tmp_path / "raw"
    a = raw / "A"
    b = raw / "B"
    (a / "data" / "chunk-000").mkdir(parents=True)
    (b / "data" / "chunk-000").mkdir(parents=True)
    a3 = a / "data" / "chunk-000" / "episode_000003.parquet"
    b3 = b / "data" / "chunk-000" / "episode_000003.parquet"
    a3.touch()
    b3.touch()

    # 质检 CSV：只标 A 的 episode_000003（path 列定位）
    csv_path = tmp_path / "outlier_frames.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "file", "path"])
        w.writerow([3, a3.name, str(a3)])

    # 校验报告：只标 B 的 episode_000003（实例相对路径定位）
    log_path = tmp_path / "report.txt"
    log_path.write_text(
        "B/data/chunk-000/episode_000003.parquet: 缺少必需列\n",
        encoding="utf-8",
    )

    skip = read_skip_episodes(raw, csv_path, log_path)

    assert skip == {a3.resolve(), b3.resolve()}