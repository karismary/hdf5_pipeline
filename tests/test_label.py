"""label 模块：MP4/HDF5 配对入库、打标、批量更新、查询。"""

import json
import shutil

from hdf5_pipeline.label.database import (
    add_label,
    add_labels,
    count_list,
    get_list,
    get_records,
    query_records,
    scan_pairs,
)


def _build_paired_dirs(tmp_path, sample_hdf5, n=3):
    """构造 mp4_dir + raw_dir 平铺目录，前 n 对同名配对。"""
    mp4_dir = tmp_path / "mp4"
    raw_dir = tmp_path / "raw"
    mp4_dir.mkdir()
    raw_dir.mkdir()
    for i in range(n):
        stem = f"episode_{i:06d}"
        shutil.copy2(sample_hdf5, raw_dir / f"{stem}.hdf5")
        (mp4_dir / f"{stem}.mp4").write_bytes(b"fake-mp4")
    return mp4_dir, raw_dir


def test_scan_pairs_and_add_label(tmp_path, sample_hdf5):
    mp4_dir, raw_dir = _build_paired_dirs(tmp_path, sample_hdf5, 3)
    db = tmp_path / "label.db"

    assert scan_pairs(str(db), str(mp4_dir), str(raw_dir)) == 3

    add_label(str(db), "episode_000000.mp4", quality="good", attr={"背景": "阳台"})
    rec = get_records(str(db), "episode_000000.mp4")

    assert rec is not None
    assert rec["quality"] == "good"
    assert json.loads(rec["attr"])["背景"] == "阳台"


def test_scan_pairs_ignores_unpaired(tmp_path, sample_hdf5):
    mp4_dir, raw_dir = _build_paired_dirs(tmp_path, sample_hdf5, 2)
    # 只有 mp4 没有 hdf5 / 只有 hdf5 没有 mp4 的孤儿文件不应配对
    (mp4_dir / "orphan_mp4_only.mp4").write_bytes(b"x")
    (raw_dir / "orphan_hdf5_only.hdf5").write_bytes(b"x")
    db = tmp_path / "label.db"

    assert scan_pairs(str(db), str(mp4_dir), str(raw_dir)) == 2


def test_add_labels_batch_quality(tmp_path, sample_hdf5):
    mp4_dir, raw_dir = _build_paired_dirs(tmp_path, sample_hdf5, 3)
    db = tmp_path / "label.db"
    scan_pairs(str(db), str(mp4_dir), str(raw_dir))
    records = get_list(str(db))

    add_labels(str(db), records, quality="bad")

    assert count_list(str(db), "quality = 'bad'") == 3


def test_add_labels_if_qualify_updates_path_not_move(tmp_path, sample_hdf5):
    mp4_dir, raw_dir = _build_paired_dirs(tmp_path, sample_hdf5, 2)
    db = tmp_path / "label.db"
    scan_pairs(str(db), str(mp4_dir), str(raw_dir))
    records = get_list(str(db))
    target = tmp_path / "good"

    add_labels(str(db), records, quality="good", if_qualify=True, target_dir=str(target))

    rec = get_records(str(db), "episode_000000.mp4")
    assert rec["quality"] == "good"
    # 归档模式只更新 DB 里的 hdf5_path，文件移动由 UI 层负责
    assert rec["hdf5_path"] == str(target / "episode_000000.hdf5")
    assert (raw_dir / "episode_000000.hdf5").exists()


def test_query_records_and_count(tmp_path, sample_hdf5):
    mp4_dir, raw_dir = _build_paired_dirs(tmp_path, sample_hdf5, 3)
    db = tmp_path / "label.db"
    scan_pairs(str(db), str(mp4_dir), str(raw_dir))
    add_label(str(db), "episode_000001.mp4", quality="bad")

    assert count_list(str(db), "quality = 'bad'") == 1

    rows, error = query_records(str(db), "*", "quality = 'bad'")
    assert error is None
    assert len(rows) == 1
    assert rows[0]["mp4_name"] == "episode_000001.mp4"