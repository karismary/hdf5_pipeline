"""完整链路集成测试：rename → quality → render → label。"""

from hdf5_pipeline.rename.engine import collect_hdf5_files, rename_files
from hdf5_pipeline.quality.checker import run_quality_check
from hdf5_pipeline.render.engine import render_mp4
from hdf5_pipeline.label.database import add_label, get_records, scan_pairs


def test_full_pipeline(raw_dir, tmp_path):
    # ---- 1. rename：递归收集 + 重命名到统一格式 ----
    renamed = tmp_path / "renamed"
    files = collect_hdf5_files(str(raw_dir))
    assert len(files) == 3
    assert rename_files(files, str(renamed)) == 3
    first = renamed / "episode_000000.hdf5"
    assert first.exists()

    # ---- 2. quality：对重命名目录做异常检测 ----
    summary = run_quality_check(
        str(renamed), "hdf5",
        str(tmp_path / "outlier_frames.csv"),
        str(tmp_path / "outlier_summary.json"),
    )
    assert summary["num_files"] == 3

    # ---- 3. render：渲染第一集，输出名与 scan_pairs 配对约定一致 ----
    mp4_dir = tmp_path / "mp4"
    mp4_dir.mkdir()
    ok, msg, _name = render_mp4(str(first), str(mp4_dir / "episode_000000.mp4"))
    assert ok, msg

    # ---- 4. label：配对入库并打标 ----
    db = tmp_path / "label.db"
    assert scan_pairs(str(db), str(mp4_dir), str(renamed)) == 1

    add_label(str(db), "episode_000000.mp4", quality="good")
    rec = get_records(str(db), "episode_000000.mp4")
    assert rec is not None
    assert rec["quality"] == "good"