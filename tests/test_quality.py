"""quality 模块：HDF5 异常帧检测 + 结果导出。"""

import shutil

from hdf5_pipeline.quality.checker import run_quality_check
from hdf5_pipeline.quality.detector import export_results, parse_mask


def test_run_hdf5_check_summary(tmp_path, sample_hdf5):
    # quality 检测面向平铺目录（重命名后），loader 非递归 glob
    flat = tmp_path / "flat"
    flat.mkdir()
    for i in range(3):
        shutil.copy2(sample_hdf5, flat / f"episode_{i:06d}.hdf5")
    csv = tmp_path / "outlier_frames.csv"
    js = tmp_path / "outlier_summary.json"

    summary = run_quality_check(str(flat), "hdf5", str(csv), str(js))

    assert summary["num_files"] == 3
    assert summary["num_frames"] > 0
    assert summary["num_outliers"] >= 0


def test_export_results_writes_when_rows(tmp_path):
    rows = [{"file": "a.hdf5", "frame": 3, "score": 0.9}]
    csv = tmp_path / "o.csv"
    js = tmp_path / "o.json"

    returned = export_results(rows, {"num_outliers": 1}, str(csv), str(js))

    # 非空列表时写入文件（实现返回 None，falsy）
    assert not returned
    assert csv.exists()
    assert js.exists()


def test_export_results_skips_when_no_rows(tmp_path):
    csv = tmp_path / "o.csv"
    js = tmp_path / "o.json"

    returned = export_results([], {"num_outliers": 0}, str(csv), str(js))

    assert returned is True
    assert not csv.exists()
    assert not js.exists()


def test_parse_mask():
    mask = parse_mask("1,0,1")
    assert mask.tolist() == [True, False, True]