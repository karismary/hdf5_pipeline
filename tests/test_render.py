"""render 模块：HDF5 → MP4 多面板渲染。"""

import cv2

from hdf5_pipeline.render.engine import render_mp4


def test_render_mp4_success(sample_hdf5, tmp_path):
    out = tmp_path / "episode_000006.mp4"

    ok, msg, name = render_mp4(str(sample_hdf5), str(out))

    assert ok, msg
    assert name == "episode_000006.hdf5"
    assert out.exists()
    assert out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    assert frame_count > 0


def test_render_mp4_missing_file(tmp_path):
    ok, msg, name = render_mp4(
        str(tmp_path / "no_such.hdf5"), str(tmp_path / "no_such.mp4")
    )

    assert ok is False
    assert msg
    assert name == "no_such.hdf5"