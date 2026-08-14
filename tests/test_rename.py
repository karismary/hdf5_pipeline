"""rename 模块：递归收集 + 自然排序 + episode_NNNNNN 重命名。"""

import shutil

from hdf5_pipeline.rename.engine import collect_hdf5_files, rename_files


def test_collect_recursive_and_natural_sorted(raw_dir):
    files = collect_hdf5_files(str(raw_dir))
    names = [p.name for p in files]
    # 递归到嵌套子目录，且按自然排序（episode_10 排在 episode_2 之后）
    assert names == [
        "episode_000001.hdf5",
        "episode_000002.hdf5",
        "episode_000010.hdf5",
    ]


def test_collect_excludes_rename_dir(tmp_path, sample_hdf5):
    raw = tmp_path / "raw"
    (raw / "rename").mkdir(parents=True)
    shutil.copy2(sample_hdf5, raw / "rename" / "episode_000003.hdf5")
    shutil.copy2(sample_hdf5, raw / "episode_000004.hdf5")

    files = collect_hdf5_files(str(raw))
    assert [p.name for p in files] == ["episode_000004.hdf5"]


def test_rename_files_copy_and_numbering(raw_dir, tmp_path):
    files = collect_hdf5_files(str(raw_dir))
    out = tmp_path / "renamed"

    n = rename_files(files, str(out))

    assert n == 3
    produced = sorted(p.name for p in out.glob("*.hdf5"))
    assert produced == [
        "episode_000000.hdf5",
        "episode_000001.hdf5",
        "episode_000002.hdf5",
    ]
    # copy 模式：源文件仍在原位
    for f in files:
        assert f.exists()


def test_rename_into_nonempty_dir_creates_timestamp_subdir(raw_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "placeholder.txt").write_text("x", encoding="utf-8")

    files = collect_hdf5_files(str(raw_dir))
    rename_files(files, str(out))

    subs = [d for d in out.iterdir() if d.is_dir()]
    assert len(subs) == 1
    assert len(list(subs[0].glob("*.hdf5"))) == 3


def test_rename_if_move_removes_source(raw_dir, tmp_path):
    files = collect_hdf5_files(str(raw_dir))
    out = tmp_path / "renamed"

    rename_files(files, str(out), if_move=True)

    for f in files:
        assert not f.exists()
    assert len(list(out.glob("*.hdf5"))) == 3