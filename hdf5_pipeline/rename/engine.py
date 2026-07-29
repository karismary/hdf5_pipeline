import shutil
from pathlib import Path
from hdf5_pipeline.core.hdf5_utils import get_hdf5_files, natural_sort_key


def collect_hdf5_files(data_dir: str) -> list[Path]:
    """扫描目录（含所有子目录），返回按自然顺序排序的 HDF5 文件列表。"""
    files = get_hdf5_files(Path(data_dir))

    for subdir in Path(data_dir).iterdir():
        if not subdir.is_dir() or subdir.name in ("rename", "__pycache__"):
            continue
        files.extend(get_hdf5_files(subdir))

    files = sorted(set(files), key=lambda f: natural_sort_key(f.name))
    return files


def rename_files(files: list[Path], output_dir: str) -> int:
    """按 episode_XXXXXX 格式重命名并复制到输出目录。

    Args:
        files: HDF5 文件路径列表。
        output_dir: 输出目录。

    Returns:
        处理的文件数量。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        dst = out / f"episode_{i:06d}.hdf5"
        shutil.copy2(src, dst)
    return len(files)