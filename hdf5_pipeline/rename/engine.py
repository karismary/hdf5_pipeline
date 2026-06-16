import os
import shutil
from pathlib import Path
from hdf5_pipeline.core.hdf5_utils import get_hdf5_files


def collect_hdf5_files(data_dir: str) -> list[Path]:
    """扫描所有子目录，返回按自然顺序排序的 HDF5 文件列表。"""
    files = []
    for name in os.listdir(data_dir):
        subdir = os.path.join(data_dir, name)
        if not os.path.isdir(subdir) or name in ("rename", "__pycache__"):
            continue
        files.extend(get_hdf5_files(Path(subdir)))

    files.sort(key=lambda f: natural_sort_key(f.name))
    return files


def rename_files(files: list[Path], output_dir: str) -> int:
    """按 episode_XXXXXX 格式重命名并复制到输出目录。

    Args:
        files: HDF5 文件路径列表。
        output_dir: 输出目录。

    Returns:
        处理的文件数量。
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, src in enumerate(files):
        dst = os.path.join(output_dir, f"episode_{i:06d}.hdf5")
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst}")
    return len(files)

