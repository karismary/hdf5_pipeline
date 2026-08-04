import shutil
import datetime
from pathlib import Path
from hdf5_pipeline.core.hdf5_utils import get_hdf5_files, natural_sort_key


def collect_hdf5_files(data_dir: str) -> list[Path]:
    """扫描目录（含所有子目录），返回按自然顺序排序的 HDF5 文件列表。

    get_hdf5_files 内部使用 rglob 递归搜索，会覆盖所有子目录。
    排除 rename 和 __pycache__ 输出目录，避免重复收集。

    Args:
        data_dir (str): 要扫描的根目录路径。

    Returns:
        list[Path]: 去重并按文件名自然排序的 HDF5 文件列表。
    """
    files = get_hdf5_files(data_dir)

    files = [f for f in files if "rename" not in f.parts and "__pycache__" not in f.parts]
    files = sorted(set(files), key=lambda f: natural_sort_key(f.name))
    return files


def rename_files(files: list[Path], output_dir: str, if_move = False) -> int:
    """按 episode_XXXXXX 格式重命名并复制到输出目录。

    如果输出目录已存在内容，会在其下新建一个带时间戳的子目录
    （如 rename_20260730_143000），避免覆盖上一次的结果。

    Args:
        files (list[Path]): HDF5 文件路径列表。
        output_dir (str): 输出目录路径。为空目录时直接使用，否则新建子目录。

    Returns:
        int: 处理的文件数量。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        out = out / f"rename_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents = True, exist_ok = True)
    for i, src in enumerate(files):
        dst = out / f"episode_{i:06d}.hdf5"
        n = 0
        while Path(dst).exists():
            n += 1
            dst = out / f"episode_{i:06d}_{n}.hdf5"
        if not if_move:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
    return len(files)