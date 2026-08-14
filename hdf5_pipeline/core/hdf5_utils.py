import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal, overload, cast
import h5py
import numpy as np
import pyarrow.parquet as pq
import re

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

FORMAT_DICT = {
    "lerobot":".parquet",
    "hdf5":".hdf5"
}

def natural_sort_key(name: str) -> list:
    """按数字大小排序，保证 2 在 10 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

def get_hdf5_files(folder: str, format: str = "hdf5") -> List[Path]:
    """递归收集目录下所有指定格式文件，按文件名普通排序后返回。

    使用 rglob 搜索，会递归所有子目录。
    支持通过 format 参数选择数据格式（对应 FORMAT_DICT）。
    注意：默认 sorted() 为字典序排序（episode_10 在 episode_2 前面），
    如需自然排序请自行调用 sorted(files, key=lambda p: natural_sort_key(p.name))。

    Args:
        folder (Path): 要搜索的根目录路径。
        format (str): 数据格式关键字，如 "hdf5" → "*.hdf5"、"lerobot" → "*.parquet"。
            不在 FORMAT_DICT 中时回退为 ".hdf5"。

    Returns:
        List[Path]: 按文件名字典序排序的匹配文件路径列表。
    """
    file_type = FORMAT_DICT.get(format, ".hdf5")
    dir_path = Path(folder)

    return sorted([f for f in dir_path.rglob(f"*{file_type}")])

@overload
def get_sorted_files(folder: str, file_type: list, return_type: Literal[0]) -> List[Path]: ...

@overload
def get_sorted_files(folder: str, file_type: list, return_type: Literal[1]) -> List[str]: ...

def get_sorted_files(folder: str, file_type: list, return_type: int) -> List[Path] | List[str]:
    """递归收集目录下所有指定类型文件，按文件名自然排序后返回。

    使用 rglob 搜索，会递归所有子目录。

    Args:
        folder (Path): 要搜索的根目录路径。
        file_type (list): 要搜索的文件后缀列表，例如 ['.mp4', '.hdf5']。
        return_type (int): 返回格式控制。
            0 — 返回完整 Path 对象列表。
            1 — 只返回文件名字符串列表。

    Returns:
        List[Path] 或 List[str]:
            当 return_type=0 时返回 Path 对象列表；
            当 return_type=1 时返回文件名字符串列表。
            两种情况下均按自然排序（保证 "episode_2" 在 "episode_10" 前面）。
    """
    dir_path = Path(folder)
    if len(file_type) == 1:
        if return_type == 0:
            return sorted([f for f in dir_path.rglob(f"*{file_type[0]}")], 
                        key = lambda p: natural_sort_key(p.name))
        else:
            return sorted([f.name for f in dir_path.rglob(f"*{file_type[0]}")], 
                key = lambda p: natural_sort_key(p))
    else:
        all_files = []
        for ext in file_type:
            all_files.extend(dir_path.rglob(f"*{ext}"))
        if return_type == 0:
            return sorted(all_files, key=lambda p: natural_sort_key(p.name))
        else:
            names: list[str] = [f.name for f in all_files]
            return sorted(names, key=lambda p: natural_sort_key(p))

def get_hdf5_frame_count(h5_file: str, format: str = "hdf5"):
    """获取数据文件的帧数，兼容 HDF5 与 LeRobot Parquet 两种格式。

    Args:
        h5_file (str): 数据文件路径（HDF5 或 Parquet）。
        format (str): 数据格式关键字，如 "hdf5" → 读取 HDF5、"lerobot" → 读取 Parquet。

    Returns:
        int 或 None: 帧数；文件不存在或无法读取时返回 None。
    """
    h5_path = Path(h5_file)
    if not h5_path.exists():
        return None
    try:
        if format == "lerobot":
            return pq.ParquetFile(h5_path).metadata.num_rows
        with h5py.File(h5_path, "r") as f:
            if "observations/pixels" in f:
                pix = "observations/pixels"
            elif "observation/pixels" in f:
                pix = "observation/pixels"
            else:
                pix = "pixels"
            cameras = list(cast(h5py.Group, f[pix]).keys())
            if not cameras:
                return None
            return min(np.asarray(f[f"{pix}/{c}"]).shape[0] for c in cameras)
    except Exception:
        return None

def normalize_image_array(arr: np.ndarray) -> np.ndarray:
    """将各种形状/类型的图像数组统一为 (T, H, W, 3) uint8。

    依次进行三步处理，以适应模型中统一的数据要求：

    1. NCHW → NHWC 转置
       如果输入为 (T, C, H, W) 且 C 为 1 或 3，自动转置为 (T, H, W, C)。
    2. float → uint8 归一化
       如果输入为 float 类型（值域通常 [0, 1]），缩放到 [0, 255] 并转为 uint8。
    3. 灰度 → RGB 扩展
       如果输入只有 1 个颜色通道，复制 3 份扩展为 RGB。

    Args:
        arr (np.ndarray): 输入图像数组，支持的形状与类型:
            - NHWC: (T, H, W, C), C=1 或 3，uint8 或 float
            - NCHW: (T, C, H, W), C=1 或 3，uint8 或 float

    Returns:
        np.ndarray: 形状为 (T, H, W, 3) 的 uint8 数组。
    """

    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
            
    if not np.issubdtype(arr.dtype, np.integer):
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, -1)

    return arr


def load_images_from_hdf5(path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """从 HDF5 文件加载所有相机图像。

    Args:
        path: HDF5 文件路径。

    Returns:
        {cam_name: ndarray (T, H, W, 3)}
    """

    with h5py.File(path, "r") as root:
        ts = np.asarray(root["timestamps"]) if "timestamps" in root else None
        pix_group = "observations/pixels" if "observations/pixels" in root else "pixels"
        cams = list(cast(h5py.Group, root[pix_group]).keys())
        imgs = {}
        for c in cams:
            raw_array = np.asarray(root[f"{pix_group}/{c}"])
            imgs[c] = normalize_image_array(raw_array)
        if ts is not None and len(ts) > 1:
            dt = ts[1] - ts[0]
            fps = int(round(1.0 / dt)) if dt > 0 else 15
        else:
            fps = 15

    return imgs, fps


def load_actions_from_hdf5(path: str, n_frames: int) -> np.ndarray:
    """加载动作数据，自动检测 'action' 或 'actions' key。

    Args:
        path: HDF5 文件路径。
        n_frames: 取前 N 帧。

    Returns:
        ndarray (n_frames, dim)
    """

    with h5py.File(path, "r") as root:
        act = np.asarray(root["action"])[:n_frames] if "action" in root else np.asarray(root["actions"])[:n_frames]

    return act


def load_joints_from_hdf5(path: str, n_frames: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """加载左右臂关节数据。

    Args:
        path: HDF5 文件路径。
        n_frames: 取前 N 帧。

    Returns:
        (left_joints, right_joints) — 不存在时为 None
    """

    with h5py.File(path, "r") as root:
        left_j = np.asarray(root["observations/left_arm_joints"])[:n_frames] if "observations/left_arm_joints" in root else None
        right_j = np.asarray(root["observations/right_arm_joints"])[:n_frames] if "observations/right_arm_joints" in root else None

    return left_j, right_j


def project_30_to_16(x30: np.ndarray) -> np.ndarray:
    """将原始 30 维向量投影到训练使用的 16 维。

    丢弃末端位姿 (end-effector pose)，只保留关节和夹爪。

    Args:
        x30: 原始 30 维数据, shape (T, 30)。
            布局: left_ee(7) + right_ee(7) + left_joints(7) + right_joints(7) + left_grip(1) + right_grip(1)

    Returns:
        16 维数据, shape (T, 16)。
            布局: left_joints(7) + left_grip(1) + right_joints(7) + right_grip(1)
    """

    left_j = x30[:, 14:21]
    left_g = x30[:, 28:29]
    right_j = x30[:, 21:28]
    right_g = x30[:, 29:30]
    return np.concatenate([left_j, left_g, right_j, right_g], axis=1)


def load_raw_30dim(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """从 HDF5 加载原始 30 维 action 和 state。

    从 observation 字段重建 state_30，与 action_30 保持相同语义布局。

    Args:
        path: HDF5 文件路径。

    Returns:
        (action_30, state_30) — shape 均为 (T, 30)
    """
    with h5py.File(path, "r") as f:
        action_30 = np.asarray(f["actions"])

        left_ee = np.asarray(f["observations/left_end_effector_pose"])
        right_ee = np.asarray(f["observations/right_end_effector_pose"])
        left_j = np.asarray(f["observations/left_arm_joints"])
        right_j = np.asarray(f["observations/right_arm_joints"])
        left_g = np.asarray(f["observations/left_gripper_state"])
        right_g = np.asarray(f["observations/right_gripper_state"])

        state_30 = np.concatenate(
            [left_ee, right_ee, left_j, right_j, left_g, right_g], axis=1
        )

    return action_30, state_30