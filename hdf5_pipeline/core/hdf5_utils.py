import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import h5py
import numpy as np
import re

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def natural_sort_key(name: str) -> list:
    """按数字大小排序，保证 2 在 10 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

def get_hdf5_files(folder: Path) -> List[Path]:
    """递归收集目录下所有 HDF5 文件，按文件名自然排序后返回。

    使用 rglob 搜索，会递归所有子目录。
    同时匹配 .h5 和 .hdf5 两种后缀。

    Args:
        folder (Path): 要搜索的根目录路径。

    Returns:
        List[Path]: 排序后的 HDF5 文件路径列表。
            按文件名进行自然排序（自然排序保证了 "episode_2" 在 "episode_10" 前面）。
    """

    return sorted([f for f in folder.rglob("*.h5")] + [f for f in folder.rglob("*.hdf5")])

def get_sorted_files(folder: Path, file_type: str, return_type: int) -> List[Path]:
    """递归收集目录下所有指定类型文件，按文件名自然排序后返回。

    使用 rglob 搜索，会递归所有子目录。

    Args:
        folder (Path): 要搜索的根目录路径。
        file_type (str): 要搜索的文件后缀，例如 '.mp4'、'.hdf5'。
        return_type (int): 返回格式控制。
            0 — 返回完整 Path 对象列表。
            1 — 只返回文件名字符串列表。

    Returns:
        List[Path] 或 List[str]:
            当 return_type=0 时返回 Path 对象列表；
            当 return_type=1 时返回文件名字符串列表。
            两种情况下均按自然排序（保证 "episode_2" 在 "episode_10" 前面）。
    """
    if return_type == 0:
        return sorted([f for f in folder.rglob(f"*{file_type}")], 
                    key = lambda p: natural_sort_key(p.name))
    else:
        return sorted([f.name for f in folder.rglob(f"*{file_type}")], 
            key = lambda p: natural_sort_key(p))

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


def load_images_from_hdf5(path: str) -> Dict[str, np.ndarray]:
    """从 HDF5 文件加载所有相机图像。

    Args:
        path: HDF5 文件路径。

    Returns:
        {cam_name: ndarray (T, H, W, 3)}
    """

    with h5py.File(path, "r") as root:
        pix_group = "observations/pixels" if "observations/pixels" in root else "pixels"
        cams = list(root[pix_group].keys())
        imgs = {}
        for c in cams:
            raw_array = root[f"{pix_group}/{c}"][:]
            imgs[c] = normalize_image_array(raw_array)

    return imgs


def load_actions_from_hdf5(path: str, n_frames: int) -> np.ndarray:
    """加载动作数据，自动检测 'action' 或 'actions' key。

    Args:
        path: HDF5 文件路径。
        n_frames: 取前 N 帧。

    Returns:
        ndarray (n_frames, dim)
    """

    with h5py.File(path, "r") as root:
        act = root["action"][:n_frames] if "action" in root else root["actions"][:n_frames]

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
        left_j = root["observations/left_arm_joints"][:n_frames] if "observations/left_arm_joints" in root else None
        right_j = root["observations/right_arm_joints"][:n_frames] if "observations/right_arm_joints" in root else None
 
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
        action_30 = f["actions"][:]

        left_ee = f["observations/left_end_effector_pose"][:]
        right_ee = f["observations/right_end_effector_pose"][:]
        left_j = f["observations/left_arm_joints"][:]
        right_j = f["observations/right_arm_joints"][:]
        left_g = f["observations/left_gripper_state"][:]
        right_g = f["observations/right_gripper_state"][:]

        state_30 = np.concatenate(
            [left_ee, right_ee, left_j, right_j, left_g, right_g], axis=1
        )

    return action_30, state_30