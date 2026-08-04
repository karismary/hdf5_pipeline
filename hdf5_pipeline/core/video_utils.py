import os
from typing import Tuple, Optional

import cv2
import numpy as np

def extract_first_last_frames(mp4_path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    提取视频文件的首帧和尾帧，并将颜色空间从默认的 BGR 转换为 RGB。

    Args:
        mp4_path (str or pathlib.Path): MP4 视频文件的绝对或相对路径。

    Returns:
        tuple: 包含两个元素的元组 `(first_frame, last_frame)`。
            - first_frame (numpy.ndarray or None): 视频的第一帧图像（RGB 格式）。
            - last_frame (numpy.ndarray or None): 视频的最后一帧有效图像（RGB 格式）。
            - 如果视频文件不存在、无法打开或帧读取失败，对应位置返回 None。

    Notes:
        - OpenCV 在处理某些视频容器时，直接读取 `total_frames - 1`（绝对最后一帧）
          有几率会返回空数据 (None) 或引发错误。因此，代码中将尾帧索引设定为 
          `total_frames - 2`，这是一种保证能够稳定读取到视频末尾画面的工程化做法。
    """
    if not os.path.exists(mp4_path): return None, None
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened(): return None, None
    ret, frame1 = cap.read()
    first_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) if ret else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_frame = None
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
        ret, frame2 = cap.read()
        if ret: last_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    cap.release()
    return first_frame, last_frame

def get_video_info(mp4_path: str) -> Tuple[int, float]:
    """获取视频的帧数和帧率。

    Args:
        mp4_path: 视频文件路径。

    Returns:
        (total_frames, fps) — 总帧数和帧率。
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return 0, 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

def get_frame(mp4_path: str, n: int) -> Optional[np.ndarray]:
    """读取视频指定帧，返回 RGB 格式的 numpy 数组。

    使用 cap.set 跳转到指定帧，只读一帧，不加载全部帧到内存。

    Args:
        mp4_path (str): MP4 视频文件路径。
        n (int): 帧索引，从 0 开始。n=0 为首帧。

    Returns:
        np.ndarray 或 None: RGB 格式的图像数组 (H, W, 3)，读取失败返回 None。
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    if n > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    cap.release()
    return frame[:, :, ::-1] if ret else None