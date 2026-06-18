"""HDF5 → MP4 视频渲染引擎。
"""

import os
import traceback
import gc
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hdf5_pipeline.core.constants import JOINT_NAMES
from hdf5_pipeline.core.hdf5_utils import (
    load_images_from_hdf5,
    load_actions_from_hdf5,
    load_joints_from_hdf5,
)

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',          # macOS 苹方
    'STHeiti',              # macOS 华文黑体
    'SimHei',               # Windows 黑体
    'Microsoft YaHei',      # Windows 微软雅黑
    'Noto Sans CJK SC',     # Linux 思源黑体
    'WenQuanYi Micro Hei',  # Linux 文泉驿
]
plt.rcParams['axes.unicode_minus'] = False
LINE_COLORS = plt.cm.tab20(np.linspace(0, 1, 16))
JOINT_COLORS = plt.cm.plasma(np.linspace(0, 1, 7))

# ==================== 辅助 ====================

def _prepare_action16(act: np.ndarray, n_frames: int) -> np.ndarray:
    """
    从 action 数组提取或填充出固定的 16 维动作空间特征。
    Args:
        act (np.ndarray): 原始的动作数据数组，可能包含任意维度。
        n_frames (int): 目标视频或序列的总帧数。
    Returns:
        np.ndarray: 形状为 (n_frames, 16) 的 numpy 数组，数据类型为 float32。
    """
    arr = np.zeros((n_frames, 16), dtype=np.float32)
    if act.ndim == 2 and act.shape[1] >= 16:
        arr[:, :] = act[:, -16:].astype(np.float32)
    elif act.ndim == 2:
        d = min(act.shape[1], 16)
        arr[:, :d] = act[:, :d].astype(np.float32)
    return arr

def _fig_to_rgb(fig):
    """将 matplotlib 的 Figure 对象转换为 numpy RGB 图像数组。

    Args:
        fig (matplotlib.figure.Figure): 需要转换的 matplotlib 图像对象。

    Returns:
        np.ndarray: 形状为 (H, W, 3) 的 numpy 数组，表示 RGB 图像。
    """
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def _render_action_curves(n_frames, act16, action_on, panel_w, panel_h):
    """渲染 16 维动作曲线，生成静态的 RGB 图像数组。

    Args:
        n_frames (int): 总帧数，用于设置 X 轴范围。
        act16 (np.ndarray): 形状为 (n_frames, 16) 的动作数据。
        action_on (list): 长度为 16 的布尔列表，控制是否显示对应维度的曲线。
        panel_w (int): 渲染面板的宽度（像素）。
        panel_h (int): 渲染面板的高度（像素）。

    Returns:
        np.ndarray: 渲染完成的 RGB 图像数组。
    """
    fig, ax = plt.subplots(figsize=(panel_w / 80, panel_h / 80), dpi=80)
    x_full = np.arange(n_frames)
    ax.set_xlim(0, n_frames - 1)
    ax.set_ylim(-3.14, 3.14)
    ax.set_title("Action 16 Dimensions", fontsize=10, pad=8)
    ax.grid(True, alpha=0.3)
    for i in range(16):
        if action_on[i]:
            ax.plot(x_full, act16[:, i], color=LINE_COLORS[i], linewidth=1.2, label=f"a{i}")
        else:
            ax.plot([], [], color=LINE_COLORS[i], linewidth=1.2, label=f"a{i}")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    fig.canvas.draw()
    img = _fig_to_rgb(fig)
    # 获取 axes 像素坐标
    ax_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_x0 = int(ax_bbox.x0 * fig.dpi)
    ax_x1 = int(ax_bbox.x1 * fig.dpi)
    plt.close(fig)
    return img, ax_x0, ax_x1


def _render_joint_curves(n_frames, joint_data, joint_on, title, panel_w, panel_h):
    """渲染单侧机械臂的 7 维关节曲线，生成静态的 RGB 图像数组。

    Args:
        n_frames (int): 总帧数，用于设置 X 轴范围。
        joint_data (np.ndarray): 形状为 (n_frames, 7) 的关节数据。如果为 None,则返回 None。
        joint_on (list): 长度为 7 的布尔列表，控制是否显示对应关节的曲线。
        title (str): 图表标题（例如 "Left Arm Joints"）。
        panel_w (int): 渲染面板的宽度（像素）。
        panel_h (int): 渲染面板的高度（像素）。

    Returns:
    tuple or None:
        - None: 如果 joint_data 为 None 或所有显示开关关闭
        - (img: np.ndarray, x0: int, x1: int): 
            img 为 (H, W, 3) 的 RGB 图像数组
            x0, x1 为 该曲线图 在图像中的像素 X 坐标范围
    """
    if joint_data is None or not any(joint_on):
        return None
    fig, jx = plt.subplots(figsize=(panel_w / 80, panel_h / 80), dpi=80)
    x_full = np.arange(n_frames)
    jx.set_xlim(0, n_frames - 1)
    jx.set_ylim(-3.14, 3.14)
    jx.set_title(title, fontsize=10, pad=8)
    jx.grid(True, alpha=0.3)
    for i in range(7):
        if joint_on[i]:
            jx.plot(x_full, joint_data[:, i], color=JOINT_COLORS[i], linewidth=1.2, label=JOINT_NAMES[i][:8])
        else:
            jx.plot([], [], color=JOINT_COLORS[i], linewidth=1.2, label=JOINT_NAMES[i][:8])
    jx.legend(loc="upper right", fontsize=7, ncol=2)
    fig.canvas.draw()
    img = _fig_to_rgb(fig)
    jx_bbox = jx.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    jx_x0 = int(jx_bbox.x0 * fig.dpi)
    jx_x1 = int(jx_bbox.x1 * fig.dpi)
    plt.close(fig)
    return img, jx_x0, jx_x1


def _draw_cursor_line(img, t, n_frames, x0, x1):
    """在曲线图上绘制红色竖线，指示当前播放到第几帧。
    
    直接在原图数组上修改（in-place），不返回值。

    Args:
        img (np.ndarray): 要画线的 RGB 图像，shape (H, W, 3)。会被原地修改。
        t (int): 当前帧索引，范围 [0, n_frames-1]。
        n_frames (int): 视频总帧数。
        x0 (int): 曲线图 axes 区域在图像中的左边界像素 X 坐标。
        x1 (int): 曲线图 axes 区域在图像中的右边界像素 X 坐标。
            红线只会画在 [x0, x1] 范围内，不会超出图表区域。

    Returns:
        None。直接修改传入的 img 数组。

    示例:
        假设有 100 帧，曲线图区域在图像上 x=[50, 500]，
        播放到第 50 帧时，红线画在 x = 50 + (50/99) * 450 ≈ 277 处。
    """

    width = x1 - x0
    x = int(round(t / max(n_frames - 1, 1) * width)) + x0
    cv2.line(img, (x, 0), (x, img.shape[0] - 1), (0, 0, 255), 1)

# ==================== 主渲染函数 ====================

def render_mp4(
    hdf5_path: Path, out_mp4: Path,
    show_img: bool = True, show_act: bool = True,
    action_on: list = None, left_on: list = None, right_on: list = None,
    abort_event=None
):
    """将 HDF5 格式的录像和传感器数据同步渲染合成 MP4 视频。

    该函数会提取 HDF5 文件中的图像帧、动作序列以及机械臂关节数据，并将这些
    信息使用 OpenCV 和 Matplotlib 组合渲染为多面板的监控视角视频。支持多进程
    安全退出机制。

    Args:
        hdf5_path (Path): 输入的 HDF5 数据文件路径。
        out_mp4 (Path): 输出生成的 MP4 视频文件的目标路径。
        show_img (bool, optional): 是否在视频中包含实拍的相机图像面板。默认为 True。
        show_act (bool, optional): 是否在视频中包含动作维度曲线图。默认为 True。
        action_on (list, optional): 长度为 16 的布尔列表，控制 16 维动作曲线的具体显示状态。默认为全部开启。
        left_on (list, optional): 长度为 7 的布尔列表，控制左臂 7 个关节曲线显示。默认为全部开启。
        right_on (list, optional): 长度为 7 的布尔列表，控制右臂 7 个关节曲线显示。默认为全部开启。
        abort_event (multiprocessing.Event, optional): 多进程事件对象，用于接收用户从外部发送的强制中断信号。

    Returns:
        tuple: 包含三个元素的元组 (success: bool, message: str, filename: str)
            - success: 视频生成是否成功（或是否被安全中断）。
            - message: 执行结果的状态信息或错误详情。
            - filename: 处理的原始 HDF5 文件名。
    """
    if action_on is None:  action_on = [True] * 16
    if left_on is None:    left_on = [True] * 7
    if right_on is None:   right_on = [True] * 7

    video_writer = None
    out_str = str(Path(out_mp4).resolve())
    hdf5_str = str(Path(hdf5_path).resolve())

    try:
        if abort_event and abort_event.is_set():
            return False, "用户手动终止", hdf5_path.name

        plt.close("all")

        # ---- 1. 加载数据 ----
        imgs = load_images_from_hdf5(hdf5_str)
        cams = list(imgs.keys())
        n_frames = min(v.shape[0] for v in imgs.values())
        act = load_actions_from_hdf5(hdf5_str, n_frames)
        left_j, right_j = load_joints_from_hdf5(hdf5_str, n_frames)
        act16 = _prepare_action16(act, n_frames)

        # ---- 2. 预渲染曲线面板 (只调一次) ----
        panel_w = 960
        rows = bool(show_img) + bool(show_act)
        if (any(left_on) and left_j is not None) or (any(right_on) and right_j is not None):
            rows += 1
        rows = max(rows, 1)
        row_h = max(200, 640 // rows)

        curve_action_img = None
        if show_act:
            curve_action_img, ax_x0, ax_x1 = _render_action_curves(n_frames, act16, action_on, panel_w, row_h)
        curve_left_img = None
        lx_x0 = lx_x1 = None
        if any(left_on) and left_j is not None:
            j_w = panel_w // 2 if (any(right_on) and right_j is not None) else panel_w
            curve_left_img, lx_x0, lx_x1 = _render_joint_curves(n_frames, left_j, left_on, "Left Arm Joints", j_w, row_h)

        curve_right_img = None
        rx_x0 = rx_x1 = None
        if any(right_on) and right_j is not None:
            j_w = panel_w // 2 if (any(left_on) and left_j is not None) else panel_w
            curve_right_img, rx_x0, rx_x1 = _render_joint_curves(n_frames, right_j, right_on, "Right Arm Joints", j_w, row_h)



        # ---- 3. VideoWriter ----
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        total_h = rows * row_h
        video_writer = cv2.VideoWriter(out_str, fourcc, 15, (panel_w, total_h))
        if not video_writer.isOpened():
            raise RuntimeError("无法初始化视频写入器")

        # ---- 4. 逐帧合成 (纯 OpenCV, 无 matplotlib draw) ----
        is_aborted = False
        for t in range(n_frames):
            if abort_event and abort_event.is_set():
                is_aborted = True
                break

            canvas = np.full((total_h, panel_w, 3), 255, dtype=np.uint8)
            y = 0

            if show_img:
                raw = np.concatenate([imgs[c][t] for c in cams], axis=1)
                rh, rw = raw.shape[:2]
                scale = min(panel_w / rw, row_h / rh)
                frame = cv2.resize(raw, (int(rw * scale), int(rh * scale)))
                x_off = (panel_w - frame.shape[1]) // 2
                canvas[y:y+frame.shape[0], x_off:x_off+frame.shape[1]] = frame
                y += row_h

            if curve_action_img is not None:
                img = curve_action_img.copy()
                _draw_cursor_line(img, t, n_frames,ax_x0,ax_x1)
                canvas[y:y+row_h, :panel_w] = img[:row_h, :panel_w]
                y += row_h

            joint_panels = []
            if curve_left_img is not None:  joint_panels.append(curve_left_img)
            if curve_right_img is not None: joint_panels.append(curve_right_img)

            if joint_panels:
                num = len(joint_panels)
                p_w = panel_w // num
            for i, img in enumerate(joint_panels):
                c = img.copy()
                if img is curve_left_img:
                    _draw_cursor_line(c, t, n_frames, lx_x0, lx_x1)
                if img is curve_right_img:
                    _draw_cursor_line(c, t, n_frames, rx_x0, rx_x1)
                canvas[y:y+row_h, i*p_w:(i+1)*p_w] = c[:row_h, :p_w]
                y += row_h


            video_writer.write(canvas)

        # ---- 6. 清理 ----
        video_writer.release()
        video_writer = None

        if is_aborted:
            if os.path.exists(out_str):
                try:
                    os.remove(out_str)
                except Exception:
                    pass
            return False, "用户手动终止", hdf5_path.name

        return True, "Video generated successfully", hdf5_path.name

    except Exception as e:
        err_msg = f"{str(e)} | {traceback.format_exc(limit=1).strip()}"
        if video_writer is not None:
            video_writer.release()
        if os.path.exists(out_str):
            try: os.remove(out_str)
            except Exception: pass
        return False, err_msg, hdf5_path.name

    finally:
        if 'imgs' in locals(): del imgs
        if 'act16' in locals(): del act16
        gc.collect()