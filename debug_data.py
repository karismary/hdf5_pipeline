"""调试脚本：查看 HDF5 数据的具体形状和内容。"""
import sys
from pathlib import Path

# # 确保能找到 hdf5_pipeline 包
# sys.path.insert(0, str(Path(__file__).parent))

# from hdf5_pipeline.core.hdf5_utils import (
#     load_images_from_hdf5,
#     load_actions_from_hdf5,
#     load_joints_from_hdf5,
#     load_raw_30dim,
#     project_30_to_16,
# )
# from hdf5_pipeline.render.engine import _prepare_action16


# # ====== 改这里：指定你要看的 HDF5 文件 ======
# hdf5_path = "./test_data_haimain_zhijia/episode_000016.hdf5"
# # ===========================================


# def main():
#     hdf5_str = str(Path(hdf5_path).resolve())
#     print(f"📁 文件: {hdf5_path}")
#     print("=" * 60)

#     # ---- 1. 图像 ----
#     print("\n📷 图像数据 (load_images_from_hdf5):")
#     imgs = load_images_from_hdf5(hdf5_str)
#     cams = list(imgs.keys())
#     print(f"   原始数据：{str(imgs)}")
#     print(f"   相机数量: {len(cams)}, 名称: {cams}")
#     n_frames = min(v.shape[0] for v in imgs.values())
#     print(f"   总帧数: {n_frames}")
#     for cam in cams:
#         arr = imgs[cam]
#         print(f"   {cam}: shape={arr.shape}, dtype={arr.dtype}, 值范围=[{arr.min()}, {arr.max()}]")

#     # ---- 2. 动作 ----
#     print("\n🎮 动作数据 (load_actions_from_hdf5):")
#     act = load_actions_from_hdf5(hdf5_str, n_frames)
#     print(f"   shape={act.shape}, dtype={act.dtype}")
#     print(f"   值范围=[{act.min():.4f}, {act.max():.4f}]")
#     print(f"   前 3 帧:\n{act[:3]}")

#     # ---- 3. 关节 ----
#     print("\n🦾 关节数据 (load_joints_from_hdf5):")
#     left_j, right_j = load_joints_from_hdf5(hdf5_str, n_frames)
#     if left_j is not None:
#         print(f"   左臂: shape={left_j.shape}, dtype={left_j.dtype}")
#         print(f"      值范围=[{left_j.min():.4f}, {left_j.max():.4f}]")
#     else:
#         print("   左臂: None")
#     if right_j is not None:
#         print(f"   右臂: shape={right_j.shape}, dtype={right_j.dtype}")
#         print(f"      值范围=[{right_j.min():.4f}, {right_j.max():.4f}]")
#     else:
#         print("   右臂: None")

#     # ---- 4. 16 维投影 ----
#     print("\n📐 16 维训练空间 (_prepare_action16):")
#     act16 = _prepare_action16(act, n_frames)
#     print(f"   shape={act16.shape}, dtype={act16.dtype}")
#     print(f"   前 3 帧:\n{act16[:3]}")

#     # ---- 5. 原始 30 维 (用于异常检测) ----
#     print("\n📏 原始 30 维 (load_raw_30dim):")
#     action_30, state_30 = load_raw_30dim(hdf5_str)
#     print(f"   action_30: shape={action_30.shape}, dtype={action_30.dtype}")
#     print(f"   state_30:  shape={state_30.shape}, dtype={state_30.dtype}")

#     # 投影到 16 维对比
#     act16_v2 = project_30_to_16(action_30)
#     print(f"   project_to_16: shape={act16_v2.shape}")
#     print(f"   前 3 帧:\n{act16_v2[:3]}")

#     print("\n✅ 完成")

#     # ====== video_utils 测试 ======
#     print("\n" + "=" * 60)
#     print("🎬 视频工具测试 (extract_first_last_frames + get_video_info)")

#     from hdf5_pipeline.core.video_utils import extract_first_last_frames, get_video_info

#     mp4_path = "./test_data_haimain_zhijia/v2/episode_000016.mp4"
#     mp4_str = str(Path(mp4_path).resolve())

#     # 1. 视频信息
#     total_frames, fps = get_video_info(mp4_str)
#     print(f"\n   文件: {mp4_path}")
#     print(f"   总帧数: {total_frames}")
#     print(f"   帧率: {fps:.2f} fps")
#     print(f"   时长: {total_frames / fps:.1f} 秒")

#     # 2. 首末帧
#     first, last = extract_first_last_frames(mp4_str)
#     print(f"\n   首帧: shape={first.shape}, dtype={first.dtype}" if first is not None else "   首帧: None")
#     print(f"   末帧: shape={last.shape}, dtype={last.dtype}" if last is not None else "   末帧: None")
#     print(f"   值范围: [{first.min()}, {first.max()}]" if first is not None else "")
#     # ====== quality/hdf5_checker 测试 ======
# print("\n" + "=" * 60)
# print("🔍 HDF5 Checker 测试 (load_hdf5_episodes)")

# from hdf5_pipeline.quality.hdf5_checker import load_hdf5_episodes

# fields,episodes = load_hdf5_episodes("./test_data_haimain_zhijia/*.hdf5")

# print(f"hdf5_files 显示如下 \n {str(fields)},共{len(fields)}条记录")
# print(f"\n   匹配到 {len(episodes)} 个文件\n")

# for path, action, state, frame_idx in episodes[:3]:  # 只看前 3 个
#     name = Path(path).name
#     print(f"   📄 {name}")
#     print(f"      action.shape = {action.shape}    (T=帧数, D=16维)")
#     print(f"      state.shape  = {state.shape}     (T=帧数, D=16维)")
#     print(f"      frame_idx    = shape {frame_idx.shape}")
#     print(f"      帧索引范围    = [{frame_idx[0]}, {frame_idx[-1]}]")
#     print(f"      action 值范围 = [{action.min():.4f}, {action.max():.4f}]")
#     print(f"      state  值范围 = [{state.min():.4f}, {state.max():.4f}]")
#     print()

# if len(episodes) > 3:
#     print(f"   ... 还有 {len(episodes) - 3} 个文件省略")

# ====== render/engine 函数测试 ======
# ====== render/engine 函数测试 ======
print("\n" + "=" * 60)
print("🎨 render/engine 函数测试 (真实数据)")

from hdf5_pipeline.render.engine import _prepare_action16, _fig_to_rgb
from hdf5_pipeline.core.hdf5_utils import load_images_from_hdf5, load_actions_from_hdf5, load_joints_from_hdf5
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

hdf5_path = "./test_data/test1.hdf5" 
hdf5_str = str(Path(hdf5_path).resolve())

# 1. 加载真实数据
print("\n📂 加载 HDF5 真实数据:")
imgs = load_images_from_hdf5(hdf5_str)
cams = list(imgs.keys())
n_frames = min(v.shape[0] for v in imgs.values())
act = load_actions_from_hdf5(hdf5_str, n_frames)
left_j, right_j = load_joints_from_hdf5(hdf5_str, n_frames)
print(f"   相机: {cams}")
print(f"   总帧数: {n_frames}")
print(f"   action shape: {act.shape}")
print(f"   left_j: {left_j.shape if left_j is not None else 'None'}")
print(f"   right_j: {right_j.shape if right_j is not None else 'None'}")

# 2. 测试 _prepare_action16
print("\n📐 _prepare_action16 真实数据:")
act16 = _prepare_action16(act, n_frames)
print(f"   输入 action shape: {act.shape}")
print(f"   输出 act16 shape: {act16.shape}")
print(f"   前 3 帧:\n{act16[:3]}")
print(f"   值范围: [{act16.min():.4f}, {act16.max():.4f}]")

# 3. 测试 _fig_to_rgb — 用真实数据渲染一帧看效果
print("\n🖼️ _fig_to_rgb — 真实曲线渲染测试:")
fig, ax = plt.subplots(figsize=(8, 3), dpi=80)
x_full = np.arange(n_frames)
for i in range(16):
    ax.plot(x_full, act16[:, i], linewidth=1.0, label=f"a{i}")
ax.set_title("Action 16 Dimensions (真实数据)", fontsize=10)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=6, ncol=4)

fig.canvas.draw()
img = _fig_to_rgb(fig)
plt.close(fig)

print(f"   渲染图尺寸: {img.shape[1]}×{img.shape[0]} 像素")
print(f"   数据类型: {img.dtype}, 值范围: [{img.min()}, {img.max()}]")
print(f"   结论: 真实曲线→numpy 数组成功 ✓")

# ====== buffer_rgba 原理测试 ======
print("\n" + "=" * 60)
print("🔬 buffer_rgba + asarray 原理演示")

import matplotlib.pyplot as plt

# 画一个简单的图
fig, ax = plt.subplots(figsize=(2, 1), dpi=30)  # 小尺寸, 低dpi, 方便看
ax.plot([0, 1], [0, 1])  # 一条从 (0,0) 到 (1,1) 的线
ax.set_title("test")
fig.canvas.draw()

# ---- buffer_rgba() 是什么 ----
buf = fig.canvas.buffer_rgba()  # 这是 matplotlib 内部的内存缓冲区
print(f"\n1. buffer_rgba() 返回: {type(buf)}")
print(f"   它是一个内存视图对象, 不是 numpy 数组")

# ---- np.asarray() 做什么 ----
arr = np.asarray(buf)
print(f"\n2. np.asarray(buf) 返回: type={type(arr)}, shape={arr.shape}")
print(f'   它没有复制数据, 而是直接"借用"了 matplotlib 的内存')
print(f"   所以 arr 和 matplotlib 内部共享同一块内存")

# ---- 验证数据内容 ----
print(f"\n3. 数组内容分析:")
print(f"   dtype={arr.dtype}")  # uint8
print(f"   总像素数 = {arr.shape[0]}×{arr.shape[1]} = {arr.shape[0]*arr.shape[1]}")
print(f"   每个像素 = {arr.shape[2]} 通道 (RGBA)")
print(f"   值范围: [{arr.min()}, {arr.max()}]")

# ---- 取特定像素看含义 ----
print(f"\n4. 看几个像素 (左上角):")
print(f"   白底区域  [0,0]   = RGBA({arr[0,0,0]}, {arr[0,0,1]}, {arr[0,0,2]}, {arr[0,0,3]})")
h, w = arr.shape[:2]
print(f"   白底区域  [0,{w-1}] = RGBA({arr[0,w-1,0]}, {arr[0,w-1,1]}, {arr[0,w-1,2]}, {arr[0,w-1,3]})")
# 找一个有颜色的像素 (轴标签附近)
print(f"   轴线附近 [{h//2},{w//4}] = RGBA({arr[h//2,w//4,0]}, {arr[h//2,w//4,1]}, {arr[h//2,w//4,2]}, {arr[h//2,w//4,3]})")

# ---- 为什么取前3通道 ----
print(f"\n5. 为什么 _fig_to_rgb 要取 [:, :, :3]:")
print(f"   第4通道 (Alpha) 全是 {arr[-1,-1,3]}, 不需要, 所以丢掉")
print(f"   只保留 RGB → shape 从 {arr.shape} 变成 ({arr.shape[0]}, {arr.shape[1]}, 3)")

plt.close(fig)
print(f"\n✅ 原理: buffer_rgba → 内存指针 → asarray 零拷贝读取 → [:,:,:3] 去 Alpha")



# if __name__ == "__main__":
#     main()