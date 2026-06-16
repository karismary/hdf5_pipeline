"""生成 test1.hdf5 测试文件，模拟机器人遥操作数据"""
import h5py
import numpy as np
from pathlib import Path

T = 10          # 帧数
H, W, C = 480, 640, 3  # 图像尺寸

h5_path = Path(__file__).parent / "test_data" / "test1.hdf5"
h5_path.parent.mkdir(exist_ok=True)

# 随机种子固定，每次生成一样的数据，便于调试
rng = np.random.RandomState(42)

with h5py.File(h5_path, "w") as f:
    # ── action (30维) ──
    # 前7: left_ee_pose, 7-14: right_ee_pose, 14-21: left_joints, 21-28: right_joints, 28: left_grip, 29: right_grip
    act = rng.uniform(-1, 1, (T, 30)).astype(np.float64)
    f.create_dataset("action", data=act)
    f.create_dataset("actions", data=act)  # fallback

    # ── timestamps ──
    ts = np.arange(T, dtype=np.float64) * 0.05  # 50ms 一帧 (20fps)
    f.create_dataset("timestamps", data=ts)

    # ── observations/pixels (2个相机) ──
    pix = f.create_group("observations/pixels")
    # cam_1: 随机彩色噪声 (看起来像画面)
    cam1 = rng.randint(0, 256, (T, H, W, C), dtype=np.uint8)
    pix.create_dataset("cam_1", data=cam1, compression="gzip")
    # cam_2: 黑白渐变模拟
    grad = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    cam2 = np.stack([grad] * 3, axis=-1)  # (H, W, 3)
    cam2 = np.tile(cam2[np.newaxis, ...], (T, 1, 1, 1))  # (T, H, W, 3)
    pix.create_dataset("cam_2", data=cam2, compression="gzip")

    # ── observations/joints + grippers + ee_pose ──
    obs = f["observations"]
    obs.create_dataset("left_arm_joints",        data=rng.uniform(-1, 1, (T, 7)).astype(np.float64))
    obs.create_dataset("right_arm_joints",       data=rng.uniform(-1, 1, (T, 7)).astype(np.float64))
    obs.create_dataset("left_end_effector_pose", data=rng.uniform(-1, 1, (T, 7)).astype(np.float64))
    obs.create_dataset("right_end_effector_pose",data=rng.uniform(-1, 1, (T, 7)).astype(np.float64))
    obs.create_dataset("left_gripper_state",     data=rng.uniform(0, 1, (T, 1)).astype(np.float64))
    obs.create_dataset("right_gripper_state",    data=rng.uniform(0, 1, (T, 1)).astype(np.float64))

# 验证
with h5py.File(h5_path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:45s} shape={obj.shape!s:20s} dtype={obj.dtype}")
    print(f"✅ 生成: {h5_path}")
    f.visititems(show)
