"""
常量定义：关节名称、维度映射、mask 配置。

原始 HDF5 30维布局:
  [0:7]   left_end_effector_pose   (训练不使用)
  [7:14]  right_end_effector_pose  (训练不使用)
  [14:21] left_arm_joints          (训练使用)
  [21:28] right_arm_joints         (训练使用)
  [28]    left_gripper             (训练使用)
  [29]    right_gripper            (训练使用)

训练空间 16维布局:
  [0:7]   left_joints
  [7]     left_gripper
  [8:15]  right_joints
  [15]    right_gripper
"""

from typing import List, Tuple

# --- 关节名称 ---
JOINT_NAMES: List[str] = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow_pitch", "wrist_yaw", "wrist_pitch", "wrist_roll",
]

# --- 30维原始布局的索引范围 ---
DIM_LEFT_EE: Tuple[int, int] = (0, 7)       # 左末端位姿 [0:7]
DIM_RIGHT_EE: Tuple[int, int] = (7, 14)     # 右末端位姿 [7:14]
DIM_LEFT_JOINTS: Tuple[int, int] = (14, 21)  # 左臂关节 [14:21]
DIM_RIGHT_JOINTS: Tuple[int, int] = (21, 28) # 右臂关节 [21:28]
DIM_LEFT_GRIPPER: int = 28                   # 左夹爪
DIM_RIGHT_GRIPPER: int = 29                  # 右夹爪

# --- 30→16 投影说明 ---
RAW_30_TO_TRAINING_16_MAP: List[str] = [
    "0-6:left_joints",
    "7:left_gripper",
    "8-14:right_joints",
    "15:right_gripper",
]

# --- 默认 Delta 掩码 (16维) ---
# 关节维度=True (计算 delta=action-state)，夹爪维度=False (保留 action 原值)
DEFAULT_DELTA_MASK_16: List[bool] = [
    True, True, True, True, True, True, True,   # 左臂7关节
    False,                                        # 左夹爪
    True, True, True, True, True, True, True,   # 右臂7关节
    False,                                        # 右夹爪
]

# --- 预设严格度 (异常检测) ---
STRICTNESS_PRESETS = {
    "loose":  {"min_score": 15.0, "top_k_per_episode": 40, "top_k_global": 5000, "min_denom": None},
    "medium": {"min_score": 40.0, "top_k_per_episode": 20, "top_k_global": 2500, "min_denom": 1e-4},
    "strict": {"min_score": 80.0, "top_k_per_episode": 10, "top_k_global": 1000, "min_denom": 5e-5},
}