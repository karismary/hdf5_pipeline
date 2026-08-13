from typing import List

MOTORS = [
    "leftarm_joint0",
    "leftarm_joint1",
    "leftarm_joint2",
    "leftarm_joint3",
    "leftarm_joint4",
    "leftarm_joint5",
    "leftarm_joint6",
    "leftarm_gripper",
    "rightarm_joint0",
    "rightarm_joint1",
    "rightarm_joint2",
    "rightarm_joint3",
    "rightarm_joint4",
    "rightarm_joint5",
    "rightarm_joint6",
    "rightarm_gripper",
    "torso_joint0",
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
    "torso_joint5",
]

CAMERAS = [
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]

SPIRIT_ACTION_COLS = [
    "leftarm_cmd_joint_pos", "leftarm_gripper_cmd_pos",
    "rightarm_cmd_joint_pos", "rightarm_gripper_cmd_pos",
    "torso_cmd_joint_pos",
]
SPIRIT_STATE_COLS = [c.replace("_cmd_", "_state_") for c in SPIRIT_ACTION_COLS]

#左臂7×True, 左爪 False, 右臂7×True, 右爪 False, 腰6×True
DEFAULT_DELTA_MASK_22: List[bool] = [
    True, True, True, True, True, True, True,
    False,
    True, True, True, True, True, True, True,
    False,
    True, True, True, True, True, True
]