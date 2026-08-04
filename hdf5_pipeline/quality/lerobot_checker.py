"""封装 — 委托给通用的 run_quality_check。

保留 ``run_lerobot_check(data_dir, out_csv, out_json, ...)``
签名以兼容旧代码，内部固定 ``format="lerobot"``。
"""
from functools import partial

from hdf5_pipeline.quality.checker import run_quality_check

run_lerobot_check = partial(run_quality_check, format="lerobot")

run_lerobot_check.__name__ = "run_lerobot_check"
run_lerobot_check.__doc__ = "LeRobot 专用封装，委托给 run_quality_check(data_dir, format='lerobot', ...)"

__all__ = ["run_lerobot_check"]
