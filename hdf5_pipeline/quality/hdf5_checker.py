"""封装 — 委托给通用的 run_quality_check。

保留 ``run_hdf5_check(data_dir, out_csv, out_json, ...)``
签名以兼容旧代码，内部固定 ``format="hdf5"``。
"""
from functools import partial

from hdf5_pipeline.quality.checker import run_quality_check

run_hdf5_check = partial(run_quality_check, format="hdf5")

# 保留 inspect.signature 可读性
run_hdf5_check.__name__ = "run_hdf5_check"
run_hdf5_check.__doc__ = "HDF5 专用封装，委托给 run_quality_check(data_dir, format='hdf5', ...)"

__all__ = ["run_hdf5_check"]
