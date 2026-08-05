import h5py
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

def validate_file(file_path: str) -> tuple[bool, list[str]]:
    """校验单个数据文件（HDF5 / Parquet）的结构完整性。

    Args:
        file_path (str): 数据文件路径。

    Returns:
        tuple: (ok, errors)
            ok — 文件是否通过校验。
            errors — 错误信息列表，为空表示通过。
    """
    errors = []
    ok = True

    file_format = Path(file_path).suffix
    if not Path(file_path).exists():
        errors.append(f"文件不存在：{file_path}\n")
        ok = False
        return ok,errors
    else:
        try:
            if file_format == ".hdf5" or file_format == ".h5":
                with h5py.File(Path(file_path), "r") as f:
                    if not "observations/pixels" in f and not "pixels" in f:
                        errors.append(f"{file_path}：图像组不存在\n")
                        ok = False
                    else:
                        pix = "observations/pixels" if "observations/pixels" in f else "pixels"
                        cams = list(f[pix].keys())
                        if not cams:
                            errors.append(f"{file_path}：图像组为空\n")
                            ok = False
                        for cam in cams:
                            n_frames = f[f"{pix}/{cam}"].shape[0]
                            if n_frames < 1:
                                errors.append(f"{file_path}：未采集到数据,帧数为 0\n")
                                ok = False
                                break
                    if not "action" in f and not "actions" in f:
                        errors.append(f"{file_path}：数据组不存在\n")
            if file_format == ".parquet":
                with pq.ParquetFile(Path(file_path)) as pf:
                    column_names = pf.schema_arrow.names
                    n_rows = pf.metadata.num_rows
                    required = {"action", "observation.state", "frame_index"}
                    if not required.issubset(column_names):
                        errors.append(f"{file_path}：数据组不存在\n")
                        ok = False
                    if n_rows < 1:
                        errors.append(f"{file_path}：未采集到数据,帧数为 0")
                        ok = False

        except Exception as e:
            errors.append(f"{file_path}：路径或文件存在问题:报错{e}\n")
            ok = False
        return ok, errors
