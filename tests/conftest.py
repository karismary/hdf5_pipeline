"""pytest 共享 fixtures。

数据来源：
- HDF5 链路用项目内 test_data 下的真实样本（复制到 tmp_path，不污染源数据）。
- spirit 专项用本地真实实例（环境变量 SPIRIT_RAW_DIR 指定，只读，
  转换/质检输出一律进 tmp_path）。

数据缺失或环境变量未设置时对应测试自动 skip（pytest.skip），不会误报失败。
"""

import os
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOOD_QUALITY_DIR = PROJECT_ROOT / "test_data" / "good_quality"
# spirit 真实实例目录不硬编码在代码里，由环境变量注入（本地或 CI 配置）
SPIRIT_RAW_DIR = os.environ.get("SPIRIT_RAW_DIR")


def _sample_hdf5() -> Path:
    p = GOOD_QUALITY_DIR / "episode_000006.hdf5"
    if not p.exists():
        pytest.skip("缺少本地 HDF5 测试样本（test_data 下）")
    return p


@pytest.fixture
def sample_hdf5(tmp_path: Path) -> Path:
    """真实 HDF5 样本的副本（30 帧 / 双相机），返回临时路径。"""
    dst = tmp_path / "episode_000006.hdf5"
    shutil.copy2(_sample_hdf5(), dst)
    return dst


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """含嵌套子目录、文件名乱序的原始数据目录（3 个 HDF5）。"""
    src = _sample_hdf5()
    raw = tmp_path / "raw"
    (raw / "session_1").mkdir(parents=True)
    (raw / "session_2" / "nested").mkdir(parents=True)
    shutil.copy2(src, raw / "session_1" / "episode_000010.hdf5")
    shutil.copy2(src, raw / "session_1" / "episode_000001.hdf5")
    shutil.copy2(src, raw / "session_2" / "nested" / "episode_000002.hdf5")
    return raw


@pytest.fixture(scope="session")
def spirit_raw() -> Path:
    """spirit 真实实例目录（环境变量 SPIRIT_RAW_DIR 指定）。"""
    if not SPIRIT_RAW_DIR:
        pytest.skip("未设置环境变量 SPIRIT_RAW_DIR，跳过 spirit 测试")
    inst = Path(SPIRIT_RAW_DIR)
    if not inst.exists():
        pytest.skip("SPIRIT_RAW_DIR 指向的目录不存在，跳过 spirit 测试")
    return inst