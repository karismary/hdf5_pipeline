"""统一配置管理。"""

import json
from pathlib import Path
from typing import Dict, Any


# ---- 第一步：定义默认值 ----

DEFAULT_PATHS = {
    "db_dir":   "./data_workspace",
    "raw_dir":  "./data_workspace/1_raw_hdf5",
    "mp4_dir":  "./data_workspace/2_input_mp4",
    "good_dir": "./data_workspace/3_good_quality",
    "bad_dir":  "./data_workspace/4_bad_quality",
}


DEFAULT_CUSTOM_COLS: Dict[str, Any] = {}
DEFAULT_CONFIG_PATH = Path("./config.json")



def get_default_config() -> Dict[str, Any]:
    """
    返回一份全新的默认配置副本。

    → 防止调用方修改返回值时污染全局常量。
    """
    return {
        "paths": dict(DEFAULT_PATHS),
        "custom_cols": dict(DEFAULT_CUSTOM_COLS),
    }

# ---- 第三步：load_config ----

def load_config(config_path: Path = None):
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config = get_default_config()

    if not config_path.exists():
        return config
    
    try:
        with open(config_path,'r',encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return config
    
    if "paths" not in data:
        data = {"paths": DEFAULT_PATHS, "custom_cols": data}

    config["paths"].update(data.get("paths", {}))

    return config   


# ---- 第四步：save_config ----

def save_config(config_data: Dict[str, Any], config_path: Path = None) -> None:
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)
