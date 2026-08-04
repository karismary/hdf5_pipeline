import argparse, importlib
import streamlit as st

DEFAULT_MODULE = "render"
MODULES = {
    "rename":  {"title": "文件重命名", "name": "rename"},
    "quality": {"title": "质量检测",   "name": "quality"},
    "render":  {"title": "视频渲染",   "name": "render"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--module", default = DEFAULT_MODULE, choices = list(MODULES.keys()))
args, _ = parser.parse_known_args()
module_name = args.module
entry = MODULES[module_name]

mod = importlib.import_module(f"hdf5_pipeline.ui.{entry.get('name', 'render')}_tab")
show = getattr(mod, f"show_tab_{entry.get('name', 'render')}")

st.set_page_config(page_title=entry["title"], layout="wide")
show()