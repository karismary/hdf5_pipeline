"""ui 标签页共享的 Streamlit 工具函数。"""

import streamlit as st
from hdf5_pipeline.core.utils import pick_folder

# ===== Widget key 前缀约定 =====
# 每个标签页一个语义前缀，保证跨页唯一、可读。
# 规则：tab + 功能缩写。新写 widget key 时必须用对应前缀。
KEY_RENAME   = "tabre"   # 📁 文件重命名（ui/rename_tab）
KEY_QUALITY  = "tabqu"   # 🧹 质量检测（ui/quality_tab）
KEY_RENDER   = "tabrd"   # 🎬 视频渲染（ui/render_tab）
KEY_LABEL    = "tabla"   # 🏷️ 视频打标（label/app.py tab4）
KEY_OVERVIEW = "tabov"   # 📊 数据总览（label/app.py tab5）
KEY_CONFIG   = "tabcf"   # ⚙️ 配置（label/app.py tab6）

def folder_callback(target_state: str) -> None:
    """选择文件夹并将路径存入 session_state。

    Args:
        target_state (str): session_state 的键名。
    """
    selected_path = pick_folder()
    if selected_path:
        st.session_state[target_state] = selected_path