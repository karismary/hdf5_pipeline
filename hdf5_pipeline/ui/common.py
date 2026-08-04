"""ui 标签页共享的 Streamlit 工具函数。"""

import streamlit as st
from hdf5_pipeline.core.utils import pick_folder

def folder_callback(target_state: str) -> None:
    """选择文件夹并将路径存入 session_state。

    Args:
        target_state (str): session_state 的键名。
    """
    selected_path = pick_folder()
    if selected_path:
        st.session_state[target_state] = selected_path