import streamlit as st
from hdf5_pipeline.ui.common import KEY_OVERVIEW

# —— tab4 记录缓存（打标工作区）——
S_RECORDS = "records"                # tab4 读/写
S_SELECTED = "selected"              # tab4 读，quality/attrs 写
S_SELECTED_INDEX = "selected_index"
S_MARKED_RED = "_marked_red"         # tab4 标红：被「不要」的 mp4_name 集合

# —— tab5 分页局部（前缀与 widget 键一致，改 KEY_OVERVIEW 一处全变）——
S_OV_RECORDS = f"{KEY_OVERVIEW}_records"
S_OV_PAGE    = f"{KEY_OVERVIEW}_page"
S_OV_WHERE   = f"{KEY_OVERVIEW}_where"

# —— 跨 tab 数据版本（广播信号）——
S_DB_VERSION = "_db_version"
S_TAB4_VERSION = "_tab4_db_version"
S_TAB5_VERSION = "_tab5_db_version"

# —— 一次性消息 ——
S_TOAST = "_toast_msg"
S_DB_WARNED = "_db_warned"        # 数据库不可用时的弹窗去重标记

def get(key, default=None): 
    return st.session_state.get(key, default)
def set(key, value):
    st.session_state[key] = value
def pop(key, default=None):
    return st.session_state.pop(key, default)


def init_state():
    st.session_state.setdefault(S_SELECTED_INDEX, 0)
    st.session_state.setdefault(S_OV_PAGE, 0)
    st.session_state.setdefault(S_OV_WHERE, None)
    st.session_state.setdefault(S_DB_VERSION, 0)
    st.session_state.setdefault(S_TAB4_VERSION, 0)
    st.session_state.setdefault(S_TAB5_VERSION, 0)
    st.session_state.setdefault(S_MARKED_RED, frozenset())

def bump_db_version():      # 任何对 label 表的写操作后调用
    set(S_DB_VERSION, get(S_DB_VERSION, 0) + 1)
