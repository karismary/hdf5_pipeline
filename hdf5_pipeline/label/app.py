import json
import streamlit as st
# from streamlit_file_browser import st_file_browser 
from pathlib import Path
import subprocess, platform
import sqlite3

from hdf5_pipeline.core.hdf5_utils import get_sorted_files
from hdf5_pipeline.label.database import init_db, get_unlabeled, add_label, scan_pairs
from hdf5_pipeline.core.config import load_config, save_config

st.set_page_config(page_title="HDF5 Labeling", layout="wide")

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 450px !important;
    }
</style>
""", unsafe_allow_html=True)

config = load_config()
path_dict = {
    "数据库路径(没有文件会在选择目录下自动创建)":"db_dir",
    "hdf5文件夹路径":"raw_dir",
    "mp4文件夹路径":"mp4_dir",
    "good_quality_hdf5存储路径":"good_dir",
    "bad_quality_hdf5存储路径":"bad_dir"
}

# def seclect_folder(path_type):
#     if st.session_state.get("_browser_for") != path_type :
#         return
#     folder = st_file_browser(
#                 key=f"browser_{path_type}",
#                 path="/",
#                 show_choose_file=True,
#                 show_choose_folder=True,
#                 show_delete_file=False,
#                 show_upload_file=False,
#             )
#     if folder:
#         f_path = folder["target"]["path"]
#         st.session_state[path_type] = f_path
#         config["paths"][path_type] = f_path
#         save_config(config)
#         st.session_state["_browser_for"] = None

def seclect_folder(path_type):
    s = platform.system()

    # 所有路径类型统一弹文件夹选择
    if s == "Darwin":
        cmd = ["osascript", "-e", 'POSIX path of (choose folder)']
    elif s == "Windows":
        cmd = ["powershell", "-Command",
               'Add-Type -AssemblyName System.Windows.Forms;'
               '$f=New-Object System.Windows.Forms.FolderBrowserDialog;'
               'if($f.ShowDialog()-eq"OK"){$f.SelectedPath}']
    else:
        cmd = ["zenity", "--file-selection", "--directory"]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return

    folder = r.stdout.strip()
    if path_type == "db_dir":
        db_files = list(Path(folder).glob("*.db"))
        db_path = db_files[0] if db_files else Path(folder) / "label.db"
        if not db_path.exists():
            init_db(str(db_path))
        folder = str(db_path)
    st.session_state[f"ui_{path_type}"] = folder
    config["paths"][path_type] = folder
    save_config(config)

    st.toast(f"已选择: {folder}", icon="📁")

st.title("HDF5 Labeling Tool")
st.write("这是一个用于给 HDF5 数据打标的工具，支持质量分类和属性的标注。")

#————侧边栏：设置工作路径，调用本地选择文件夹————
with st.sidebar:
    st.subheader("工作路径配置")
    for keys in path_dict:
        path = path_dict[keys]

        st.write(keys)
        col01,col02 = st.columns([4,1])
        with col01:
            st.text_input(keys, key=f"ui_{path}", label_visibility="collapsed", 
                          value=st.session_state.get(f"ui_{path}",config['paths'][path]))
        with col02:
            st.button("📁浏览", key=f"btn_{path}", on_click=seclect_folder, args=(path,))
    
    if st.button("扫描文件夹并同步数据库", key="btn_sync", width="stretch"):
        n = scan_pairs(
            st.session_state.get("ui_db_dir", config['paths']['db_dir']),
            st.session_state.get("ui_mp4_dir", config['paths']['mp4_dir']),
            st.session_state.get("ui_raw_dir", config['paths']['raw_dir'])
        )
        st.toast(f"新增 {n} 对文件", icon="✅")


#————主页面：————
tab1, tab2, tab3 = st.tabs(["视频打标","历史记录","⚙️配置"])

##————标签页1:打标界面————
with tab1:
    st.subheader("开始打标记录")
    col11, col12 = st.columns([1,3])
    # with col11:
    #     mp4_dir = Path(config["paths"]["mp4_dir"])
    #     raw_dir = Path(config["paths"]["raw_dir"])
    #     db_path = Path(config["paths"]["db_dir"])
    #     mp4_list = get_sorted_files(mp4_dir,".mp4")
    #     raw_list = get_sorted_files(raw_dir,".hdf5")

    #     # for 

    #     conn = sqlite3.connect(db_path)
    #     cursor = conn.cursor()

    #     st.button

    #     # for i,mp4_name in enumerate(mp4_dir):


    #     st.write("list")
    # with col12:
    #     st.write("video")
    #     st.write("graphs")
    #     st.write("progressing...")
##————标签页2:打标历史记录操作界面————
with tab2:
    st.write("历史记录")

##————标签页3:配置界面————
with tab3:
    st.write("配置")
