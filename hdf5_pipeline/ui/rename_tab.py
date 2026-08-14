"""文件重命名 — 将原始 HDF5 统一命名为 episode_NNNNNN 格式。"""

import streamlit as st
from pathlib import Path
from hdf5_pipeline.ui.common import folder_callback, KEY_RENAME
from hdf5_pipeline.rename.engine import collect_hdf5_files, rename_files

def show_tab_rename() -> None:
    """文件重命名标签页主入口。

    扫描源目录下的所有 HDF5 文件，统一重命名为 episode_NNNNNN 格式
    并复制到输出目录。完成后可锁定界面防止误操作。
    """

    if st.session_state.get("locked"):
        with st.container(key = f"{KEY_RENAME}_tip", border = True, horizontal_alignment = "center"):
            st.subheader("该界面已被锁定")
            st.caption("解锁后即可重新扫描与重命名。")
            if st.button("取消锁定", key = f"{KEY_RENAME}_unlock", icon = ":material/lock_open:"):
                st.session_state["locked"] = False
                st.rerun()
    else:
        st.subheader("文件整理与重命名", divider = True)
        st.caption("扫描源目录中的 HDF5 文件，统一重命名并汇总到输出目录。")
        with st.container(key = f"{KEY_RENAME}_container", border = True):
            st.markdown("**路径设置**")

            colr1, colr2 = st.columns([4, 1], vertical_alignment = "center")
            with colr1:
                src_dir = st.text_input("源目录",
                                        key = f"{KEY_RENAME}_src_dir_ti",
                                        placeholder = "未选择 - hdf5源目录（含子目录的原始 HDF5 文件夹）",
                                        label_visibility = "collapsed")
            with colr2:
                st.button("浏览", key = f"{KEY_RENAME}_srcdir_bt", width = "content",
                          on_click = folder_callback, args = (f"{KEY_RENAME}_src_dir_ti", ),
                          icon = ":material/folder_open:")

            colr3, colr4 = st.columns([4, 1], vertical_alignment = "center")
            with colr3:
                dst_dir = st.text_input("输出总目录",
                                        key = f"{KEY_RENAME}_dst_dir_ti",
                                        placeholder = "未选择 - 输出总目录（将hdf5文件重命名汇总到统一文件夹下）",
                                        label_visibility = "collapsed")
            with colr4:
                st.button("浏览", key = f"{KEY_RENAME}_dstdir_bt", width = "content",
                          on_click = folder_callback, args = (f"{KEY_RENAME}_dst_dir_ti", ),
                          icon = ":material/folder_open:")

            if st.button("扫描文件", width = "stretch", icon = ":material/search:"):
                if not src_dir or not Path(src_dir).exists():
                    st.warning("请先选择有效的源目录")
                else:
                    files = collect_hdf5_files(src_dir)
                    if files:
                        st.success(f"发现 {len(files)} 个 HDF5 文件")
                        for f in files[:10]:
                            st.markdown(f"- `{f.name}`")
                        if len(files) > 10:
                            st.markdown(f"- ...还有 {len(files)-10} 个文件")
                        st.session_state["rename_files"] = files
                    else:
                        st.warning("未找到 HDF5 文件")

            if "rename_files" in st.session_state and st.button("执行重命名", type = "primary", width = "stretch", icon = ":material/drive_file_move:"):
                n = rename_files(st.session_state["rename_files"], dst_dir)
                st.success(f"成功重命名 {n} 个文件到 {dst_dir}")

            if st.button("锁定界面", key = f"{KEY_RENAME}_lock", width = "stretch", icon = ":material/lock:"):
                st.session_state["locked"] = True
                st.rerun()
