"""文件重命名 — 将原始 HDF5 统一命名为 episode_NNNNNN 格式。"""

import streamlit as st
from pathlib import Path
from hdf5_pipeline.ui.common import folder_callback, KEY_RENAME
from hdf5_pipeline.rename.engine import collect_hdf5_files, rename_files

def show_tab_rename() -> None:
    """📁 文件重命名标签页主入口。

    扫描源目录下的所有 HDF5 文件，统一重命名为 episode_NNNNNN 格式
    并复制到输出目录。完成后可锁定界面防止误操作。
    """

    if st.session_state.get("locked"):
        with st.container(key = f"{KEY_RENAME}_tip", border = True, horizontal_alignment = "center"):
            st.subheader("该界面已被锁定")
            if st.button("取消锁定", key = f"{KEY_RENAME}_unlock"):
                st.session_state["locked"] = False
                st.rerun()
    else:
        st.subheader("文件整理与重命名")
        with st.container(key = f"{KEY_RENAME}_container", border = True):

            colre1, colre2 = st.columns([4,1])
            with colre1:
                src_dir = st.text_input("源目录",
                                        key = f"{KEY_RENAME}_src_dir_ti",
                                        placeholder = "未选择 - hdf5源目录（含子目录的原始 HDF5 文件夹）",
                                        label_visibility = "collapsed")
                dst_dir = st.text_input("输出总目录",
                                        key = f"{KEY_RENAME}_dst_dir_ti",
                                        placeholder = "未选择 - 输出总目录（将hdf5文件重命名汇总到统一文件夹下）",
                                        label_visibility = "collapsed")
            
            with colre2:
                st.button("📂浏览", key = f"{KEY_RENAME}_srcdir_bt", width = "stretch", on_click = folder_callback, args = (f"{KEY_RENAME}_src_dir_ti", ))
                st.button("📂浏览", key = f"{KEY_RENAME}_dstdir_bt", width = "stretch", on_click = folder_callback, args = (f"{KEY_RENAME}_dst_dir_ti", ))

            if st.button("🔍 扫描文件", use_container_width=True):
                if not src_dir or not Path(src_dir).exists():
                    st.warning("请先选择有效的源目录")
                else:
                    files = collect_hdf5_files(src_dir)
                    if files:
                        st.success(f"发现 {len(files)} 个 HDF5 文件")
                        for f in files[:10]:
                            st.write(f"  {f.name}")
                        if len(files) > 10:
                            st.write(f"  ...还有 {len(files)-10} 个文件")
                        st.session_state["rename_files"] = files
                    else:
                        st.warning("未找到 HDF5 文件")
            
            if "rename_files" in st.session_state and st.button("执行重命名", type="primary", use_container_width=True):
                n = rename_files(st.session_state["rename_files"], dst_dir)
                st.success(f"成功重命名 {n} 个文件到 {dst_dir}")

            if st.button("锁定界面", key = f"{KEY_RENAME}_lock", width = "stretch"):
                st.session_state["locked"] = True
                st.rerun()