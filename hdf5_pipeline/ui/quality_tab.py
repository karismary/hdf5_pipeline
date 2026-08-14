"""🧹 质量检测 — 对 HDF5 文件进行异常帧检测。"""

import streamlit as st
from pathlib import Path
from hdf5_pipeline.ui.common import folder_callback, KEY_QUALITY
from hdf5_pipeline.quality.checker import run_quality_check

def show_tab_quality() -> None:
    """🧹 质量检测标签页主入口。

    支持 HDF5 和 LeRobot Parquet 两种格式的异常帧检测。
    选择源目录和数据格式，设置严格度，执行检测后显示统计摘要。
    """
    st.subheader("数据质量检测")
    with st.expander("**路径设置**", key = f"{KEY_QUALITY}_path_set", expanded = True):
        colqu1_1, colqu1_2 = st.columns([4,1])
        with colqu1_1:
            raw_folder = st.text_input("数据目录",
                                    key = f"{KEY_QUALITY}_src_dir_ti",
                                    placeholder = "包含所有数据的文件夹目录（可选.hdf5或.parque格式，在下方选择栏选择数据格式）",
                                    label_visibility = "collapsed")
            csv_out_folder = st.text_input("csv文档导出目录",
                                    key = f"{KEY_QUALITY}_csv_dir_ti",
                                    placeholder = "质量评价文档（.csv格式）目录",
                                    label_visibility = "collapsed",
                                    help = "不选择则在默认路径下新建csv文档")
            json_out_folder = st.text_input("json文档导出目录",
                                    key = f"{KEY_QUALITY}_json_dir_ti",
                                    placeholder = "质量评价文档（.json格式）目录",
                                    label_visibility = "collapsed",
                                    help = "不选择则在默认路径下新建json文档")
        
        with colqu1_2:
            st.button("📂浏览", key = f"{KEY_QUALITY}_srcdir_bt", width = "stretch", on_click = folder_callback, args = (f"{KEY_QUALITY}_src_dir_ti", ))
            st.button("📂浏览", key = f"{KEY_QUALITY}_csvdir_bt", width = "stretch", on_click = folder_callback, args = (f"{KEY_QUALITY}_csv_dir_ti", ))
            st.button("📂浏览", key = f"{KEY_QUALITY}_jsondir_bt", width = "stretch", on_click = folder_callback, args = (f"{KEY_QUALITY}_json_dir_ti", ))

    colqu2_1, colqu2_2 = st.columns([2,3])
    with colqu2_1:
        with st.container(key = f"{KEY_QUALITY}_container2_1", border = True):
            st.markdown("**检查设置（必选）**")
            st.selectbox("数据检查严格度", key = f"{KEY_QUALITY}_strict_sb", options = ["loose", "medium", "strict"])
            st.selectbox("数据格式",key = f"{KEY_QUALITY}_format_sb", options = ["HDF5", "LeRobot Parquet"])
            
            disabled=not (st.session_state.get(f"{KEY_QUALITY}_strict_sb") and st.session_state.get(f"{KEY_QUALITY}_format_sb"))
            if st.button("执行检测", key = f"{KEY_QUALITY}_run_detector", width = "stretch", disabled = disabled):
                if Path(raw_folder).exists():
                    src = st.session_state.get(f"{KEY_QUALITY}_src_dir_ti", "")
                    csv_path = st.session_state.get(f"{KEY_QUALITY}_csv_dir_ti", "")
                    json_path = st.session_state.get(f"{KEY_QUALITY}_json_dir_ti", "")

                    if csv_path and Path(csv_path).is_dir():
                        csv_path = str(Path(csv_path) / "outlier_frames.csv")
                    if json_path and Path(json_path).is_dir():
                        json_path = str(Path(json_path) / "outlier_summary.json")

                    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

                    strictness = str(st.session_state.get(f"{KEY_QUALITY}_strict_sb"))
                    file_format = str(st.session_state.get(f"{KEY_QUALITY}_format_sb"))

                    with st.spinner("检测中......"):
                        if not src:
                            st.warning("请先选择数据目录")
                        elif file_format == "HDF5":
                            st.session_state["q_summary"] = run_quality_check(
                                src, "hdf5", csv_path, json_path, strictness=strictness)
                            st.success("检测完成")
                        else:
                            st.session_state["q_summary"] = run_quality_check(
                                src, "lerobot", csv_path, json_path, strictness=strictness)
                            st.success("检测完成")
                        if not Path(csv_path).exists():
                            st.toast(f"已导出: {json_path},不存在异常帧")
                            st.session_state.update(check_result = f"已导出: {json_path},不存在异常帧")
                        else:
                            st.toast(f"已导出: {csv_path} / {json_path}")
                            st.session_state.update(check_result = f"已导出: {csv_path} / {json_path}")

                
    with colqu2_2:
        with st.container(key = f"{KEY_QUALITY}_container2_2", border = True):
            if "q_summary" in st.session_state:
                s = st.session_state["q_summary"]
                st.metric("扫描文件数", s.get("num_files", 0))
                st.metric("异常帧数", s.get("num_outliers", 0))
                st.metric("总帧数", s.get("num_frames", 0))

                st.markdown("**异常最多的维度：**")
                for d in s.get("top_dims", []):
                    st.write(f"  维度 {d['dim']}: {d['count']} 帧")

                st.markdown("**异常最多的文件：**")
                for ep in s.get("top_episodes", [])[:5]:
                    st.write(f"  episode_{ep['episode']:06d}: {ep['count']} 帧")

                st.markdown(f"_**{st.session_state.get('check_result')}**_")

            else:
                st.info("检测结果将在此处显示")