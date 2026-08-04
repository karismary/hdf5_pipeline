"""🎬 视频渲染 — 将 HDF5 文件渲染为 MP4 视频。"""

import streamlit as st
from pathlib import Path
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, CancelledError
from multiprocessing import Manager
from hdf5_pipeline.core.config import load_config
from hdf5_pipeline.ui.common import folder_callback
from hdf5_pipeline.core.hdf5_utils import get_sorted_files, get_hdf5_frame_count
from hdf5_pipeline.render.engine import render_mp4

RENDER_LOG = Path("./_render_log.txt")

CHECK_INTERVAL = 5
PER_FRAME_BUDGET = 0.3
BASE_OVERHEAD = 60
MIN_TIMEOUT = 60
MAX_TIMEOUT = 3600

def _render_deadlines(h5_file, format):
    n = get_hdf5_frame_count(h5_file, format)
    if n is None:
        budget = MAX_TIMEOUT
    else:
        budget = n * PER_FRAME_BUDGET + BASE_OVERHEAD
    budget = min(max(budget, MIN_TIMEOUT), MAX_TIMEOUT)
    return time.time() + budget

@st.fragment(run_every=2)
def render_status() -> None:
    """渲染进度面板，每 2 秒自动刷新。

    显示渲染日志、进度条和调试信息。
    渲染完成后自动停止刷新。
    """
    with st.container(key="tabrd_log", border=True):
        col_t, col_d = st.columns([3, 1])
        with col_t:
            st.markdown("**📋 渲染日志**")
        with col_d:
            st.checkbox("调试", key="rd_debug", value=False)

        if RENDER_LOG.exists():
            raw = RENDER_LOG.read_text().strip()
            done = max(0, raw.count("\n"))
        else:
            st.info("日志文件不存在，可能还未开始渲染")

        if st.session_state.get("rd_debug"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("并发数", st.session_state.get("tabrd_sub_progress_cb", 2))
            c2.metric("总文件", st.session_state.get("_render_total", 0))
            c3.metric("已完成", done if st.session_state.get("_rendering") else "-")
            c4.metric("日志行", len(raw.split(chr(10))) if RENDER_LOG.exists() else 0)

        if st.session_state.get("_rendering") and RENDER_LOG.exists():
            total = st.session_state.get("_render_total", 0)
            if total:
                st.progress(min(done / total, 1.0), text=f"进度: {done}/{total}")
            st.text_area("日志", value = raw[-3000:], height = 150, disabled = True, label_visibility = "collapsed")

        if not st.session_state.get("_rendering") and not RENDER_LOG.exists():
            st.info("点击「开始转换」启动渲染")


def render_all(files, src, out, show_image, show_action, action_on, left_on, right_on, skip_existed, stop_event, n_workers) -> None:
    """后台线程：多进程渲染 HDF5 文件为 MP4。

    使用 ProcessPoolExecutor 并发执行，每处理一个文件写一条日志。
    支持通过 stop_event 终止。

    Args:
        files: HDF5 文件名列表。
        src: 源目录路径。
        out: 输出目录路径。
        show_image: 是否显示相机图像。
        show_action: 是否显示动作曲线。
        action_on: 16 维动作维度布尔列表。
        left_on / right_on: 7 维关节布尔列表。
        skip_existed: 是否跳过已生成的 MP4。
        stop_event: 跨进程终止信号（Manager().Event()）。
        n_workers: 并发进程数。
    """
    RENDER_LOG.write_text("")
    def write_log(content):
        with open(RENDER_LOG, "a") as f:
            f.write(content)
    success = 0
    fail = 0
    skip = 0

    with ProcessPoolExecutor(max_workers = n_workers) as executor:
        pending = {}
        for fname in files:
            if stop_event.is_set():
                executor.shutdown(wait = False, cancel_futures = True)
                write_log(f"终止：{fname}\n")
                break
            mp4 = f"{out}/{Path(fname).stem}.mp4"
            if skip_existed and Path(mp4).exists():
                write_log(f"跳过：{fname}\n")
                skip += 1
                continue
            deadline = _render_deadlines(Path(src)/fname, "hdf5")
            future = executor.submit(render_mp4, Path(src)/fname, Path(mp4),
                                    show_image, show_action, action_on, left_on, right_on, stop_event)
            pending[future] = (fname, deadline)
        while pending:
            if stop_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                write_log("终止\n")
                break
            done, not_done = wait(pending, CHECK_INTERVAL, FIRST_COMPLETED)
            for future in done:
                fname, _ = pending.pop(future)
                try:
                    ok, msg, _ = future.result()
                    if ok:
                        write_log(f"成功：{fname}\n")
                        success += 1
                    else:
                        write_log(f"失败：{fname} | {msg}\n")
                        fail += 1
                except CancelledError:
                    pass
                except Exception:
                    write_log(f"失败：{fname}\n")
                    fail += 1

            now = time.time()
            for future in not_done:
                fname, deadline = pending[future]
                if now > deadline:
                    pending.pop(future)
                    write_log(f"超时：{fname}\n")
                    fail += 1

def show_tab_render() -> None:
    """🎬 视频渲染标签页主入口。

    提供路径选择、渲染选项配置、多进程并发渲染和日志监控。
    """
    config = load_config()
    st.subheader("视频渲染")
    with st.expander("**路径设置**", key = "tabrd_path_set", expanded = True):
        colqu1_1, colqu1_2 = st.columns([4,1])
        with colqu1_1:
            st.text_input("数据目录（HDF5）", key = "tabrd_src_dir_ti", placeholder = "hdf5原始数据的文件夹目录", label_visibility = "collapsed")
            st.text_input("视频导出目录（MP4）", key = "tabrd_mp4_dir_ti", placeholder = "导出hdf5转换mp4视频文件的文件夹目录", label_visibility = "collapsed")
        
        with colqu1_2:
            st.button("📂浏览", key = "tabrd_src_dir_bt", width = "stretch", on_click = folder_callback, args = ("tabrd_src_dir_ti", ))
            st.button("📂浏览", key = "tabrd_mp4_dir_bt", width = "stretch", on_click = folder_callback, args = ("tabrd_mp4_dir_ti", ))

    colrd2_1, colrd2_2 = st.columns([1,1])
    with colrd2_1:
        with st.container(key = "tabrd_container1", border = True):
            st.markdown("**图像渲染选项**")
            colrd2_1_1, colrd2_1_2= st.columns(2)
            with colrd2_1_1:
                st.checkbox("显示顶端图像", key = "tabrd_show_image_cb", value = True)
            with colrd2_1_2:
                st.checkbox("显示动作曲线", key = "tabrd_show_action_cb", value = True)

            with st.expander("动作维度（16维）", key = "tabrd_action_dim_ep", expanded = False):
                st.session_state["action_on"] = []
                action_cols = st.columns(8)
                for i in range(16):
                    with action_cols[i % 8]:
                        st.session_state["action_on"].append(st.checkbox(f"a{i}", key = f"action_check_a{i}", value = True))

            left_right_dict = {0:"should", 1:"should", 2:"should", 3:"elbow_", 4:"wrist_", 5:"wrist_", 6:"wrist_", }

            with st.expander("左机械臂关节（7）", key = "tabrd_left_dim_ep", expanded = False):
                st.session_state.update(left_on = [])
                left_cols = st.columns(7)
                for i in range(7):
                    with left_cols[i]:
                        st.session_state["left_on"].append(st.checkbox(left_right_dict.get(i), key = f"lefton_check_a{i}", value = True))

            with st.expander("右机械臂关节（7）", key = "tabrd_right_dim_ep", expanded = False):
                st.session_state.update(right_on = [])
                right_cols = st.columns(7)
                for i in range(7):
                    with right_cols[i]:
                        st.session_state["right_on"].append(st.checkbox(left_right_dict.get(i), key = f"righton_check_a{i}", value = True))

    with colrd2_2:
        with st.container(key = "tabrd_container2", border = True):
            st.markdown("**视频输出选项**")
            colrd3_1, colrd3_2 = st.columns(2, vertical_alignment = "center")
            with colrd3_1:
                st.number_input("并发数", key = "tabrd_sub_progress_cb", min_value = 1, max_value = os.cpu_count(), value = 2, help = "同时渲染的文件数量，建议不超过 CPU 核心数")
            with colrd3_2:
                st.checkbox("跳过已生成的视频(断点续传)", key = "tabrd_skip_exist_cb", value = True)

        with st.container(key = "tabrd_container3", border = True):
            st.markdown("**视频输出**")
            colrd4_1, colrd4_2 = st.columns(2)
            with colrd4_1:
                if st.button("开始转换", key = "tabrd_start_transfer_bt", width = "stretch"):
                    st.session_state.update(is_aborted = False)
                    src = st.session_state.get("tabrd_src_dir_ti")
                    out = st.session_state.get("tabrd_mp4_dir_ti")
                    if not src or not out:
                        st.warning("请选择正确的数据文件夹")
                    else:
                        files = get_sorted_files(src, [".hdf5", ".h5"], 1)
                        file_count = len(files)
                        st.session_state.update(_render_total = file_count)
                        if not files:
                            st.warning("未找到 HDF5 文件")
                        else:
                            stop_ev = Manager().Event()
                            st.session_state.update(_render_stop = stop_ev)
                            st.session_state.update(_rendering = True)

                            threading.Thread(
                                target = render_all,
                                args = (
                                    files,
                                    src,
                                    out,
                                    st.session_state.get("tabrd_show_image_cb", True),
                                    st.session_state.get("tabrd_show_action_cb", True),
                                    st.session_state.get("action_on", [True] * 16),
                                    st.session_state.get("left_on", [True] * 7),
                                    st.session_state.get("right_on", [True] * 7),
                                    st.session_state.get("tabrd_skip_exist_cb", True),
                                    stop_ev,
                                    st.session_state.get("tabrd_sub_progress_cb", 2)
                                ),
                                daemon = True
                            ).start()
                            st.toast(f"开始渲染，共{file_count}个文件")

            with colrd4_2:
                if st.button("终止转换", key = "tabrd_abort_transfer_bt", width = "stretch"):
                    ev = st.session_state.get("_render_stop")
                    if ev:
                        ev.set()
                        st.session_state.update(_rendering = False)
                        st.toast("已发送停止信号")

    render_status()