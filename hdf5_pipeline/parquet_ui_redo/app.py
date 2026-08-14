"""Spirit（千寻智能 moz1）LeRobot parquet 数据的独立 Streamlit UI。

三个 tab：转换 / 质检 / 校验，运行在同一 raw_dir 上。
不挂在 ``hdf5_pipeline/ui/module_app.py``（HDF5 双臂流水线 UI）上。

运行::

    streamlit run hdf5_pipeline/parquet_ui_redo/app.py

"""

import csv
from pathlib import Path

import streamlit as st

from hdf5_pipeline.core.utils import pick_file, pick_folder
from hdf5_pipeline.parquet_ui_redo.constants import (
    DEFAULT_QUALITY_CSV,
    DEFAULT_VALIDATION_REPORT,
)
from hdf5_pipeline.parquet_ui_redo.convert import convert_spirit
from hdf5_pipeline.parquet_ui_redo.quality import run_spirit_quality
from hdf5_pipeline.parquet_ui_redo.validator import validate_spirit_dataset, read_skip_episodes

KEY_PREFIX = "pui"
STATE_VERSION = f"{KEY_PREFIX}_version"

CACHE_CONVERT = f"{KEY_PREFIX}_cache_convert"
CACHE_QUALITY = f"{KEY_PREFIX}_cache_quality"
CACHE_VALID = f"{KEY_PREFIX}_cache_valid"

STALE_NOTE = "该结果来自上一轮操作（其他 tab 有更新），建议重新执行。"
# 版本号广播 + 结果缓存

def _bump_version() -> None:
    st.session_state[STATE_VERSION] = st.session_state.get(STATE_VERSION, 0) + 1

def _current_version() -> int:
    return st.session_state.get(STATE_VERSION, 0)

def _cache_set(key: str, data) -> None:
    st.session_state[key] = {"version": _current_version(), "data": data}

def _cache_get(key: str):
    entry = st.session_state.get(key)
    if not entry:
        return None, False
    stale = entry["version"] != _current_version()
    return entry["data"], stale

def _folder_callback(target_key: str) -> None:
    folder = pick_folder()
    if folder:
        st.session_state[target_key] = folder

def _file_callback(target_key: str) -> None:
    file = pick_file()
    if file:
        st.session_state[target_key] = file

def _resolve_output_path(value: str, default_name: str) -> str:
    """目录则拼接默认文件名，否则按文件路径使用。"""
    if not value:
        return value
    p = Path(value)
    if p.is_dir():
        return str(p / default_name)
    return str(p)
# ---------------------------------------------------------------------------
# tab * 3
def _tab_validate(raw_dir: str) -> None:
    st.caption("检查原始数据的结构完整性与 event_log 一致性；可把错误报告写入文件，供转换页跳过问题集。")
    with st.expander("**报告输出（可选，供转换跳过使用）**",
                     key=f"{KEY_PREFIX}_valid_report", icon=":material/summarize:"):
        c1, c2, c3 = st.columns([2, 4, 1], vertical_alignment="center")
        with c1:
            st.markdown("**报告输出路径**")
        with c2:
            report_path = st.text_input(
                "报告输出路径",
                key=f"{KEY_PREFIX}_valid_report_path",
                placeholder="校验错误写入此 .txt（转换页可据此跳过问题集）",
                label_visibility="collapsed",
            )
        with c3:
            st.button("浏览报告目录", key=f"{KEY_PREFIX}_valid_report_browse",
                      icon=":material/folder_open:",
                      on_click=_folder_callback, args=(f"{KEY_PREFIX}_valid_report_path",))
    if st.button("执行校验", key=f"{KEY_PREFIX}_run_valid", type="primary",
                 icon=":material/fact_check:"):
        if not raw_dir or not Path(raw_dir).is_dir():
            st.warning("请先选择有效的原始数据目录 raw_dir", icon=":material/warning:")
        else:
            with st.spinner("校验中……"):
                _bump_version()
                try:
                    ok, errors = validate_spirit_dataset(raw_dir)
                except Exception as exc:
                    st.error(f"校验失败: {exc}")
                    return
                _cache_set(CACHE_VALID, {"ok": ok, "errors": errors})
                report_path = _resolve_output_path(report_path, DEFAULT_VALIDATION_REPORT)
                if report_path:
                    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(errors) + ("\n" if errors else ""))
                st.success("校验完成", icon=":material/check_circle:")

    data, stale = _cache_get(CACHE_VALID)
    if data:
        if stale:
            st.info(STALE_NOTE, icon=":material/refresh:")
        if data["ok"]:
            st.success("校验通过，未发现问题。", icon=":material/verified:")
        else:
            errors = data["errors"]
            st.error(f"发现 {len(errors)} 个问题：", icon=":material/error:")
            with st.container(border=True):
                for i, err in enumerate(errors, 1):
                    st.markdown(f"{i}. {err}")
            st.download_button(
                "导出错误日志（.txt）",
                data="\n".join(f"{i}. {err}" for i, err in enumerate(errors, 1)),
                file_name="spirit_validation_errors.txt",
                mime="text/plain",
                key=f"{KEY_PREFIX}_valid_download",
                icon=":material/download:",
            )

def _tab_quality(raw_dir: str) -> None:
    st.caption("对原始数据做异常帧检测（22 维 delta = cmd − state），导出 CSV / JSON 检测报告。")
    with st.expander("**检测设置**", key=f"{KEY_PREFIX}_quality_setup", expanded=True,
                     icon=":material/tune:"):
        strictness = st.selectbox("严格度", key=f"{KEY_PREFIX}_strictness",
                                  options=["loose", "medium", "strict"])
        r1 = st.columns([2, 4, 1], vertical_alignment="center")
        with r1[0]:
            st.markdown("**CSV 检测报告 导出路径**")
        with r1[1]:
            csv_out = st.text_input("CSV 导出路径", key=f"{KEY_PREFIX}_csv_out",
                                    placeholder="异常帧明细（缺省文件名 outlier_frames.csv）",
                                    label_visibility="collapsed")
        with r1[2]:
            st.button("浏览 CSV 目录", key=f"{KEY_PREFIX}_csv_browse",
                      icon=":material/folder_open:",
                      on_click=_folder_callback, args=(f"{KEY_PREFIX}_csv_out",))
        r2 = st.columns([2, 4, 1], vertical_alignment="center")
        with r2[0]:
            st.markdown("**JSON 检测报告 导出路径**")
        with r2[1]:
            json_out = st.text_input("JSON 导出路径", key=f"{KEY_PREFIX}_json_out",
                                     placeholder="统计摘要（缺省文件名 outlier_summary.json）",
                                     label_visibility="collapsed")
        with r2[2]:
            st.button("浏览 JSON 目录", key=f"{KEY_PREFIX}_json_browse",
                      icon=":material/folder_open:",
                      on_click=_folder_callback, args=(f"{KEY_PREFIX}_json_out",))

    if st.button("执行质检", key=f"{KEY_PREFIX}_run_quality", type="primary",
                 icon=":material/analytics:"):
        if not raw_dir or not Path(raw_dir).is_dir():
            st.warning("请先选择有效的原始数据目录 raw_dir", icon=":material/warning:")
        else:
            csv_path = _resolve_output_path(csv_out, DEFAULT_QUALITY_CSV)
            json_path = _resolve_output_path(json_out, "outlier_summary.json")
            if csv_path:
                Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            if json_path:
                Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with st.spinner("检测中……"):
                _bump_version()
                try:
                    summary = run_spirit_quality(
                        raw_dir, csv_path, json_path, strictness=strictness
                    )
                except Exception as exc:
                    st.error(f"质检失败: {exc}")
                    return
                _cache_set(CACHE_QUALITY, {
                    "summary": summary,
                    "csv": csv_path,
                    "json": json_path,
                })
                st.success("质检完成", icon=":material/check_circle:")

    data, stale = _cache_get(CACHE_QUALITY)
    if data:
        if stale:
            st.info(STALE_NOTE, icon=":material/refresh:")
        summary = data["summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("扫描集数", summary.get("num_files", 0), border=True)
        c2.metric("总帧数", summary.get("num_frames", 0), border=True)
        c3.metric("异常帧数", summary.get("num_outliers", 0), border=True)

        col_d, col_e = st.columns(2)
        with col_d:
            st.markdown("**异常最多的维度**")
            top_dims = summary.get("top_dims", [])
            if top_dims:
                st.dataframe(
                    [{"维度": f"dim {d['dim']}", "异常帧数": d["count"]} for d in top_dims],
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("无")
        with col_e:
            st.markdown("**异常最多的集**")
            top_eps = summary.get("top_episodes", [])[:5]
            if top_eps:
                st.dataframe(
                    [{"集": f"episode_{ep['episode']:06d}", "异常帧数": ep["count"]}
                     for ep in top_eps],
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("无")

        csv_path = data.get("csv")
        if csv_path and Path(csv_path).exists():
            st.markdown("**Top 异常帧明细**")
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                st.dataframe(rows[:50], width="stretch", hide_index=True)
        if summary.get("num_outliers", 0) == 0:
            st.info("未检出异常帧（CSV/JSON 未生成）。", icon=":material/info:")

def _tab_convert(raw_dir: str) -> None:
    st.caption("将原始 parquet 转为标准 LeRobot v2.1 数据集；可跳过检测文档判定的问题集。")
    with st.expander("**路径设置**", key=f"{KEY_PREFIX}_convert_paths", expanded=True,
                     icon=":material/folder:"):
        c1, c2, c3 = st.columns([2, 4, 1], vertical_alignment="center")
        with c1:
            st.markdown("**输出目录（out_dir）**")
        with c2:
            out_dir = st.text_input(
                "输出目录（out_dir）",
                key=f"{KEY_PREFIX}_out_dir",
                placeholder="转换结果写入此目录（存在则先清空）",
                label_visibility="collapsed",
            )
        with c3:
            st.button("浏览", key=f"{KEY_PREFIX}_out_browse",
                      icon=":material/folder_open:",
                      on_click=_folder_callback, args=(f"{KEY_PREFIX}_out_dir",))
        link_videos = st.checkbox("视频用软链接（symlink，默认）", value=True,
                                  key=f"{KEY_PREFIX}_link_videos")

    skip_on = st.checkbox("跳过检测文档中涉及的问题文件", key=f"{KEY_PREFIX}_skip_on")
    skip_csv = ""
    skip_log = ""
    if skip_on:
        with st.expander("**检测文档（质检 CSV / 校验报告）**", key=f"{KEY_PREFIX}_skip_docs",
                         icon=":material/rule:"):
            st.caption("勾选后，转换将跳过这些文档中判定有问题的 episode（质检 CSV 的 episode 列 + 校验报告的 episode_XXXXXX）。")
            c1, c2, c3 = st.columns([2, 4, 1], vertical_alignment="center")
            with c1:
                st.markdown("**质检 CSV（异常帧明细）**")
            with c2:
                skip_csv = st.text_input(
                    "质检 CSV（异常帧明细）",
                    key=f"{KEY_PREFIX}_skip_csv",
                    placeholder="选择质检导出的 CSV 文件",
                    label_visibility="collapsed",
                )
            with c3:
                st.button("选择 CSV 文件", key=f"{KEY_PREFIX}_skip_csv_browse",
                          icon=":material/description:",
                          on_click=_file_callback, args=(f"{KEY_PREFIX}_skip_csv",))
            c4, c5, c6 = st.columns([2, 4, 1], vertical_alignment="center")
            with c4:
                st.markdown("**校验报告（.txt）**")
            with c5:
                skip_log = st.text_input(
                    "校验报告（.txt）",
                    key=f"{KEY_PREFIX}_skip_log",
                    placeholder="选择校验页导出的报告 .txt 文件",
                    label_visibility="collapsed",
                )
            with c6:
                st.button("选择报告文件", key=f"{KEY_PREFIX}_skip_log_browse",
                          icon=":material/description:",
                          on_click=_file_callback, args=(f"{KEY_PREFIX}_skip_log",))

    if st.button("开始转换", key=f"{KEY_PREFIX}_run_convert", type="primary",
                 icon=":material/transform:"):
        if not raw_dir or not Path(raw_dir).is_dir():
            st.warning("请先选择有效的原始数据目录 raw_dir", icon=":material/warning:")
        elif not out_dir:
            st.warning("请先填写输出目录 out_dir", icon=":material/warning:")
        else:
            skip_set = read_skip_episodes(raw_dir, skip_csv, skip_log) if skip_on else set()
            with st.spinner("转换中……"):
                _bump_version()
                try:
                    stats = convert_spirit(raw_dir, out_dir, link_videos=link_videos,
                                           skip_episodes=skip_set)
                except Exception as exc:
                    st.error(f"转换失败: {exc}")
                    return
                _cache_set(CACHE_CONVERT, {"stats": stats, "out_dir": stats["out_dir"]})
                st.success(f"转换完成 → {stats['out_dir']}", icon=":material/check_circle:")

    data, stale = _cache_get(CACHE_CONVERT)
    if data:
        if stale:
            st.info(STALE_NOTE, icon=":material/refresh:")
        stats = data["stats"]
        c1, c2, c3 = st.columns(3)
        c1.metric("转换集数", stats["converted"], border=True)
        c2.metric("跳过集数", stats["skipped"], border=True)
        c3.metric("视频数", stats["videos"], border=True)
        st.caption(f"输出目录：{data['out_dir']}")
        if stats["converted"] == 0:
            st.warning("没有任何集被转换（可能全部是 is_mistake 或数据不可读）。",
                       icon=":material/warning:")

# 主入口
def main() -> None:
    st.set_page_config(page_title="Spirit Parquet 工具", page_icon=":material/science:",
                       layout="wide")
    st.title("Spirit（千寻 moz1）LeRobot Parquet 工具")
    st.caption("独立链路：原始 parquet → 校验 / 质检 / 转换。不涉及 HDF5 双臂流水线。")

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 4, 1], vertical_alignment="center")
        with c1:
            st.markdown("**原始数据目录（raw_dir）**")
        with c2:
            st.text_input(
                "原始数据目录（raw_dir）",
                key=f"{KEY_PREFIX}_raw_dir",
                placeholder="raw_dir 或单个实例目录（含 event_log.jsonl），三个 tab 共用",
                label_visibility="collapsed",
            )
        with c3:
            st.button("浏览", key=f"{KEY_PREFIX}_raw_browse",
                      icon=":material/folder_open:",
                      on_click=_folder_callback, args=(f"{KEY_PREFIX}_raw_dir",))

    raw_dir = st.session_state.get(f"{KEY_PREFIX}_raw_dir", "")

    tab_validate, tab_quality, tab_convert = st.tabs(["校验", "质检", "转换"])
    with tab_validate:
        _tab_validate(raw_dir)
    with tab_quality:
        _tab_quality(raw_dir)
    with tab_convert:
        _tab_convert(raw_dir)


if __name__ == "__main__":
    main()