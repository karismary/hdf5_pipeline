"""Spirit（千寻智能 moz1）LeRobot parquet 数据的独立 Streamlit UI。

三个 tab：转换 / 质检 / 校验，运行在同一 raw_dir 上。
不挂在 ``hdf5_pipeline/ui/module_app.py``（HDF5 双臂流水线 UI）上。

运行::

    streamlit run hdf5_pipeline/parquet_ui_redo/app.py

"""

import csv
from pathlib import Path

import streamlit as st

from hdf5_pipeline.core.utils import pick_folder
from hdf5_pipeline.parquet_ui_redo.convert import convert_spirit
from hdf5_pipeline.parquet_ui_redo.quality import run_spirit_quality
from hdf5_pipeline.parquet_ui_redo.validator import validate_spirit_dataset

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
def _tab_convert(raw_dir: str) -> None:
    st.markdown("**转换** — 将原始 parquet 转为标准 LeRobot v2.1 数据集。")
    with st.expander("**路径设置**", key=f"{KEY_PREFIX}_convert_paths", expanded=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            out_dir = st.text_input(
                "输出目录（out_dir）",
                key=f"{KEY_PREFIX}_out_dir",
                placeholder="转换结果写入此目录（存在则先清空）",
                label_visibility="collapsed",
            )
            link_videos = st.checkbox("视频用软链接（symlink，默认）", value=True,
                                      key=f"{KEY_PREFIX}_link_videos")
        with col2:
            st.button("浏览", key=f"{KEY_PREFIX}_out_browse",
                      on_click=_folder_callback, args=(f"{KEY_PREFIX}_out_dir",))

    if st.button("开始转换", key=f"{KEY_PREFIX}_run_convert", type="primary"):
        if not raw_dir or not Path(raw_dir).is_dir():
            st.warning("请先选择有效的原始数据目录 raw_dir")
        elif not out_dir:
            st.warning("请先填写输出目录 out_dir")
        else:
            with st.spinner("转换中……"):
                _bump_version()
                try:
                    stats = convert_spirit(raw_dir, out_dir, link_videos=link_videos)
                except Exception as exc:
                    st.error(f"转换失败: {exc}")
                    return
                _cache_set(CACHE_CONVERT, {"stats": stats, "out_dir": out_dir})
                st.success(f"转换完成 → {out_dir}")

    data, stale = _cache_get(CACHE_CONVERT)
    if data:
        if stale:
            st.info(STALE_NOTE)
        stats = data["stats"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("转换集数", stats["converted"])
        c2.metric("跳过集数", stats["skipped"])
        c3.metric("视频数", stats["videos"])
        c4.metric("输出目录", data["out_dir"])
        if stats["converted"] == 0:
            st.warning("没有任何集被转换（可能全部是 is_mistake 或数据不可读）。")

def _tab_quality(raw_dir: str) -> None:
    st.markdown("**质检** — 对原始数据做异常帧检测（22 维 delta = cmd − state）。")
    with st.expander("**检测设置**", key=f"{KEY_PREFIX}_quality_setup", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            strictness = st.selectbox("严格度", key=f"{KEY_PREFIX}_strictness",
                                      options=["loose", "medium", "strict"])
            csv_out = st.text_input("CSV 导出路径", key=f"{KEY_PREFIX}_csv_out",
                                    placeholder="异常帧明细（缺省文件名 outlier_frames.csv）")
        with col2:
            json_out = st.text_input("JSON 导出路径", key=f"{KEY_PREFIX}_json_out",
                                     placeholder="统计摘要（缺省文件名 outlier_summary.json）")

    if st.button("执行质检", key=f"{KEY_PREFIX}_run_quality", type="primary"):
        if not raw_dir or not Path(raw_dir).is_dir():
            st.warning("请先选择有效的原始数据目录 raw_dir")
        else:
            csv_path = _resolve_output_path(csv_out, "outlier_frames.csv")
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
                st.success("质检完成")

    data, stale = _cache_get(CACHE_QUALITY)
    if data:
        if stale:
            st.info(STALE_NOTE)
        summary = data["summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("扫描集数", summary.get("num_files", 0))
        c2.metric("总帧数", summary.get("num_frames", 0))
        c3.metric("异常帧数", summary.get("num_outliers", 0))

        st.markdown("**异常最多的维度：**")
        for d in summary.get("top_dims", []):
            st.write(f"  维度 {d['dim']}: {d['count']} 帧")
        st.markdown("**异常最多的集：**")
        for ep in summary.get("top_episodes", [])[:5]:
            st.write(f"  episode_{ep['episode']:06d}: {ep['count']} 帧")

        csv_path = data.get("csv")
        if csv_path and Path(csv_path).exists():
            st.markdown("**Top 异常帧：**")
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                st.dataframe(rows[:50], width="stretch")
        if summary.get("num_outliers", 0) == 0:
            st.info("未检出异常帧（CSV/JSON 未生成）。")

def _tab_validate(raw_dir: str) -> None:
    st.markdown("**校验** — 检查原始数据的结构完整性与 event_log 一致性。")
    if st.button("执行校验", key=f"{KEY_PREFIX}_run_valid", type="primary"):
        if not raw_dir or not Path(raw_dir).is_dir():
            st.warning("请先选择有效的原始数据目录 raw_dir")
        else:
            with st.spinner("校验中……"):
                _bump_version()
                try:
                    ok, errors = validate_spirit_dataset(raw_dir)
                except Exception as exc:
                    st.error(f"校验失败: {exc}")
                    return
                _cache_set(CACHE_VALID, {"ok": ok, "errors": errors})
                st.success("校验完成")

    data, stale = _cache_get(CACHE_VALID)
    if data:
        if stale:
            st.info(STALE_NOTE)
        if data["ok"]:
            st.success("校验通过，未发现问题。")
        else:
            st.error(f"发现 {len(data['errors'])} 个问题：")
            for err in data["errors"]:
                st.write(f"- {err}")

# 主入口
def main() -> None:
    st.set_page_config(page_title="Spirit Parquet 工具", layout="wide")
    st.title("Spirit（千寻 moz1）LeRobot Parquet 工具")
    st.caption("独立链路：原始 parquet → 转换 / 质检 / 校验。不涉及 HDF5 双臂流水线。")

    with st.expander("**原始数据目录（raw_dir，三个 tab 共用）**",
                     key=f"{KEY_PREFIX}_raw_dir_box", expanded=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_input(
                "原始数据目录",
                key=f"{KEY_PREFIX}_raw_dir",
                placeholder="raw_dir 或单个实例目录（含 event_log.jsonl）",
                label_visibility="collapsed",
            )
        with col2:
            st.button("浏览", key=f"{KEY_PREFIX}_raw_browse",
                      on_click=_folder_callback, args=(f"{KEY_PREFIX}_raw_dir",))

    raw_dir = st.session_state.get(f"{KEY_PREFIX}_raw_dir", "")

    tab_convert, tab_quality, tab_validate = st.tabs(["转换", "质检", "校验"])
    with tab_convert:
        _tab_convert(raw_dir)
    with tab_quality:
        _tab_quality(raw_dir)
    with tab_validate:
        _tab_validate(raw_dir)


if __name__ == "__main__":
    main()