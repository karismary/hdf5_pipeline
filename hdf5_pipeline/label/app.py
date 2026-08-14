import json
import copy
import csv
import sqlite3
import streamlit as st
# from streamlit_file_browser import st_file_browser
from pathlib import Path
# import cv2
import shutil

from hdf5_pipeline.core.video_utils import get_video_info,get_frame
from hdf5_pipeline.label.database import init_db, add_label, add_labels, scan_pairs, get_list, get_records, translate_where, query_records, count_list, update_attrs
from hdf5_pipeline.core.config import load_config, save_config
from hdf5_pipeline.core.utils import pick_folder
from hdf5_pipeline.ui.rename_tab import show_tab_rename
from hdf5_pipeline.ui.quality_tab import show_tab_quality
from hdf5_pipeline.ui.render_tab import show_tab_render
from hdf5_pipeline.ui.common import KEY_LABEL, KEY_OVERVIEW, KEY_CONFIG
from hdf5_pipeline.label.state import (
    get, set, pop, init_state, bump_db_version,
    S_RECORDS, S_SELECTED, S_SELECTED_INDEX,
    S_OV_RECORDS, S_OV_PAGE, S_OV_WHERE,
    S_DB_VERSION, S_TAB4_VERSION, S_TAB5_VERSION, S_TOAST, S_MARKED_RED, S_DB_WARNED,
)

# def select_folder(path_type):
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

TYPE_DICT = {
    "icon":{"good" : "✅", "bad" : "❌", "pending" : "❎", "unlabeled" : "☑️"},
    "path":{"good" : "good_dir", "bad" : "bad_dir", "pending" : "raw_dir", "unlabeled" : "raw_dir"}
}
PAGE_SIZE = 10

def select_folder(path_type: str) -> None:
    """选择文件夹并保存到 config。

    调用 pick_folder() 获取路径，db_dir 特殊处理（自动创建 label.db），
    然后写入 session_state 和 config.json。

    Args:
        path_type (str): 配置键名，如 'db_dir'、'raw_dir'。

    Returns:
        None。
    """
    folder = pick_folder()
    if not folder:
        return

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

def qualify_and_move(
    db_path: str,
    mp4_name: str | None = None,
    raw_path: str | None = None,
    target_dir: str | None = None,
    quality: str | None = None,
    compare_quality: str | None = None,
    if_quantity: bool = False,
    quantity_records: list[sqlite3.Row] | None = None,
) -> list[tuple] | None:
    """移动 HDF5 文件到目标目录，并更新数据库的 quality 和 hdf5_path。

    支持单条 / 批量两种模式：
    - 单条（if_quantity=False）：移动 raw_path 指向的文件，更新该条记录。
    - 批量（if_quantity=True）：移动 quantity_records 中每个文件，
      通过 add_labels 一次事务更新所有记录。

    DB 更新失败时回滚文件移动。不负责刷新 session_state（由调用方处理）。

    Args:
        db_path (str): 数据库文件路径。
        mp4_name (str | None, optional): MP4 文件名，单条模式定位记录用。
        raw_path (str | None, optional): HDF5 文件当前完整路径（单条模式）。
        target_dir (str | None, optional): 目标文件夹（good_dir 或 bad_dir）。
        quality (str | None, optional): 目标质量标签，'good' 或 'bad'。
        compare_quality (str | None, optional): 当前质量值，用于跳过重复操作；
            不传时单条模式自动从 st.session_state["selected"] 读取，批量模式不生效。
        if_quantity (bool, optional): 是否批量模式，默认 False。
        quantity_records (list[sqlite3.Row] | None, optional): 批量模式必传的记录列表
            （get_list/query_records 返回的 Row），按列名取值：移动用
            record["hdf5_path"]，回滚用 record["hdf5_name"] 与 record["hdf5_path"]。

    Returns:
        list[tuple] | None:
            单条模式返回 None。
            批量模式返回重建后的记录元组列表（quality 已替换为新值），
            供调用方刷新界面；DB 更新失败时已回滚文件移动。
    """
    if compare_quality is None and not if_quantity:
        selected = get(S_SELECTED)
        if quality == selected["quality"]:
            st.toast(f'已设置为"{selected["quality"]}"')
            return
    else:
        if compare_quality == quality:
            st.rerun()
            return
    if not if_quantity:
        # 单条模式
        if mp4_name is None or raw_path is None or target_dir is None or quality is None:
            st.error("单条模式缺少 mp4_name / raw_path / target_dir / quality")
            return
        if not Path(target_dir).exists():
            st.error(f"文件夹：{target_dir}不存在，请检查")
            return
        if not (Path(target_dir) / Path(raw_path).name).exists():
            shutil.move(raw_path, target_dir)
            nhdf5_path = Path(target_dir) / Path(raw_path).name
            try:
                add_label(db_path, mp4_name, str(nhdf5_path), quality, None)
            except Exception:
                shutil.move(str(nhdf5_path), str(Path(raw_path).parent))
                raise
            bump_db_version()
        return
    # 批量模式
    if target_dir is None or quality is None or quantity_records is None:
        st.error("批量模式缺少 target_dir / quality / quantity_records")
        return
    if not Path(target_dir).exists():
        st.error(f"文件夹：{target_dir}不存在，请检查")
        return
    new_records = []
    for record in quantity_records:
        if (Path(target_dir) / record["hdf5_name"]).exists():
            continue
        else:
            shutil.move(Path(record["hdf5_path"]), target_dir)
            record = record[:5] + (quality,) + record[6:]
            new_records.append(record)
    try:
        add_labels(db_path, quantity_records, quality, True, target_dir)
    except Exception:
        for record in quantity_records:
            shutil.move(Path(target_dir) / record["hdf5_name"], Path(record["hdf5_path"]))
        raise
    bump_db_version()
    return new_records

def quality_module(key_name: str, if_session_state: bool = True, session_state: str = "selected", target: sqlite3.Row | None = None) -> None:
    """渲染质量标记按钮。

    支持两种模式：
    1. session_state 模式（if_session_state=True）— 从 session_state 读当前记录。
    2. target 模式（if_session_state=False）— 从 target 参数取记录数据。

    Args:
        key_name (str): 按钮 key 前缀，确保唯一性。
        if_session_state (bool): True 从 session_state 读记录，False 从 target 取记录。
        session_state (str): session_state 的键名，默认 "selected"。
        target (sqlite3.Row, optional): 单条记录（get_records 返回的 Row），if_session_state=False 时必传。

    Returns:
        None。
    """

    def button_quality(type: str, sel: sqlite3.Row):
        if st.button(f"{TYPE_DICT['icon'].get(type)} {type}", key = f"botton_{type}_{key_name}:{sel['mp4_name']}", width = "stretch"):
            target_dir = TYPE_DICT["path"].get(type)
            qualify_and_move(
                config["paths"]["db_dir"],   # 数据库路径
                sel["mp4_name"],              # mp4_name
                sel["hdf5_path"],             # hdf5 当前路径
                str(Path(config["paths"][target_dir])),  # hdf5 目标路径
                type,                     # 质量标签
                compare_quality = sel["quality"]
            )
            set(S_SELECTED, get_records(config["paths"]["db_dir"], sel["mp4_name"]))
            st.rerun()

    sel = get(S_SELECTED) if if_session_state else target
    if sel is None:
        return
    colnq1, colnq2 = st.columns(2)
    with colnq1:
        button_quality("good", sel)
        button_quality("pending", sel)
    with colnq2:
        button_quality("bad", sel)
        button_quality("unlabeled", sel)

def attrs_module(template_attrs: dict, key_name: str, if_session_state: bool = True, session_state: str = "selected", target: sqlite3.Row | None = None) -> None:
    """渲染属性修改下拉框 + 确认按钮。

    支持两种模式：
    1. session_state 模式 — 从 session_state 读当前记录。
    2. target 模式 — 从 target 取记录数据。

    Args:
        template_attrs (dict): config["custom_cols"] 属性配置。
        key_name (str): 按钮 key 前缀。
        if_session_state (bool): True 从 session_state 读，False 从 target 取。
        session_state (str): session_state 键名，默认 "selected"。
        target (sqlite3.Row, optional): 单条记录（get_records 返回的 Row），target 模式时必传。

    Returns:
        None。
    """
    sel = get(S_SELECTED) if if_session_state else target
    if sel is None:
        return
    try:
        db_attrs = json.loads(sel["attr"]) if sel and sel["attr"] and sel["attr"] != "{}" else {}
    except:
        db_attrs = {}

    for keys, attrs in template_attrs.items():

        raw_val = db_attrs.get(keys, "未选择")
        current_val = raw_val.get("option", "未选择") if isinstance(raw_val, dict) else raw_val
        original_options =attrs["option"]
        new_options = ["未选择"] + attrs["option"]
        current_idx = original_options.index(current_val) + 1 if current_val in original_options else 0

        widget_key = f"attrs_{key_name}_{keys}"

        cola1, cola2, cola3 = st.columns([1, 4, 1], gap="xxsmall")
        with cola1:
            st.markdown(f"**{attrs['label']}:**")
        with cola2:
            attr_selected = st.selectbox(
                f"{attrs['label']}",
                key = widget_key,
                options = new_options,
                index = current_idx,
                label_visibility="collapsed"
            )
        with cola3:
            if st.button("确认", key=f"confirm_{key_name}_{keys}", width="stretch"):
                payload_to_save = copy.deepcopy(db_attrs)
                payload_to_save[keys] = {
                    "label": attrs['label'],
                    "option": attr_selected if attr_selected else "未选择",
                    "type": attrs.get('type', 'text')
                }
                mp4_name = sel["mp4_name"]
                add_label(config["paths"]["db_dir"], mp4_name, attr=payload_to_save)
                bump_db_version()
                if if_session_state:
                    if widget_key in st.session_state:
                        del st.session_state[widget_key]
                    set(S_TOAST, f"属性 **{attrs['label']}** 已设为：**{attr_selected}**")
                    st.rerun()
                else:
                    # 清掉 attrs widget 缓存，下次打开从数据库读最新值
                    for wk in list(st.session_state.keys()):
                        if isinstance(wk, str) and wk.startswith(f"attrs_{key_name}_"):
                            del st.session_state[wk]

st.set_page_config(page_title="HDF5 Labeling Tool",
                   page_icon=":material/brush:",
                   layout="wide",
                   initial_sidebar_state="expanded")

init_state()

_ENABLE_SELECT_COLUMNS = False

config = load_config("config.json")
path_dict = {
    "数据库路径(没有文件会在选择目录下自动创建)":"db_dir",
    "hdf5文件夹路径":"raw_dir",
    "mp4文件夹路径":"mp4_dir",
    "good_quality_hdf5存储路径":"good_dir",
    "bad_quality_hdf5存储路径":"bad_dir"
}

def _db_ready() -> bool:
    """db_dir 是否指向含 label 表的有效 SQLite 数据库。"""
    db_path = Path(config["paths"]["db_dir"])
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='label'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False

@st.dialog("数据库不可用", icon=":material/error:")
def _show_db_dialog() -> None:
    st.warning("当前数据库路径无效，或尚未包含 label 表。")
    st.markdown(
        "请在**左侧边栏「工作路径配置」**中重新设置「数据库路径」：\n\n"
        "- 用 **浏览** 选择文件夹，缺少的 `label.db` 会自动创建；\n"
        "- 或直接指向已有的 `.db` 文件。\n\n"
        "设置完成后即可正常使用。"
    )
    if st.button("知道了", key="db_dialog_ok"):
        st.rerun()

st.title("HDF5 Labeling Tool")
st.caption("给 HDF5 数据打标的工具：质量分类 + 属性标注，配套文件重命名、质量检测、视频渲染等完整流水线。")

# 数据库有效性：不可用则在侧边栏提示 + 首次弹窗（修复后自动清除标记）
_db_ok = _db_ready()
if _db_ok:
    if pop(S_DB_WARNED, None):
        set(S_TAB4_VERSION, -1)
else:
    st.error("数据库不可用：文件缺失或尚未包含 label 表，请在左侧边栏「工作路径配置」中设置正确的数据库路径。")
    if not get(S_DB_WARNED, False):
        set(S_DB_WARNED, True)
        _show_db_dialog()

_toast_msg = pop(S_TOAST, None)
if _toast_msg:
    st.toast(_toast_msg)

#————侧边栏：设置工作路径，调用本地选择文件夹————
with st.sidebar:
    st.subheader("工作路径配置", divider=True)
    with st.container(border=True):
        for keys in path_dict:
            path = path_dict[keys]
            st.markdown(f"**{keys}**")
            col01, col02 = st.columns([4, 1], vertical_alignment="center")
            with col01:
                st.text_input(keys, key=f"ui_{path}", label_visibility="collapsed",
                              value=st.session_state.get(f"ui_{path}", config['paths'][path]))
            with col02:
                st.button("浏览", key=f"btn_{path}", on_click=select_folder, args=(path,),
                          icon=":material/folder_open:", width="content")

    if st.button("扫描文件夹并同步数据库", key="btn_sync", width="stretch", icon=":material/sync:"):
        n = scan_pairs(
            st.session_state.get("ui_db_dir", config['paths']['db_dir']),
            st.session_state.get("ui_mp4_dir", config['paths']['mp4_dir']),
            st.session_state.get("ui_raw_dir", config['paths']['raw_dir'])
        )
        if n > 0:
            bump_db_version()
        st.toast(f"新增 {n} 对文件", icon="✅")

#————主页面：————（根据需求添加后续标签）
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "文件重命名", "质量检测", "视频渲染",
    "视频打标", "数据总览", "配置"
])

#————标签页1:文件重命名————
with tab1: show_tab_rename()
#————标签页2:质量检测————
with tab2: show_tab_quality()
#————标签页3:视频渲染————
with tab3: show_tab_render()
#————标签页4:打标界面————
with tab4:
    # DB 版本变化则重查记录缓存（覆盖打标、批量修改、扫描同步等所有写操作）
    if _db_ok and get(S_DB_VERSION, 0) != get(S_TAB4_VERSION, 0):
        try:
            set(S_RECORDS, get_list(str(Path(config["paths"]["db_dir"]))))
        except Exception:
            pass
        set(S_TAB4_VERSION, get(S_DB_VERSION, 0))
    st.subheader("标记记录")
    st.caption("从文件列表选择一条记录，查看首末帧与视频，完成质量分类与属性标注。")

    # —— 异常文件处理（CSV 自动导入）——
    with st.expander("异常文件处理（CSV 自动导入）", icon=":material/report:"):
        csv_path = Path(config["paths"]["csv_dir"]) / "outlier_frames.csv"
        if not csv_path.exists():
            st.info("未找到异常报告 CSV，请先在「质量检测」运行检测并导出到配置的 csv_dir\n若已经运行，则说明数据不存在质量问题")
        else:
            with open(csv_path, "r", encoding = "utf-8") as f:
                abnormal = {row["file"] for row in csv.DictReader(f)}
                placeholder = ", ".join(f"'{n.replace(chr(39), chr(39)*2)}'" for n in abnormal)
                abnormal_records, _ = query_records(config["paths"]["db_dir"], "*", f"hdf5_name in ({placeholder})")
            col001, col002, col003 = st.columns(3)
            with col001:
                if st.button("扫描并标红", key = f"{KEY_LABEL}_csv_scan_red", icon=":material/flag:"):
                    set(S_MARKED_RED, get(S_MARKED_RED, frozenset()) | {r["mp4_name"] for r in abnormal_records})
                    st.rerun()
            with col002:
                if st.button("全部标记 bad", key = f"{KEY_LABEL}_qualify_all", icon=":material/block:"):
                    qualify_and_move(
                        config["paths"]["db_dir"],
                        target_dir = config["paths"]["bad_dir"],
                        quality = "bad",
                        if_quantity = True,
                        quantity_records = abnormal_records
                    )
                    set(S_MARKED_RED, get(S_MARKED_RED, frozenset()) - {r["mp4_name"] for r in abnormal_records})
                    st.rerun()
            with col003:
                if st.button("取消标红", key = f"{KEY_LABEL}_csv_clear_red", icon=":material/undo:"):
                    set(S_MARKED_RED, frozenset())
                    st.rerun()
            marked = get(S_MARKED_RED, frozenset())
            if marked:
                st.markdown("**检测到异常的记录：**")
                for name in sorted(marked):
                    st.markdown(f'<span style="color:red">🔴 {name}</span>', unsafe_allow_html=True)

    col11, col12 = st.columns([1.5, 3])
#————标签页4-第一列：工作记录选择————
    with col11:
        with st.container(key = f"{KEY_LABEL}_col1_container", border = True):
            next_flag = False
            st.markdown("**文件列表**")
            if st.button("获取并更新记录列表", key="byn_fetch", width="stretch", icon=":material/refresh:"):
                if _db_ready():
                    set(S_RECORDS, get_list(config["paths"]["db_dir"]))
                    set(S_TAB4_VERSION, get(S_DB_VERSION, 0))
                else:
                    st.error("数据库不可用（文件缺失或没有 label 表），请在左侧边栏设置正确的数据库路径")
            if S_RECORDS in st.session_state:
                next_flag = True
                records = get(S_RECORDS)
                selected_idx = st.selectbox(
                    "**选择文件**",
                    range(len(records)),                       # 实际选项：0, 1, 2...
                    format_func=lambda i: f"{'🔴 ' if records[i]['mp4_name'] in get(S_MARKED_RED, frozenset()) else ''}{records[i]['quality']} - {records[i]['id']}.{records[i]['mp4_name'].replace('.mp4','')}",         # 显示：文件名
                    label_visibility="visible",
                    index = get(S_SELECTED_INDEX, 0)
                )
                set(S_SELECTED_INDEX, selected_idx)

                records = get(S_RECORDS)
                total = len(records)
                good = sum(1 for r in records if r["quality"] == "good")
                bad = sum(1 for r in records if r["quality"] == "bad")
                labeled = good + bad
                progress = labeled / total if total > 0 else 0
                st.progress(progress, text = f"{labeled}/{total} 已完成标记")
                st.caption(f"good {good} ｜ bad {bad} ｜ 待标 {total - labeled}")

                col1101, col1102 = st.columns(2)
                with col1101:
                    if st.button("上一条", key=f"{KEY_LABEL}_prev", width="stretch", icon=":material/skip_previous:"):
                        if selected_idx > 0:
                            new_idx = selected_idx - 1
                            set(S_SELECTED_INDEX, new_idx)
                            set(S_SELECTED, records[new_idx])
                            st.rerun()
                with col1102:
                    if st.button("下一条", key=f"{KEY_LABEL}_next", width="stretch", icon=":material/skip_next:"):
                        if selected_idx < len(records) - 1:
                            new_idx = selected_idx + 1
                            set(S_SELECTED_INDEX, new_idx)
                            set(S_SELECTED, records[new_idx])
                            st.rerun()

                if selected_idx is not None:
                    next_flag = True
                    selected = records[get(S_SELECTED_INDEX, 0)]
                    set(S_SELECTED, selected)
                    q_icon = {"good": "✅", "bad": "❌", "pending": "❎", "unlabeled": "☑️"}.get(selected['quality'], "☑️")
                    st.markdown(f"{q_icon} **{selected['quality']}**　·　`{selected['hdf5_name']}`")
#————标签页4-第一列：属性设置-质量分类————
        if next_flag:
            with st.container(key = f"{KEY_LABEL}_col1_attr_checker", border = True):
                with st.container(key = f"{KEY_LABEL}_col1_qualities", border = False):
                    st.markdown("**质量分类**")
                    quality_module(f"{KEY_LABEL}_quality_button")
                with st.container(key = f"{KEY_LABEL}_col1_attr_settings", border = False):
                    st.markdown("**属性设置**")
                    if config["custom_cols"]:
                        attrs_module(
                            config["custom_cols"],
                            f"{KEY_LABEL}_attr_{get(S_SELECTED)['id']}",
                            if_session_state=True
                        )
                    else:
                        st.caption("暂无自定义属性，请到「配置」页添加")
#————标签页4-第二列：工作区域————
    with col12:
        with st.container(key = f"{KEY_LABEL}_col2_graphbox", border = next_flag):
            sel = get(S_SELECTED)
            if sel:
                video_path = sel["mp4_path"]
                n_frames = get_video_info(video_path)[0]
                st.markdown("**首 / 末帧预览**")
                col121, col122 = st.columns([1, 1], vertical_alignment="center")
                with col121:
                    first_frame = get_frame(video_path, 0)
                    if first_frame is not None:
                        st.image(first_frame, caption = "首帧", width="stretch")
                    else:
                        st.info("无法读取首帧")
                with col122:
                    last_frame = get_frame(video_path, n_frames - 1)
                    if last_frame is not None:
                        st.image(last_frame, caption = "末帧", width="stretch")
                    else:
                        st.info("无法读取末帧")
        with st.container(key = f"{KEY_LABEL}_col2_videobox", border = next_flag):
            sel = get(S_SELECTED)
            if sel:
                st.markdown(f"**视频：`{(sel['mp4_path'].split('/'))[-1]}`**")
                st.video(sel['mp4_path'])

##————标签页5:数据总览操作界面————
with tab5:
    st.subheader("数据总览")
    st.caption("自定义 SQL 查询与批量修改，右侧实时查看记录列表。")
    col21, col22 = st.columns([1.5, 3])
##————标签页5-第一列:数据批量处理————
##————标签页5-第一列-第一项:查询筛选————
    with col21:
        # 清除筛选信号：须在 SELECT/WHERE 控件实例化之前消费（改 widget key 有 Streamlit 限制）
        if pop(f"{KEY_OVERVIEW}_clear_sync", None):
            st.session_state[f"{KEY_OVERVIEW}_select_cols"] = []
            st.session_state[f"{KEY_OVERVIEW}_where_input"] = ""
        with st.container(key = f"{KEY_OVERVIEW}_sql_choose_container", border = True):
            st.markdown("**自定义查询**")
            all_columns: dict[str, str] = {}
            selected_cols: list[str] = []
            if _ENABLE_SELECT_COLUMNS:
                all_columns = {
                    "ID": "id", "HDF5名": "hdf5_name", "HDF5路径": "hdf5_path",
                    "MP4名": "mp4_name", "MP4路径": "mp4_path", "质量": "quality",
                    "属性": "attr", "创建时间": "created_at", "打标时间": "labeled_at"
                }
                selected_cols = st.multiselect(
                    "**SELECT**（筛选目标列）",
                    options = list(all_columns.keys()),
                    default = None,
                    placeholder = "可多选",
                    key = f"{KEY_OVERVIEW}_select_cols",
                )
            sql_where = st.text_area("**WHERE**（筛选条件）",
                                    key = f"{KEY_OVERVIEW}_where_input",
                                    placeholder='输入筛选条件例: quality = "good" \n留空则查询所有数据',
                                    help="""
                                    SQL 筛选语法说明：
                                    普通列查询（直接用列名）：
                                    quality = "good"          ← 质量标签
                                    id > 5                    ← 记录 ID

                                    属性查询（@开头）：
                                    @天气 = "晴天"             ← 查属性 JSON 里的值
                                    @背景 LIKE "%室%"          ← 属性模糊匹配

                                    支持完整的 SQL 逻辑运算：
                                    quality = "good" AND @天气 = "晴天"
                                    NOT quality = "unlabeled"
                                    """
                                        )
            attr_map = {}
            for k, v in config["custom_cols"].items():
                attr_map[v["label"]] = f"$.{k}.option"
            col211, col212 = st.columns(2)
            with col211:
                if st.button("执行查询", key=f"{KEY_OVERVIEW}_query", width="stretch", type="primary", icon=":material/search:"):
                    where_condition: str | None = None
                    col_names = "*"
                    if _ENABLE_SELECT_COLUMNS:
                        col_names = ", ".join(all_columns[c] for c in selected_cols) if selected_cols else "*"
                    if sql_where.strip():
                        where_condition = translate_where(sql_where.strip(), attr_map)
                    st.toast(f"执行 SQL: SELECT {col_names} FROM label WHERE {where_condition}")

                    result, err = query_records(
                        str(config["paths"]["db_dir"]),
                        col_names,
                        where_condition,
                        limit = PAGE_SIZE,
                        offset = 0
                    )
                    if err:
                        st.error(f"报错：{err}")
                    else:
                        set(S_OV_WHERE, where_condition if where_condition else "")
                        set(S_OV_PAGE, 0)
                        set(f"{KEY_OVERVIEW}_jump_sync", 0)
                        set(S_OV_RECORDS, result)
            with col212:
                if st.button("清除所有筛选", key=f"{KEY_OVERVIEW}_clear_filter", width="stretch", icon=":material/restart_alt:"):
                    set(f"{KEY_OVERVIEW}_clear_sync", True)   # 信号：下轮 rerun 在控件实例化前清空输入
                    set(S_OV_WHERE, "")
                    set(S_OV_PAGE, 0)
                    set(f"{KEY_OVERVIEW}_jump_sync", 0)
                    result, err = query_records(
                        str(config["paths"]["db_dir"]),
                        "*",
                        None,
                        limit = PAGE_SIZE,
                        offset = 0,
                    )
                    if err:
                        st.error(f"报错：{err}")
                    else:
                        set(S_OV_RECORDS, result)
                    st.rerun()

##————标签页5-第一列-第二项:批量修改————
        # with st.container(key = f"{KEY_OVERVIEW}_sql_control_container", border = True):
        with st.expander("**批量修改**", False, key = f"{KEY_OVERVIEW}_sql_control_expander", icon=":material/select_all:"):
            st.markdown("**质量分类批量修改**")
            st.caption("作用于当前筛选结果（WHERE 条件）命中的全部记录。")
            col23, col24 = st.columns([4, 1], vertical_alignment="center")
            with col23:
                batch_quality = st.selectbox("quality_batch", ["good", "bad", "pending", "unlabeled"], key = f"{KEY_OVERVIEW}_batch_qualities", label_visibility = "collapsed")
            with col24:
                page = get(S_OV_PAGE, 0)
                where = get(S_OV_WHERE)
                if st.button("确认", key=f"{KEY_OVERVIEW}_bq_confirm_button", width="stretch", icon=":material/check:"):
                    full_selected, err = query_records(config["paths"]["db_dir"], "*", where)
                    if err:
                        st.error(f"错误:{err}")
                    elif full_selected:
                        qualify_and_move(
                            config["paths"]["db_dir"],
                            target_dir = config["paths"][f"{batch_quality}_dir" if batch_quality == "good" or batch_quality == "bad" else "raw_dir"],
                            quality = batch_quality,
                            if_quantity = True,
                            quantity_records = full_selected
                        )
                        new_list, err = query_records(
                                            str(config["paths"]["db_dir"]),
                                            "*",
                                            where,
                                            limit = PAGE_SIZE,
                                            offset = PAGE_SIZE * page
                                        )
                        if err:
                            st.error(f"错误:{err}")
                        else:
                            set(S_OV_RECORDS, new_list)
                            set(S_TAB5_VERSION, get(S_DB_VERSION, 0))
                            st.toast(f"已完成修改")

            st.markdown("**属性批量修改**")
            attrs_config = config["custom_cols"]
            for key, attr in attrs_config.items():
                attr_config = attrs_config[key]
                col25, col26, col27 = st.columns([1, 3, 1], vertical_alignment="center")
                with col25:
                    label = attr.get('label')
                    st.markdown(f"**{label}:**")
                with col26:
                    multi_attrs_select = st.selectbox(
                        f"multi_attrs_select_{key}",
                        key = f"attrs_select_of_{KEY_OVERVIEW}_col1_{key}",
                        index = None,
                        options = attr_config.get("option", []),
                        label_visibility = "collapsed",
                        width = "stretch"
                    )
                with col27:
                    where = get(S_OV_WHERE, "")
                    page = get(S_OV_PAGE, 0)
                    full_selected, _ = query_records(config["paths"]["db_dir"], "*", where)
                    if st.button(
                        "确认",
                        key = f"attrs_select_button_{KEY_OVERVIEW}_col1_{key}",
                        width = "stretch",
                        icon = ":material/check:"
                    ):
                        if multi_attrs_select is None:
                            st.rerun()
                        else:
                            update_attrs(config["paths"]["db_dir"], full_selected, (key, multi_attrs_select))
                            bump_db_version()
                            new_list, err = query_records(
                                                str(config["paths"]["db_dir"]),
                                                "*",
                                                where,
                                                limit = PAGE_SIZE,
                                                offset = PAGE_SIZE * page
                                            )
                            if err:
                                st.warning(f"错误:{err}")
                            else:
                                set(S_OV_RECORDS, new_list)
                                set(S_TAB5_VERSION, get(S_DB_VERSION, 0))
                                st.toast(f"已批量修改 {len(full_selected)} 条记录的属性")
                                st.rerun()
##————标签页5-第二列:数据列表显示————
    with col22:
        page = get(S_OV_PAGE, 0)
        where = get(S_OV_WHERE)
        with st.container(key = f"{KEY_OVERVIEW}_database_viewer_container", border = True):
            st.markdown("**所有记录**")
            if st.button("获取/刷新数据", key=f"{KEY_OVERVIEW}_refresh", icon=":material/refresh:"):
                records, err = query_records(
                    config["paths"]["db_dir"], "*", where,
                    limit = PAGE_SIZE, offset = page * PAGE_SIZE)
                if err:
                    st.error(err)
                else:
                    set(S_OV_RECORDS, records)
                    set(S_TAB5_VERSION, get(S_DB_VERSION, 0))
            if S_OV_RECORDS in st.session_state:
                records = get(S_OV_RECORDS)
                total = count_list(config["paths"]["db_dir"], where)
                total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

                # 翻页/跳转后同步跳转输入框（须在 number_input 实例化之前设置）
                _sync_page = pop(f"{KEY_OVERVIEW}_jump_sync", None)
                if _sync_page is not None:
                    set(f"{KEY_OVERVIEW}_jump", _sync_page + 1)

                with st.container(border = True):
                    h_id, h_name, h_qual, h_attr = st.columns([1, 3, 2, 3], vertical_alignment = "center")
                    with h_id: st.markdown("**ID**")
                    with h_name: st.markdown("**文件名**")
                    with h_qual: st.markdown("**质量**")
                    with h_attr: st.markdown("**属性**")
                
                if not records:
                    st.caption("当前筛选无记录，可点击「获取/刷新数据」或「清除所有筛选」。")
                with st.container(border = True):
                    for rec in records:
                        col_id, col_name, col_qual, col_attr = st.columns([1, 3, 2, 3], vertical_alignment = "center")
                        with col_id: st.markdown(f"**{rec['id']}**")
                        with col_name:
                            with st.popover(rec['mp4_name'],key = f"{KEY_OVERVIEW}_listname_{rec['id']}", width = "stretch"):
                                st.video(rec['mp4_path'])
                                n_frames = get_video_info(rec['mp4_path'])[0]
                                c22n1, c22n2 = st.columns(2)
                                with c22n1:
                                    frame = get_frame(rec['mp4_path'], 0)
                                    if frame is not None: st.image(frame, width="stretch")
                                with c22n2:
                                    frame = get_frame(rec['mp4_path'], max(0, n_frames - 1))
                                    if frame is not None: st.image(frame, width="stretch")
                        with col_qual:
                            icon = {"good": "✅ ", "bad": "❌ ", "pending": "❎ ", "unlabeled": "⬜ "}.get(rec['quality'], "")
                            with st.popover(f"{icon}{rec['quality']}", key = f"{KEY_OVERVIEW}_listqual_{rec['id']}", width = "stretch"):
                                st.markdown("**质量选择**")
                                quality_module(f"popover_{rec['id']}", False, target = rec)

                        with col_attr:
                            attrs = json.loads(rec['attr']) if rec['attr'] and rec['attr'] != "{}" else {}
                            if attrs:
                                attrs_show = []
                                for k, v in attrs.items():
                                    label = v.get("label", "-") if isinstance(v, dict) else "-"
                                    option = v.get("option", "") if isinstance(v, dict) else ""
                                    attrs_show.append(f"{label}:{option}")
                                text_show = " | ".join(attrs_show) if attrs_show else "-"
                                with st.popover(text_show, key = f"{KEY_OVERVIEW}_listattr_{rec['id']}", width = "stretch"):
                                    st.markdown("**属性修改**")
                                    attrs_module(config["custom_cols"], f"popover_attr_{rec['id']}", if_session_state=False, target=rec)

                st.divider()
                c_p, c_info, c_jump, c_n = st.columns([1, 2, 3, 1], vertical_alignment="center")
                with c_p:
                    if page > 0 and st.button("", key=f"{KEY_OVERVIEW}_page_prev", width = "content",
                                              icon=":material/chevron_left:", help="上一页"):
                        new_page = page - 1
                        records, err = query_records(
                            config["paths"]["db_dir"], "*", where,
                            limit = PAGE_SIZE, offset = new_page * PAGE_SIZE)
                        if not err:
                            set(S_OV_RECORDS, records)
                            set(S_OV_PAGE, new_page)
                            set(f"{KEY_OVERVIEW}_jump_sync", new_page)
                        st.rerun()
                with c_info:
                    st.markdown(f"**第 {page + 1} / {total_pages} 页 · 共 {total} 条**")
                with c_jump:
                    colj1, colj2 = st.columns([2, 1], vertical_alignment="center")
                    with colj1:
                        target = st.number_input("跳转页", min_value = 1, max_value = total_pages,
                                                 step = 1,
                                                 key = f"{KEY_OVERVIEW}_jump",
                                                 label_visibility = "collapsed",
                                                 help = f"输入目标页码（1 ~ {total_pages}）")
                    with colj2:
                        if st.button("跳转", key=f"{KEY_OVERVIEW}_jump_btn", width = "stretch",
                                     icon=":material/arrow_forward:"):
                            if target != page + 1:
                                new_page = target - 1
                                records, err = query_records(
                                    config["paths"]["db_dir"], "*", where,
                                    limit = PAGE_SIZE, offset = new_page * PAGE_SIZE)
                                if not err:
                                    set(S_OV_RECORDS, records)
                                    set(S_OV_PAGE, new_page)
                                    set(f"{KEY_OVERVIEW}_jump_sync", new_page)
                            st.rerun()
                with c_n:
                    if page < total_pages - 1 and st.button("", key=f"{KEY_OVERVIEW}_page_next", width = "content",
                                                             icon=":material/chevron_right:", help="下一页"):
                        new_page = page + 1
                        records, err = query_records(
                            config["paths"]["db_dir"], "*", where,
                            limit = PAGE_SIZE, offset = new_page * PAGE_SIZE)
                        if not err:
                            set(S_OV_RECORDS, records)
                            set(S_OV_PAGE, new_page)
                            set(f"{KEY_OVERVIEW}_jump_sync", new_page)
                        st.rerun()

# 弹窗质量/属性修改、其他 tab 写库后统一刷新当前页
if get(S_DB_VERSION, 0) != get(S_TAB5_VERSION, 0):
    set(S_TAB5_VERSION, get(S_DB_VERSION, 0))
    if S_OV_RECORDS in st.session_state:
        _new_records, _err = query_records(
            config["paths"]["db_dir"], "*", get(S_OV_WHERE),
            limit = PAGE_SIZE, offset = get(S_OV_PAGE, 0) * PAGE_SIZE)
        if not _err:
            set(S_OV_RECORDS, _new_records)
        st.rerun()

##————标签页6:配置界面————
with tab6:
    st.subheader("自定义属性配置")
    st.caption("新增 / 编辑 / 删除自定义属性，或导入导出配置备份。")
##————标签页6:配置界面 - 新建属性模块————
    with st.container(key = f"{KEY_CONFIG}_settings_container", border = True):
        with st.expander("新建属性", expanded = False, icon=":material/add_box:"):
            col311, col312 = st.columns(2, vertical_alignment="center")
            with col311:
                new_key = st.text_input("属性键名（英文名）", placeholder="例: attr_method")
                new_label = st.text_input("显示名称", placeholder="例: 朝向")
            with col312:
                new_type = st.selectbox("输入类型", [
                    "select",         # 下拉单选（最常用）
                    "multi_select",   # 多选标签
                    "text",           # 短文本
                    "number",         # 数字
                    "boolean",        # 是/否
                ])
                new_options = st.text_input("选项（逗号分隔）", placeholder="例: 垂直,倾斜,水平")
            if st.button("确认添加", icon=":material/add:"):
                if not new_key or not new_label:
                    st.error("属性键名和显示名称不能为空")
                else:
                    config["custom_cols"][new_key] = {
                        "label" : new_label,
                        "option" : [ opt.strip() for opt in new_options.split(",") if opt.strip()],
                        "type" : new_type
                    }
                    save_config(config)
                    st.toast(f"已添加{new_label}属性")
##————标签页6:配置界面 - 属性编辑模块————
        if config["custom_cols"]:
            st.markdown("**已保存自定义属性**")
            for col_name, col_info in config["custom_cols"].items():
                with st.container(key = f"{KEY_CONFIG}_para1_container{col_name}", border = True):
                    col321, col322, col323 = st.columns([4, 1, 1], vertical_alignment="center")
                    with col321:
                        st.text_input("已经保存自定义属性", placeholder = f"{col_info['label']} — {'/ '.join(col_info['option'])} — type：{col_info['type']}", label_visibility = "collapsed")
                    with col322:
                        with st.popover(f"编辑: {col_info['label']} 属性", icon=":material/build:"):
                            new_label = st.text_input("显示名称", value=col_info.get("label", ""), key=f"el_{col_name}")
                            new_type = st.selectbox("输入类型", ["select","multi_select","text","number","boolean"],
                                                    index=["select","multi_select","text","number","boolean"].index(col_info.get("type","select")),
                                                    key=f"et_{col_name}")
                            new_options = st.text_input("选项（逗号分隔）",
                                                        value=", ".join(col_info.get("options", [])),
                                                        key=f"eo_{col_name}")
                            if st.button("保存修改", key=f"save_{col_name}", icon=":material/save:"):
                                config["custom_cols"][col_name].update({
                                    "label": new_label,
                                    "type": new_type,
                                    "option": [o.strip() for o in new_options.split(",") if o.strip()]
                                })
                                save_config(config)
                                st.session_state.pop(f"editing_{col_name}", None)
                                st.rerun()
                    with col323:
                        if st.button("删除", key = f"{KEY_CONFIG}_para1_delete{col_name}", width="stretch", icon=":material/delete:"):
                            st.session_state[f"confirm_del_{col_name}"] = True
                    if st.session_state.get(f"confirm_del_{col_name}"):
                        st.warning(f"确定删除「{col_info['label']}」吗？")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("确认删除", key=f"confirm_yes_{col_name}", icon=":material/check:"):
                                del config["custom_cols"][col_name]
                                save_config(config)
                                st.session_state.pop(f"confirm_del_{col_name}", None)
                                st.rerun()
                        with c2:
                            if st.button("取消", key=f"confirm_no_{col_name}", icon=":material/close:"):
                                st.session_state.pop(f"confirm_del_{col_name}", None)
                                st.rerun()
        else:
            st.info("暂无自定义属性，在上方添加")
            
        st.divider()
        st.markdown("**导入 / 导出配置**")
        cola, colb = st.columns(2, vertical_alignment="center")
        with cola:
            st.download_button(
                "导出属性配置",
                data=json.dumps(config["custom_cols"], ensure_ascii=False, indent=2),
                file_name="custom_cols_backup.json",
                mime="application/json",
                width="stretch",
                icon=":material/download:",
            )
        with colb:
            uploaded = st.file_uploader("导入属性配置", type="json", label_visibility="collapsed")
            if uploaded:
                try:
                    data = json.load(uploaded)
                    if isinstance(data, dict):
                        config["custom_cols"] = data
                        save_config(config)
                        st.success(f"成功导入 {len(data)} 个属性")
                        st.rerun()
                    else:
                        st.error("格式错误：需要 JSON 对象")
                except:
                    st.error("文件解析失败")

# with st.expander("🔍 调试：查看 session_state"):
#     st.json(st.session_state.to_dict())