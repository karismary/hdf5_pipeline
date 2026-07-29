#该文件大部分为古法手搓，请放心食用

import json
import copy
import streamlit as st
# from streamlit_file_browser import st_file_browser 
from pathlib import Path
import sqlite3
# import cv2
import shutil
import pandas as pd

from hdf5_pipeline.core.hdf5_utils import get_sorted_files
from hdf5_pipeline.core.video_utils import get_video_info,get_frame
from hdf5_pipeline.label.database import init_db, get_unlabeled, add_label, scan_pairs, get_list, get_records, translate_where, query_records
from hdf5_pipeline.core.config import load_config, save_config
from hdf5_pipeline.core.utils import pick_folder
from hdf5_pipeline.preview.rename_tab import show_tab as show_rename
from hdf5_pipeline.preview.quality_tab import show_tab as show_quality
from hdf5_pipeline.preview.render_tab import show_tab as show_render

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

def seclect_folder(path_type: str) -> None:
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

def qualify_and_move(db_path: str, mp4_name: str, raw_path: str, target_dir: str, quality: str, if_rerun: bool = True, compare_quality: str = None) -> None:
    """标记质量并移动 HDF5 文件到对应目录。

    如果目标质量与当前相同则跳过。
    移动 HDF5 文件后更新数据库的 quality 和 hdf5_path。

    Args:
        db_path (str): 数据库文件路径。
        mp4_name (str): MP4 文件名，用于定位数据库记录。
        raw_path (str): HDF5 文件当前完整路径。
        target_dir (str): 目标文件夹（good_dir 或 bad_dir）。
        quality (str): 目标质量标签，'good' 或 'bad'。
        if_rerun (bool, optional): 是否触发 st.rerun()，默认 True。
        compare_quality (str, optional): 当前质量值，用于判断重复操作。
            不传时自动从 st.session_state["selected"] 读取。

    Returns:
        None。
    """
    if compare_quality is None:
        if quality == st.session_state["selected"][5]:
            st.toast(f'已设置为"{st.session_state["selected"][5]}"')
            return
    else:
        if compare_quality == quality:
            st.rerun()
            return
    if Path(target_dir).exists():
        shutil.move(raw_path, target_dir)
        nhdf5_path = Path(target_dir)/Path(raw_path).name
        add_label(db_path, mp4_name, str(nhdf5_path), quality, None)
        lists = get_list(db_path)
        st.session_state["records"] = lists
        st.session_state["selected"] = get_records(db_path, mp4_name)
        if if_rerun:
            st.rerun()
    else:
        st.error(f"文件夹：{target_dir}不存在，请检查")

# def qualitify(db_paths, mp4_names, qualities):
#     add_label(db_paths, mp4_names, quality = qualities)
#     lists = get_list(db_paths)
#     st.session_state["records"] = lists
#     st.session_state["selected"] = get_records(db_paths, mp4_names)
#     st.toast(f"{st.session_state['selected'][3]}已完成质量分类，至{f'✅ GOOD QUALITY' if qualities == 'good' else '❌ BAD QUALITY'}")

def quality_module(key_name: str, if_session_state: bool = True, session_state: str = "selected", target: tuple = None) -> None:
    """渲染 GOOD/BAD 两个质量标记按钮。

    支持两种模式：
    1. session_state 模式（if_session_state=True）— 从 session_state 读当前记录，用于 tab1。
    2. target 模式（if_session_state=False）— 从 target 参数取记录数据，用于 tab2 popover。

    Args:
        key_name (str): 按钮 key 前缀，确保唯一性。
        if_session_state (bool): True 从 session_state 读记录，False 从 target 取记录。
        session_state (str): session_state 的键名，默认 "selected"。
        target (tuple, optional): 记录元组，if_session_state=False 时必传。

    Returns:
        None。
    """
    sel = st.session_state.get("selected") if if_session_state else target
    colnq1, colnq2 = st.columns(2)
    with colnq1:
        if st.button(" ✅ ", key = f"botton_good_{key_name}:{sel[3]}", width = "stretch"):
                if not if_session_state:
                    st.session_state["_popover_quality_changed"] = True
                qualify_and_move(
                    config["paths"]["db_dir"],   # 数据库路径
                    sel[3],                       # mp4_name
                    sel[2],                       # hdf5 当前路径
                    str(Path(config["paths"]["good_dir"])),  # hdf5 目标路径
                    "good",                     # 质量标签
                    compare_quality = sel[5]
                )
                st.rerun()
    with colnq2:
        if st.button(" ❌ ", key = f"botton_bad_{key_name}:{sel[3]}", width = "stretch"):
                if not if_session_state:
                    st.session_state["_popover_quality_changed"] = True
                qualify_and_move(
                    config["paths"]["db_dir"],   # 数据库路径
                    sel[3],                       # mp4_name
                    sel[2],                       # hdf5 当前路径
                    str(Path(config["paths"]["bad_dir"])),  # hdf5 目标路径
                    "bad",                     # 质量标签
                    compare_quality = sel[5]
                )
                st.rerun()

def attrs_module(template_attrs: dict, key_name: str, if_session_state: bool = True, session_state: str = "selected", target: tuple = None) -> None:
    """渲染属性修改下拉框 + 确认按钮。

    支持两种模式：
    1. session_state 模式 — 从 session_state 读当前记录，用于 tab1。
    2. target 模式 — 从 target 取记录数据，用于 tab2 popover。

    Args:
        template_attrs (dict): config["custom_cols"] 属性配置。
        key_name (str): 按钮 key 前缀。
        if_session_state (bool): True 从 session_state 读，False 从 target 取。
        session_state (str): session_state 键名，默认 "selected"。
        target (tuple, optional): 记录元组，target 模式时必传。

    Returns:
        None。
    """
    sel = st.session_state.get("selected") if if_session_state else target
    try:
        db_attrs = json.loads(sel[6]) if sel and sel[6] and sel[6] != "{}" else {}
    except:
        db_attrs = {}

    for keys, attrs in template_attrs.items():

        raw_val = db_attrs.get(keys, "未选择")
        current_val = raw_val.get("option", "未选择") if isinstance(raw_val, dict) else raw_val
        original_options = attrs["option"]
        current_idx = original_options.index(current_val) if current_val in original_options else 0

        widget_key = f"attrs_{key_name}_{keys}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = original_options.index(current_val) if current_val in original_options else 0

        cola1, cola2, cola3 = st.columns([1, 4, 1], gap="xxsmall")
        with cola1:
            st.markdown(f"**{attrs['label']}:**")
        with cola2:
            attr_selected = st.selectbox(
                f"{attrs['label']}",
                key=widget_key,
                options=original_options,
                index = current_idx,
                label_visibility="collapsed"
            )
        with cola3:
            if st.button("确认", key=f"confirm_{key_name}_{keys}", use_container_width=True):
                payload_to_save = copy.deepcopy(db_attrs)
                payload_to_save[keys] = {
                    "label": attrs['label'],
                    "option": attr_selected if attr_selected else "未选择",
                    "type": attrs.get('type', 'text')
                }
                mp4_name = sel[3]
                add_label(config["paths"]["db_dir"], mp4_name, attr=payload_to_save)
                if if_session_state:
                    st.session_state["records"] = get_list(config["paths"]["db_dir"])
                    if widget_key in st.session_state:
                        del st.session_state[widget_key]
                    st.session_state["_toast_msg"] = f"属性 **{attrs['label']}** 已设为：**{attr_selected}**"
                    st.rerun()
                if not if_session_state:
                    st.session_state["_popover_attrs_changed"] = True
                    # 清掉 attrs widget 缓存，下次打开从数据库读最新值
                    for wk in list(st.session_state.keys()):
                        if wk.startswith(f"attrs_{key_name}_"):
                            del st.session_state[wk]

st.set_page_config(page_title="HDF5 Labeling",
                   layout="wide",
                   initial_sidebar_state="expanded")

with open("./hdf5_pipeline/label/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

config = load_config("config.json")
path_dict = {
    "数据库路径(没有文件会在选择目录下自动创建)":"db_dir",
    "hdf5文件夹路径":"raw_dir",
    "mp4文件夹路径":"mp4_dir",
    "good_quality_hdf5存储路径":"good_dir",
    "bad_quality_hdf5存储路径":"bad_dir"
}

st.title("🗿HDF5 Labeling Tool")
st.write("这是一个用于给 HDF5 数据打标的工具，支持质量分类和属性的标注。")

# 处理跨 rerun 的 toast 消息
if "_toast_msg" in st.session_state:
    st.toast(st.session_state.pop("_toast_msg"))

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

#————主页面：————（根据需求添加后续标签）
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 文件重命名", "🧹 质量检测", "🎬 视频渲染",
    "🏷️ 视频打标", "📊 数据总览", "⚙️ 配置"
])

#————标签页1:文件重命名————
with tab1: show_rename()
#————标签页2:质量检测————
with tab2: show_quality()
#————标签页3:视频渲染————
with tab3: show_render()
#————标签页4:打标界面————
with tab4:
    # 检测数据是否需要刷新
    if "records" in st.session_state:
        current_count = len(st.session_state["records"])
        db_count = len(get_list(str(Path(config["paths"]["db_dir"]))))
        if current_count != db_count:
            st.session_state["records"] = get_list(str(Path(config["paths"]["db_dir"])))
    st.subheader("标记记录")
    col11, col12 = st.columns([1.5,3])
#————标签页1-第一列：工作记录选择————
    with col11:
        with st.container(key = "tag1_col1_container", border = True):
            next_flag = False
            # st.markdown("<span class='tag1_col1_border'></span>", unsafe_allow_html=True)
            st.markdown("**文件列表**")
            if st.button("获取并更新记录列表", key="byn_fetch", width="stretch"):
                if Path(config["paths"]["db_dir"]).exists():
                    next_flag = True
                    db_path = Path(config["paths"]["db_dir"])
                    records_list = get_list(str(db_path))
                    st.session_state["records"] = records_list
                else:
                    st.error("未查找到对应数据库，请展开侧边栏选择路径并创建")
            if "records" in st.session_state:
                records = st.session_state["records"]
                selected_idx = st.selectbox(
                    "**选择文件**",
                    range(len(records)),                       # 实际选项：0, 1, 2...
                    format_func=lambda i: f"{records[i][5]} - {records[i][0]}.{records[i][3].replace('.mp4','')}",         # 显示：文件名
                    label_visibility="visible",
                    index = st.session_state.get("selected_index", 0)
                )
                st.session_state["selected_index"] = selected_idx

                records = st.session_state["records"]
                total = len(records)
                good = sum(1 for r in records if r[5] == "good")
                bad = sum(1 for r in records if r[5] == "bad")
                labeled =good + bad
                progress = labeled / total if total > 0 else 0
                st.progress(progress, text = f"📊 {labeled}/{total} 已完成标记")

                col1101, col1102 = st.columns(2)
                with col1101:
                    if st.button("⬅️ 上一条", use_container_width = True):
                        if selected_idx > 0:
                            st.session_state["selected_index"] = selected_idx - 1
                            new_idx = st.session_state["selected_index"]
                            st.session_state["selected"] = records[new_idx]
                            st.rerun()
                with col1102:       
                    if st.button("➡️ 下一条", use_container_width = True):
                        if selected_idx < len(records) - 1:
                            st.session_state["selected_index"] = selected_idx + 1
                            new_idx = st.session_state["selected_index"]
                            st.session_state["selected"] = records[new_idx]
                            st.rerun()
                        
                if selected_idx is not None:
                    next_flag = True
                    st.session_state["selected"] = records[st.session_state["selected_index"]]
                    st.markdown(f"""
                                🗿标注状态：**{'✅' if st.session_state['selected'][5]=='good' else '❌' if st.session_state['selected'][5]=='bad' else '☑️'}-{st.session_state['selected'][5]}**\n
                                🗿名称：**{st.session_state['selected'][1]}**
                                """)
#————标签页1-第一列：属性设置-质量分类————
        if next_flag:
            with st.container(key = "tag1_col1_attr_checker", border = True):
                    with st.container(key = "tab1_col1_qualities", border = False):
                        st.markdown("**质量分类**")
                        quality_module("tab1_quality_button")
#————标签页1-第一列：属性设置-属性修改————
                    with st.container(key = "tag1_col1_attr_settings", border = False):
                        st.markdown("**属性设置**")
                        if config["custom_cols"]:
                            attrs_module(
                                config["custom_cols"],
                                f"tab1_attr_{st.session_state['selected'][0]}",
                                if_session_state=True
                            )
#————标签页1-第二列：工作区域————
#————标签页1-第二列：工作区域-视频区域
    with col12:
        with st.container(key = "tag1_col2_videobox", border = next_flag):
            sel = st.session_state.get("selected")
            if sel:
                st.markdown(f"**{(sel[4].split('/'))[-1]}**")
                st.video(sel[4])
#————标签页1-第二列：工作区域-图片区域
        with st.container(key = "tag1_col2_graphbox",border = next_flag):
            sel = st.session_state.get("selected")
            if sel:
                video_path = sel[4]
                n_frames = get_video_info(video_path)[0]
                col121, col122 = st.columns([1,1])
                with col121:
                    first_frame = get_frame(video_path,0)
                    st.image(first_frame, caption = "首帧", use_container_width = True)
                with col122:
                    last_frame = get_frame(video_path, n_frames - 1)
                    st.image(last_frame, caption = "末帧", use_container_width = True)

##————标签页5:数据总览操作界面————
with tab5:
    st.subheader("数据总览")
    col21, col22 = st.columns([1.5,3])
##————标签页2-第一列:数据批量处理————
##————标签页2-第一列-第一项:查询筛选————
    with col21:
        with st.container(key = "tag2_sql_choose_container", border = True):
            st.markdown("**自定义查询**")
        #     all_columns = {
        #         "ID": "id", "HDF5名": "hdf5_name", "HDF5路径": "hdf5_path",
        #         "MP4名": "mp4_name", "MP4路径": "mp4_path", "质量": "quality",
        #         "属性": "attr", "创建时间": "created_at", "打标时间": "labeled_at"
        #     }
        #     selected_cols = st.multiselect(
        #         "**SELECT**（筛选目标列）",
        #         options = list(all_columns.keys()),
        #         default = None,
        #         placeholder = "可多选"
        #     )
            sql_where = st.text_area("**WHERE**（筛选条件）",
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
            if st.session_state.get("selected_button") is None:
                st.session_state["selected_button"] = False
            if st.button("执行查询", key="tag2_query", use_container_width=True):
                try:
                    col_names = ", ".join(all_columns[c] for c in selected_cols) if selected_cols else "*"
                except NameError:
                    col_names = "*"
                where_condition = None
                if sql_where.strip():
                    where_condition = translate_where(sql_where.strip(), attr_map)
                st.toast(f"执行 SQL: SELECT {col_names} FROM label WHERE {where_condition}")
                result, err = query_records(
                    str(config["paths"]["db_dir"]),
                    col_names,
                    where_condition
                )
                if err:
                    st.error(f"报错：{err}")
                else:
                    st.session_state["tag2_records"] = result
                    st.session_state["selected_button"] = True
        if st.session_state["selected_button"] and sql_where:
##————标签页2-第一列-第二项:批量修改————
            with st.container(key = "tag2_sql_control_container", border = True):
                st.markdown("**质量分类批量修改**")
                col23, col24 = st.columns([4,1])
                with col23:
                    batch_quality = st.selectbox("quality_batch", ["good", "bad", "unlabeled"], key = "tag2_batch_qualities", label_visibility = "collapsed")
                with col24:
                    if st.button("确认", "tag2_bq_confirm_button", use_container_width = True):
                        if st.session_state["tag2_records"] and st.session_state["tag2_records"] != []:
                            count = 0
                            new_list = []
                            tag2_selected = st.session_state.get("tag2_records")
                            for selected_record in tag2_selected:
                                qualify_and_move(config["paths"]["db_dir"],
                                                selected_record[3],
                                                selected_record[2],
                                                config["paths"][f"{batch_quality}_dir" if batch_quality == "good" or batch_quality == "bad" else "raw_dir"],
                                                batch_quality,
                                                False,
                                                selected_record[5]
                                                )
                                count += 1
                                new_record = get_records(config["paths"]["db_dir"], selected_record[3])
                                new_list.append(new_record)
                                new_quality = new_record[5]
                                st.session_state[f"t2_q_{new_record[0]}"] = (
                                    ["unlabeled", "good", "bad"].index(new_quality)
                                    if new_quality in ["unlabeled", "good", "bad"] else 0
                                )
                            st.session_state["tag2_records"] = new_list
                            st.toast(f"已修改 {count} 条记录，点击「获取/刷新数据」查看最新结果")
                st.markdown("**属性批量修改**")
                attrs_config = config["custom_cols"]
                for key, attr in attrs_config.items():
                    attr_config = attrs_config[key]
                    col25, col26, col27 = st.columns([1,3,1])
                    with col25:
                        st.write(f"{attr.get('label')}:")
                    with col26:
                        multi_attrs_select = st.selectbox(
                            f"multi_attrs_select_{key}",
                            key = f"attrs_select_of_tab2_col1_{key}",
                            index = None,
                            options = attr_config.get("option", []),
                            label_visibility = "collapsed",
                            width = "stretch"
                        )
                    with col27:
                        selected_records = st.session_state.get("tag2_records", [])
                        if st.button(
                            f"确认",
                            key = f"attrs_select_button_tab2_col1_{key}",
                            width = "stretch"
                        ):
                            if multi_attrs_select is None:
                                st.rerun()
                            else:
                                new_records = []
                                for record in selected_records:
                                    raw_attr = record[6]
                                    raw_attr_dict = json.loads(raw_attr)
                                    if key not in raw_attr_dict:
                                        raw_attr_dict[key] = {"label": attr_config["label"], "option": "未选择", "type": attr_config.get("type", "text")}
                                    raw_attr_dict[key]["option"] = multi_attrs_select
                                    add_label(config["paths"]["db_dir"], record[3], attr = raw_attr_dict)
                                new_records = []
                                for rec in st.session_state["tag2_records"]:
                                    latest = get_records(config["paths"]["db_dir"], rec[3])
                                    new_records.append(latest if latest else rec)
                                st.session_state["tag2_records"] = new_records
                                st.toast(f"已批量修改 {len(selected_records)} 条记录的属性")
                                st.rerun()
##————标签页2-第二列:数据列表显示————
    with col22:
        with st.container(key = "tag2_database_viewer_container", border = True):
            st.markdown("**所有记录**")
            if st.button("获取/刷新数据", key="tag2_refresh"):
                records = get_list(str(Path(config["paths"]["db_dir"])))
                st.session_state["tag2_records"] = records.copy()
            if "tag2_records" in st.session_state:
                records = st.session_state["tag2_records"]
                page = st.session_state.get("tag2_page", 0)
                page_size = 10
                
                total = len(records)
                total_pages = max(1, (total + page_size - 1) // page_size)
                start = page * page_size
                end = min(start + page_size, total)
                
                with st.container(border = True):
                    h_id, h_name, h_qual, h_attr = st.columns([1, 3, 2, 3], vertical_alignment = "center")
                    with h_id: st.markdown("**ID**")
                    with h_name: st.markdown("**文件名**")
                    with h_qual: st.markdown("**质量**")
                    with h_attr: st.markdown("**属性**")
                
                with st.container(border = True):
                    for rec in records[start:end]:
                        # with st.container(border = True, gap = "xxsmall",height = "stretch"):
                        col_id, col_name, col_qual, col_attr = st.columns([1, 3, 2, 3], vertical_alignment = "center")
                        with col_id: st.markdown(f"**{rec[0]}**")
                        # with col_name: st.button(f"{rec[3]}", key = f"t2_ln_{rec[0]}", disabled = True, width = "stretch")
                        with col_name:
                            with st.popover(rec[3],key = f"tab2_listname_{rec[0]}", width = "stretch"):
                                st.video(rec[4])
                                n_frames = get_video_info(rec[4])[0]
                                c22n1, c22n2 = st.columns(2)
                                with c22n1:
                                    frame = get_frame(rec[4], 0)
                                    if frame is not None: st.image(frame, use_container_width=True)
                                with c22n2:
                                    frame = get_frame(rec[4], max(0, n_frames - 1))
                                    if frame is not None: st.image(frame, use_container_width=True)
                        with col_qual:
                            icon = {"good": "✅ ", "bad": "❌ ", "unlabeled": "⬜ "}.get(rec[5], "")
                            # st.button(f"{icon}{rec[5]}", key = f"t2_lq_{rec[0]}", disabled = True, width = "stretch")
                            with st.popover(f"{icon}{rec[5]}", key = f"tab2_listqual_{rec[0]}", width = "stretch"):
                                st.markdown("**质量选择**")
                                quality_module(f"popover_{rec[0]}", False, target = rec)

                        if st.session_state.pop("_popover_quality_changed", False):
                            st.session_state["tag2_records"] = get_list(str(Path(config["paths"]["db_dir"])))
                            st.rerun()

                                
                        with col_attr:
                            # if  f"attrs_of_{rec[1]}" not in st.session_state:
                            #     st.session_state[f"attrs_of_{rec[1]}"] = []
                            attrs = json.loads(rec[6]) if rec[6] and rec[6] != "{}" else {}
                            if attrs:
                                attrs_show = []
                                for k, v in attrs.items():
                                    label = v.get("label", "-") if isinstance(v, dict) else "-"
                                    option = v.get("option", "") if isinstance(v, dict) else ""
                                    attrs_show.append(f"{label}:{option}")
                                text_show = " | ".join(attrs_show) if attrs_show else "-"
                                with st.popover(text_show, key = f"tab2_listattr_{rec[0]}", width = "stretch"):
                                    st.markdown("**质量选择**")
                                    attrs_module(config["custom_cols"], f"popover_attr_{rec[0]}", if_session_state=False, target=rec)

                                if st.session_state.pop("_popover_attrs_changed", False):
                                    st.session_state["tag2_records"] = get_list(str(Path(config["paths"]["db_dir"])))
                                    st.rerun()
                                    # st.text_input(f"attrs_show_{rec[1]}",
                                    #             placeholder = ' | '.join(attrs_show) if attrs_show else "-",
                                    #             key = f"show_attrs_{rec}",
                                    #             disabled =True,
                                    #             label_visibility = "collapsed")
                
                st.divider()
                c_space_left, c_p, c_info, c_n, c_space_right = st.columns([3, 1, 2, 1, 3], vertical_alignment="center")
                with c_p:
                    if page > 0 and st.button("⬅️", width = "content"):
                        st.session_state["tag2_page"] = page - 1
                        st.rerun() 
                with c_info:
                    st.button(
                        f"第 {page+1}/{total_pages} 页（共 {total} 条）", 
                        disabled=True, 
                        width = "content"
                    )
                with c_n:
                    if page < total_pages - 1 and st.button("➡️", width = "content"):
                        st.session_state["tag2_page"] = page + 1
                        st.rerun()

##————标签页6:配置界面————
with tab6:
    st.subheader("自定义属性配置")
##————标签页3:配置界面 - 新建属性模块————
    with st.container(key = "tag3_settings_container", border = True):
        with st.expander("➕ 新建属性", expanded = False):
            col311,col312 = st.columns(2)
            with col311:
                new_key = st.text_input("属性键名（英文名）", placeholder="例: attr_method")
                new_label = st.text_input("显示名称", placeholder="例: 朝向")
            with col312:
                new_type = st.selectbox("输入类型", [#重力束缚的灵魂啊
                    "select",         # 下拉单选（最常用）
                    "multi_select",   # 多选标签
                    "text",           # 短文本
                    "number",         # 数字
                    "boolean",        # 是/否
                ])
                new_options = st.text_input("选项（逗号分隔）", placeholder="例: 垂直,倾斜,水平")   
            if st.button("✅ 确认添加"):
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
##————标签页3:配置界面 - 属性编辑模块————
        if config["custom_cols"]:
            st.markdown("**已保存自定义属性**")
            for col_name, col_info in config["custom_cols"].items():
                with st.container(key = f"tag3_para1_container{col_name}", border = True):
                    col321, col322,col323= st.columns([4,1,1])
                    with col321:
                        st.text_input("已经保存自定义属性", placeholder = f"{col_info['label']} — {'/ '.join(col_info['option'])} — type：{col_info['type']}", label_visibility = "collapsed")
                    with col322:
                        # if st.button("✏️编辑", key=f"edit_{col_name}", use_container_width=True):
                        #     st.session_state[f"editing_{col_name}"] = not st.session_state.get(f"editing_{col_name}", False)
                        # if st.session_state.get(f"editing_{col_name}"):
                        with st.popover(f"展开以 🔧 编辑: {col_info['label']}属性"):
                            new_label = st.text_input("显示名称", value=col_info.get("label", ""), key=f"el_{col_name}")
                            new_type = st.selectbox("输入类型", ["select","multi_select","text","number","boolean"],
                                                    index=["select","multi_select","text","number","boolean"].index(col_info.get("type","select")),
                                                    key=f"et_{col_name}")
                            new_options = st.text_input("选项（逗号分隔）", 
                                                        value=", ".join(col_info.get("options", [])),
                                                        key=f"eo_{col_name}")
                            if st.button("保存修改", key=f"save_{col_name}"):
                                config["custom_cols"][col_name].update({
                                    "label": new_label,
                                    "type": new_type,
                                    "option": [o.strip() for o in new_options.split(",") if o.strip()]
                                })
                                save_config(config)
                                st.session_state.pop(f"editing_{col_name}", None)
                                st.rerun()
                    with col323:
                        if st.button("🗑️删除", key = f"tag3_para1_delete{col_name}", use_container_width = True):
                            st.session_state[f"confirm_del_{col_name}"] = True
                    if st.session_state.get(f"confirm_del_{col_name}"):
                        st.warning(f"确定删除「{col_info['label']}」吗？")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("✅ 确认删除", key=f"confirm_yes_{col_name}"):
                                del config["custom_cols"][col_name]
                                save_config(config)
                                st.session_state.pop(f"confirm_del_{col_name}", None)
                                st.rerun()
                        with c2:
                            if st.button("❌ 取消", key=f"confirm_no_{col_name}"):
                                st.session_state.pop(f"confirm_del_{col_name}", None)
                                st.rerun()
        else:
            st.info("暂无自定义属性，在上方添加")
            
        st.divider()
        st.markdown("**导入/导出配置**")
        cola, colb = st.columns(2)
        with cola:
            st.download_button(
                "📤 导出属性配置",
                data=json.dumps(config["custom_cols"], ensure_ascii=False, indent=2),
                file_name="custom_cols_backup.json",
                mime="application/json",
                use_container_width=True
            )
        with colb:
            uploaded = st.file_uploader("📥 导入属性配置", type="json", label_visibility="collapsed")
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

with st.expander("🔍 调试：查看 session_state"):
    st.json(st.session_state.to_dict())
