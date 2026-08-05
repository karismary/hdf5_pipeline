# 启动: streamlit run hdf5_pipeline/label/app.py --server.port=8501

import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from hdf5_pipeline.core.hdf5_utils import get_sorted_files

def init_db(db_path: str) -> None:
    """初始化数据库，创建 label 表（如果不存在）。

    自动建表，不会覆盖已有数据。
    表结构记录了每个视频文件对应的 HDF5、MP4 的完整路径、
    打标结果、自定义属性以及时间信息。

    Args:
        db_path (str): SQLite 数据库文件路径。文件不存在时会自动创建。

    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS label (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        hdf5_name TEXT NOT NULL UNIQUE,
                        hdf5_path TEXT NOT NULL,
                        mp4_name TEXT NOT NULL,
                        mp4_path TEXT NOT NULL,
                        quality TEXT DEFAULT 'unlabeled',
                        attr TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT '',
                        labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     )''')
    conn.commit()
    conn.close()

def get_unlabeled(db_path: str, mp4_dir: str) -> List[str]:
    """扫描 mp4_dir，返回尚未打标的 MP4 文件名列表。

    通过对比 mp4_dir 目录下所有 .mp4 文件与数据库记录，
    筛除已有打标记录的文件，只返回标记为 'unlabeled' 的文件。

    Args:
        db_path (str): SQLite 数据库文件路径。
        mp4_dir (str): 存放 MP4 视频文件的目录路径。

    Returns:
        List[str]: 未打标的 MP4 文件名列表，例如 ['episode_000000.mp4', ...]。
    """
    all_names = {n.name for n in Path(mp4_dir).glob("*.mp4")}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT mp4_name FROM label WHERE quality != 'unlabeled'")
    rows = cursor.fetchall()
    labeled_names = {row[0] for row in rows}
    unlabeled_names = all_names - labeled_names
    conn.close()

    return list(unlabeled_names)

def scan_pairs(db_path: str, mp4_dir: str, raw_dir: str) -> int:
    """扫描 mp4_dir 与 raw_dir，将配对的 MP4+HDF5 注册到数据库。

    遍历所有 MP4 文件，在 raw_dir 中查找同名的 .hdf5 文件。
    找到配对后将文件名和完整路径写入数据库，跳过已有记录，不会覆盖已打标的文件。

    Args:
        db_path (str): SQLite 数据库文件路径。
        mp4_dir (str): MP4 视频文件所在目录。
        raw_dir (str): 原始 HDF5 文件所在目录。

    Returns:
        int: 新注册的文件对数。
    """
    init_db(db_path)
    mp4_list = get_sorted_files(Path(mp4_dir), ".mp4", 1)
    raw_set = set(get_sorted_files(Path(raw_dir), ".hdf5", 1))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    count = 0

    for m in mp4_list:
        h = m.replace(".mp4", ".hdf5")
        if h not in raw_set:
            continue
        try:
            m_path = f"{mp4_dir}{m}"
            h_path = f"{raw_dir}{h}"
            # 读取 MP4 文件修改时间作为 created_at
            mp4_file = Path(m_path)
            created_at = datetime.fromtimestamp(mp4_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S') if mp4_file.exists() else '-'
            cursor.execute(
                "INSERT INTO label (hdf5_name, hdf5_path, mp4_name, mp4_path, created_at) VALUES (?, ?, ?, ?, ?)",
                (h, h_path, m, m_path, created_at)
            )
            count += 1
        except sqlite3.IntegrityError:   # 已存在，跳过
            pass

    conn.commit()
    conn.close()
    return count

def add_label(db_path: str, mp4_name: str, hdf5_path: str = None, quality: str = None, attr: dict = None) -> None:
    """更新一个视频文件的打标结果。

    根据 mp4_name 定位记录，只更新传入的字段，不传的保持不变。
    labeled_at 自动设为当前时间（GMT+8）。

    Args:
        db_path (str): SQLite 数据库文件路径。
        mp4_name (str): MP4 文件名，例如 'episode_000000.mp4'（必填，用于定位记录）。
        hdf5_path (str, optional): HDF5 文件更新后的完整路径。文件移动后更新此字段。
        quality (str, optional): 打标结果，'good' 或 'bad'。不传则不更新。
        attr (dict, optional): 自定义属性字典。不传则不更新。

    Returns:
        None
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    set_clauses = ["labeled_at = ?"]
    values = [now]

    if hdf5_path is not None:
        set_clauses.append("hdf5_path = ?")
        values.append(hdf5_path)
    if quality is not None:
        set_clauses.append("quality = ?")
        values.append(quality)
    if attr is not None:
        set_clauses.append("attr = ?")
        values.append(json.dumps(attr))

    if len(set_clauses) == 1:
        return

    values.append(mp4_name)
    sql = f"UPDATE label SET {', '.join(set_clauses)} WHERE mp4_name = ?"

    conn = sqlite3.connect(db_path)
    conn.execute(sql, tuple(values))
    conn.commit()
    conn.close()

def get_list(db_path: str) -> List[Tuple[int, str, str, str, str, str, str, str, str]]:
    """从数据库获取所有记录的完整列表。

    查询 label 表中全部字段。

    Args:
        db_path (str): SQLite 数据库文件路径。

    Returns:
        list[tuple]: [(id, hdf5_name, hdf5_path, mp4_name, mp4_path, quality, attr, created_at, labeled_at), ...]。
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, hdf5_name, hdf5_path, mp4_name, mp4_path, quality, attr, created_at, labeled_at FROM label")
    row = cursor.fetchall()
    conn.close()

    return row

def get_records(db_path: str, mp4_name: str):
    """按 mp4_name 查询单条记录。

    Args:
        db_path (str): 数据库文件路径。
        mp4_name (str): MP4 文件名。

    Returns:
        tuple 或 None: 记录元组 (id, hdf5_name, ..., labeled_at)，不存在返回 None。
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, hdf5_name, hdf5_path, mp4_name, mp4_path, quality, attr, created_at, labeled_at FROM label WHERE mp4_name =?", (mp4_name, ))
    row = cursor.fetchone()
    conn.close()

    return row

def translate_where(condition, attr_map):
    result = condition
    for label, json_path in attr_map.items():
        pattern = rf'@{re.escape(label)}\s*(=|!=|>=|<=|>|<| LIKE | NOT LIKE | IS | IS NOT )\s*"([^"]*)"'
        replacement = rf"json_extract(attr, '{json_path}') \1 '\2'"
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result

def query_records(db_path: str, columns: str, where_clause: str | None = None):
    """执行自定义查询，返回结果列表。

    Args:
        db_path (str): 数据库路径。
        columns (str): SELECT 的列名，如 "id, quality, attr"。
        where_clause (str, optional): WHERE 条件（不含 WHERE 关键字）。

    Returns:
        tuple: (records, error_msg)
            records — 查询结果列表，失败时为空列表。
            error_msg — 成功为 None，失败为错误信息。
    """
    query = f"SELECT {columns} FROM label"
    if where_clause:
        query += f" WHERE {where_clause}"
    
    conn = sqlite3.connect(db_path)
    try:
        result = conn.execute(query).fetchall()
        return result, None
    except Exception as e:
        return [], str(e)
    finally:
        conn.close()

# if __name__ == "__main__":
#     n = scan_pairs("./test_data/db/label.db", "./test_data/mp4", "./test_data/raw")
#     print(f"✅ 新增 {n} 对文件")
    
#     conn = sqlite3.connect("./test_label.db")
#     for row in conn.execute("SELECT * FROM label"):
#         print(row)
#     conn.close()