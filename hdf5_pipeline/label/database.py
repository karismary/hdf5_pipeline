# 启动: streamlit run hdf5_pipeline/label/app.py --server.port=8501

import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from hdf5_pipeline.core.hdf5_utils import get_sorted_files

def _connect(db_path: str) -> sqlite3.Connection:
    """统一的连接工厂：所有数据库访问都从这里拿连接。

    设置 row_factory=sqlite3.Row，使查询结果支持按列名取值
    （row["mp4_name"]），也保留下标取值（row[3]），迁移期间两种都能用。
    集中在一处设置，避免漏设某个连接导致 Row/tuple 混用的坑。
    """
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c

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
    conn = _connect(db_path)
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_label_mp4_name ON label(mp4_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_label_quality_name ON label(quality)")

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
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT mp4_name FROM label WHERE quality != 'unlabeled'")
    rows = cursor.fetchall()
    labeled_names = {row["mp4_name"] for row in rows}
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
    mp4_list = set(get_sorted_files(mp4_dir, [".mp4"], 1))
    raw_set = set(get_sorted_files(raw_dir, [".hdf5"], 1))
    pairs = sorted(m for m in mp4_list if m.replace(".mp4", ".hdf5") in raw_set)

    conn = _connect(db_path)
    cursor = conn.cursor()

    existing = {row["mp4_name"] for row in cursor.execute("SELECT mp4_name FROM label")}
    new_pairs = [p for p in pairs if p not in existing]
    new_pairs_data = [
        (
            p.replace(".mp4", ".hdf5"),
            str(Path(raw_dir) / p.replace(".mp4", ".hdf5")),
            p,
            str(Path(mp4_dir) / p),
            datetime.fromtimestamp((Path(mp4_dir) / p).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        )
        for p in new_pairs
    ]
    cursor.executemany("INSERT OR IGNORE INTO label (hdf5_name, hdf5_path, mp4_name, mp4_path, created_at) VALUES (?, ?, ?, ?, ?)", new_pairs_data)

    conn.commit()
    conn.close()

    return len(new_pairs)

def add_label(db_path: str, mp4_name: str, hdf5_path: str | None = None, quality: str | None = None, attr: dict | None = None) -> None:
    """更新一个视频文件的打标结果，单次调用。

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

    conn = _connect(db_path)
    conn.execute(sql, tuple(values))
    conn.commit()
    conn.close()

def add_labels(db_path: str, records: list, quality: str | None = None, if_qualify: bool = False, target_dir: str | None = None, attr: dict | None = None) -> None:
    """批量更新多条记录的打标结果，单连接单事务执行。

    对 records 中每条记录按 mp4_name 定位，只更新传入的字段，不传的保持不变。
    labeled_at 自动设为当前时间。

    Args:
        db_path (str): SQLite 数据库文件路径。
        records (list[sqlite3.Row]): 记录列表，由 get_list/query_records 等
            返回（row_factory=Row），按列名取值：record["mp4_name"]、
            record["quality"]、record["hdf5_name"]。
        quality (str, optional): 目标质量标签，不传则不更新。
        if_qualify (bool, optional): True 为打标归档模式——同时把 hdf5_path
            更新为 target_dir 下的 hdf5_name，并跳过已是目标 quality 的记录。
        target_dir (str, optional): HDF5 目标目录，if_qualify=True 时必传。
        attr (dict, optional): 统一写入所有记录的属性字典，不传则不更新。

    Returns:
        None
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    clause_list = []
    set_clauses = ["labeled_at = ?"]
    values = [now]

    if quality is not None:
        set_clauses.append("quality = ?")
        values.append(quality)
    if attr is not None:
        set_clauses.append("attr = ?")
        values.append(json.dumps(attr))

    if len(set_clauses) == 1:
        return
    
    conn = _connect(db_path)
    sqls = f"UPDATE label SET {', '.join(set_clauses)} WHERE mp4_name = ?"

    if not if_qualify:
        for record in records:
            mp4_name = record["mp4_name"]
            value = values.copy()
            value.append(mp4_name)
            clause_list.append(value)
        conn.executemany(sqls, clause_list)
    else:
        if target_dir is None:
            return
        sql = f"UPDATE label SET {', '.join(set_clauses)}, hdf5_path = ? WHERE mp4_name = ?"
        for record in records:
            if record["quality"] == quality:
                continue
            new_hdf5_path = str(Path(target_dir) / record["hdf5_name"])
            value = values.copy()
            value.append(new_hdf5_path)
            value.append(record["mp4_name"])
            clause_list.append(value)
        if clause_list:
            conn.executemany(sql, clause_list)

    conn.commit()
    conn.close()

def update_attrs(db_path: str, records: list, new_attr: tuple) -> None:
    """批量更新多条记录中某属性的选项，单连接单事务执行。

    对 records 中每条记录，解析其 attr JSON，把 new_attr 指定配置键
    子对象的 option 更新为新值，并刷新 labeled_at。所有 UPDATE 在同一连接、
    同一次 commit 内完成，相比逐条调用 add_label 省去 N 次开连接 / 落盘。

    Args:
        db_path (str): SQLite 数据库文件路径。
        records (list[sqlite3.Row]): 记录列表，每条取 record["mp4_name"]
            为 mp4_name、record["attr"] 为原 attr JSON。
        new_attr (tuple): (config_key, option_value)。config_key 是 custom_cols
            的配置键（如 "attr_weather"），option_value 是所选选项字符串。

    Returns:
        None
    """

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = _connect(db_path)
    cursor = conn.cursor()

    for record in records:
        mp4_name = record["mp4_name"]
        raw_attr = json.loads(record["attr"])
        key, value = new_attr
        if key in raw_attr:
            raw_attr[key]["option"] = value
            cursor.execute("UPDATE label SET attr = ?, labeled_at = ? WHERE mp4_name = ?", (json.dumps(raw_attr), now, mp4_name))
        else:
            continue

    conn.commit()
    conn.close()


def count_list(db_path: str, where_clause: str | None = None) -> int:
    """返回 label 表中满足条件的记录总数。

    where_clause 为 None 时统计全表，否则统计 WHERE 条件筛选后的行数。
    数据库文件不存在时返回 0。

    Args:
        db_path (str): SQLite 数据库文件路径。
        where_clause (str | None, optional): WHERE 条件（不含 WHERE 关键字）。

    Returns:
        int: 符合条件的记录总数，数据库不存在时为 0。
    """
    if Path(db_path).exists():
        conn = _connect(db_path)
        cursor = conn.cursor()
        if where_clause is None or where_clause == "":
            where = ""
        else:
            where = f" WHERE {where_clause}"
        cursor.execute(f"SELECT COUNT(*) FROM label {where}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    else:
        return 0

def get_list(db_path: str) -> List[Tuple[int, str, str, str, str, str, str, str, str]]:
    """从数据库获取所有记录的完整列表。

    查询 label 表中全部字段。

    Args:
        db_path (str): SQLite 数据库文件路径。

    Returns:
        list[sqlite3.Row]: 全表记录，按列名取值
            （record["mp4_name"]、record["quality"] 等）。
    """

    conn = _connect(db_path)
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
        sqlite3.Row 或 None: 单条记录（按列名取值），不存在返回 None。
    """
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, hdf5_name, hdf5_path, mp4_name, mp4_path, quality, attr, created_at, labeled_at FROM label WHERE mp4_name =?", (mp4_name, ))
    row = cursor.fetchone()
    conn.close()

    return row

def translate_where(condition: str, attr_map: dict[str, str]) -> str:
    """把用户 WHERE 条件里的 @属性引用翻译成 SQL 可执行的 json_extract。

    属性存在 attr 的 JSON 里，SQL 无法直接写 @天气 = "晴天" 查询，
    所以把每个 @标签 的引用替换成 json_extract(attr, '路径') 的等价条件。
    属性显示名 → JSON 路径 的映射由调用方从 config 构建传入。

    例：condition = 'quality = "good" AND @天气 = "晴天"'
        attr_map = {"天气": "$.weather.option"}
        → 'quality = "good" AND json_extract(attr, \'$.weather.option\') = \'晴天\''

    Args:
        condition (str): 用户写的 WHERE 条件片段（不含 WHERE 关键字），
            可能含 @标签 形式的属性引用。
        attr_map (dict[str, str]): 属性显示名 → attr JSON 路径 的映射。

    Returns:
        str: 翻译后的 WHERE 条件，@引用已替换为 json_extract(attr, ...)；
            未匹配到 @引用的部分原样保留。
    """
    result = condition
    for label, json_path in attr_map.items():
        pattern = rf'@{re.escape(label)}\s*(=|!=|>=|<=|>|<| LIKE | NOT LIKE | IS | IS NOT )\s*"([^"]*)"'
        replacement = rf"json_extract(attr, '{json_path}') \1 '\2'"
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result

def query_records(db_path: str, columns: str, where_clause: str | None, limit: int | None = None, offset: int = 0):
    """执行自定义查询，返回结果列表（可选分页）。

    默认返回全部匹配行；传入 limit 时按 id 稳定排序取当前页
    （ORDER BY id LIMIT ? OFFSET ?），保证翻页期间行不重复、不遗漏。
    where_clause 为 None 时查询全表。

    Args:
        db_path (str): 数据库路径。
        columns (str): SELECT 的列名，如 "id, quality, attr"。
        where_clause (str | None, optional): WHERE 条件（不含 WHERE 关键字）。
        limit (int | None, optional): 每页行数，None 表示不分页返回全部。
        offset (int, optional): 跳过的行数，默认 0。

    Returns:
        tuple: (records, error_msg)
            records — 查询结果列表，失败时为空列表。
            error_msg — 成功为 None，失败为错误信息。
    """
    if columns is None:
        columns = "*"
    query = f"SELECT {columns} FROM label"
    if where_clause:
        query += f" WHERE {where_clause}"
    
    conn = _connect(db_path)
    if limit is None:
        try:
            result = conn.execute(query).fetchall()
            return result, None
        except Exception as e:
            return [], str(e)
    else:
        query += f" ORDER BY id LIMIT {limit} OFFSET {offset}"
        try:
            result = conn.execute(query).fetchall()
        except Exception as e:
            return [], str(e)
        finally:
            conn.close()
    return result, None

# if __name__ == "__main__":
#     n = scan_pairs("./test_data/db/label.db", "./test_data/mp4", "./test_data/raw")
#     print(f"✅ 新增 {n} 对文件")
    
#     conn = sqlite3.connect("./test_label.db")
#     for row in conn.execute("SELECT * FROM label"):
#         print(row)
#     conn.close()