import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from hdf5_pipeline.core.hdf5_utils import get_sorted_files

def init_db(db_path):
    """初始化数据库，创建 label 表（如果不存在）。

    自动建表，不会覆盖已有数据。
    表结构记录了每个视频文件对应的 HDF5、打标结果、自定义属性以及时间信息。

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
                        mp4_name TEXT NOT NULL,
                        quality TEXT DEFAULT 'unlabeled',
                        attr TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT '',
                        labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     )''')
    conn.commit()
    conn.close()

def get_unlabeled(db_path, mp4_dir) -> List[str]:
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

def add_label(db_path, mp4_dir, mp4_name, quality, attr):
    """记录或更新一个视频文件的打标结果。

    使用 INSERT OR REPLACE，同一 mp4_name 会覆盖旧记录。
    自动从 mp4 文件名推导对应的 hdf5 文件名（.mp4 → .hdf5）。
    自动记录 MP4 文件的创建时间作为 created_at。

    Args:
        db_path (str): SQLite 数据库文件路径。
        mp4_dir (str): MP4 文件所在目录（用于读取文件时间戳）。
        mp4_name (str): MP4 文件名，例如 'episode_000000.mp4'。
        quality (str): 打标结果，'good' 或 'bad'。
        attr (dict): 自定义属性字典，例如 {'attr_method': '垂直'}。

    Returns:
        None
    """
    attrs = json.dumps(attr)
    hdf5_name = mp4_name.replace('.mp4', '.hdf5')
    mp4_path = Path(mp4_dir) / mp4_name
    if  mp4_path.exists():
        stat = mp4_path.stat()
        create_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    else:
        create_time = '-'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO label (hdf5_name, mp4_name, quality, attr, created_at) VALUES (?, ?, ?, ?, ?)",
                    (hdf5_name, mp4_name, quality, attrs, create_time))
    conn.commit()
    conn.close()

def scan_pairs(db_path, mp4_dir, raw_dir):
    """扫描 mp4_dir 与 raw_dir，将配对的 MP4+HDF5 注册到数据库。

    遍历所有 MP4 文件，在 raw_dir 中查找同名的 .hdf5 文件。
    找到配对后写入数据库，跳过已有记录，不会覆盖已打标的文件。

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
    row = []

    for m in mp4_list:
        hdf5 = m.replace(".mp4", ".hdf5")
        if hdf5 not in raw_set:
            continue
        try:
            cursor.execute(
                "INSERT INTO label (hdf5_name, mp4_name) VALUES (?, ?)",
                (hdf5, m)
            )
            count += 1
        except sqlite3.IntegrityError:   # 已存在，跳过
            pass

    conn.commit()
    conn.close()
    return count

# def get_list(db_path)

if __name__ == "__main__":
    n = scan_pairs("./test_data/db/label.db", "./test_data/mp4", "./test_data/raw")
    print(f"✅ 新增 {n} 对文件")
    
    conn = sqlite3.connect("./test_label.db")
    for row in conn.execute("SELECT * FROM label"):
        print(row)
    conn.close()

