# MIGRATION_CONTEXT.md — 会话上下文迁移档案

> 用途：跨会话/跨模型切换时，把整个优化项目的上下文一次性带走。
> 更新时间：2026-08-11（专项 01/02 均已完成并验证，专项 03 挂起等 .parquet）。

---

## 一、核心任务

**项目**：`hdf5_pipeline`（/Users/karis/projects/test）——机器人数据流水线工具，HDF5 重命名 → 异常帧质量检测 → 渲染 MP4 → 打标归档，底层数据还有 LeRobot (.parquet) 格式。

**协作模式（用户硬性要求）**：
- ❌ **不要直接给代码**。要告诉用什么方法、什么库/模块、什么技能技术栈，用户自己实现，我 review，验证通过再下一条。
- 一次只处理一条，不做批量打包。
- 所有优化依据来自 `todo/README.md` 的条目清单（按优先级 P0→P3 排序），专项 01/02/03 最后做。

**技术栈**：Python 3.11.15（conda 环境 `./conda_env/bin/python`）、Streamlit 1.58.0、h5py、pyarrow、OpenCV、numpy、matplotlib、pytest。

---

## 二、协作规则（用户反复强调，必须遵守）

1. **看函数实现，别信旧 docstring**：用户原话 *"你看我的 docstring 干什么，我函数里面写的是什么就是什么，docstring 就是之前的，错的"*。docstring/类型注解经常滞后于实现，判断代码行为以**当前函数体**为准。
2. **先验证前提再优化**：多次拦下伪需求——5.5（widget key 兼容性，实测证伪）、4.5（normalize_image_array，唯一调用方不触发）、8.1（README 过时，已实现）。不要为了改而改。
3. **别用过头语法糖**：用户原话 *"你这个语法糖用的太过分了"*。写显式可读的循环/条件，别炫技。
4. **模块职责单一**：DB 模块（database.py）只碰数据库，文件移动/UI 在别处处理。原话 *"add_labels 里面不能加上 qualify_move 的内容"*。
5. **单元测试最后统一做**：1.3 推迟到所有优化完成后再写 pytest。用户已定：最后写一份**测试文档**做完整链路回归（用户原话"最后 pytest 的时候我们写一个测试文档来完整链路的测试"）。
6. 每改完一处，用户会说"改完了/往下"，我再进入下一条；用户没让做的部分（如 tab5 调用代码）先不做。

---

## 三、已完成并验证的改动（README 全队列 ✅）

### P0
- **5.3 SQL 查询坏掉**：`query_records` 默认 `col_names="*"`，开发者可经 `_ENABLE_SELECT_COLUMNS` 开关启用列筛选，无 NameError。
- **2.4 Render 批量进度无超时**：`render_all` 改为 wait 轮询 + 每文件 deadline（常量 `CHECK_INTERVAL=5 / PER_FRAME_BUDGET=0.3 / BASE_OVERHEAD=60 / MIN_TIMEOUT=60 / MAX_TIMEOUT=3600`），进度靠日志行数（=进度契约）。

### P1
- **2.1** 硬编码参数 → 命名常量 `JOINT_ANGLE_ENVELOPE`、`MIN_ROW_H`。
- **5.5** widget key 兼容性——前提证伪，移除 safe_key 全部机制。
- **6.3** 重命名覆盖风险：while + 后缀碰撞防护。
- **3.1** `compute_outliers` 内存优化：`MAX_QUANTILE_SAMPLES=200_000` 降采样拟合分位数（`[::stride]`），评分仍走全量（采样会漏离群帧）。
- **4.3** `video_utils` 加 `isOpened()` 保护 + 类型提示（`-> Optional[np.ndarray]`）。
- **5.4** 打标移动回滚：补偿模式（try/except + 移回原父目录 + raise）。
- **6.2** `rename_files(files, out, if_move=False)` copy→move 参数化。
- **7.1** pending 状态：打标区加"待复议"，不移动文件。

### P2
- **1.2** `__main__.py`：`python -m hdf5_pipeline` 入口。
- **2.5** `render/__init__.py` 补齐；**2.3** 随 batch_gui 删除一并解决。
- **7.2** 新建 `core/validator.py`：`validate_file() -> (bool, list[str])` 双格式校验（HDF5 像素+action；Parquet `required.issubset(columns)`+num_rows）。
- **7.3** `cli.py` pipeline 一键流程：rename→quality→render(串行)→提示进入打标。
- **8.1** Quality Tab 结果持久化：自动载入上次检测结果 + 导出 toast。
- **5.1** widget key 前缀常量（ui/common.py：`tabre/tabqu/tabrd/tabla/tabov/tabcf`）。
- **1.4** `requirements.txt` 锁定：h5py==3.16.0, matplotlib==3.10.9, numpy==2.4.6, opencv-python==4.13.0.92, pyarrow==24.0.0, streamlit==1.58.0。

### P3
- **5.2** `seclect_folder` → `select_folder` + 全局引用。
- **8.3** 暗色模式：`.streamlit/config.toml` `[theme] base="light"`（streamlit 原生，已删 style.css，本地保留但不 git 跟踪）。
- **9.1/9.2** `debug_data.py`、`test_data_副本/` 加入 `.gitignore`。

### 收尾/结构（2.4b）
- **preview → ui 整包重命名**（`git mv` + 全部引用）；删除 `render/batch_gui.py`（本地保留、.gitignore 不跟踪）。
- **`ui/module_app.py`** 通用模块入口：`importlib.import_module` + `getattr` dispatch 注册表，argparse `--module`。注意 f-string 内嵌引号在 3.11 是 SyntaxError（PEP 701 需 3.12+）。
- **`cli.py`**：`ui --module X` / `render` 别名。`_run_ui_module` 必须带 `"--", "--module", name` 分隔符（否则 Click 报 "No such option '--module'"）。
- `render_mp4` 签名统一为 `(hdf5_path: str, out_mp4: str)`，`.name` 一律 `Path(x).name`。

---

## 四、专项 01 / 02 已完成（本次会话，均通过运行时验证）

### 专项 01：session_state 统一管理（StateManager，形态 2）
- 新建 `label/state.py`：key 常量（`S_RECORDS`/`S_SELECTED`/`S_SELECTED_INDEX`/`S_OV_*`/`S_DB_VERSION`/`S_TAB4_VERSION`/`S_TAB5_VERSION`/`S_TOAST`）+ `get/set/pop` 安全访问器 + `init_state()` + `bump_db_version()`。
- **widget key 不可抽象**（`ui_*`/`attrs_*`/`botton_*`/`confirm_del_*` 等保持裸字符串）——Streamlit 在 widget 声明处用 `key=` 字符串绑定，抽象出来会断绑定。
- **跨 tab 数据共享的真相**：`st.tabs` 每次 rerun **全渲染**（非懒加载），所有 tab 本就共享 session_state——真正的缺口是**变更通知**。解法 = **版本号广播**：任何写库操作调 `bump_db_version()`；各 tab 顶部/尾部把 `S_DB_VERSION` 与自己的 `S_TABn_VERSION` 比对，不等则刷新自身缓存。
- **消费先于检查**：刷新前先 `set(S_TABn_VERSION, get(S_DB_VERSION))` 再重查，避免无限 rerun；tab5 自动刷新用 `if S_OV_RECORDS in st.session_state` 守卫（用户没查过就不自动填）。
- **手动重查后也要消费版本**（批量确认/刷新按钮），否则 post-render 检查会二次重查。
- 删掉 3 个 point-to-point 布尔 flag 和 count 比对，统一走版本号；顺带删了死 key `selected_button`。
- `bump_db_version()` 是 session_state 层行为，**必须放 UI 调用点（app.py）**，不能进 database.py（DB 模块职责单一）。
- AppTest 端到端 24 项全过：启动无异常、质量按钮真移动文件 + bump + tab4/tab5 版本消费、跨 tab 列表自动刷新、attrs 确认 toast。

### 专项 02：数据库优化（database.py）
- **A1 索引**：`init_db` 加 `idx_label_mp4_name(mp4_name)`、`idx_label_quality_name(quality)`（`CREATE INDEX IF NOT EXISTS`）。
- **A2 scan_pairs 批量**：set 交集求配对 + 现有 mp4_name 预过滤 + `executemany` + `INSERT OR IGNORE`。
- **A3 add_labels 批量**：单连接单事务；if_qualify 分支 tuple 索引已修（原 `record.get("hdf5_name")` AttributeError 已消除，见下方 Row 重构后彻底按列名取值）。
- **A4**：`count_list(db_path, where)` 计数；`query_records(..., limit, offset)` 服务端分页（`ORDER BY id LIMIT ? OFFSET ?` 稳定排序，翻页不重不漏）。tab5 用 `PAGE_SIZE=10` 分页 + `count_list` 算总页数。`get_list` **未加分页**——tab4 selectbox 需要全列表做进度统计（`labeled/total`），分页反而破坏设计，且当前无调用方需要（不为假想需求设计）。
- **sqlite3.Row 重构**（用户此前认定"最大技术债"，本次已做）：
  - `_connect(db_path)` 连接工厂集中设 `conn.row_factory = sqlite3.Row`，10 个连接点全走它。
  - 所有魔法索引改按列名：app.py 约 30 处（`sel[3]`→`sel["mp4_name"]`、`r[5]`→`r["quality"]`、`rec[0]`→`rec["id"]`…）+ database.py 内部。
  - Row 同时支持下标/列名 → **渐进迁移安全**（改一半旧代码也不崩）；列名错立即抛 IndexError（魔法数字错位是静默错，更难查）。
  - ⚠️ **盲 replace_all 会递归**：`_connect` 函数体里也有一行 `conn = sqlite3.connect(...)`，全局替换成 `_connect(...)` 即无限递归（用户当场抓出）。解法：工厂内部局部变量用不同名（`c`），函数体不匹配替换模式。
  - 剩余 3 处下标是合法误报：`db_files[0]`（文件列表下标）、`get_video_info(...)[0]`（`(帧数,帧率)` 元组解包）×2。
- DB 层验证 `verify_spec02.py`：**17 通过 / 1 by-design**。by-design = `update_attrs` 对缺失属性键 `continue` 跳过、不新建默认子对象（已知行为，批量改属性时未设过该属性的记录会被静默跳过）。

### 其它收尾（本次会话）
- **`use_container_width` 弃用清理**：app.py + rename_tab.py 共 14 处 → `width="stretch"`（Streamlit 1.58 支持）。
- **`check_session_state_rules` 警告清理**：`attrs_module` 里 `st.session_state[widget_key] = current_idx` 预置块冗余——widget key 内嵌记录 id（`attrs_{key_name}_{keys}`，key_name 含 id），`index=` 在 key 不存在时本就自动初始化。删除后警告消失、行为不变。

---

## 五、尚未开始的专项

- **专项 03 parquet 支持**：**用户明确挂起**——等用户找到真实 `.parquet` 实例文件后再做，届时可能有大幅重构。`core/hdf5_utils.py` 已有 `get_hdf5_frame_count(h5_file, format)` 双格式雏形、`FORMAT_DICT={"lerobot":".parquet","hdf5":".hdf5"}`。

---

## 六、关键技术点与决策记录（讲给用户听过的）

1. **deadline 超时**：慢≠卡死。`budget = PER_FRAME_BUDGET*n_frames + BASE_OVERHEAD`，clip 到 [MIN,MAX]，`wait(FIRST_COMPLETED, timeout)` 轮询，`not_done` 只有 `now>deadline` 才算超时。`as_completed` 卡在迭代器层，`result(timeout)` 救不了。
2. **降采样拟合**：分位数对样本量鲁棒，`[::stride]` 是 numpy 视图不复制；`stride = max(1, n_total // MAX)`。小数据 stride=1 与原来逐字节一致。用户实测 500 条×2min×30Hz ≈ 180 万帧 ≈ 230MB，非真瓶颈，但加了保险丝。
3. **SQLite**：索引 = B-tree O(log n)；`executemany` 复用同一模板；COMMIT 是昂贵操作；`sqlite3` 绑定拒绝 Path 对象（要 `str()`）；`INSERT OR IGNORE` 靠 UNIQUE 约束去重。
4. **补偿模式**（5.4）：先动文件/DB，失败就把文件移回原父目录再 raise——恢复"权威状态"而非"字节原样"。
5. **tuple vs list**：定长记录用 tuple、增长集合用 list、成员判断用 set。反复给用户讲解。
6. **Python 陷阱清单**（几乎都踩过）：
   - `list.append()` 返回 None；方法要加 `()`（`st.rerun`、`values.copy`、`get_records`）。
   - `json.dumps` vs `json.dump`（dumps 序列化为字符串）。
   - f-string 内嵌双引号在 3.11 是 SyntaxError（PEP 701 需 3.12）。
   - `from datetime import datetime` vs `import datetime`（模块/类同名）。
   - `while pending is not None`（dict 永不为 None，死循环）→ `while pending:`。
   - `record[5] = quality` 对 tuple 是 TypeError → 元组重建 `record[:5] + (quality,) + record[6:]`。
   - `sqlite3` UPDATE 每条 SQL 模板不同 → 不能用 executemany，只能单事务循环。
7. **widget key 前缀**：跨页唯一约定，常量在 `ui/common.py`。
8. **版本号广播**（专项 01）：`st.tabs` 每次 rerun 全渲染、共享 session_state，跨 tab 同步靠"DB 版本号 + 各 tab 消费"比对；**消费先于检查**防无限 rerun；`bump_db_version()` 只在 UI 写库处调用，不进 DB 模块。
9. **`sqlite3.row_factory` 是连接级设置**：必须收进 `_connect` 工厂一处集中设，漏设一个连接就 Row/tuple 混用、报错隐晦。⚠️ 盲 `replace_all` 会把工厂函数体里的 `conn = sqlite3.connect(...)` 一起换成 `_connect(...)` → 无限递归；函数体局部变量用不同名（`c`）规避。
10. **widget key 内嵌记录 id**（`attrs_{key_name}_{keys}` 的 key_name 含记录 id）→ 每条记录独立 key，`st.selectbox(index=...)` 在 key 不存在时自动初始化，**无需 session_state 预置**（预置反而触发 `check_session_state_rules` 警告）。

---

## 七、关键文件路径速查

| 文件 | 状态 |
|---|---|
| `hdf5_pipeline/label/app.py` | ✅ 专项01 迁移（版本号跨 tab 同步）+ 专项02 Row 按列名取值 + width="stretch" |
| `hdf5_pipeline/label/database.py` | ✅ `_connect` 工厂 + Row + 批量/分页/计数全齐 |
| `hdf5_pipeline/label/state.py` | ✅ 新建（专项01 StateManager，git 未跟踪） |
| `hdf5_pipeline/ui/rename_tab.py` | ✅ width="stretch" |
| `hdf5_pipeline/cli.py` | ✅ 已改（ui/render 入口） |
| `hdf5_pipeline/rename/engine.py` | ✅ 已改（datetime import 修复） |
| `hdf5_pipeline/ui/common.py` | ✅ widget key 常量 + folder_callback |
| `hdf5_pipeline/ui/module_app.py` | ✅ 通用模块入口 |
| `hdf5_pipeline/ui/render_tab.py` | ✅ deadline 超时 |
| `hdf5_pipeline/quality/detector.py` | ✅ 降采样 |
| `hdf5_pipeline/quality/checker.py` | ✅ 统一 run_quality_check |
| `hdf5_pipeline/core/validator.py` | ✅ 新建 |
| `hdf5_pipeline/core/hdf5_utils.py` | ✅ get_hdf5_frame_count 双格式 |
| `hdf5_pipeline/core/video_utils.py` | ✅ isOpened + 类型提示 |
| `hdf5_pipeline/__main__.py` | ✅ 新建 |
| `requirements.txt` / `.streamlit/config.toml` / `.gitignore` | ✅ 新建 |
| `todo/README.md` | 条目清单（已全清） |
| `tests/` | 待最后统一 pytest（目录已建，未写用例） |

**当前 git 未提交**：M `cli.py`、`label/app.py`、`label/database.py`、`rename/engine.py`、`ui/rename_tab.py`；未跟踪 `MIGRATION_CONTEXT.md`、`label/state.py`。

---

## 八、环境与数据

- conda：`./conda_env/bin/python`（Python 3.11.15）。
- 用户数据规模：**数量上限约 500 条，每条约 2 分钟**，采集 15/30Hz。
- `config.json` 在项目根，含 paths（db_dir/raw_dir/mp4_dir/good_dir/bad_dir 等）与 custom_cols。
- label 表结构：`(id, hdf5_name, hdf5_path, mp4_name, mp4_path, quality, attr, created_at, labeled_at)`，quality ∈ {unlabeled, good, bad, pending}。

---

## 九、下一步（衔接点）

1. **专项 03 parquet**：挂起中，等用户提供真实 `.parquet` 实例文件后开始。
2. **最终 pytest 完整链路回归**：用户已定最后写一份**测试文档**做完整链路测试（rename→quality→render→label 全流程），统一 pytest 回归。
3. 已知边界（未定）：`qualify_and_move` 单条模式"目标文件已存在 → `pass` 静默跳过（不更新 DB、不 bump）"，与批量模式（仍走 add_labels 更新 DB）语义不一致；`update_attrs` 对缺失属性键 `continue` 跳过（by-design）。
