import argparse
import subprocess

UI_MODULES = ["rename", "quality", "render"]


def _run_ui_module(module_name: str) -> None:
    """启动独立 UI 页面：streamlit run module_app.py -- --module <name>。

    注意：`--` 是必须的——它告诉 Click "后面的参数留给脚本"，
    否则 --module 会被当成 streamlit 自己的选项而报错。
    """
    subprocess.run([
        "streamlit", "run", "hdf5_pipeline/ui/module_app.py",
        "--", "--module", module_name
    ])


def main():
    parser = argparse.ArgumentParser(description="HDF5 Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    rename_p = sub.add_parser("rename", help = "递归获取目录下的所有数据文件，并根据自然排序重命名放置于同一文件夹下")
    rename_p.add_argument("src", help = "源目录")
    rename_p.add_argument("out", help = "输出目录")

    check_p = sub.add_parser("check", help = "对目录下所有文件的异常帧进行检测")
    check_p.add_argument("dir", help = "数据目录")
    check_p.add_argument("--strictness", default = "medium", choices = ["loose", "medium", "strict"], help = "判断严格程度")
    check_p.add_argument("--format", default = "hdf5", choices = ["hdf5", "lerobot"], help = "数据格式（默认 hdf5）")

    pipeline = sub.add_parser("pipeline", help = "自动执行：对.hdf5文件的重命名 → 质量检测 → 渲染 → 打开打标界面")
    pipeline.add_argument("src", help = "包含原始数据的文件夹（支持递归搜索目录下所有的文件）")
    pipeline.add_argument("out", help = "重命名后的数据文件目录")
    pipeline.add_argument("csv_json_out", help = "输出检测报告.csv和.json文件的路径")
    pipeline.add_argument("mp4_out", help = "输出渲染视频路径")
    
    ui_p = sub.add_parser("ui", help = "启动独立模块页面（重命名/质量检测/渲染）")
    ui_p.add_argument("--module", default = "render", choices = UI_MODULES, help = "要显示的模块")

    sub.add_parser("render", help="视频渲染（启动独立渲染页面）")
    sub.add_parser("label", help="完整打标系统（启动 Streamlit）")

    args = parser.parse_args()

    if args.command == "rename":
        print(f"重命名: {args.src} → {args.dst}")
        from hdf5_pipeline.rename.engine import collect_hdf5_files, rename_files
        files = collect_hdf5_files(args.src)
        n = rename_files(files, args.out)
        print(f"完成: {n} 个文件已重命名")
    elif args.command == "check":
        print(f"检测: {args.dir}, 格式: {args.format}, 严格度: {args.strictness}")
        from hdf5_pipeline.quality.checker import run_quality_check
        summary = run_quality_check(args.dir, args.format, strictness=args.strictness)
        print(f"完成: {summary['num_files']} 个文件, {summary['num_outliers']} 个异常帧")
    elif args.command == "pipeline":
        from hdf5_pipeline.rename.engine import collect_hdf5_files, rename_files
        from hdf5_pipeline.quality.checker import run_quality_check
        from hdf5_pipeline.render.engine import render_mp4
        from pathlib import Path
        files = collect_hdf5_files(args.src)
        n = rename_files(files, args.out)
        print(f"完成: {n} 个文件已重命名")
        summary = run_quality_check(args.out, "hdf5", args.csv_json_out, args.csv_json_out)
        Path(args.mp4_out).mkdir(parents=True, exist_ok=True)   # 渲染输出目录可能不存在
        for hdf5 in Path(args.out).rglob("*.hdf5"):
            mp4 = Path(args.mp4_out) / (hdf5.stem + ".mp4")
            ok, msg, fname = render_mp4(str(hdf5), str(mp4))   # 按 str 契约传
            print(f"{'✓' if ok else '✗'} {fname}: {msg}")
        print("所有任务已完成\n进入打标请运行 label 指令")
    elif args.command == "ui":
        _run_ui_module(args.module)
    elif args.command == "render":
        _run_ui_module("render")
    elif args.command == "label":
        subprocess.run(["streamlit", "run", "hdf5_pipeline/label/app.py"])


if __name__ == "__main__":
    main()