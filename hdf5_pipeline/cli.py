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


def _run_spirit(args) -> None:
    """执行 spirit 子命令：convert / check / validate / ui。"""
    from hdf5_pipeline.parquet_ui_redo.convert import convert_spirit
    from hdf5_pipeline.parquet_ui_redo.quality import run_spirit_quality
    from hdf5_pipeline.parquet_ui_redo.validator import validate_spirit_dataset

    if args.spirit_cmd == "convert":
        print(f"spirit 转换: {args.raw_dir} → {args.out_dir}")
        stats = convert_spirit(args.raw_dir, args.out_dir, link_videos=not args.copy_videos)
        print(f"完成: 转换 {stats['converted']}, 跳过 {stats['skipped']}, 视频 {stats['videos']}")
    elif args.spirit_cmd == "check":
        print(f"spirit 质检: {args.raw_dir}, 严格度: {args.strictness}")
        summary = run_spirit_quality(
            args.raw_dir, args.out_csv, args.out_json, strictness=args.strictness
        )
        print(f"完成: {summary['num_files']} 个文件, {summary['num_frames']} 帧, "
              f"{summary['num_outliers']} 个异常帧")
    elif args.spirit_cmd == "validate":
        ok, errors = validate_spirit_dataset(args.raw_dir)
        if ok:
            print("校验通过，未发现问题。")
        else:
            print(f"发现 {len(errors)} 个问题:")
            for err in errors:
                print(f"  - {err}")
            raise SystemExit(1)
    elif args.spirit_cmd == "ui":
        subprocess.run(["streamlit", "run", "hdf5_pipeline/parquet_ui_redo/app.py"])


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

    spirit_p = sub.add_parser("spirit", help="千寻(moz1) spirit parquet 数据：转换/质检/校验/UI")
    spirit_sub = spirit_p.add_subparsers(dest="spirit_cmd", required=True)

    sp_convert = spirit_sub.add_parser("convert", help="原始 parquet → 标准 LeRobot v2.1 数据集")
    sp_convert.add_argument("raw_dir", help="原始数据目录（或单个实例目录）")
    sp_convert.add_argument("out_dir", help="输出数据集目录（存在则先清空）")
    sp_convert.add_argument("--copy-videos", action="store_true",
                            help="视频用复制而非软链接（默认软链接）")

    sp_check = spirit_sub.add_parser("check", help="原始数据异常帧检测")
    sp_check.add_argument("raw_dir", help="原始数据目录")
    sp_check.add_argument("out_csv", help="异常帧明细 CSV 路径")
    sp_check.add_argument("out_json", help="统计摘要 JSON 路径")
    sp_check.add_argument("--strictness", default="strict",
                          choices=["loose", "medium", "strict"], help="判断严格程度（默认 strict）")

    sp_validate = spirit_sub.add_parser("validate", help="结构完整性 + event_log 一致性校验")
    sp_validate.add_argument("raw_dir", help="原始数据目录")

    spirit_sub.add_parser("ui", help="启动 spirit 独立 Streamlit UI")

    args = parser.parse_args()

    if args.command == "rename":
        print(f"重命名: {args.src} → {args.out}")
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
    elif args.command == "spirit":
        _run_spirit(args)


if __name__ == "__main__":
    main()