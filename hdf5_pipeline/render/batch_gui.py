import os
import time
import threading
import concurrent.futures
import gc
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from hdf5_pipeline.render.engine import render_mp4
from hdf5_pipeline.core.constants import JOINT_NAMES
from hdf5_pipeline.core.hdf5_utils import get_hdf5_files

class BatchApp:
    def __init__(self, root, abort_event):
        self.root = root
        self.abort_event = abort_event
        self.root.title("HDF5 to MP4")
        self.root.geometry("850x680")
        
        self._progress_gen = 0   # 每次 start_batch 递增

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.is_processing = False
        self.start_time = 0.0
        self.futures = []

        self.show_img = tk.BooleanVar(value=True)
        self.show_act = tk.BooleanVar(value=True)
        self.action_dims = [tk.BooleanVar(value=True) for _ in range(16)]
        self.left_j = [tk.BooleanVar(value=True) for _ in range(7)]
        self.right_j = [tk.BooleanVar(value=True) for _ in range(7)]

        # 【新增】：跳过已存在文件的开关变量
        self.skip_existing = tk.BooleanVar(value=True)

        default_workers = max(1, min(2, os.cpu_count() or 2))
        self.max_workers_var = tk.IntVar(value=default_workers)

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=15)
        main.pack(fill="both", expand=True)

        dir_frame = ttk.LabelFrame(main, text="目录设置", padding=10)
        dir_frame.pack(fill="x", pady=5)

        ttk.Label(dir_frame, text="输入文件夹 (HDF5):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(dir_frame, textvariable=self.input_dir, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2, pady=5)

        ttk.Label(dir_frame, text="输出文件夹 (MP4):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(dir_frame, textvariable=self.output_dir, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, pady=5)

        opt_frame = ttk.LabelFrame(main, text="图表渲染与性能选项", padding=10)
        opt_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(opt_frame, text="显示顶端图像", variable=self.show_img).grid(row=0, column=0, sticky="w",
                                                                                     pady=2)
        ttk.Checkbutton(opt_frame, text="显示动作曲线", variable=self.show_act).grid(row=0, column=1, sticky="w",
                                                                                     pady=2)

        worker_frame = ttk.Frame(opt_frame)
        worker_frame.grid(row=0, column=2, columnspan=2, sticky="e", pady=2)
        act_frame = ttk.Frame(main, padding=5)
        act_frame.pack(fill="x", pady=10)
        
        ttk.Label(worker_frame, text="并发进程:").pack(side="left", padx=(0, 2))
        ttk.Spinbox(worker_frame, from_=1, to=os.cpu_count() or 4, textvariable=self.max_workers_var, width=3).pack(
            side="left")
        
        ttk.Label(opt_frame, text="动作维度 (16):").grid(row=1, column=0, sticky="w", pady=(8, 2))
        af = ttk.Frame(opt_frame)
        af.grid(row=2, column=0, columnspan=4, sticky="w")
        for i in range(16):
            ttk.Checkbutton(af, text=f"a{i}", variable=self.action_dims[i]).pack(side="left", padx=1)

        ttk.Label(opt_frame, text="左机械臂关节:").grid(row=3, column=0, sticky="w", pady=(8, 2))
        lf = ttk.Frame(opt_frame)
        lf.grid(row=4, column=0, columnspan=4, sticky="w")
        for i in range(7):
            ttk.Checkbutton(lf, text=JOINT_NAMES[i][:6], variable=self.left_j[i]).pack(side="left", padx=2)

        ttk.Label(opt_frame, text="右机械臂关节:").grid(row=5, column=0, sticky="w", pady=(8, 2))
        rf = ttk.Frame(opt_frame)
        rf.grid(row=6, column=0, columnspan=4, sticky="w")
        for i in range(7):
            ttk.Checkbutton(rf, text=JOINT_NAMES[i][:6], variable=self.right_j[i]).pack(side="left", padx=2)

        self.btn_start = ttk.Button(act_frame, text="▶ 开始批量转换", command=self.start_batch)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(act_frame, text="终止转换", command=self.stop_batch, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        # 【新增】：界面打钩选项
        ttk.Checkbutton(act_frame, text="跳过已生成的视频 (断点续传)", variable=self.skip_existing).pack(side="left",
                                                                                                         padx=15)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(act_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=10)

        self.lbl_progress = ttk.Label(act_frame, text="0 / 0")
        self.lbl_progress.pack(side="left", padx=5)

        self.log = scrolledtext.ScrolledText(main, height=12, bg="#f5f5f5")
        self.log.pack(fill="both", expand=True, pady=5)

    def browse_input(self):
        d = filedialog.askdirectory(title="选择包含 HDF5 文件的输入文件夹")
        if d:
            self.input_dir.set(d)
            # 预览文件信息
            files = get_hdf5_files(Path(d))
            if files:
                self.safe_log_print(f"发现 {len(files)} 个 HDF5 文件。")
                for f in files[:5]:
                    self.safe_log_print(f"   → {f.name}")
                if len(files) > 5:
                    self.safe_log_print(f"   ... 还有 {len(files) - 5} 个文件")
            else:
                self.safe_log_print("⚠️ 未找到 HDF5 文件。")

    def browse_output(self):
        d = filedialog.askdirectory(title="选择 MP4 输出文件夹")
        if d: self.output_dir.set(d)

    def safe_log_print(self, msg):
        def _log():
            self.log.insert("end", msg + "\n")
            self.log.see("end")

        self.root.after(0, _log)

    def update_progress(self, current, total):
        gen = self._progress_gen           # 捕获当前代次
        def _update():
            if gen != self._progress_gen:  # 旧回调，跳过
                return
            if total > 0:
                self.progress_var.set((current / total) * 100)
                elapsed = time.time() - self.start_time
                if current > 0:
                    eta = (elapsed / current) * (total - current)
                    eta_str = f"{int(eta // 60)}分{int(eta % 60)}秒"
                else:
                    eta_str = "计算中..."
                self.lbl_progress.config(text=f"{current} / {total} | 剩余 ≈ {eta_str}")
        self.root.after(0, _update)

    
    def stop_batch(self):
        if not self.is_processing: return
        self.btn_stop.config(state="disabled")
        self.safe_log_print("正在发送终止信号，清空排队任务，并自动清理残缺文件...")

        self.abort_event.set()
        for f in self.futures:
            f.cancel()

    def process_files_thread(self, input_folder, output_folder, files):
        total_files = len(files)
        self.log.delete(1.0, tk.END)  # 移到线程开头
        self.update_progress(0, total_files)

        show_img_val = self.show_img.get()
        show_act_val = self.show_act.get()
        action_on_vals = [v.get() for v in self.action_dims]
        left_on_vals = [v.get() for v in self.left_j]
        right_on_vals = [v.get() for v in self.right_j]
       
        failed_names = []    
        
        success_count = 0
        completed_count = 0
        was_aborted = False

        current_workers = max(1, self.max_workers_var.get())
        self.safe_log_print(f"共 {total_files} 个文件待处理，并发数: {current_workers}")

        if self.abort_event.is_set():
            return
        
        else:
            self.safe_log_print(f"\n--- 当前并发数: {current_workers} ---")

            with concurrent.futures.ProcessPoolExecutor(max_workers=current_workers) as executor:
                for src in files:
                    video_path = output_folder / f"{src.stem}.mp4"
                    self.futures.append(
                        executor.submit(
                            render_mp4, src, video_path,
                            show_img_val, show_act_val,
                            action_on_vals, left_on_vals, right_on_vals,
                            self.abort_event,
                        )
                    )

                for future in concurrent.futures.as_completed(self.futures):
                    try:
                        ok, msg, filename = future.result()
                        if ok:
                            success_count += 1
                            self.safe_log_print(f"完成: {filename}")
                        else:
                            if "手动终止" in msg:
                                was_aborted = True
                                self.safe_log_print(f"中止 ({filename})")
                            else:
                                self.safe_log_print(f"失败 ({filename}): {msg}")
                                failed_names.append(filename)
                        completed_count += 1
                        self.update_progress(completed_count, total_files)
                    except concurrent.futures.CancelledError:
                        was_aborted = True
                        completed_count += 1
                        self.update_progress(completed_count, total_files)
                    except Exception as e:
                        self.safe_log_print(f"⚠️ 严重错误: {str(e)}")
                        completed_count += 1

            gc.collect()

        elapsed_time = time.time() - self.start_time
        mins, secs = divmod(elapsed_time, 60)
        time_str = f"{int(mins)}分 {secs:.2f}秒" if mins > 0 else f"{elapsed_time:.2f}秒"

        if was_aborted:
            self.safe_log_print(f"\n任务被手动终止！本次成功处理: {success_count} 个文件。耗时: {time_str}")
        else:
            self.safe_log_print(f"\n全部完毕！成功转换: {success_count}/{total_files}。耗时: {time_str}")

        if failed_names:
            self.safe_log_print(f"\n失败文件汇总 ({len(failed_names)} 个):")
            for name in failed_names:
                self.safe_log_print(f"   × {name}")


        def _reset_btn():
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.is_processing = False
            title = "已终止" if was_aborted else "完成"
            messagebox.showinfo(title, f"执行结束。\n本次成功: {success_count}/{total_files}\n总耗时: {time_str}")

        self.root.after(0, _reset_btn)

    def start_batch(self):

        if self.is_processing: return

        in_dir = self.input_dir.get()
        out_dir = self.output_dir.get()

        if not in_dir or not out_dir:
            messagebox.showwarning("警告", "请先选择输入和输出文件夹。")
            return

        in_path = Path(in_dir)
        out_path = Path(out_dir)

        if not in_path.exists():
            messagebox.showerror("错误", "输入文件夹不存在。")
            return

        out_path.mkdir(parents=True, exist_ok=True)
        files = get_hdf5_files(in_path)

        if not files:
            messagebox.showinfo("提示", "在输入文件夹中没有找到 .h5 或 .hdf5 文件。")
            return

        self.log.delete(1.0, tk.END)

        # 【核心逻辑】：如果在界面勾选了跳过已存在，则预先过滤掉这些文件
        if self.skip_existing.get():
            pending_files = []
            skipped_count = 0
            for f in files:
                expected_mp4 = out_path / f"{f.stem}.mp4"
                # 如果这个 MP4 在目标文件夹里已经存在了，就不加到待办列表里
                if expected_mp4.exists():
                    skipped_count += 1
                else:
                    pending_files.append(f)
            files = pending_files

            if skipped_count > 0:
                self.safe_log_print(f"发现 {skipped_count} 个已存在的视频文件，已自动跳过。")

            if not files:
                self.safe_log_print("所有文件均已转换完毕，无需重复处理！")
                messagebox.showinfo("提示", "目标文件夹中已包含所有对应的视频文件，无需重复转换。")
                return

        self.is_processing = True
        self._progress_gen += 1   # 递增代次，上一轮的旧回调全部失效
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.abort_event.clear()

        self.start_time = time.time()

        threading.Thread(
            target=self.process_files_thread,
            args=(in_path, out_path, files),
            daemon=True
        ).start()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    manager = multiprocessing.Manager()
    global_abort_event = manager.Event()

    root = tk.Tk()
    app = BatchApp(root, global_abort_event)
    root.mainloop()