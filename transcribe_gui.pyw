#!/usr/bin/env python3
"""
视频/音频语音转文字工具 - 图形界面
双击运行，选择文件或文件夹，批量转写为 .txt 和 .srt 文件。
"""

import json
import logging
import os
import queue
import sys
import tempfile
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 自动定位同目录下的 .venv，确保用 venv 的依赖运行
_script_dir = Path(__file__).resolve().parent
_venv_site = _script_dir / ".venv" / "Lib" / "site-packages"
if _venv_site.is_dir() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

_file_logger = None
_log_path = None


def _init_file_logger():
    global _file_logger, _log_path
    if _file_logger is not None:
        return
    _log_path = Path(tempfile.gettempdir()) / "transcribe_gui.log"
    _file_logger = logging.getLogger("transcribe_gui")
    _file_logger.setLevel(logging.DEBUG)
    _file_logger.propagate = False
    handler = RotatingFileHandler(_log_path, maxBytes=2 * 1024 * 1024,
                                  backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _file_logger.addHandler(handler)


def _log_to_file(msg):
    _init_file_logger()
    _file_logger.info(msg)

# Windows 高 DPI 适配（必须在创建 Tk 窗口前设置）
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

__version__ = "1.0.0"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import transcribe_core as core

# 设置文件路径：%APPDATA%/transcribe/settings.json
if sys.platform == "win32":
    _settings_dir = Path(os.environ.get("APPDATA", str(Path.home()))) / "transcribe"
else:
    _settings_dir = Path.home() / ".transcribe"
_settings_path = _settings_dir / "settings.json"


class TranscribeApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"视频/音频语音转文字 v{__version__}")
        self.root.geometry("640x680")
        self.root.minsize(560, 600)
        self.root.resizable(True, True)

        self.files = []
        self.running = False
        self.worker_thread = None
        self._run_options = None
        self._model_refs = []
        self._msg_queue = queue.Queue()
        self._cancel = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_settings()

        _log_to_file("application started")
        self._log(f"日志文件: {_log_path}")

        # 轮询消息队列，处理后台线程发来的 GUI 更新
        self._poll_queue()

    def _poll_queue(self):
        """每 100ms 检查消息队列，将后台线程的 GUI 请求在主线程执行"""
        try:
            while True:
                callback = self._msg_queue.get_nowait()
                try:
                    callback()
                except Exception as e:
                    _log_to_file(f"GUI callback error: {e}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _post_to_gui(self, callback):
        """后台线程调用：将一个无参回调投递到主线程执行"""
        self._msg_queue.put(callback)

    def _release_model(self):
        """释放当前持有的模型引用，触发显存回收。"""
        self._model_refs.clear()
        import gc
        gc.collect()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 3}

        # 标题
        tk.Label(self.root, text="视频/音频语音转文字工具", font=("微软雅黑", 14, "bold")).pack(pady=(10, 2))
        tk.Label(self.root, text="基于 faster-whisper，完全本地运行，支持中英文和音频文件", fg="gray").pack()

        # 文件选择
        frame_files = tk.LabelFrame(self.root, text="视频/音频文件", padx=8, pady=4)
        frame_files.pack(fill="x", **pad)

        btn_row = tk.Frame(frame_files)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="选择文件", command=self._pick_files, width=10).pack(side="left")
        tk.Button(btn_row, text="选择文件夹", command=self._pick_folder, width=10).pack(side="left", padx=(6, 0))
        tk.Button(btn_row, text="清空", command=self._clear_files, width=6).pack(side="right")

        self.lbl_files = tk.Label(frame_files, text="未选择文件", fg="gray", anchor="w", justify="left", height=2)
        self.lbl_files.pack(fill="x", pady=(2, 0))

        # 参数设置
        frame_opts = tk.LabelFrame(self.root, text="参数设置", padx=8, pady=4)
        frame_opts.pack(fill="x", **pad)

        # 第一行：模型 + 设备
        row1 = tk.Frame(frame_opts)
        row1.pack(fill="x", pady=2)
        tk.Label(row1, text="模型:", width=6, anchor="e").pack(side="left")
        self.var_model = tk.StringVar(value="medium")
        ttk.Combobox(row1, textvariable=self.var_model, state="readonly", width=12,
                     values=["tiny", "base", "small", "medium", "large-v2", "large-v3"]).pack(side="left", padx=(4, 0))
        tk.Label(row1, text="(中文推荐 medium / large-v3)", fg="gray").pack(side="left", padx=(8, 0))

        row1b = tk.Frame(frame_opts)
        row1b.pack(fill="x", pady=2)
        tk.Label(row1b, text="设备:", width=6, anchor="e").pack(side="left")
        self.var_device = tk.StringVar(value="自动")
        ttk.Combobox(row1b, textvariable=self.var_device, state="readonly", width=12,
                     values=["自动", "GPU (CUDA)", "CPU"]).pack(side="left", padx=(4, 0))
        tk.Label(row1b, text="无显卡选 CPU", fg="gray").pack(side="left", padx=(8, 0))

        # 第二行：语言 + 简繁
        row2 = tk.Frame(frame_opts)
        row2.pack(fill="x", pady=2)
        tk.Label(row2, text="语言:", width=6, anchor="e").pack(side="left")
        self.var_lang = tk.StringVar(value="自动检测")
        ttk.Combobox(row2, textvariable=self.var_lang, state="readonly", width=12,
                     values=["自动检测", "中文 (zh)", "英文 (en)"]).pack(side="left", padx=(4, 0))

        tk.Label(row2, text="简繁:", width=6, anchor="e").pack(side="left", padx=(16, 0))
        self.var_variant = tk.StringVar(value="简体中文")
        ttk.Combobox(row2, textvariable=self.var_variant, state="readonly", width=12,
                     values=["简体中文", "繁体中文"]).pack(side="left", padx=(4, 0))

        # VAD 设置
        frame_vad = tk.LabelFrame(self.root, text="VAD 语音检测", padx=8, pady=4)
        frame_vad.pack(fill="x", **pad)

        vad_row0 = tk.Frame(frame_vad)
        vad_row0.pack(fill="x", pady=2)
        self.var_vad = tk.BooleanVar(value=True)
        tk.Checkbutton(vad_row0, text="启用 VAD（加速转写）", variable=self.var_vad,
                       command=self._toggle_vad).pack(side="left")

        self.frame_vad_params = tk.Frame(frame_vad)
        self.frame_vad_params.pack(fill="x", pady=(2, 0))

        vp1 = tk.Frame(self.frame_vad_params)
        vp1.pack(fill="x", pady=1)
        tk.Label(vp1, text="静音分段:", width=10, anchor="e").pack(side="left")
        self.var_silence = tk.StringVar(value="700")
        tk.Entry(vp1, textvariable=self.var_silence, width=8).pack(side="left", padx=(4, 0))
        tk.Label(vp1, text="ms（静音超过此时长才分段）", fg="gray").pack(side="left", padx=(4, 0))

        vp2 = tk.Frame(self.frame_vad_params)
        vp2.pack(fill="x", pady=1)
        tk.Label(vp2, text="语音填充:", width=10, anchor="e").pack(side="left")
        self.var_pad = tk.StringVar(value="300")
        tk.Entry(vp2, textvariable=self.var_pad, width=8).pack(side="left", padx=(4, 0))
        tk.Label(vp2, text="ms（语音前后保留的缓冲）", fg="gray").pack(side="left", padx=(4, 0))

        vp3 = tk.Frame(self.frame_vad_params)
        vp3.pack(fill="x", pady=1)
        tk.Label(vp3, text="检测灵敏度:", width=10, anchor="e").pack(side="left")
        self.var_threshold = tk.StringVar(value="0.50")
        tk.Entry(vp3, textvariable=self.var_threshold, width=8).pack(side="left", padx=(4, 0))
        tk.Label(vp3, text="（0~1，越低越灵敏）", fg="gray").pack(side="left", padx=(4, 0))

        # 选项
        frame_misc = tk.Frame(self.root)
        frame_misc.pack(fill="x", **pad)
        self.var_skip = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_misc, text="跳过已有 .txt/.srt 的文件", variable=self.var_skip).pack(side="left")

        # 按钮 + 状态
        frame_action = tk.Frame(self.root)
        frame_action.pack(fill="x", **pad)

        self.btn_start = tk.Button(frame_action, text="开始转写", command=self._start,
                                   font=("微软雅黑", 11), bg="#4CAF50", fg="white", width=14)
        self.btn_start.pack(side="left")

        self.btn_stop = tk.Button(frame_action, text="停止", command=self._stop,
                                  font=("微软雅黑", 11), bg="#f44336", fg="white", width=8,
                                  state="disabled")
        self.btn_stop.pack(side="left", padx=(6, 0))

        self.lbl_status = tk.Label(frame_action, text="就绪", fg="gray")
        self.lbl_status.pack(side="left", padx=(12, 0))

        # 日志输出
        frame_log = tk.LabelFrame(self.root, text="运行日志", padx=6, pady=4)
        frame_log.pack(fill="both", expand=True, **pad)

        self.log_text = scrolledtext.ScrolledText(frame_log, height=10, font=("Consolas", 9),
                                                  state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True)

        # 菜单栏
        self.root.option_add("*tearOff", False)
        menubar = tk.Menu(self.root)
        help_menu = tk.Menu(menubar)
        help_menu.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)

    def _toggle_vad(self):
        state = "normal" if self.var_vad.get() else "disabled"
        for child in self.frame_vad_params.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, tk.Entry):
                    widget.config(state=state)

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _show_about(self):
        messagebox.showinfo(
            "关于",
            f"视频/音频语音转文字 v{__version__}\n\n"
            f"基于 faster-whisper，完全本地运行。\n"
            f"支持中文/英文，输出 .txt 和 .srt 字幕。\n\n"
            f"默认内置 medium 模型，其他模型首次使用时自动下载。\n"
            f"日志文件: {_log_path}",
        )

    def _load_settings(self):
        """从 %APPDATA% 读取上次的参数设置，恢复到 UI 控件。"""
        try:
            data = json.loads(_settings_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.var_model.set(data.get("model", "medium"))
        self.var_device.set(data.get("device", "自动"))
        self.var_lang.set(data.get("language", "自动检测"))
        self.var_variant.set(data.get("variant", "简体中文"))
        self.var_vad.set(data.get("vad_enabled", True))
        self.var_silence.set(str(data.get("vad_silence", "700")))
        self.var_pad.set(str(data.get("vad_pad", "300")))
        self.var_threshold.set(str(data.get("vad_threshold", "0.50")))
        self.var_skip.set(data.get("skip_existing", True))
        self._toggle_vad()
        geo = data.get("window_geometry")
        if geo:
            try:
                self.root.geometry(geo)
            except Exception:
                pass

    def _save_settings(self):
        """把当前 UI 参数写入 %APPDATA%，下次启动恢复。"""
        try:
            _settings_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "model": self.var_model.get(),
                "device": self.var_device.get(),
                "language": self.var_lang.get(),
                "variant": self.var_variant.get(),
                "vad_enabled": self.var_vad.get(),
                "vad_silence": self.var_silence.get(),
                "vad_pad": self.var_pad.get(),
                "vad_threshold": self.var_threshold.get(),
                "skip_existing": self.var_skip.get(),
                "window_geometry": self.root.geometry(),
            }
            _settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            _log_to_file(f"save settings failed: {e}")

    def _pick_files(self):
        if self.running:
            return
        files = filedialog.askopenfilenames(title="选择视频/音频文件", filetypes=core.MEDIA_FILETYPES)
        if files:
            self.files = list(files)
            self._update_file_label()

    def _pick_folder(self):
        if self.running:
            return
        folder = filedialog.askdirectory(title="选择包含视频/音频的文件夹")
        if folder:
            self.files = [str(p) for p in core.find_media(folder)]
            self._update_file_label()

    def _clear_files(self):
        if self.running:
            return
        self.files = []
        self._update_file_label()

    def _update_file_label(self):
        if not self.files:
            self.lbl_files.config(text="未选择文件", fg="gray")
        elif len(self.files) <= 3:
            names = "\n".join(Path(f).name for f in self.files)
            self.lbl_files.config(text=names, fg="black")
        else:
            names = "\n".join(Path(f).name for f in self.files[:3])
            self.lbl_files.config(text=f"{names}\n... 共 {len(self.files)} 个文件", fg="black")

    def _get_device(self):
        mapping = {"自动": "auto", "GPU (CUDA)": "cuda", "CPU": "cpu"}
        return mapping.get(self.var_device.get(), "auto")

    def _get_vad_params(self):
        if not self.var_vad.get():
            return None
        try:
            return {
                "min_silence_duration_ms": int(self.var_silence.get()),
                "speech_pad_ms": int(self.var_pad.get()),
                "threshold": float(self.var_threshold.get()),
            }
        except ValueError:
            return None

    def _start(self):
        if self.running:
            return
        if not self.files:
            messagebox.showwarning("提示", "请先选择视频/音频文件或文件夹")
            return

        vad_params = self._get_vad_params()
        if self.var_vad.get() and vad_params is None:
            messagebox.showwarning("提示", "VAD 参数格式不正确，请检查")
            return

        self.running = True
        self._save_settings()
        self._run_options = {
            "files": list(self.files),
            "model_name": self.var_model.get(),
            "device": self._get_device(),
            "vad_enabled": self.var_vad.get(),
            "vad_params": vad_params,
            "skip_existing": self.var_skip.get(),
            "language_label": self.var_lang.get(),
            "variant": self.var_variant.get(),
        }
        self.btn_start.config(state="disabled", text="转写中...")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="正在加载模型...", fg="blue")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

        self.worker_thread = threading.Thread(target=self._run_transcribe, daemon=True)
        self.worker_thread.start()

    def _run_transcribe(self):
        error_msg = None
        try:
            _log_to_file("thread started")

            # 准备参数
            options = self._run_options or {}
            model_name = options["model_name"]
            device = options["device"]
            vad_enabled = options["vad_enabled"]
            vad_params = options["vad_params"]
            skip_existing = options["skip_existing"]

            lang_map = {"中文 (zh)": "zh", "英文 (en)": "en", "自动检测": None}
            language = lang_map.get(options["language_label"])

            initial_prompt = None
            if options["variant"] == "简体中文":
                initial_prompt = "以下是普通话的句子，请使用简体中文输出。"
            elif options["variant"] == "繁体中文":
                initial_prompt = "以下是普通话的句子。"
            variant = options["variant"]

            # 加载模型
            self._post_to_gui(lambda: self._log(f"正在加载模型 ({model_name})..."))
            model, actual_device, compute_type = core.load_model(model_name, device)
            self._model_refs.append(model)
            self._post_to_gui(lambda: self._log(f"模型加载完成，使用 {actual_device} ({compute_type})"))
            _log_to_file(f"model loaded: {actual_device}")

            # 过滤已处理文件
            files_to_process = []
            skipped = 0
            for f in options["files"]:
                p = Path(f)
                if skip_existing and core.is_output_valid(p.with_suffix(".txt"), p.with_suffix(".srt")):
                    skipped += 1
                else:
                    files_to_process.append(f)

            if skipped:
                self._post_to_gui(lambda s=skipped: self._log(f"跳过 {s} 个已处理文件"))

            if not files_to_process:
                self._post_to_gui(lambda: self._log("所有文件已有输出，无需处理"))
                _log_to_file("no files to process, returning")
                return

            total = len(files_to_process)
            self._post_to_gui(lambda t=total: self._log(f"共 {t} 个文件待处理"))
            self._post_to_gui(lambda t=total: self.lbl_status.config(text=f"正在转写 (0/{t})...", fg="blue"))
            _log_to_file(f"processing {total} files")

            # 逐个处理
            success = 0
            failed_files = []
            for idx, file_path in enumerate(files_to_process, 1):
                if self._cancel:
                    self._post_to_gui(lambda: self._log("已停止"))
                    break

                p = Path(file_path)
                self._post_to_gui(lambda i=idx, t=total, n=p.name: (
                    self._log(f"[{i}/{t}] 正在转写: {n}"),
                    self.lbl_status.config(text=f"正在转写 ({i}/{t}): {n}", fg="blue"),
                ))
                _log_to_file(f"transcribing {p.name}")

                def _on_seg(seg_idx, start, end, text, _idx=idx, _total=total, _name=p.name):
                    ts = core.format_timestamp(start)
                    short = text if len(text) <= 30 else text[:30] + "..."
                    self._post_to_gui(lambda: self._log(f"  [{_idx}/{_total}] {ts} {short}"))

                def _on_prog(seg_count, current_time, total_dur, _idx=idx, _total=total, _name=p.name):
                    if total_dur > 0:
                        pct = min(100.0, current_time / total_dur * 100)
                        self._post_to_gui(lambda p=pct, i=_idx, t=_total, n=_name:
                            self.lbl_status.config(text=f"正在转写 ({i}/{t}): {n} {p:.0f}%", fg="blue"))

                result = None
                file_error = None
                try:
                    result = core.transcribe_file(
                        model, p,
                        language=language,
                        initial_prompt=initial_prompt,
                        vad_enabled=vad_enabled,
                        vad_params=vad_params,
                        on_segment=_on_seg,
                        on_progress=_on_prog,
                        variant=variant,
                    )
                except Exception as e:
                    if actual_device == "cuda" and core.is_memory_error(e):
                        self._post_to_gui(lambda: self._log("GPU 显存不足，自动切换到 CPU 重试当前文件..."))
                        _log_to_file(f"cuda memory error, retrying on cpu: {e}")
                        self._release_model()
                        model, actual_device, compute_type = core.load_model(model_name, "cpu")
                        self._model_refs = [model]
                        self._post_to_gui(lambda d=actual_device, c=compute_type:
                            self._log(f"模型加载完成，使用 {d} ({c})"))
                        try:
                            result = core.transcribe_file(
                                model, p,
                                language=language,
                                initial_prompt=initial_prompt,
                                vad_enabled=vad_enabled,
                                vad_params=vad_params,
                                on_segment=_on_seg,
                                on_progress=_on_prog,
                                variant=variant,
                            )
                        except Exception as e2:
                            file_error = f"转写失败(CPU重试后): {e2}"
                    else:
                        file_error = f"转写失败: {e}"
                _log_to_file(f"transcribe_file returned: {result is not None}, error: {file_error}")

                if result is not None and file_error is None:
                    try:
                        core.atomic_write_text(p.with_suffix(".txt"), result["txt_text"])
                        core.atomic_write_text(p.with_suffix(".srt"), result["srt_text"])
                        success += 1
                        elapsed = result['elapsed']
                        segs = result['segments']
                        self._post_to_gui(lambda i=idx, t=total, n=p.name, e=elapsed, s=segs:
                                          self._log(f"[{i}/{t}] 完成: {n} ({s}句, {e:.1f}s)"))
                    except Exception as e:
                        file_error = f"写文件失败: {e}"

                if file_error:
                    failed_files.append((p.name, file_error))
                    self._post_to_gui(lambda e=file_error, n=p.name: self._log(f"[失败] {n}: {e}"))

            summary = f"结束: 成功 {success}/{total}"
            if failed_files:
                summary += f"，失败 {len(failed_files)}"
            if self._cancel:
                summary += "（已停止）"
            self._post_to_gui(lambda s=summary: self._log(s))
            if failed_files:
                self._post_to_gui(lambda: self._log("失败明细:"))
                for name, err in failed_files:
                    self._post_to_gui(lambda n=name, e=err: self._log(f"  - {n}: {e}"))
            self._post_to_gui(lambda s=success, f=len(failed_files):
                self.lbl_status.config(
                    text=("就绪" if not failed_files and not self._cancel else
                          f"成功 {s}" + (f"，失败 {f}" if failed_files else "")),
                    fg="gray"))
            _log_to_file(f"done: {success}/{total}, failed: {len(failed_files)}")

        except Exception as e:
            error_msg = str(e)
            self._post_to_gui(lambda msg=str(e): self._log(f"出错: {msg}"))
            _log_to_file(f"exception in thread: {e}")
        finally:
            self._post_to_gui(lambda m=error_msg: self._on_thread_done(m))
            _log_to_file("thread finished (finally)")

    def _on_thread_done(self, error_msg=None):
        self.running = False
        self._cancel = False
        self.btn_start.config(state="normal", text="开始转写")
        self.btn_stop.config(state="disabled")
        if error_msg:
            self.lbl_status.config(text=f"出错: {error_msg}", fg="red")
        else:
            self.lbl_status.config(text="就绪", fg="gray")

    def _stop(self):
        if not self.running:
            return
        if messagebox.askokcancel("确认停止", "确定停止当前转写任务吗？\n当前文件完成后中止。"):
            self._cancel = True
            self.btn_stop.config(state="disabled")
            self._log("正在停止（当前文件完成后中止）...")

    def _on_close(self):
        self._save_settings()
        if self.running:
            if not messagebox.askokcancel("确认退出", "转写进行中，确定退出吗？\n将停止当前任务。"):
                return
            self._cancel = True
            self.root.after(300, self._maybe_close)
            return
        self.root.destroy()

    def _maybe_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(300, self._maybe_close)
        else:
            self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TranscribeApp()
    app.run()
