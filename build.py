"""打包脚本：复制内置 medium 模型 + PyInstaller 打包成单 exe。

用法:
    .venv\\Scripts\\python.exe build.py                    # 带内置模型（~1.5 GB）
    .venv\\Scripts\\python.exe build.py --no-model         # 不带模型（~200 MB），首次运行从 HF 下载

前置条件（内置模型版）: 先运行一次程序用 medium 模型触发下载到 HF cache。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = sys.executable  # 兜底
BUNDLED_DIR = ROOT / "bundled_models" / "medium"

NO_MODEL = "--no-model" in sys.argv


def prepare_model():
    """从 HF cache 复制 medium 模型到 bundled_models/medium/。"""
    if NO_MODEL:
        print("跳过模型复制（--no-model）")
        return
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-medium"
    snapshots = hf_cache / "snapshots"
    if not snapshots.is_dir():
        print("错误：未找到 medium 模型缓存。请先运行一次程序用 medium 模型触发下载。")
        sys.exit(1)

    snap_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    if not snap_dirs:
        print("错误：medium 模型 snapshot 目录为空")
        sys.exit(1)
    src = snap_dirs[0]

    BUNDLED_DIR.mkdir(parents=True, exist_ok=True)
    files = ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]
    for f in files:
        s = src / f
        if not s.exists():
            print(f"错误：缺少模型文件 {f}")
            sys.exit(1)
        dst = BUNDLED_DIR / f
        if not dst.exists() or dst.stat().st_size != s.stat().st_size:
            print(f"复制 {f} ({s.stat().st_size / 1e6:.0f} MB)...")
            shutil.copy2(s, dst)
    print(f"模型准备完成: {BUNDLED_DIR}")


def ensure_pyinstaller():
    """确保 pyinstaller 已安装。"""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("安装 PyInstaller...")
        subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "pyinstaller", "-q"])


def build():
    """执行 PyInstaller 打包。"""
    sep = ";" if sys.platform == "win32" else ":"
    cmd = [
        str(VENV_PY), "-m", "PyInstaller",
        "--onefile", "--windowed",
        "--collect-all", "faster_whisper",
        "--collect-all", "ctranslate2",
        "--collect-all", "onnxruntime",
        "--collect-all", "av",
        "--noconfirm",
        "--clean",
    ]
    if NO_MODEL:
        cmd += ["--name", "语音转文字_无模型版"]
    else:
        cmd += ["--name", "语音转文字"]
        cmd += ["--add-data", f"bundled_models/medium{sep}bundled_models/medium"]

    cmd.append("transcribe_gui.pyw")
    print("执行打包...")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    exe_name = "语音转文字_无模型版.exe" if NO_MODEL else "语音转文字.exe"
    exe = ROOT / "dist" / exe_name
    print(f"\n打包完成: {exe}")
    if exe.exists():
        size_mb = exe.stat().st_size / (1024 * 1024)
        print(f"体积: {size_mb:.0f} MB")


if __name__ == "__main__":
    prepare_model()
    ensure_pyinstaller()
    build()
