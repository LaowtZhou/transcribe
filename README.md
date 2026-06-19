# 视频/音频语音转文字工具

基于 [faster-whisper](https://github.com/SYstran/faster-whisper) 的本地语音转文字工具。模型下载完成后完全离线运行，不联网、不传云端。支持中文和英文，输出 `.txt` 纯文本和 `.srt` 字幕文件。

## 下载使用（普通用户）

1. 到 [Releases](../../releases) 页面下载最新的 `语音转文字_无模型版.exe`
2. 双击运行
3. 点「选择文件」或「选择文件夹」添加音视频
4. 点「开始转写」

首次运行选择 `medium` 或 `large-v3` 模型时，程序会自动从国内镜像下载（约 1.5~3 GB），下载一次后永久离线使用。

**输出**：`.txt` 和 `.srt` 自动生成在源文件同目录。

## 功能

- 批量转写视频/音频 → `.txt` + `.srt`
- 支持格式：mp4 / mkv / avi / mov / mp3 / wav / flac / m4a 等常见格式
- 自动检测语言，或手动指定中文/英文
- 简体中文 / 繁体中文输出（OpenCC 后处理，确保字符统一）
- GPU（CUDA）加速，显存不足自动回退 CPU
- VAD 语音活动检测，过滤静音段加速转写
- 转写过程实时显示进度和百分比
- 中途可停止，单文件失败不影响整批
- 跳过已处理文件（可关闭）
- 设置自动保存，下次启动恢复

## 源码运行（开发者）

需要 Python 3.10+。

```bash
# 创建虚拟环境
python -m venv .venv

# 激活（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行
.venv\Scripts\pythonw.exe transcribe_gui.pyw
```

运行测试：

```bash
.venv\Scripts\python.exe -m pytest test_transcribe_core.py -v
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 模型 | medium | 首次使用自动从镜像下载到本地缓存。tiny 最快最差，large-v3 最慢最准。中文推荐 medium / large-v3 |
| 设备 | 自动 | 有 NVIDIA 显卡自动用 GPU，否则 CPU |
| 语言 | 自动检测 | 可手动指定中文/英文 |
| 简繁 | 简体中文 | OpenCC 统一字符变体 |
| VAD 静音分段 | 700ms | 静音超过此时长才切断分段 |
| VAD 语音填充 | 300ms | 语音前后保留的缓冲 |
| VAD 灵敏度 | 0.50 | 0~1，越低越灵敏 |

## 打包成 exe

```bash
.venv\Scripts\pip.exe install pyinstaller
.venv\Scripts\python.exe build.py              # 内置 medium 模型版（1.44 GB）
.venv\Scripts\python.exe build.py --no-model   # 无模型版（96 MB），首次运行下载模型
```

打包产物在 `dist/` 目录。

## 技术栈

- [faster-whisper](https://github.com/SYstran/faster-whisper) — Whisper 模型的 CTranslate2 实现
- [OpenCC](https://github.com/yichen0831/opencc-python) — 简繁转换
- Tkinter — GUI

## 许可证

[MIT](LICENSE)
