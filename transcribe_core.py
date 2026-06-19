"""
转录核心模块
提供模型加载、文件发现、转录执行等基础功能，供 GUI 调用。
"""

import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# === 环境初始化 ===

# Hugging Face 镜像（国内加速）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# NVIDIA CUDA DLL 搜索路径（Windows）
# 从当前文件位置推导 venv 路径，不依赖 site.getsitepackages()（非激活 venv 下不准确）
if sys.platform == "win32":
    _this_dir = Path(__file__).resolve().parent
    _venv_site = _this_dir / ".venv" / "Lib" / "site-packages"
    nvidia_dirs = []
    for lib in ("cublas", "cudnn", "cuda_nvrtc"):
        bin_dir = _venv_site / "nvidia" / lib / "bin"
        if bin_dir.is_dir():
            nvidia_dirs.append(str(bin_dir))
    if nvidia_dirs:
        os.environ["PATH"] = ";".join(nvidia_dirs) + ";" + os.environ.get("PATH", "")

# === 常量 ===

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".ape", ".aiff"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

MEDIA_FILETYPES = [
    ("视频/音频文件", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv *.m4v *.ts *.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma *.opus *.ape *.aiff"),
    ("视频文件", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv *.m4v *.ts"),
    ("音频文件", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma *.opus *.ape *.aiff"),
    ("所有文件", "*.*"),
]

# === 工具函数 ===


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式 HH:MM:SS,mmm"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> None:
    """原子写文件：先写 .tmp 再 os.replace，避免中途崩溃留半截文件。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def is_output_valid(txt_path, srt_path) -> bool:
    """校验已存在的输出文件是否完整（存在 + 非空 + srt 至少 4 行）。"""
    txt_path, srt_path = Path(txt_path), Path(srt_path)
    if not txt_path.exists() or not srt_path.exists():
        return False
    if txt_path.stat().st_size == 0 or srt_path.stat().st_size == 0:
        return False
    try:
        content = srt_path.read_text(encoding="utf-8").strip()
        if len(content.split("\n")) < 3:
            return False
    except Exception:
        return False
    return True


TERMINAL_PUNCTUATION = "。｡！？!?…"
PHRASE_PUNCTUATION = "，,﹐：:；;"
ENUM_PUNCTUATION = "、､"
BREAK_PUNCTUATION = TERMINAL_PUNCTUATION + PHRASE_PUNCTUATION + ENUM_PUNCTUATION
BREAK_RE = re.compile(rf"[{re.escape(BREAK_PUNCTUATION)}]+[”’\"']?$")
MIN_SUBTITLE_CHARS = 8
TARGET_SUBTITLE_CHARS = 14
MAX_SUBTITLE_CHARS = 24
MIN_SUBTITLE_DURATION = 1.2
TARGET_SUBTITLE_DURATION = 2.2
MAX_SUBTITLE_DURATION = 5.0
MIN_SUBTITLE_GAP = 0.08
PUNCTUATION_TRANSLATION = str.maketrans({
    "﹐": "，",
    "､": "、",
    "｡": "。",
    "﹒": "。",
    "﹕": "：",
    "﹔": "；",
    "！": "！",
    "？": "？",
})


def _clean_subtitle_text(text: str) -> str:
    """清理字幕文本，避免导入剪辑软件后出现多余空格。"""
    text = text.translate(PUNCTUATION_TRANSLATION)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    return text


def _text_len(text: str) -> int:
    return len(re.sub(rf"[\s{re.escape(BREAK_PUNCTUATION)}]", "", text.translate(PUNCTUATION_TRANSLATION)))


def _word_text(word) -> str:
    return getattr(word, "word", "").strip()


def _word_start(word) -> float:
    value = getattr(word, "start", None)
    if value is None:
        value = getattr(word, "end", None)
    return 0.0 if value is None else float(value)


def _word_end(word) -> float:
    value = getattr(word, "end", None)
    if value is None:
        value = getattr(word, "start", None)
    return 0.0 if value is None else float(value)


def _chunk_start(words: list) -> float:
    for word in words:
        value = getattr(word, "start", None)
        if value is not None:
            return float(value)
    return _word_start(words[0]) if words else 0.0


def _chunk_end(words: list) -> float:
    for word in reversed(words):
        value = getattr(word, "end", None)
        if value is not None:
            return float(value)
    return _word_end(words[-1]) if words else 0.0


def _chunk_text(words: list) -> str:
    return _clean_subtitle_text("".join(_word_text(w) for w in words))


def _ending_punctuation_type(text: str) -> str | None:
    text = _clean_subtitle_text(text).rstrip("”’\"'")
    if not text:
        return None
    char = text[-1]
    if char in TERMINAL_PUNCTUATION:
        return "terminal"
    if char in PHRASE_PUNCTUATION:
        return "phrase"
    if char in ENUM_PUNCTUATION:
        return "enum"
    return None


def _subtitle_duration(words: list) -> float:
    return _chunk_end(words) - _chunk_start(words)


def _append_subtitle(items: list, words: list):
    text = _chunk_text(words)
    if text:
        items.append({
            "start": _chunk_start(words),
            "end": _chunk_end(words),
            "text": text,
        })


def _best_break_index(words: list, allow_last: bool = True) -> int | None:
    """选择最适合切开的候选点，避免短碎片和顿号碎切。"""
    candidates = []
    for index in range(len(words)):
        head = words[:index + 1]
        tail = words[index + 1:]
        if not allow_last and not tail:
            continue
        head_len = _text_len(_chunk_text(head))
        tail_len = _text_len(_chunk_text(tail)) if tail else 0
        punct_type = _ending_punctuation_type(_chunk_text(head))
        if not punct_type:
            continue
        if head_len < MIN_SUBTITLE_CHARS:
            continue
        if tail and tail_len < MIN_SUBTITLE_CHARS:
            continue
        if punct_type == "enum" and head_len < TARGET_SUBTITLE_CHARS:
            continue

        priority = {"terminal": 3, "phrase": 2, "enum": 1}[punct_type]
        distance = abs(TARGET_SUBTITLE_CHARS - head_len)
        candidates.append((priority, -distance, index))

    if not candidates:
        return None
    return max(candidates)[2]


def _split_word_at_punctuation(word) -> list:
    """把一个含多个标点的 word token 拆成更小片段，时间按字符数近似分配。"""
    text = _clean_subtitle_text(_word_text(word))
    if not text:
        return []

    parts = re.findall(rf".*?[{re.escape(BREAK_PUNCTUATION)}]+[”’\"']?|.+$", text)
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return [SimpleNamespace(word=text, start=_word_start(word), end=_word_end(word))]

    start = _word_start(word)
    end = max(start, _word_end(word))
    duration = end - start
    total_len = max(1, sum(len(part) for part in parts))
    cursor = start
    split_words = []

    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            part_end = end
        else:
            part_end = start + duration * (sum(len(p) for p in parts[:index + 1]) / total_len)
        split_words.append(SimpleNamespace(word=part, start=cursor, end=part_end))
        cursor = part_end

    return split_words


def _split_words_to_subtitles(words: list) -> list[dict]:
    """用 word timestamps 重新切字幕，每条字幕使用首尾字的真实时间。"""
    items = []
    chunk = []
    split_words = []
    for word in words:
        split_words.extend(_split_word_at_punctuation(word))

    def flush():
        if not chunk:
            return
        _append_subtitle(items, chunk)
        chunk.clear()

    def flush_until(index: int):
        if index < 0:
            return
        _append_subtitle(items, chunk[:index + 1])
        del chunk[:index + 1]

    for word in split_words:
        text = _word_text(word)
        if not text:
            continue

        if chunk:
            gap = _word_start(word) - _chunk_end(chunk)
            chunk_len = _text_len(_chunk_text(chunk))
            chunk_duration = _subtitle_duration(chunk)
            if (
                gap >= 0.55
                and chunk_len >= MIN_SUBTITLE_CHARS
            ) or (
                gap >= 0.25
                and (chunk_len >= TARGET_SUBTITLE_CHARS or chunk_duration >= TARGET_SUBTITLE_DURATION)
            ):
                flush()

        chunk.append(word)
        current_text = _chunk_text(chunk)
        current_len = _text_len(current_text)
        current_duration = _subtitle_duration(chunk)
        punct_type = _ending_punctuation_type(current_text)

        if current_len >= MAX_SUBTITLE_CHARS or current_duration >= MAX_SUBTITLE_DURATION:
            break_index = _best_break_index(chunk, allow_last=False)
            if break_index is not None:
                flush_until(break_index)
                continue

        should_flush = False
        if punct_type == "terminal":
            # 字数和时长都达标才单独成行，避免 4字1秒 的碎片被放出来
            # 不达标的继续累积，由 MAX 兜底强切或末尾 merge 合并
            should_flush = (
                current_len >= MIN_SUBTITLE_CHARS
                and current_duration >= MIN_SUBTITLE_DURATION
            )
        elif punct_type == "phrase":
            should_flush = (
                current_len >= TARGET_SUBTITLE_CHARS
                or current_duration >= TARGET_SUBTITLE_DURATION
            )
        elif punct_type == "enum":
            should_flush = (
                current_len >= MAX_SUBTITLE_CHARS
                or current_duration >= MAX_SUBTITLE_DURATION
            )

        if should_flush:
            flush()
            continue

    flush()
    return _merge_short_subtitles(items)


def _merge_short_subtitles(items: list[dict]) -> list[dict]:
    """把过短字幕并入相邻字幕，避免 0.3 秒/两三个字的碎片。"""
    merged = []
    for item in items:
        text_len = _text_len(item["text"])
        duration = item["end"] - item["start"]
        if (
            merged
            and (text_len < MIN_SUBTITLE_CHARS or duration < MIN_SUBTITLE_DURATION)
            and _text_len(merged[-1]["text"] + item["text"]) <= MAX_SUBTITLE_CHARS + 4
            and item["end"] - merged[-1]["start"] <= MAX_SUBTITLE_DURATION + 1.0
        ):
            merged[-1]["end"] = item["end"]
            merged[-1]["text"] = _clean_subtitle_text(merged[-1]["text"] + item["text"])
        else:
            merged.append(item)

    return merged


def _normalize_subtitle_timing(items: list[dict]) -> list[dict]:
    """修正重叠/贴边时间，给剪辑软件保留可识别的字幕间隙。"""
    normalized = []
    for item in sorted(items, key=lambda x: (x["start"], x["end"])):
        start = max(0.0, float(item["start"]))
        end = max(start + 0.1, float(item["end"]))
        text = _clean_subtitle_text(item["text"])
        if not text:
            continue

        if normalized:
            prev = normalized[-1]
            if prev["end"] > start - MIN_SUBTITLE_GAP:
                prev["end"] = max(prev["start"] + 0.1, start - MIN_SUBTITLE_GAP)
            if start <= prev["end"]:
                start = prev["end"] + MIN_SUBTITLE_GAP
                end = max(end, start + 0.1)

        normalized.append({"start": start, "end": end, "text": text})

    return normalized


def find_media(path: str) -> list[Path]:
    """查找目录下所有视频和音频文件（递归）"""
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in MEDIA_EXTENSIONS else []

    media = []
    for ext in MEDIA_EXTENSIONS:
        media.extend(p.rglob(f"*{ext}"))
        media.extend(p.rglob(f"*{ext.upper()}"))
    return sorted(set(media))


# === 设备与模型 ===


def detect_device(requested: str = "auto") -> str:
    """
    检测计算设备。
    requested: "auto" | "cuda" | "cpu"
    返回实际可用的设备 "cuda" 或 "cpu"。
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    # auto
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def is_memory_error(exc) -> bool:
    """统一判断是否为显存/内存不足类错误（core 与 GUI 共用）。"""
    msg = str(exc).lower()
    return any(k in msg for k in ("bad allocation", "out of memory", "cuda", "cublas", "cudnn", "memory"))


def _bundled_model_path(model_name: str) -> str | None:
    """打包模式下返回内置模型本地路径，源码模式返回 None（走 HuggingFace 下载）。"""
    if getattr(sys, "frozen", False):
        candidate = Path(sys._MEIPASS) / "bundled_models" / model_name
        if (candidate / "model.bin").exists():
            return str(candidate)
    return None


def load_model(model_name: str = "medium", device: str = "auto"):
    """
    加载 Whisper 模型，GPU 失败时自动回退 CPU。
    返回 (model, actual_device, compute_type)。
    """
    from faster_whisper import WhisperModel

    actual_device = detect_device(device)
    compute_type = "float16" if actual_device == "cuda" else "int8"

    # 打包模式优先用内置模型，源码模式 model_path == model_name（触发 HF 下载）
    model_path = _bundled_model_path(model_name) or model_name

    try:
        model = WhisperModel(model_path, device=actual_device, compute_type=compute_type)
    except (RuntimeError, OSError) as e:
        if actual_device == "cuda" and is_memory_error(e):
            actual_device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_path, device="cpu", compute_type="int8")
        else:
            raise

    return model, actual_device, compute_type


# === 转录 ===

_s2t_converter = None
_t2s_converter = None


def _get_s2t_converter():
    """懒加载简繁转换器（进程内只初始化一次）。"""
    global _s2t_converter
    if _s2t_converter is None:
        from opencc import OpenCC
        _s2t_converter = OpenCC("s2t")
    return _s2t_converter


def _get_t2s_converter():
    """懒加载繁简转换器（进程内只初始化一次）。"""
    global _t2s_converter
    if _t2s_converter is None:
        from opencc import OpenCC
        _t2s_converter = OpenCC("t2s")
    return _t2s_converter


def convert_to_traditional(text: str) -> str:
    """简体转繁体（用于繁体中文输出模式）。"""
    if not text:
        return text
    return _get_s2t_converter().convert(text)


def convert_to_simplified(text: str) -> str:
    """繁体转简体（用于简体中文输出模式，确保 Whisper 输出的零星繁体字被统一）。"""
    if not text:
        return text
    return _get_t2s_converter().convert(text)


def transcribe_file(
    model,
    file_path: str | Path,
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_enabled: bool = True,
    vad_params: dict | None = None,
    on_segment=None,
    on_progress=None,
    variant: str = "简体中文",
) -> dict | None:
    """
    转写单个文件，返回结果字典。

    参数:
        model: 已加载的 WhisperModel
        file_path: 音视频文件路径
        language: 语言代码，None 表示自动检测
        initial_prompt: 初始提示词（用于引导简繁体输出）
        vad_enabled: 是否启用 VAD
        vad_params: VAD 参数字典，None 使用默认值
        on_segment: 回调函数 on_segment(index, start, end, text)，转录过程中每段实时调用
        on_progress: 回调函数 on_progress(seg_count, current_time, total_duration)，用于进度展示
        variant: 输出变体，"简体中文" 或 "繁体中文"（繁体走 OpenCC 后处理）

    返回:
        dict 包含 txt_text, srt_text, language, language_probability, duration, segments, elapsed
        文件不存在时返回 None
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    # 默认 VAD 参数
    if vad_enabled and vad_params is None:
        vad_params = {
            "min_silence_duration_ms": 700,
            "speech_pad_ms": 300,
            "threshold": 0.50,
        }

    start_time = time.time()
    segments, info = model.transcribe(
        str(file_path),
        language=language,
        beam_size=5,
        vad_filter=vad_enabled,
        vad_parameters=vad_params if vad_enabled else None,
        initial_prompt=initial_prompt,
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0 if vad_enabled else None,
    )

    subtitle_items = []
    total_duration = float(getattr(info, "duration", 0.0) or 0.0)
    seg_count = 0

    # 根据 variant 决定字符变体转换函数（简体走 t2s，繁体走 s2t，统一压制 Whisper 输出里的零星异体字）
    _variant_convert = None
    if variant == "简体中文":
        _variant_convert = convert_to_simplified
    elif variant == "繁体中文":
        _variant_convert = convert_to_traditional

    for segment in segments:
        seg_count += 1
        words = getattr(segment, "words", None) or []
        if words:
            subtitle_items.extend(_split_words_to_subtitles(words))
        else:
            text = _clean_subtitle_text(segment.text)
            if text:
                subtitle_items.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": text,
                })

        # 实时回调：转录过程中每段即时推送（faster-whisper 的 segments 是惰性 generator）
        if on_segment:
            seg_text = _clean_subtitle_text(segment.text)
            if _variant_convert:
                seg_text = _variant_convert(seg_text)
            on_segment(seg_count, segment.start, segment.end, seg_text)
        if on_progress:
            on_progress(seg_count, float(segment.end), total_duration)

    subtitle_items = _merge_short_subtitles(subtitle_items)
    subtitle_items = _normalize_subtitle_timing(subtitle_items)
    if _variant_convert:
        for item in subtitle_items:
            item["text"] = _variant_convert(item["text"])

    txt_lines = []
    srt_lines = []
    for idx, item in enumerate(subtitle_items, 1):
        text = item["text"]
        txt_lines.append(text)
        srt_lines.append(str(idx))
        srt_lines.append(f"{format_timestamp(item['start'])} --> {format_timestamp(item['end'])}")
        srt_lines.append(text)
        srt_lines.append("")

    elapsed = time.time() - start_time

    txt_text = "\n".join(txt_lines)
    srt_text = "\n".join(srt_lines)

    return {
        "txt_text": txt_text,
        "srt_text": srt_text,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": len(subtitle_items),
        "elapsed": elapsed,
    }
