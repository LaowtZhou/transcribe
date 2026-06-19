"""transcribe_core 核心切分函数单元测试。

覆盖 _normalize_subtitle_timing / _best_break_index / _merge_short_subtitles / _split_words_to_subtitles。
运行: .venv\\Scripts\\python.exe -m pytest test_transcribe_core.py -v
"""

from types import SimpleNamespace

import transcribe_core as core


def make_word(text, start, end):
    """构造模拟的 word 对象（与 faster-whisper 的 word segment 接口一致）。"""
    return SimpleNamespace(word=text, start=start, end=end)


# === _normalize_subtitle_timing ===

def test_normalize_two_non_overlapping():
    items = [
        {"start": 0.0, "end": 2.0, "text": "第一条字幕测试"},
        {"start": 2.5, "end": 4.0, "text": "第二条字幕测试"},
    ]
    result = core._normalize_subtitle_timing(items)
    assert len(result) == 2
    assert result[0]["start"] == 0.0 and result[0]["end"] == 2.0
    assert result[1]["start"] == 2.5 and result[1]["end"] == 4.0


def test_normalize_overlapping_adjusts_prev_end():
    items = [
        {"start": 0.0, "end": 2.0, "text": "第一条字幕测试"},
        {"start": 1.5, "end": 3.0, "text": "第二条字幕测试"},
    ]
    result = core._normalize_subtitle_timing(items)
    assert len(result) == 2
    # prev.end 被回退到 start - MIN_SUBTITLE_GAP
    assert result[0]["end"] < 2.0
    assert result[1]["start"] - result[0]["end"] >= core.MIN_SUBTITLE_GAP - 1e-9


def test_normalize_empty_text_skipped():
    items = [
        {"start": 0.0, "end": 1.0, "text": ""},
        {"start": 1.0, "end": 2.0, "text": "有效字幕内容"},
    ]
    result = core._normalize_subtitle_timing(items)
    assert len(result) == 1
    assert result[0]["text"] == "有效字幕内容"


def test_normalize_negative_start_clamped():
    items = [{"start": -1.0, "end": 2.0, "text": "负时间测试"}]
    result = core._normalize_subtitle_timing(items)
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 2.0


def test_normalize_empty_input():
    assert core._normalize_subtitle_timing([]) == []


# === _best_break_index ===

def test_best_break_terminal_punct_returned():
    words = [
        make_word("今天天气真的很不错", 0, 1),
        make_word("我们去公园玩吧。", 1, 2),
        make_word("明天继续来玩好不好", 2, 3),
    ]
    idx = core._best_break_index(words)
    assert idx == 1


def test_best_break_terminal_beats_phrase():
    words = [
        make_word("今天天气", 0, 1),
        make_word("很好，", 1, 2),
        make_word("真的很不错。", 2, 3),
        make_word("我们去公园玩好不好", 3, 4),
    ]
    idx = core._best_break_index(words)
    # terminal(3) 优先于 phrase(2)，返回句号所在 index
    assert idx == 2


def test_best_break_no_punctuation_returns_none():
    words = [make_word("你好", 0, 1), make_word("世界", 1, 2), make_word("测试", 2, 3)]
    assert core._best_break_index(words) is None


def test_best_break_tail_too_short_skipped():
    words = [
        make_word("今天天气真的很不错", 0, 1),
        make_word("你好。", 1, 2),
        make_word("测试一下", 2, 3),  # tail 长度 4 < MIN_SUBTITLE_CHARS(8)
    ]
    assert core._best_break_index(words) is None


# === _merge_short_subtitles ===

def test_merge_all_normal_kept():
    items = [
        {"start": 0.0, "end": 2.0, "text": "正常长度的字幕测试一"},
        {"start": 2.5, "end": 4.5, "text": "正常长度的字幕测试二"},
    ]
    result = core._merge_short_subtitles(items)
    assert len(result) == 2


def test_merge_short_into_prev():
    items = [
        {"start": 0.0, "end": 2.0, "text": "正常长度的字幕测试"},
        {"start": 2.0, "end": 2.3, "text": "短"},  # 文本短 + 时长短
    ]
    result = core._merge_short_subtitles(items)
    assert len(result) == 1
    assert "短" in result[0]["text"]
    assert result[0]["end"] == 2.3


def test_merge_short_not_merged_when_prev_full():
    # 前条已达 MAX_SUBTITLE_CHARS，合并后超出 MAX+4，不合并
    long_text = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八"  # 27 字
    items = [
        {"start": 0.0, "end": 5.0, "text": long_text},
        {"start": 5.0, "end": 5.3, "text": "短句子四个字"},
    ]
    result = core._merge_short_subtitles(items)
    assert len(result) == 2


def test_merge_empty_input():
    assert core._merge_short_subtitles([]) == []


# === _split_words_to_subtitles ===

def test_split_basic_two_subtitles():
    words = [
        make_word("今天天气真的很不错", 0, 2),
        make_word("我们去公园玩吧。", 2, 4),
        make_word("明天继续来玩好不好", 4, 6),
    ]
    result = core._split_words_to_subtitles(words)
    assert len(result) == 2
    assert "。" in result[0]["text"]
    assert result[0]["start"] == 0.0
    assert result[1]["end"] == 6.0


def test_split_punctuation_triggers_flush():
    words = [
        make_word("一二三四五六七八九十一二三四五。", 0, 3),
        make_word("六七八九十一二三四五六七八九十一二三四", 3, 6),
    ]
    result = core._split_words_to_subtitles(words)
    # 句号触发第一条 flush（15字 3s 达标），第二条独立
    assert len(result) == 2
    assert "。" in result[0]["text"]


def test_split_empty_input():
    assert core._split_words_to_subtitles([]) == []


def test_split_single_short_word_kept():
    words = [make_word("你好", 0, 1)]
    result = core._split_words_to_subtitles(words)
    assert len(result) == 1
    assert result[0]["text"] == "你好"
