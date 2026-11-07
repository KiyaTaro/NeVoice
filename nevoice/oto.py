# -*- coding: utf-8 -*-
"""自动生成 CV oto.ini。
经典 UTAU 行格式: wav=alias,Offset,Consonant,Cutoff,Preutter,Overlap
- Offset/Overlap/Preutter/Consonant 为相对文件起点的绝对位置(ms)
- Cutoff 为「尾部留白长度」(正, ms)，右边界 = 总时长 - Cutoff
正 Cutoff 同时兼容传统 UTAU 与 OpenUtau。每个 wav 写假名+罗马字双别名。"""
import numpy as np


def _voice_end_ms(y, sr):
    """基于能量找有效结束点(ms)。"""
    total = len(y) / sr * 1000.0
    win = int(0.01 * sr)
    if win < 2:
        return total - 15.0
    n = len(y) // win
    rms = np.array([np.sqrt(np.mean(y[i * win:(i + 1) * win] ** 2))
                    for i in range(n)])
    thr = max(rms.max() * 0.05, 1e-4)
    active = np.where(rms > thr)[0]
    if len(active) == 0:
        return total - 15.0
    return min((active[-1] + 1) * win / sr * 1000.0, total - 5.0)


def _geometry(y, sr, ac):
    """返回 offset, consonant, cutoff, preutter, overlap（整数 ms）。"""
    total = len(y) / sr * 1000.0
    c_end = ac["c_end"]
    fp = (ac["t"][1] - ac["t"][0]) * 1000.0
    middle = c_end * fp                       # 辅音->元音过渡位置
    voice_end = _voice_end_ms(y, sr)

    if c_end == 0:                            # 纯元音：参照原站给小值
        offset = 0
        overlap = 8
        preutter = 15
        consonant = 40
    else:
        offset = 0
        overlap = max(4, middle * 0.35)
        preutter = middle
        consonant = middle + 20.0

    cutoff = max(8.0, total - voice_end)      # 尾部留白长度
    right = total - cutoff
    # 几何校验：offset < overlap < preutter < consonant < right
    if consonant >= right:
        consonant = right - 10.0
    if preutter >= consonant:
        preutter = consonant - 8.0
    if overlap >= preutter:
        overlap = preutter - 4.0
    return [int(round(v)) for v in
            (offset, consonant, cutoff, preutter, overlap)]


def build_oto_lines(wav_name, kana, romaji, y, sr, ac):
    o, con, cut, pre, ov = _geometry(y, sr, ac)
    tail = f"{o},{con},{cut},{pre},{ov}"
    aliases = (kana, romaji) if kana != romaji else (romaji,)
    return [f"{wav_name}={a},{tail}" for a in aliases]


def write_oto(lines, path):
    with open(path, "w", encoding="shift_jis", errors="replace") as f:
        f.write("\n".join(lines) + "\n")
