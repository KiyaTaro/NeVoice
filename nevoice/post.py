# -*- coding: utf-8 -*-
"""后处理：去 DC、去爆音、RMS 响度归一化 + lookahead peak limiter。"""
import numpy as np
from scipy.signal import lfilter


def remove_dc(y):
    return y - np.mean(y)


def lookahead_limiter(y, sr, ceiling_db=-1.0, lookahead_ms=2.0, release_ms=80.0):
    ceiling = 10 ** (ceiling_db / 20.0)
    la = max(1, int(lookahead_ms / 1000 * sr))
    rel = max(1, int(release_ms / 1000 * sr))
    peak = np.maximum(1e-9, np.abs(y))
    env = np.empty_like(peak)
    cur = 0.0
    for i in range(len(y)):
        ahead = peak[min(i + la, len(y) - 1)]
        cur = max(ahead, cur - cur / rel)
        env[i] = cur
    gain = np.minimum(1.0, ceiling / env)
    gain = lfilter([1.0 / la] * la, [1.0], gain)
    gain = np.clip(gain, 0, 1)
    return y * gain


def rms_normalize(y, target_rms, sr, ceiling_db=-1.0):
    active = np.abs(y) > max(np.max(np.abs(y)) * 0.05, 1e-4)
    if active.sum() < 4:
        return y
    cur_rms = np.sqrt(np.mean(y[active] ** 2)) + 1e-9
    g = target_rms / cur_rms
    y = y * g
    return lookahead_limiter(y, sr, ceiling_db=ceiling)


def master(y, sr, target_rms, depop=True):
    y = remove_dc(y)
    if depop:
        y = lookahead_limiter(y, sr)
    y = rms_normalize(y, target_rms, sr)
    m = np.max(np.abs(y)) + 1e-9
    if m > 0.99:
        y = y / m * 0.99
    return y
