# -*- coding: utf-8 -*-
"""声音分析：加载/重采样、鲁棒 F0、谱包络 sp、非周期 ap、音色指纹。"""
import numpy as np
import pyworld as pw
import librosa
from scipy.ndimage import median_filter, gaussian_filter1d


def load_audio(src, target_sr=44100):
    """加载任意格式 -> 单声道 float64，重采样到 target_sr。"""
    if isinstance(src, np.ndarray):
        x = src.astype(np.float64)
        sr = target_sr
    else:
        x, sr = librosa.load(src, sr=None, mono=True)
        x = x.astype(np.float64)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    # 去 DC，归一化到 0.9 峰值
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 0:
        x = x / peak * 0.9
    return x, target_sr


def _smooth_f0(f0, win=5):
    """中值滤波去 F0 毛刺，清音(0)段保留。"""
    voiced = f0 > 0
    out = f0.copy()
    if voiced.sum() > win:
        vals = out[voiced]
        sm = median_filter(vals, size=win, mode="nearest")
        out[voiced] = sm
    return out


def estimate_f0(x, sr, frame_period=5.0, f0_min=50.0, f0_max=1200.0):
    """harvest 与 dio 融合的鲁棒 F0；非人声也尽量给主导基频。"""
    # harvest：更准、对复杂音色鲁棒
    f0_h, t_h = pw.harvest(x, sr, f0_floor=f0_min, f0_ceil=f0_max,
                           frame_period=frame_period)
    # dio：交叉校验
    _f0_d, t_d = pw.dio(x, sr, f0_floor=f0_min, f0_ceil=f0_max,
                        frame_period=frame_period)
    f0_d = pw.stonemask(x, _f0_d, t_d, sr)
    f0 = f0_h.copy()
    # 两者都浊音且接近时取均值；harvest 清音但 dio 浊音时补 dio
    both = (f0_h > 0) & (f0_d > 0)
    close = both & (np.abs(f0_h - f0_d) / np.maximum(f0_h, 1) < 0.03)
    f0[close] = (f0_h[close] + f0_d[close]) / 2.0
    fill = (f0_h <= 0) & (f0_d > 0)
    f0[fill] = f0_d[fill]
    f0 = _smooth_f0(f0)
    return f0, t_h


def spectral_envelope(x, f0, t, sr, fft_size=None, frame_period=5.0):
    """cheaptrick 谱包络 + d4c 非周期成分。"""
    if fft_size is None:
        fft_size = pw.get_cheaptrick_fft_size(sr)
    sp = pw.cheaptrick(x, f0, t, sr, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, sr, fft_size=fft_size)
    return sp, ap, fft_size


def timbre_fingerprint(sp, f0, sr, fft_size):
    """从浊音帧提取粗粒度音色基底（对数域强平滑，剥离具体共振峰）。"""
    freqs = np.linspace(0, sr / 2, sp.shape[1])
    voiced = f0 > 0
    if voiced.sum() < 2:  # 几乎无浊音（纯噪声/物体声），用全部帧
        voiced = np.ones(len(f0), dtype=bool)
    log_sp = np.log(sp[voiced] + 1e-10)
    log_mean = np.median(log_sp, axis=0)
    # 沿频率轴强平滑：sigma 约 120Hz，保留音色倾斜与粗共振
    bins_hz = sr / 2 / sp.shape[1]
    sigma_bins = max(8.0, 120.0 / max(bins_hz, 1))
    timbre = gaussian_filter1d(log_mean, sigma=sigma_bins)
    return timbre, freqs


def analyze(src, target_sr=44100, frame_period=5.0):
    """完整分析，返回 VoiceProfile dict。"""
    x, sr = load_audio(src, target_sr)
    f0, t = estimate_f0(x, sr, frame_period=frame_period)
    sp, ap, fft_size = spectral_envelope(x, f0, t, sr, frame_period=frame_period)
    timbre, freqs = timbre_fingerprint(sp, f0, sr, fft_size)
    voiced_f0 = f0[f0 > 0]
    f0_med = float(np.median(voiced_f0)) if len(voiced_f0) else 180.0
    # 谐波音强度：浊音帧占比（非人声可能偏低）
    harmonicity = float(np.mean(f0 > 0))
    return dict(
        sr=sr, x=x, f0=f0, t=t, sp=sp, ap=ap, fft_size=fft_size,
        timbre=timbre, freqs=freqs, f0_med=f0_med,
        harmonicity=harmonicity, frame_period=frame_period,
    )
