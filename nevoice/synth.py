# -*- coding: utf-8 -*-
"""谐波-噪声声码器（DDSP 思路，numpy / CPU）+ 日语音节声学目标。"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .phonetics import VOWELS, CONS


# ---------------- 共振峰曲线 ----------------
_FORMANT_BW = (80.0, 110.0, 150.0)   # F1/F2/F3 带宽（收窄让峰更突出）
_FORMANT_A = (3.2, 3.0, 1.8)        # 对数增益强度（大幅提升，主导发音）

def formant_curve(vowel, freqs):
    """目标元音在频域的对数增益（高斯谐振峰叠加），强力主导元音色彩。"""
    if vowel not in VOWELS:
        return np.zeros_like(freqs)
    v = VOWELS[vowel]
    centers = (v["f1"], v["f2"], v["f3"])
    g = np.zeros_like(freqs)
    for F, B, A in zip(centers, _FORMANT_BW, _FORMANT_A):
        g += A * np.exp(-((freqs - F) ** 2) / (2.0 * B ** 2))
    return g


def timbre_tilt(timbre, sr, nb):
    """提取参考音色的温和倾斜 + 小幅粗质感；限制高频衰减，避免压垮共振峰。"""
    freqs = np.linspace(1.0, sr / 2, nb)
    lf = np.log(freqs)
    # 拟合 log 幅度 ~ log 频率 的整体倾斜
    M = np.vstack([lf, np.ones(nb)]).T
    slope, intercept = np.linalg.lstsq(M, timbre, rcond=None)[0]
    # 去趋势后的粗质感（物体腔体共振），压缩到小幅
    resid = timbre - (slope * lf + intercept)
    coarse = gaussian_filter1d(resid, sigma=350.0 / (sr / 2 / nb))
    coarse = np.clip(coarse, -2.0, 2.0)
    # 只保留 30% 倾斜并限制范围，让高频 F2 能立起来
    tilt = 0.3 * slope * lf
    tilt = tilt - tilt[0]
    tilt = np.clip(tilt, -6.0, 6.0)
    return tilt + coarse


# ---------------- 单 mora 声学目标 ----------------
def mora_acoustics(mora, profile, dur=0.44, semitones=0, glide=0.28):
    """
    生成帧级: f0(Hz,0=清), 目标线性幅度谱 n_frames x n_bins,
    浊音权重 vw, 噪声权重 nw, 帧时间。
    """
    sr = profile["sr"]; fp = profile["frame_period"] / 1000.0
    n_frames = max(8, int(round(dur / fp)))
    nb = profile["timbre"].shape[0]
    freqs = profile["freqs"]

    # 参考音色：仅保留频谱倾斜 + 粗质感（已剥离其自身元音共振峰）
    base = timbre_tilt(profile["timbre"], sr, nb)

    onset = mora["onset"]; vowel = mora["vowel"]; is_glide = mora["glide"]
    c_info = CONS.get(onset)
    # 辅音/滑音/元音 帧边界
    if c_info is None:
        c_end = 0
    else:
        c_end = int(round(n_frames * c_info["rel"]))
    g_end = c_end + (int(round(n_frames * 0.12)) if is_glide else 0)

    # 目标 F0（参考基频 + 移调）
    base_f0 = profile["f0_med"] * (2.0 ** (semitones / 12.0))
    f0 = np.zeros(n_frames)
    vw = np.zeros(n_frames)   # 浊音权重
    nw = np.ones(n_frames)    # 噪声权重
    for i in range(n_frames):
        frac = i / max(n_frames - 1, 1)
        # 稳音为主，glide 控制轻微自然下滑
        bend = 1.0
        if i >= g_end:
            bend = 2.0 ** (-(glide * 0.04) * frac * 12.0 / 12.0)
        voiced = True
        if c_info is not None and i < c_end:
            voiced = bool(c_info["voiced"])
        if voiced:
            f0[i] = base_f0 * bend
            vw[i] = 1.0
            nw[i] = 0.15
    # 辅音->元音过渡带混合
    if c_end > 0:
        trans = max(1, int(round(n_frames * 0.08)))
        for i in range(max(0, c_end - trans), c_end):
            k = (i - (c_end - trans)) / trans
            vw[i] = vw[i] * (1 - k) + (1.0 * k)
            nw[i] = nw[i] * (1 - k) + (0.15 * k)

    # 帧级目标谱
    sp_frames = np.zeros((n_frames, nb))
    vcurve = formant_curve(vowel if vowel in VOWELS else "a", freqs)
    for i in range(n_frames):
        logsp = base.copy()
        if i >= g_end:
            # 元音稳定段：强共振峰主导发音
            logsp += vcurve
        elif c_info is not None and i < c_end:
            manner = c_info["manner"]; voiced_c = c_info["voiced"]
            if not voiced_c:
                # 清辅音：噪声激励，按 cutoff 衰减高频
                logsp += np.where(freqs <= c_info["cutoff"], 0.0, -6.0)
            elif manner == "nasal":
                # 鼻音：u 类共振 + 整体压低
                logsp += formant_curve("u", freqs) * 0.7 - 0.8
            else:
                # 浊爆破/闪音：带目标元音共振峰（略弱）
                logsp += vcurve * 0.8
        else:
            # 过渡/滑音段
            logsp += vcurve * 0.7
        sp_frames[i] = np.exp(logsp)
    # 归一化每帧峰值
    sp_frames /= (np.max(sp_frames, axis=1, keepdims=True) + 1e-9)
    t_frame = np.arange(n_frames) * fp
    return dict(f0=f0, sp=sp_frames, vw=vw, nw=nw, t=t_frame,
                c_end=c_end, g_end=g_end)


# ---------------- numpy 谐波-噪声渲染 ----------------
class HarmonicNoiseVocoder:
    def __init__(self, sr, n_harm=100, device="cpu"):
        self.sr = sr
        self.n_harm = n_harm
        self.dev = device

    def _sample_mag(self, sp_frames, t_frame, f, t):
        """在 (时间 t, 频率 f) 双线性采样目标幅度谱。"""
        n_frames, nb = sp_frames.shape
        # 时间索引
        ti = np.clip(np.interp(t, t_frame, np.arange(n_frames)), 0, n_frames - 1)
        t0 = np.floor(ti).astype(int); t1 = np.minimum(t0 + 1, n_frames - 1)
        wt = ti - t0
        # 频率索引
        freqs = np.linspace(0, self.sr / 2, nb)
        fi = np.clip(np.interp(f, freqs, np.arange(nb)), 0, nb - 1)
        f0i = np.floor(fi).astype(int); f1i = np.minimum(f0i + 1, nb - 1)
        wf = fi - f0i
        # 逐点取四个角
        A00 = sp_frames[t0, f0i]; A01 = sp_frames[t0, f1i]
        A10 = sp_frames[t1, f0i]; A11 = sp_frames[t1, f1i]
        A = (A00 * (1 - wf) + A01 * wf) * (1 - wt) \
          + (A10 * (1 - wf) + A11 * wf) * wt
        return A

    def render(self, ac, noise_seed=0):
        sr = self.sr
        dur = ac["t"][-1] + (ac["t"][1] - ac["t"][0])
        N = int(round(dur * sr))
        t = np.arange(N) / sr
        # 样本级 F0 与权重
        f0_s = np.interp(t, ac["t"], ac["f0"], left=0, right=0)
        vw_s = np.interp(t, ac["t"], ac["vw"], left=0, right=0)
        nw_s = np.interp(t, ac["t"], ac["nw"], left=1, right=1)

        rng = np.random.default_rng(noise_seed)
        y = np.zeros(N)

        # ---- 谐波部分 ----
        if np.any(f0_s > 0):
            # 瞬时相位：cumsum 2π f0/sr
            inst_phase = 2 * np.pi * np.cumsum(f0_s) / sr
            harm_idx = np.arange(1, self.n_harm + 1)
            # 谐波频率矩阵 N x H
            Hmat = f0_s[:, None] * harm_idx[None, :]
            valid = Hmat < sr / 2
            # 幅度采样
            flat_f = Hmat.ravel()
            flat_t = np.repeat(t[:, None], self.n_harm, axis=1).ravel()
            Aflat = self._sample_mag(ac["sp"], ac["t"], flat_f, flat_t)
            A = Aflat.reshape(N, self.n_harm)
            A = np.where(valid, A, 0.0)
            phases = inst_phase[:, None] * harm_idx[None, :]
            harmonic = np.sum(A * np.sin(phases), axis=1)
            y += harmonic * vw_s

        # ---- 噪声部分（时变滤波，分帧 overlap-add 简化为谱整形）----
        noise = rng.standard_normal(N)
        # 用 STFT 按目标谱整形
        nfft = 512; hop = 128
        win = np.hanning(nfft)
        nout = np.zeros(N)
        wsum = np.zeros(N)
        for pos in range(0, N - nfft, hop):
            frame_noise = noise[pos:pos + nfft] * win
            spec = np.fft.rfft(frame_noise)
            # 该帧目标谱（取中点时间）
            mid_t = (pos + nfft / 2) / sr
            fi = np.clip(np.interp(mid_t, ac["t"], np.arange(len(ac["t"]))),
                         0, len(ac["t"]) - 1)
            i0 = int(np.floor(fi)); i1 = min(i0 + 1, len(ac["t"]) - 1)
            w = fi - i0
            target = ac["sp"][i0] * (1 - w) + ac["sp"][i1] * w
            shape = target / (np.max(target) + 1e-9)
            # 重采样到 nfft 频点
            shape_n = np.interp(np.linspace(0, sr / 2, nfft // 2 + 1),
                                np.linspace(0, sr / 2, len(shape)), shape)
            spec = spec * shape_n
            out_f = np.fft.irfft(spec, nfft) * win
            nout[pos:pos + nfft] += out_f
            wsum[pos:pos + nfft] += win ** 2
        wsum[wsum == 0] = 1
        nout /= wsum
        y += nout * nw_s

        # 软饱和（暖声）+ 起音/收尾包络
        y = np.tanh(y * 1.1) / np.tanh(1.1)
        attack = int(0.005 * sr); release = int(0.02 * sr)
        env = np.ones(N)
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        y *= env
        return y.astype(np.float64)


_VOC = {}
def synth_mora(mora, profile, dur=0.44, semitones=0, glide=0.28, seed=0):
    sr = profile["sr"]
    ac = mora_acoustics(mora, profile, dur, semitones, glide)
    key = (sr, )
    if key not in _VOC:
        _VOC[key] = HarmonicNoiseVocoder(sr)
    y = _VOC[key].render(ac, noise_seed=seed)
    return y, ac
