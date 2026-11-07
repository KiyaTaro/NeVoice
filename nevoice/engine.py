# -*- coding: utf-8 -*-
"""引擎编排：参考音 -> 完整日语五十音 UTAU 声库目录 + zip。"""
import os
import zipfile
import numpy as np
import soundfile as sf

from . import analyze as A
from . import synth as S
from . import post as P
from . import oto as O
from .phonetics import MORA, ORDER


def _target_rms(raws):
    vals = []
    for y in raws:
        act = np.abs(y) > max(np.abs(y).max() * 0.05, 1e-4)
        if act.sum() > 4:
            vals.append(np.sqrt(np.mean(y[act] ** 2)))
    return float(np.median(vals)) if vals else 0.2


def build_voicebank(source, out_dir, singer="nevoice", dur=0.44,
                    semitones=0, glide=0.28, depop=True,
                    include_oto=True, target_sr=44100, progress=None):
    os.makedirs(out_dir, exist_ok=True)
    profile = A.analyze(source, target_sr=target_sr)

    # 第一遍：合成原始音，收集 RMS
    rendered = []
    for idx, romaji in enumerate(ORDER):
        mora = MORA[romaji]
        y, ac = S.synth_mora(mora, profile, dur=dur,
                             semitones=semitones, glide=glide, seed=idx)
        rendered.append((romaji, mora, y, ac))
        if progress:
            progress(idx + 1, len(ORDER), romaji)

    tgt = _target_rms([r[2] for r in rendered])

    # 第二遍：后处理 + 写 wav
    oto_lines = []
    for romaji, mora, y, ac in rendered:
        y = P.master(y, target_sr, target_rms=tgt, depop=depop)
        wav_name = f"{romaji}.wav"
        sf.write(os.path.join(out_dir, wav_name),
                 (y * 32767).astype(np.int16), target_sr, subtype="PCM_16")
        if include_oto:
            oto_lines.extend(O.build_oto_lines(
                wav_name, mora["kana"], romaji, y, target_sr, ac))

    if include_oto:
        O.write_oto(oto_lines, os.path.join(out_dir, "oto.ini"))

    # character.txt
    with open(os.path.join(out_dir, "character.txt"), "w",
              encoding="shift_jis", errors="replace") as f:
        f.write(f"{singer}\nnevoice AI engine\n")

    # 打包 zip
    zip_path = out_dir + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.join(singer, os.path.relpath(full, out_dir)))

    return dict(
        zip_path=zip_path, out_dir=out_dir, count=len(rendered),
        f0=round(profile["f0_med"], 1),
        harmonicity=round(profile["harmonicity"], 2),
        target_rms=round(tgt, 4),
    )
