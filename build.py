# -*- coding: utf-8 -*-
"""命令行一键生成：python build.py 参考音频.wav [歌手名]"""
import sys
import time
from nevoice.engine import build_voicebank


def main():
    if len(sys.argv) < 2:
        print("用法: python build.py <参考音频> [歌手名] [输出目录]")
        sys.exit(1)
    source = sys.argv[1]
    singer = sys.argv[2] if len(sys.argv) > 2 else "nevoice"
    out = sys.argv[3] if len(sys.argv) > 3 else f"./{singer}_voicebank"

    def prog(i, n, r):
        print(f"\r  合成 {i}/{n} ({r})", end="", flush=True)

    t0 = time.time()
    res = build_voicebank(source, out, singer=singer, progress=prog)
    print("\n完成，耗时", round(time.time() - t0, 1), "秒")
    print("样本包:", res["zip_path"], "| 假名", res["count"], "| F0", res["f0"], "Hz")


if __name__ == "__main__":
    main()
