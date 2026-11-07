# -*- coding: utf-8 -*-
"""本地 AI 声库服务器（Python 标准库，零额外 web 框架）。"""
import io
import os
import re
import json
import zipfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

from nevoice import analyze as A, synth as S, post as P, oto as O
from nevoice.phonetics import MORA, ORDER

SR = 44100
MAX_UPLOAD = 24 * 1024 * 1024
WEB = os.path.join(os.path.dirname(__file__), "web")
STATE = {"lock": threading.Lock(), "ready": False, "moras": {},
         "zip": b"", "info": {}, "singer": "nevoice"}
BAKE_LOCK = threading.Lock()


def wav_bytes(y, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, (y * 32767).astype(np.int16), sr,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()


def demo_source(sr=SR):
    dur, f0 = 1.2, 170.0
    t = np.arange(int(sr * dur)) / sr
    x = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 25))
    x *= np.minimum(1, t / 0.03) * np.minimum(1, (dur - t) / 0.08)
    return x / np.abs(x).max() * 0.9


def parse_multipart(body, boundary):
    parts = body.split(b"--" + boundary)
    out = {}
    for p in parts:
        if not p or p in (b"--", b"--\r\n", b"--\n"):
            continue
        if b"\r\n\r\n" not in p:
            continue
        head, content = p.strip(b"\r\n").split(b"\r\n\r\n", 1)
        name = filename = None
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if "content-disposition" in line.lower():
                m = re.search(r'name="([^"]*)"', line)
                name = m.group(1) if m else None
                f = re.search(r'filename="([^"]*)"', line)
                filename = f.group(1) if f else None
        if name:
            out[name] = dict(filename=filename, data=content)
    return out


def run_bake(form, tmp_path=None):
    glide = float(form.get("glide", 0.28))
    dur = float(form.get("duration", 0.44))
    semitones = int(float(form.get("semitones", 0)))
    depop = form.get("depop", "1") != "0"
    if "file" in form and form["file"]["filename"]:
        data = form["file"]["data"]
        path = tmp_path or "/tmp/_nevoice_ref.wav"
        with open(path, "wb") as f:
            f.write(data)
        source = path
    else:
        source = demo_source()

    profile = A.analyze(source, target_sr=SR)
    rendered = []
    for idx, romaji in enumerate(ORDER):
        y, ac = S.synth_mora(MORA[romaji], profile, dur=dur,
                             semitones=semitones, glide=glide, seed=idx)
        rendered.append((romaji, y, ac))
    tgt = _rms([r[1] for r in rendered])
    moras = {}
    oto_lines = []
    for romaji, y, ac in rendered:
        y2 = P.master(y, SR, target_rms=tgt, depop=depop)
        moras[romaji] = y2
        oto_lines.extend(O.build_oto_lines(
            f"{romaji}.wav", MORA[romaji]["kana"], romaji, y2, SR, ac))

    zbuf = io.BytesIO()
    singer = STATE.get("singer", "nevoice")
    with zipfile.ZipFile(zbuf, zipfile.ZIP_DEFLATED) as z:
        if STATE.get("include_oto", True):
            z.writestr(f"{singer}/oto.ini",
                       "\n".join(oto_lines).encode("shift_jis", "replace"))
        for romaji, y2 in moras.items():
            z.writestr(f"{singer}/{romaji}.wav", wav_bytes(y2))
    return dict(moras=moras, zip=zbuf.getvalue(),
                info=dict(f0=round(profile["f0_med"], 1),
                          harmonicity=round(profile["harmonicity"], 2),
                          count=len(moras), rev="NeVoice AI"))


def _rms(ys):
    vals = []
    for y in ys:
        act = np.abs(y) > max(np.abs(y).max() * 0.05, 1e-4)
        if act.sum() > 4:
            vals.append(np.sqrt(np.mean(y[act] ** 2)))
    return float(np.median(vals)) if vals else 0.2


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path = u.path
        if path in ("/", "/index.html"):
            return self._serve("index.html", "text/html; charset=utf-8")
        if path.startswith("/api/mora/"):
            rom = path.split("/")[-1].replace(".wav", "")
            with STATE["lock"]:
                if not STATE["ready"] or rom not in STATE["moras"]:
                    return self._json(404, {"error": "not ready"})
                return self._send(200, wav_bytes(STATE["moras"][rom]),
                                  "audio/wav")
        if path == "/api/bank.zip":
            with STATE["lock"]:
                if not STATE["ready"]:
                    return self._json(404, {"error": "not ready"})
                return self._send(200, STATE["zip"], "application/zip")
        if path == "/api/aiueo.wav":
            with STATE["lock"]:
                if not STATE["ready"]:
                    return self._json(404, {"error": "not ready"})
                y = np.concatenate([STATE["moras"][v]
                                    for v in ("a", "i", "u", "e", "o")])
                return self._send(200, wav_bytes(y), "audio/wav")
        if path == "/api/chart":
            data = [dict(kana=MORA[r]["kana"], romaji=r) for r in ORDER]
            return self._json(200, data)
        return self._json(404, {"error": "unknown"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD:
            return self._json(413, {"error": "文件太大，请控制在 24MB 内"})
        body = self.rfile.read(length)
        ct = self.headers.get("Content-Type", "")
        form = {}
        if "multipart/form-data" in ct:
            b = re.search(r'boundary=([^;]+)', ct).group(1).strip('"').encode()
            raw = parse_multipart(body, b)
            form = {k: (v["data"].decode("utf-8", "replace")
                        if not v["filename"] else v)
                    for k, v in raw.items()}
        else:
            form = urllib.parse.parse_qs(body.decode("utf-8"))
            form = {k: v[0] for k, v in form.items()}

        if u.path in ("/api/bake", "/api/bake/"):
            if not BAKE_LOCK.acquire(blocking=False):
                return self._json(503, {"error": "服务器正忙，请稍后再试"})
            try:
                singer = form.get("singer", "nevoice")
                if not isinstance(singer, str) or not singer.strip():
                    singer = "nevoice"
                STATE["singer"] = re.sub(r'[\\/:*?"<>|]', "_", singer).strip()
                STATE["include_oto"] = form.get("oto", "1") != "0"
                result = run_bake(form)
                with STATE["lock"]:
                    STATE.update(ready=True, **result)
                return self._json(200, result["info"])
            except Exception as e:
                return self._json(500, {"error": str(e)})
            finally:
                BAKE_LOCK.release()
        return self._json(404, {"error": "unknown"})

    def _serve(self, name, ctype):
        p = os.path.join(WEB, name)
        if not os.path.exists(p):
            return self._json(404, {"error": "missing"})
        with open(p, "rb") as f:
            self._send(200, f.read(), ctype)


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"NeVoice 已启动：http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
