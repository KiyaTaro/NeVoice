# -*- coding: utf-8 -*-
"""日语五十音语音学定义：mora 表、元音共振峰、辅音发音方法。"""

# ---- 元音目标共振峰 (F1/F2/F3, Hz)，中性成人参考 ----
VOWELS = {
    "a": dict(f1=800, f2=1200, f3=2500, len=1.00),
    "i": dict(f1=300, f2=2300, f3=3000, len=0.92),
    "u": dict(f1=350, f2=1200, f3=2400, len=0.95),
    "e": dict(f1=480, f2=1800, f3=2700, len=0.98),
    "o": dict(f1=450, f2=900,  f3=2400, len=1.00),
}

# ---- 辅音：发音方法 / 是否带声 / 相对时长 / 噪声截止(Hz) / 强度 ----
# manner: stop 爆破, fric 摩擦, affric 塞擦, nasal 鼻音, liquid 闪音, glide 近音
CONS = {
    "k":  dict(voiced=False, manner="stop",   rel=0.34, cutoff=3500, amp=0.80),
    "t":  dict(voiced=False, manner="stop",   rel=0.30, cutoff=4200, amp=0.85),
    "p":  dict(voiced=False, manner="stop",   rel=0.30, cutoff=3000, amp=0.85),
    "g":  dict(voiced=True,  manner="stop",   rel=0.34, cutoff=3000, amp=0.70),
    "d":  dict(voiced=True,  manner="stop",   rel=0.30, cutoff=3600, amp=0.75),
    "b":  dict(voiced=True,  manner="stop",   rel=0.30, cutoff=2800, amp=0.75),
    "s":  dict(voiced=False, manner="fric",   rel=0.42, cutoff=5500, amp=0.70),
    "sh": dict(voiced=False, manner="fric",   rel=0.42, cutoff=4000, amp=0.70),
    "h":  dict(voiced=False, manner="fric",   rel=0.36, cutoff=6000, amp=0.55),
    "f":  dict(voiced=False, manner="fric",   rel=0.36, cutoff=2500, amp=0.55),
    "z":  dict(voiced=True,  manner="fric",   rel=0.40, cutoff=4500, amp=0.60),
    "ch": dict(voiced=False, manner="affric", rel=0.40, cutoff=3800, amp=0.75),
    "ts": dict(voiced=False, manner="affric", rel=0.40, cutoff=4800, amp=0.75),
    "j":  dict(voiced=True,  manner="affric", rel=0.40, cutoff=3500, amp=0.65),
    "m":  dict(voiced=True,  manner="nasal",  rel=0.36, cutoff=2000, amp=0.80),
    "n":  dict(voiced=True,  manner="nasal",  rel=0.36, cutoff=2200, amp=0.80),
    "N":  dict(voiced=True,  manner="nasal",  rel=0.60, cutoff=1800, amp=0.80),
    "r":  dict(voiced=True,  manner="liquid", rel=0.24, cutoff=2400, amp=0.70),
    "y":  dict(voiced=True,  manner="glide",  rel=0.22, cutoff=2600, amp=0.70),
    "w":  dict(voiced=True,  manner="glide",  rel=0.22, cutoff=1800, amp=0.70),
}

# ---- 五十音 mora 程序化生成 ----
# (kana, romaji, onset 辅音或 None, vowel, glide 拗音)
def _rows():
    out = []
    # あ行纯元音
    for k, r, v in [("あ","a","a"),("い","i","i"),("う","u","u"),("え","e","e"),("お","o","o")]:
        out.append((k, r, None, v, False))
    table = [
        ("か","き","く","け","こ", "k","k","k","k","k", "ka","ki","ku","ke","ko"),
        ("さ","し","す","せ","そ", "s","sh","s","s","s", "sa","shi","su","se","so"),
        ("た","ち","つ","て","と", "t","ch","ts","t","t", "ta","chi","tsu","te","to"),
        ("な","に","ぬ","ね","の", "n","n","n","n","n", "na","ni","nu","ne","no"),
        ("は","ひ","ふ","へ","ほ", "h","h","f","h","h", "ha","hi","fu","he","ho"),
        ("ま","み","む","め","も", "m","m","m","m","m", "ma","mi","mu","me","mo"),
        ("や","ゆ","よ", "y","y","y", "ya","yu","yo"),
        ("ら","り","る","れ","ろ", "r","r","r","r","r", "ra","ri","ru","re","ro"),
        ("わ","を", "w","w", "wa","wo"),
    ]
    vowels5 = ["a","i","u","e","o"]
    for row in table:
        if len(row) == 15:  # 5 列: 5假名+5辅音+5罗马字
            ks = row[0:5]; cs = row[5:10]; romas = row[10:15]
            for i in range(5):
                out.append((ks[i], romas[i], cs[i], vowels5[i], False))
        elif len(row) == 9:  # や行 3 列
            ks = row[0:3]; cs = row[3:6]; romas = row[6:9]
            for i, v in enumerate(["a","u","o"]):
                out.append((ks[i], romas[i], cs[i], v, False))
        elif len(row) == 6:  # わ行 2 列
            ks = row[0:2]; cs = row[2:4]; romas = row[4:6]
            for i, v in enumerate(["a","o"]):
                out.append((ks[i], romas[i], cs[i], v, False))
    # ん
    out.append(("ん","n","N","N",False))
    return out

def _daku():
    out = []
    specs = [
        ("が","ぎ","ぐ","げ","ご", "g","g","g","g","g", "ga","gi","gu","ge","go"),
        ("ざ","じ","ず","ぜ","ぞ", "z","j","z","z","z", "za","ji","zu","ze","zo"),
        ("ば","び","ぶ","べ","ぼ", "b","b","b","b","b", "ba","bi","bu","be","bo"),
        ("ぱ","ぴ","ぷ","ぺ","ぽ", "p","p","p","p","p", "pa","pi","pu","pe","po"),
    ]
    vowels5 = ["a","i","u","e","o"]
    for row in specs:
        ks = row[0:5]; cs = row[5:10]; romas = row[10:]
        for i in range(5):
            out.append((ks[i], romas[i], cs[i], vowels5[i], False))
    # だ行只列 da de do（ぢづ与じず同音）
    for k, r, c, v in [("だ","da","d","a"),("で","de","d","e"),("ど","do","d","o")]:
        out.append((k, r, c, v, False))
    return out

# 拗音：在辅音后插入 y 滑音
_YOUON_BASE = [
    ("き","k","きゃ","きゅ","きょ","kya","kyu","kyo"),
    ("し","sh","しゃ","しゅ","しょ","sha","shu","sho"),
    ("ち","ch","ちゃ","ちゅ","ちょ","cha","chu","cho"),
    ("に","n","にゃ","にゅ","にょ","nya","nyu","nyo"),
    ("ひ","h","ひゃ","ひゅ","ひょ","hya","hyu","hyo"),
    ("み","m","みゃ","みゅ","みょ","mya","myu","myo"),
    ("り","r","りゃ","りゅ","りょ","rya","ryu","ryo"),
    ("ぎ","g","ぎゃ","ぎゅ","ぎょ","gya","gyu","gyo"),
    ("じ","j","じゃ","じゅ","じょ","ja","ju","jo"),
    ("び","b","びゃ","びゅ","びょ","bya","byu","byo"),
    ("ぴ","p","ぴゃ","ぴゅ","ぴょ","pya","pyu","pyo"),
]

def _youon():
    out = []
    for spec in _YOUON_BASE:
        c = spec[1]; ks = spec[2:5]; rs = spec[5:]
        for i, v in enumerate(["a","u","o"]):
            out.append((ks[i], rs[i], c, v, True))
    return out

def _build():
    mora = {}
    order = []
    for src in (_rows(), _daku(), _youon()):
        for kana, romaji, onset, vowel, glide in src:
            if romaji in mora:  # 去重
                continue
            mora[romaji] = dict(kana=kana, romaji=romaji, onset=onset,
                                vowel=vowel, glide=glide)
            order.append(romaji)
    return mora, order

MORA, ORDER = _build()

# 五十音图网格（用于 UI 展示，空串占位）
GRID_GOJUON = [
    ["a","ka","sa","ta","na","ha","ma","ya","ra","wa","n"],
    ["i","ki","shi","chi","ni","hi","mi","","ri","",""],
    ["u","ku","su","tsu","nu","fu","mu","yu","ru","",""],
    ["e","ke","se","te","ne","he","me","","re","",""],
    ["o","ko","so","to","no","ho","mo","yo","ro","wo",""],
]

if __name__ == "__main__":
    print("mora 总数:", len(MORA))
    print("样例:", MORA["ka"], MORA["sha"], MORA["n"])
