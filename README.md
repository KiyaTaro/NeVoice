# NeVoice · AI 音源工房

给一段任意声音（人声 / 物体 / 电子 / 自然声，0.4–3 秒），生成日语五十音 **102 音** 的 UTAU / OpenUtau 声库（WAV + 自动 oto.ini），打包 ZIP。

## 启动（推荐网页版）
- Windows：双击 run.bat
- Mac / Linux：运行 bash run.sh
- 首次自动装依赖，浏览器打开 http://127.0.0.1:8765

## 命令行
python build.py 参考音频.wav 歌手名

## 引擎说明
- 分析端：harvest/dio 双算法鲁棒 F0 + cheaptrick 谱包络 + d4c 非周期成分，非人声也稳。
- 合成端：谐波-噪声声码器（连续相位，DDSP 思路）+ 日语共振峰发音模型，消除原站 WORLD 电子味，元音共振峰经校验准确。
- 后处理：去爆音、RMS 响度归一化、无削波；oto 正 cutoff 兼容 UTAU 与 OpenUtau。
