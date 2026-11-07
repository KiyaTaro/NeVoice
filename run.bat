@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在检查依赖（首次运行会自动安装）...
python -m pip install -r requirements.txt
echo.
echo 启动 NeVoice...
python server.py
pause
