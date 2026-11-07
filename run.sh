#!/usr/bin/env bash
# Mac / Linux 一键启动
cd "$(dirname "$0")" || exit 1
python3 -m pip install -r requirements.txt
python3 server.py
