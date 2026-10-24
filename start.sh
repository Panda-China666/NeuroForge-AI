#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
if [ ! -f .venv/.neuroforge-ready ]; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.neuroforge-ready
fi
exec .venv/bin/python run.py
