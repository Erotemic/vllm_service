#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python manage.py benchmark --model qwen3.5-9b
