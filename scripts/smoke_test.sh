#!/usr/bin/env bash
set -euo pipefail
python manage.py smoke-test "$@"
