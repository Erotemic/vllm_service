from __future__ import annotations

import secrets
from pathlib import Path


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k] = v
    return data


def ensure_secret(env: dict[str, str], key: str, length: int = 32) -> str:
    value = env.get(key, "").strip()
    return value if value else secrets.token_urlsafe(length)


def write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{k}={v}\n" for k, v in values.items())
    print(f'Write .env to {path}')
    path.write_text(text, encoding="utf-8")
