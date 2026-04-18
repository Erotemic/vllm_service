from __future__ import annotations

from pathlib import Path

from .backends import render_compose_artifacts, render_kubeai_artifacts


def render_from_lock(root: Path, lock_data: dict) -> None:
    backend = str(lock_data.get("deployment", {}).get("backend", "compose")).lower()
    if backend == "kubeai":
        render_kubeai_artifacts(root, lock_data)
        return
    render_compose_artifacts(root, lock_data)
