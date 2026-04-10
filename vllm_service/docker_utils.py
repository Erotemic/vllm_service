from __future__ import annotations

import subprocess
from pathlib import Path


class DockerCommandError(RuntimeError):
    pass


def _cmd(compose_cmd: str, compose_file: Path, env_file: Path, *args: str) -> list[str]:
    return compose_cmd.split() + ["-f", str(compose_file), "--env-file", str(env_file), *args]


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise DockerCommandError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def compose_up(
    compose_cmd: str,
    compose_file: Path,
    env_file: Path,
    *,
    detach: bool = False,
    remove_orphans: bool = True,
) -> None:
    args = ["up"]
    if detach:
        args.append("-d")
    if remove_orphans:
        args.append("--remove-orphans")
    run(_cmd(compose_cmd, compose_file, env_file, *args))


def compose_down(compose_cmd: str, compose_file: Path, env_file: Path) -> None:
    run(_cmd(compose_cmd, compose_file, env_file, "down", "--remove-orphans"))
