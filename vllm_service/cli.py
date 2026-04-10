from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark import run_benchmark
from .config import (
    CONFIG_FILE,
    GENERATED_DIR,
    MODELS_FILE,
    PLAN_FILE,
    initial_config,
    normalized_state,
    load_yaml,
    save_yaml,
)
from .docker_utils import compose_down, compose_up
from .env_utils import parse_env_file
from .hardware import simulate_inventory
from .renderer import render_from_lock
from .resolver import resolve
from .validator import validate_resolved


def root_dir() -> Path:
    return Path.cwd()


def config_path() -> Path:
    return root_dir() / CONFIG_FILE


def models_path() -> Path:
    return root_dir() / MODELS_FILE


def generated_dir() -> Path:
    return root_dir() / GENERATED_DIR


def plan_path() -> Path:
    return root_dir() / PLAN_FILE


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        raise SystemExit("No config.yaml found. Run `python manage.py init` first.")
    return load_yaml(path)


def runtime_dir_for_config(cfg: dict[str, Any]) -> Path:
    state = normalized_state(root_dir(), cfg.get("state", {}))
    return Path(state["runtime"])


def runtime_env_path(cfg: dict[str, Any]) -> Path:
    return generated_dir() / ".env"


def runtime_litellm_config_path(cfg: dict[str, Any]) -> Path:
    return runtime_dir_for_config(cfg) / "litellm_config.yaml"


def effective_allow_unsupported(args: argparse.Namespace | None, cfg: dict[str, Any]) -> bool:
    arg_value = bool(getattr(args, "allow_unsupported", False)) if args is not None else False
    policy_value = bool(cfg.get("policy", {}).get("allow_unsupported_render", False))
    return arg_value or policy_value


def effective_inventory(args: argparse.Namespace | None) -> dict[str, Any] | None:
    spec = getattr(args, "simulate_hardware", None) if args is not None else None
    if not spec:
        return None
    return simulate_inventory(spec)


def build_plan(
    cfg: dict[str, Any],
    *,
    profile_name: str | None = None,
    allow_unsupported: bool = False,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve(root_dir(), cfg, inventory=inventory, profile_name=profile_name)
    report = validate_resolved(resolved)
    return {
        "schema_version": 1,
        "allow_unsupported": bool(allow_unsupported),
        "validated": report,
        "deployment": resolved,
    }


def save_plan(plan: dict[str, Any]) -> Path:
    path = plan_path()
    save_yaml(path, plan)
    return path


def ensure_renderable(plan: dict[str, Any]) -> None:
    validated = plan.get("validated", {}) or {}
    if validated.get("errors") and not plan.get("allow_unsupported", False):
        raise SystemExit(
            "Refusing to render because the resolved plan contains validation errors. "
            "Use `--allow-unsupported` to override."
        )


def render_is_stale(cfg: dict[str, Any] | None = None) -> bool:
    cfg = load_config() if cfg is None else cfg
    cfg_path = config_path()
    current_plan = plan_path()
    compose_file = generated_dir() / "docker-compose.yml"
    runtime_env = runtime_env_path(cfg)
    runtime_litellm_cfg = runtime_litellm_config_path(cfg)

    required_outputs = [
        current_plan,
        compose_file,
        runtime_env,
        runtime_litellm_cfg,
    ]
    if any(not p.exists() for p in required_outputs):
        return True

    if cfg_path.exists():
        oldest_generated = min(p.stat().st_mtime for p in required_outputs)
        if cfg_path.stat().st_mtime > oldest_generated:
            return True

    if current_plan.stat().st_mtime > compose_file.stat().st_mtime:
        return True
    if current_plan.stat().st_mtime > runtime_env.stat().st_mtime:
        return True
    if current_plan.stat().st_mtime > runtime_litellm_cfg.stat().st_mtime:
        return True

    return False


def cmd_init(args: argparse.Namespace) -> int:
    cfg_path = config_path()
    if cfg_path.exists() and not args.force:
        raise SystemExit("config.yaml already exists. Use --force to overwrite.")
    save_yaml(cfg_path, initial_config())
    if not models_path().exists():
        save_yaml(models_path(), {"models": {}, "profiles": {}})
    print(f"Wrote {cfg_path}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    cfg = load_config()
    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    save_plan(plan)
    print(json.dumps(plan["deployment"], indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config()
    plan = build_plan(
        cfg,
        profile_name=getattr(args, "profile", None),
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    save_plan(plan)
    print(json.dumps(plan["validated"], indent=2))
    return 0 if plan["validated"]["ok"] else 2


def cmd_lock(args: argparse.Namespace) -> int:
    cfg = load_config()
    plan = build_plan(
        cfg,
        profile_name=getattr(args, "profile", None),
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    if not plan["validated"]["ok"] and not plan["allow_unsupported"]:
        raise SystemExit(
            "Refusing to write plan.yaml because validation failed. "
            "Use --allow-unsupported to override."
        )
    save_plan(plan)
    print(json.dumps(plan, indent=2))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    cfg = load_config()
    plan = build_plan(
        cfg,
        profile_name=getattr(args, "profile", None),
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    ensure_renderable(plan)
    save_plan(plan)
    render_from_lock(root_dir(), plan)
    print(f"Wrote {plan_path()}")
    print(f"Rendered Compose into {generated_dir()}")
    print(f"Rendered mounted runtime files into {runtime_dir_for_config(cfg)}")
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    cfg = load_config()
    if render_is_stale():
        render_args = argparse.Namespace(
            profile=None,
            allow_unsupported=effective_allow_unsupported(args, cfg),
        )
        cmd_render(render_args)
    compose_up(
        cfg["runtime"]["compose_cmd"],
        generated_dir() / "docker-compose.yml",
        generated_dir() / ".env",
        detach=args.detach,
        remove_orphans=True,
    )
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    cfg = load_config()
    compose_down(cfg["runtime"]["compose_cmd"], generated_dir() / "docker-compose.yml", runtime_env_path(cfg))
    return 0


def cmd_switch(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg["active_profile"] = args.profile
    save_yaml(config_path(), cfg)
    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    ensure_renderable(plan)
    save_plan(plan)
    render_from_lock(root_dir(), plan)
    if args.apply:
        compose_down(
            cfg["runtime"]["compose_cmd"],
            generated_dir() / "docker-compose.yml",
            generated_dir() / ".env",
        )
        compose_up(
            cfg["runtime"]["compose_cmd"],
            generated_dir() / "docker-compose.yml",
            generated_dir() / ".env",
            detach=False,
            remove_orphans=True,
        )
    print(f"Switched active_profile to {args.profile}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    from .config import merged_catalogs

    cfg = load_config() if config_path().exists() else initial_config()
    cats = merged_catalogs(root_dir(), cfg)
    for name, model in cats.get("models", {}).items():
        print(f"{name}: {model['hf_model_id']}")
    return 0


def cmd_list_profiles(args: argparse.Namespace) -> int:
    from .config import merged_catalogs

    cfg = load_config() if config_path().exists() else initial_config()
    cats = merged_catalogs(root_dir(), cfg)
    profiles = {**cats.get("profiles", {}), **cfg.get("profiles", {})}
    for name, profile in profiles.items():
        print(f"{name}: {profile.get('description', '')}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    target = root_dir() / (args.file or PLAN_FILE)
    if not target.exists():
        raise SystemExit(f"Missing file: {target}")
    print(json.dumps(load_yaml(target), indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    prompts = json.loads((root_dir() / "benchmark_prompts.json").read_text(encoding="utf-8"))
    cfg = load_config()
    env = parse_env_file(runtime_env_path(cfg))
    base_url = args.base_url or f"http://127.0.0.1:{cfg['ports']['litellm']}/v1"
    api_key = args.api_key or env.get("LITELLM_MASTER_KEY", "")
    data = run_benchmark(base_url, api_key, args.model, prompts)
    print(json.dumps(data, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Primary workflow: init -> edit config.yaml -> render -> up. "
            "Advanced commands like resolve/validate/lock still exist for inspection."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("resolve")
    s.add_argument("--profile", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM",
                   help="Simulate N GPUs with M GiB each (e.g. 4x96). Skips nvidia-smi detection.")
    s.set_defaults(func=cmd_resolve)

    s = sub.add_parser("validate")
    s.add_argument("--profile", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM",
                   help="Simulate N GPUs with M GiB each (e.g. 4x96). Skips nvidia-smi detection.")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("lock")
    s.add_argument("--profile", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM",
                   help="Simulate N GPUs with M GiB each (e.g. 4x96). Skips nvidia-smi detection.")
    s.set_defaults(func=cmd_lock)

    s = sub.add_parser("render")
    s.add_argument("--profile", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM",
                   help="Simulate N GPUs with M GiB each (e.g. 4x96). Skips nvidia-smi detection.")
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("up")
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("-d", "--detach", action="store_true", help="Run in background instead of attaching to logs")
    s.set_defaults(func=cmd_up)

    s = sub.add_parser("down")
    s.set_defaults(func=cmd_down)

    s = sub.add_parser("switch")
    s.add_argument("profile")
    s.add_argument("--apply", action="store_true")
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM",
                   help="Simulate N GPUs with M GiB each (e.g. 4x96). Skips nvidia-smi detection.")
    s.set_defaults(func=cmd_switch)

    s = sub.add_parser("list-models")
    s.set_defaults(func=cmd_list_models)

    s = sub.add_parser("list-profiles")
    s.set_defaults(func=cmd_list_profiles)

    s = sub.add_parser("explain")
    s.add_argument("--file", default=None)
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("benchmark")
    s.add_argument("--model", required=True)
    s.add_argument("--base-url", default=None)
    s.add_argument("--api-key", default=None)
    s.set_defaults(func=cmd_benchmark)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    main()
