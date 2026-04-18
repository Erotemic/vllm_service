from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import requests

from .catalog import PROFILE_NAME_ALIASES, profile_summary
from .benchmark import run_benchmark
from .config import (
    CONFIG_FILE,
    GENERATED_DIR,
    KUBEAI_GENERATED_DIR,
    MODELS_FILE,
    PLAN_FILE,
    initial_config,
    load_yaml,
    normalized_catalogs,
    save_yaml,
)
from .docker_utils import compose_down, compose_up
from .env_utils import parse_env_file
from .exporters import export_benchmark_bundle
from .hardware import simulate_inventory
from .kubeai_ops import deploy_rendered_artifacts, print_status as kubeai_print_status
from .profile_runtime import default_base_url
from .renderer import render_from_lock
from .resolver import resolve
from .validator import validate_resolved
from .verification import verify_profile


def root_dir() -> Path:
    return Path.cwd()


def config_path() -> Path:
    return root_dir() / CONFIG_FILE


def models_path() -> Path:
    return root_dir() / MODELS_FILE


def generated_dir() -> Path:
    return root_dir() / GENERATED_DIR


def kubeai_generated_dir() -> Path:
    return root_dir() / KUBEAI_GENERATED_DIR


def plan_path() -> Path:
    return root_dir() / PLAN_FILE


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        raise SystemExit("No config.yaml found. Run `python manage.py init` first.")
    return load_yaml(path)


def runtime_dir_for_config(cfg: dict[str, Any]) -> Path:
    state = cfg.get("state", {})
    runtime = state.get("runtime")
    if not runtime:
        return root_dir() / "state" / "runtime"
    p = Path(runtime)
    if p.is_absolute():
        return p
    return root_dir() / p


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


def backend_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("backend", "compose")).lower()


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
    backend = backend_name(cfg)

    if backend == "kubeai":
        required_outputs = [
            current_plan,
            kubeai_generated_dir() / "namespace.yaml",
            kubeai_generated_dir() / "kubeai-values.yaml",
            kubeai_generated_dir() / "models.yaml",
        ]
    else:
        required_outputs = [
            current_plan,
            generated_dir() / "docker-compose.yml",
            runtime_env_path(cfg),
            runtime_litellm_config_path(cfg),
        ]

    if any(not p.exists() for p in required_outputs):
        return True

    if cfg_path.exists():
        oldest_generated = min(p.stat().st_mtime for p in required_outputs)
        if cfg_path.stat().st_mtime > oldest_generated:
            return True

    if any(current_plan.stat().st_mtime > p.stat().st_mtime for p in required_outputs if p != current_plan):
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
            "Refusing to write plan.yaml because validation failed. Use --allow-unsupported to override."
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
    if backend_name(cfg) == "kubeai":
        print(f"Rendered KubeAI artifacts into {kubeai_generated_dir()}")
    else:
        print(f"Rendered Compose into {generated_dir()}")
        print(f"Rendered mounted runtime files into {runtime_dir_for_config(cfg)}")
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    cfg = load_config()
    if backend_name(cfg) != "compose":
        raise SystemExit("`up` only supports the compose backend. Use `deploy` for kubeai.")
    if render_is_stale(cfg):
        render_args = argparse.Namespace(profile=None, allow_unsupported=effective_allow_unsupported(args, cfg), simulate_hardware=None)
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
    if backend_name(cfg) != "compose":
        raise SystemExit("`down` only supports the compose backend.")
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
        if backend_name(cfg) == "compose":
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
        else:
            deploy_rendered_artifacts(root_dir(), plan["deployment"])
    print(f"Switched active_profile to {args.profile}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    cfg = load_config() if config_path().exists() else initial_config()
    cats = normalized_catalogs(root_dir(), cfg)
    for name, model in cats.get("models", {}).items():
        ref = model.get("hf_model_id") or model.get("url", "")
        print(f"{name}: {ref}")
    return 0


def cmd_list_profiles(args: argparse.Namespace) -> int:
    cfg = load_config() if config_path().exists() else initial_config()
    cats = normalized_catalogs(root_dir(), cfg)
    profiles = cats.get("profiles", {})
    hidden_legacy = set(PROFILE_NAME_ALIASES)
    for name, profile in profiles.items():
        if name in hidden_legacy:
            continue
        if profile.get("kind") == "invalid-profile":
            continue
        summary = profile_summary(profile)
        print(
            f"{name}: public={summary['public_name']} logical={summary['logical_model_name']} "
            f"protocol={summary['protocol_mode']} base_model={summary['base_model']}"
        )
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    target = root_dir() / (args.file or PLAN_FILE)
    if not target.exists():
        raise SystemExit(f"Missing file: {target}")
    print(json.dumps(load_yaml(target), indent=2))
    return 0


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    cfg = load_config()
    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    ensure_renderable(plan)
    result = export_benchmark_bundle(
        root_dir(),
        plan["deployment"],
        base_url=args.base_url,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Wrote {result['bundle_path']}")
    print(f"Wrote {result['model_deployments_path']}")
    return 0


def cmd_export_benchmark_bundle(args: argparse.Namespace) -> int:
    return _cmd_export_bundle(args)


def cmd_export_helm_bundle(args: argparse.Namespace) -> int:
    return _cmd_export_bundle(args)


def cmd_verify_profile(args: argparse.Namespace) -> int:
    cfg = load_config()
    plan = build_plan(
        cfg,
        profile_name=args.profile,
        allow_unsupported=effective_allow_unsupported(args, cfg),
        inventory=effective_inventory(args),
    )
    result = verify_profile(root_dir(), plan["deployment"])
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def cmd_benchmark(args: argparse.Namespace) -> int:
    prompts = json.loads((root_dir() / "benchmark_prompts.json").read_text(encoding="utf-8"))
    cfg = load_config()
    env = parse_env_file(runtime_env_path(cfg))
    base_url = args.base_url or f"http://127.0.0.1:{cfg['ports']['litellm']}/v1"
    api_key = args.api_key or env.get("LITELLM_MASTER_KEY", "")
    data = run_benchmark(base_url, api_key, args.model, prompts)
    print(json.dumps(data, indent=2))
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    cfg = load_config()
    if render_is_stale(cfg):
        render_args = argparse.Namespace(profile=None, allow_unsupported=effective_allow_unsupported(args, cfg), simulate_hardware=None)
        cmd_render(render_args)
    if backend_name(cfg) == "kubeai":
        plan = load_yaml(plan_path())
        deploy_rendered_artifacts(root_dir(), plan["deployment"])
        return 0
    compose_up(
        cfg["runtime"]["compose_cmd"],
        generated_dir() / "docker-compose.yml",
        generated_dir() / ".env",
        detach=args.detach,
        remove_orphans=True,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config()
    if backend_name(cfg) == "kubeai":
        namespace = cfg.get("cluster", {}).get("namespace", "kubeai")
        kubeai_print_status(namespace)
        return 0
    proc = subprocess.run(cfg["runtime"]["compose_cmd"].split() + ["-f", str(generated_dir() / "docker-compose.yml"), "ps"])
    return int(proc.returncode)


def _infer_default_base_url(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    deployment = {"backend": backend_name(cfg), "cluster": cfg.get("cluster", {}), "ports": cfg.get("ports", {})}
    return default_base_url(deployment, explicit=args.base_url)


def cmd_smoke_test(args: argparse.Namespace) -> int:
    cfg = load_config()
    env = parse_env_file(runtime_env_path(cfg)) if backend_name(cfg) == "compose" else {}
    base_url = _infer_default_base_url(cfg, args)
    headers = {"Content-Type": "application/json"}
    api_key = args.api_key or env.get("LITELLM_MASTER_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    models_resp = requests.get(f"{base_url}/models", headers=headers, timeout=30)
    models_resp.raise_for_status()
    models = models_resp.json().get("data", [])
    print(json.dumps(models_resp.json(), indent=2))
    if args.skip_chat:
        return 0
    if not models:
        raise SystemExit("No models returned from /models")
    model_name = args.model or models[0]["id"]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Primary workflow: init -> edit config.yaml -> render -> deploy. Compose and KubeAI backends are both supported."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    for name, func in [("resolve", cmd_resolve), ("validate", cmd_validate), ("lock", cmd_lock), ("render", cmd_render)]:
        s = sub.add_parser(name)
        s.add_argument("--profile", default=None)
        s.add_argument("--allow-unsupported", action="store_true")
        s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
        s.set_defaults(func=func)

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
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.set_defaults(func=cmd_switch)

    s = sub.add_parser("list-models")
    s.set_defaults(func=cmd_list_models)

    s = sub.add_parser("list-profiles")
    s.set_defaults(func=cmd_list_profiles)

    s = sub.add_parser("explain")
    s.add_argument("--file", default=None)
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("export-benchmark-bundle")
    s.add_argument("profile")
    s.add_argument("--base-url", default=None)
    s.add_argument("--output-dir", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.set_defaults(func=cmd_export_benchmark_bundle)

    s = sub.add_parser("export-helm-bundle")
    s.add_argument("profile")
    s.add_argument("--base-url", default=None)
    s.add_argument("--output-dir", default=None)
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.set_defaults(func=cmd_export_helm_bundle)

    s = sub.add_parser("verify-profile")
    s.add_argument("profile")
    s.add_argument("--allow-unsupported", action="store_true")
    s.add_argument("--simulate-hardware", default=None, metavar="NxM", help="Simulate N GPUs with M GiB each (e.g. 4x96, 2x80).")
    s.set_defaults(func=cmd_verify_profile)

    s = sub.add_parser("benchmark")
    s.add_argument("--model", required=True)
    s.add_argument("--base-url", default=None)
    s.add_argument("--api-key", default=None)
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("deploy")
    s.add_argument("-d", "--detach", action="store_true")
    s.add_argument("--allow-unsupported", action="store_true")
    s.set_defaults(func=cmd_deploy)

    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("smoke-test")
    s.add_argument("--base-url", default=None)
    s.add_argument("--api-key", default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--prompt", default="Say hello in one sentence.")
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--skip-chat", action="store_true")
    s.set_defaults(func=cmd_smoke_test)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    main()
