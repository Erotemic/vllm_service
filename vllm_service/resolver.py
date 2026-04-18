from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import merged_catalogs, normalized_cluster, normalized_state
from .hardware import detect_inventory


def _available_gpu_indices(inventory: dict[str, Any], reserve_display_gpu: str | bool | None) -> list[int]:
    gpus = deepcopy(inventory.get("gpus", []))
    if reserve_display_gpu == "auto":
        return [g["index"] for g in gpus if not g.get("display_active")]
    if reserve_display_gpu is True:
        return [g["index"] for g in gpus if not g.get("display_active")]
    return [g["index"] for g in gpus]


def _first_fit(available: list[int], count: int) -> tuple[list[int], str | None]:
    if len(available) < count:
        return available[:], f"need {count} GPUs but only {len(available)} available"
    return available[:count], None


def _runtime_value(service: dict[str, Any], model: dict[str, Any], key: str, default: Any) -> Any:
    runtime = service.get("runtime", {})
    if key in runtime:
        return runtime[key]
    if key in service:
        return service[key]
    return model.get("defaults", {}).get(key, default)


def _resolve_service(service: dict[str, Any], models: dict[str, Any], inventory: dict[str, Any], policy: dict[str, Any], used: set[int]) -> dict[str, Any]:
    model_key = service["model"]
    if model_key not in models:
        raise KeyError(f"Unknown model: {model_key}")
    model = deepcopy(models[model_key])
    placement = deepcopy(service.get("placement", {}))
    topology = deepcopy(service.get("topology", {}))
    available = [i for i in _available_gpu_indices(inventory, policy.get("reserve_display_gpu", "auto")) if i not in used]

    if "tp" in topology and "tensor_parallel_size" not in topology:
        topology["tensor_parallel_size"] = topology["tp"]
    if "dp" in topology and "data_parallel_size" not in topology:
        topology["data_parallel_size"] = topology["dp"]

    strategy = placement.get("strategy", "first_fit")
    placement_error = None
    gpu_indices: list[int] = []
    if strategy in {"exact", "multi_gpu", "single_gpu"}:
        gpu_indices = list(placement.get("gpu_indices", []))
        if not gpu_indices:
            placement_error = f"service {service['service_name']} uses {strategy} placement but no gpu_indices were provided"
    else:
        gpu_count = int(placement.get("gpu_count", topology.get("tensor_parallel_size", model.get("preferred_gpu_count", 1)) or 1))
        gpu_indices, placement_error = _first_fit(available, gpu_count)

    used.update(gpu_indices)
    tp = int(topology.get("tensor_parallel_size", max(1, len(gpu_indices) or placement.get("gpu_count", 1))))
    dp = int(topology.get("data_parallel_size", 1))

    tool_calling = deepcopy(model.get("tool_calling", {}))
    tool_calling = {**tool_calling, **deepcopy(service.get("tool_calling", {}))}
    tool_call_parser = tool_calling.get("parser")
    enable_auto_tool_choice = bool(tool_calling.get("auto", False) and tool_call_parser)

    hf_model_id = service.get("hf_model_id", model.get("hf_model_id", ""))
    model_url = service.get("url") or model.get("url") or (f"hf://{hf_model_id}" if hf_model_id else "")

    return {
        "service_name": service["service_name"],
        "model_ref": model_key,
        "hf_model_id": hf_model_id,
        "model_url": model_url,
        "served_model_name": service.get("served_model_name", model.get("served_model_name", model_key)),
        "modalities": model.get("modalities", ["text"]),
        "features": deepcopy(model.get("features", ["TextGeneration"])),
        "engine": str(service.get("engine", model.get("engine", "VLLM"))).upper(),
        "memory_class_gib": model.get("memory_class_gib"),
        "min_vram_gib_per_replica": model.get("min_vram_gib_per_replica", 0),
        "context_window": model.get("context_window"),
        "notes": model.get("notes", []),
        "gpu_indices": gpu_indices,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "resource_profile": service.get("resource_profile", model.get("resource_profile", "")),
        "min_replicas": int(service.get("min_replicas", model.get("defaults", {}).get("min_replicas", 0))),
        "max_replicas": int(service.get("max_replicas", model.get("defaults", {}).get("max_replicas", 1))),
        "priority_class_name": service.get("priority_class_name", model.get("priority_class_name")),
        "max_model_len": int(_runtime_value(service, model, "max_model_len", 32768)),
        "gpu_memory_utilization": float(_runtime_value(service, model, "gpu_memory_utilization", 0.9)),
        "enable_prefix_caching": bool(_runtime_value(service, model, "enable_prefix_caching", True)),
        "max_num_batched_tokens": int(_runtime_value(service, model, "max_num_batched_tokens", 8192)),
        "max_num_seqs": int(_runtime_value(service, model, "max_num_seqs", 16)),
        "thinking_history_policy": model.get("thinking_history_policy", "keep_final_only"),
        "placement_error": placement_error,
        "enable_auto_tool_choice": enable_auto_tool_choice,
        "tool_call_parser": tool_call_parser,
        "extra_args": deepcopy(service.get("extra_args", model.get("defaults", {}).get("extra_args", []))),
    }


def resolve(root: Path, config: dict[str, Any], inventory: dict[str, Any] | None = None, profile_name: str | None = None) -> dict[str, Any]:
    catalogs = merged_catalogs(root, config)
    inventory = deepcopy(inventory) if inventory is not None else detect_inventory()
    effective_profile_name = profile_name or config.get("active_profile")
    profiles = deepcopy(catalogs.get("profiles", {}))
    profiles = {**profiles, **deepcopy(config.get("profiles", {}))}
    if effective_profile_name not in profiles:
        raise KeyError(f"Unknown profile: {effective_profile_name}")
    profile = deepcopy(profiles[effective_profile_name])

    merged_policy = {**deepcopy(config.get("policy", {})), **deepcopy(profile.get("policy", {}))}
    backend = str(config.get("backend", "compose")).lower()
    used: set[int] = set()
    services = []
    for service in profile.get("services", []):
        service_name = service.get("service_name") or service.get("name")
        if not service_name:
            raise KeyError("Each service must define service_name")
        normalized_service = deepcopy(service)
        normalized_service["service_name"] = service_name
        service_used = set(used) if backend == "compose" else set()
        resolved_service = _resolve_service(normalized_service, catalogs.get("models", {}), inventory, merged_policy, service_used)
        if backend == "compose":
            used.update(resolved_service.get("gpu_indices", []))
        services.append(resolved_service)

    aliases = deepcopy(profile.get("router", {}).get("aliases", {}))
    if not aliases:
        aliases = {svc["served_model_name"]: svc["service_name"] for svc in services}

    return {
        "schema_version": 3,
        "source": {
            "config_file": "config.yaml",
            "active_profile": effective_profile_name,
        },
        "backend": backend,
        "images": deepcopy(config.get("images", {})),
        "ports": deepcopy(config.get("ports", {})),
        "policy": merged_policy,
        "vllm": {
            "enable_responses_api_store": bool(profile.get("vllm", {}).get("enable_responses_api_store", False)),
            "logging_level": str(profile.get("vllm", {}).get("logging_level", "INFO")),
        },
        "state": normalized_state(root, config.get("state", {})),
        "cluster": normalized_cluster(config.get("cluster", {})),
        "resource_profiles": deepcopy(config.get("resource_profiles", {})),
        "inventory": inventory,
        "profile": {
            "name": effective_profile_name,
            "description": profile.get("description", ""),
        },
        "services": services,
        "router": {
            "enabled": True,
            "type": "litellm" if backend == "compose" else "kubeai",
            "aliases": aliases,
        },
        "open_webui": {
            "enabled": backend == "compose",
            "auth": True,
        },
    }
