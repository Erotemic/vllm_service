from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import merged_catalogs, normalized_state
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


def _resolve_service(service: dict[str, Any], models: dict[str, Any], inventory: dict[str, Any], policy: dict[str, Any], used: set[int]) -> dict[str, Any]:
    model_key = service["model"]
    if model_key not in models:
        raise KeyError(f"Unknown model: {model_key}")
    model = deepcopy(models[model_key])
    placement = deepcopy(service.get("placement", {}))
    topology = deepcopy(service.get("topology", {}))
    available = [i for i in _available_gpu_indices(inventory, policy.get("reserve_display_gpu", "auto")) if i not in used]

    # Normalize topology aliases (tp -> tensor_parallel_size, dp -> data_parallel_size)
    if "tp" in topology and "tensor_parallel_size" not in topology:
        topology["tensor_parallel_size"] = topology["tp"]
    if "dp" in topology and "data_parallel_size" not in topology:
        topology["data_parallel_size"] = topology["dp"]

    strategy = placement.get("strategy", "first_fit")
    placement_error = None
    if strategy == "exact" or strategy == "multi_gpu":
        gpu_indices = list(placement.get("gpu_indices", []))
        if not gpu_indices:
            placement_error = f"service {service['service_name']} uses {strategy} placement but no gpu_indices were provided"
    else:
        gpu_count = int(placement.get("gpu_count", topology.get("tensor_parallel_size", model.get("preferred_gpu_count", 1)) or 1))
        gpu_indices, placement_error = _first_fit(available, gpu_count)

    used.update(gpu_indices)
    tp = int(topology.get("tensor_parallel_size", max(1, len(gpu_indices))))
    dp = int(topology.get("data_parallel_size", 1))

    tool_calling = deepcopy(model.get("tool_calling", {}))
    tool_calling = {**tool_calling, **deepcopy(service.get("tool_calling", {}))}
    tool_call_parser = tool_calling.get("parser")
    enable_auto_tool_choice = bool(tool_calling.get("auto", False) and tool_call_parser)

    return {
        "service_name": service["service_name"],
        "model_ref": model_key,
        "hf_model_id": model["hf_model_id"],
        "served_model_name": service.get("served_model_name", model.get("served_model_name", model_key)),
        "modalities": model.get("modalities", ["text"]),
        "memory_class_gib": model.get("memory_class_gib"),
        "min_vram_gib_per_replica": model.get("min_vram_gib_per_replica", 0),
        "context_window": model.get("context_window"),
        "notes": model.get("notes", []),
        "gpu_indices": gpu_indices,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "max_model_len": int(service.get("max_model_len", model.get("defaults", {}).get("max_model_len", 32768))),
        "gpu_memory_utilization": float(service.get("gpu_memory_utilization", model.get("defaults", {}).get("gpu_memory_utilization", 0.9))),
        "enable_prefix_caching": bool(service.get("enable_prefix_caching", model.get("defaults", {}).get("enable_prefix_caching", True))),
        "max_num_batched_tokens": int(service.get("max_num_batched_tokens", model.get("defaults", {}).get("max_num_batched_tokens", 8192))),
        "max_num_seqs": int(service.get("max_num_seqs", model.get("defaults", {}).get("max_num_seqs", 16))),
        "thinking_history_policy": model.get("thinking_history_policy", "keep_final_only"),
        "placement_error": placement_error,
        "enable_auto_tool_choice": enable_auto_tool_choice,
        "tool_call_parser": tool_call_parser,
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

    used: set[int] = set()
    services = []
    for service in profile.get("services", []):
        services.append(_resolve_service(service, catalogs.get("models", {}), inventory, config.get("policy", {}), used))

    aliases = deepcopy(profile.get("router", {}).get("aliases", {}))
    if not aliases:
        aliases = {svc["served_model_name"]: svc["service_name"] for svc in services}

    return {
        "schema_version": 2,
        "source": {
            "config_file": "config.yaml",
            "active_profile": effective_profile_name,
        },
        "images": deepcopy(config.get("images", {})),
        "ports": deepcopy(config.get("ports", {})),
        "policy": deepcopy(config.get("policy", {})),
        "vllm": {
            "enable_responses_api_store": bool(profile.get("vllm", {}).get("enable_responses_api_store", False)),
            "logging_level": str(profile.get("vllm", {}).get("logging_level", "INFO")),
        },
        "state": normalized_state(root, config.get("state", {})),
        "inventory": inventory,
        "profile": {
            "name": effective_profile_name,
            "description": profile.get("description", ""),
        },
        "services": services,
        "router": {
            "enabled": True,
            "type": "litellm",
            "aliases": aliases,
        },
        "open_webui": {
            "enabled": True,
            "auth": True,
        },
    }
