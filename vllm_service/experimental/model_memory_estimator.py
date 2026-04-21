from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Literal, Pattern, Sequence

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

try:
    from huggingface_hub import HfApi, hf_hub_download
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore
    hf_hub_download = None  # type: ignore

GiB = 1024 ** 3

@dataclass(frozen=True)
class ModelDiscoveryRule:
    author: str
    search: str
    include: Pattern[str]


CATEGORY_STYLES: Final[dict[str, str]] = {
    "weights": "cyan",
    "cache": "magenta",
    "overhead": "yellow",
}
DISPLAY_CATEGORY_STYLES: Final[dict[str, str]] = {
    "weights": "bold cyan",
    "cache": "bold magenta",
    "overhead": "bold yellow",
    "all": "bold white",
}
EXACTNESS_STYLES: Final[dict[str, str]] = {
    "exact-ish": "green",
    "modeled": "cyan",
    "heuristic": "yellow",
    "mixed": "white",
}

DEFAULT_MODEL_DISCOVERY_RULES: Final[tuple[ModelDiscoveryRule, ...]] = (
    ModelDiscoveryRule(
        author="Qwen",
        search="Qwen3.5",
        include=re.compile(r"^Qwen3\.5-.*$", re.IGNORECASE),
    ),
    ModelDiscoveryRule(
        author="Qwen",
        search="Qwen3.6",
        include=re.compile(r"^Qwen3\.6-.*$", re.IGNORECASE),
    ),
    ModelDiscoveryRule(
        author="google",
        search="gemma-4",
        include=re.compile(r"^gemma-4-.*$", re.IGNORECASE),
    ),
)
DEFAULT_MODEL_DISCOVERY_LIMIT = 100


def fit_markup(fits: bool | None) -> str:
    if fits is True:
        return "[bold green]yes[/bold green]"
    if fits is False:
        return "[bold red]no[/bold red]"
    return "[bold yellow]?[/bold yellow]"


def fit_border_style(fits: bool | None) -> str:
    if fits is True:
        return "green"
    if fits is False:
        return "red"
    return "yellow"


def category_markup(category: str) -> str:
    style = DISPLAY_CATEGORY_STYLES.get(category, "white")
    return f"[{style}]{category}[/{style}]"


def exactness_markup(exactness: str) -> str:
    style = EXACTNESS_STYLES.get(exactness, "white")
    return f"[{style}]{exactness}[/{style}]"


def gib_markup(value_gib: float, *, gpu_budget_gib: float | None = None, category: str | None = None) -> str:
    base_style = CATEGORY_STYLES.get(category or "", "white")
    if gpu_budget_gib and gpu_budget_gib > 0:
        ratio = value_gib / gpu_budget_gib
        if ratio > 1.0:
            style = "bold red"
        elif ratio > 0.85:
            style = "bold yellow"
        else:
            style = f"bold {base_style}"
    else:
        style = f"bold {base_style}"
    return f"[{style}]{value_gib:.2f}[/{style}]"



@dataclass(frozen=True)
class DTypeSpec:
    name: str
    bytes_per_element: float
    notes: str = ""


DTYPE_F16 = DTypeSpec("fp16/bf16", 2.0, "Standard half precision cache/state storage")
DTYPE_FP8 = DTypeSpec("fp8", 1.0, "Approximate FP8 storage; scale tensors are ignored here")
DTYPE_F32 = DTypeSpec("fp32", 4.0, "Single precision storage")


@dataclass(frozen=True)
class WeightFootprint:
    total_bytes: int | None
    source: str
    notes: str = ""


@dataclass(frozen=True)
class CacheGroupSpec:
    name: str
    kind: Literal["full_kv", "sliding_kv", "linear_recurrent", "linear_conv"]
    layer_type: str
    total_layers: int
    unique_cache_layers: int
    num_heads: int | None = None
    head_dim: int | None = None
    seq_len_mode: Literal["full", "sliding", "fixed"] = "full"
    sliding_window: int | None = None
    fixed_elements_per_sequence: int | None = None
    kv_copies: int = 2
    dtype_source: Literal["kv", "linear_state"] = "kv"
    notes: str = ""

    def elements_total(self, deployment: "DeploymentSpec") -> float:
        batch = deployment.concurrent_sequences
        if self.kind in {"full_kv", "sliding_kv"}:
            if self.num_heads is None or self.head_dim is None:
                raise ValueError(f"Missing num_heads/head_dim for cache group {self.name}")
            seq_len = deployment.total_sequence_tokens
            if self.seq_len_mode == "sliding":
                if self.sliding_window is None:
                    raise ValueError(f"Sliding window not set for {self.name}")
                seq_len = min(seq_len, self.sliding_window)
            return (
                self.kv_copies
                * self.unique_cache_layers
                * batch
                * seq_len
                * self.num_heads
                * self.head_dim
            )
        if self.fixed_elements_per_sequence is None:
            raise ValueError(f"Missing fixed_elements_per_sequence for cache group {self.name}")
        return self.unique_cache_layers * batch * self.fixed_elements_per_sequence

    def total_bytes(self, deployment: "DeploymentSpec") -> float:
        dtype = deployment.kv_cache_dtype if self.dtype_source == "kv" else deployment.linear_state_dtype
        return self.elements_total(deployment) * dtype.bytes_per_element


@dataclass(frozen=True)
class ModelMemorySpec:
    repo_id: str
    family: Literal["qwen3.5", "qwen3.6", "gemma4"]
    architecture: str
    max_position_embeddings: int
    text_hidden_size: int
    layer_types: tuple[str, ...]
    weight_footprint: WeightFootprint
    cache_groups: tuple[CacheGroupSpec, ...]
    media_token_hint: int | None = None
    multimodal_resident_margin_fraction: float = 0.0
    multimodal_resident_margin_note: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OverheadPolicy:
    fixed_bytes_per_gpu: int = int(4.0 * GiB)
    weight_fraction: float = 0.02
    note: str = (
        "Heuristic bucket for non-KV runtime overhead: activations/workspaces during warmup, CUDA graphs, "
        "allocator fragmentation, scheduler metadata, and other non-model buffers. This intentionally excludes "
        "the main preallocated vLLM KV cache budget."
    )

    def estimate_bytes_per_gpu(self, weight_like_bytes_per_gpu: float) -> float:
        return self.fixed_bytes_per_gpu + self.weight_fraction * weight_like_bytes_per_gpu


@dataclass(frozen=True)
class DeploymentSpec:
    name: str
    tensor_parallel_size: int
    data_parallel_size: int = 1
    concurrent_sequences: int = 1
    prompt_text_tokens: int = 8192
    media_soft_tokens: int = 0
    max_new_tokens: int = 0
    kv_cache_dtype: DTypeSpec = DTYPE_F16
    linear_state_dtype: DTypeSpec = DTYPE_F16
    gpu_memory_bytes: int | None = None
    gpu_memory_utilization: float | None = None
    language_model_only: bool = False
    use_vllm_preallocation: bool = True
    overhead_policy: OverheadPolicy = OverheadPolicy()

    @property
    def total_sequence_tokens(self) -> int:
        return self.prompt_text_tokens + self.media_soft_tokens + self.max_new_tokens

    @property
    def gpu_memory_gib(self) -> float | None:
        if self.gpu_memory_bytes is None:
            return None
        return self.gpu_memory_bytes / GiB

    @property
    def managed_budget_bytes_per_gpu(self) -> float | None:
        if self.gpu_memory_bytes is None or self.gpu_memory_utilization is None:
            return None
        return self.gpu_memory_bytes * self.gpu_memory_utilization


@dataclass(frozen=True)
class MemoryComponent:
    name: str
    category: Literal["weights", "cache", "overhead"]
    total_bytes: float
    bytes_per_gpu: float
    exactness: Literal["exact-ish", "modeled", "heuristic"]
    notes: str = ""



@dataclass(frozen=True)
class MemoryEstimate:
    model: ModelMemorySpec
    deployment: DeploymentSpec
    components: tuple[MemoryComponent, ...]
    physical_budget_bytes_per_gpu: float | None = None
    managed_budget_bytes_per_gpu: float | None = None
    request_floor_cache_bytes_per_gpu: float = 0.0
    token_cache_floor_bytes_per_gpu: float = 0.0
    fixed_cache_bytes_per_gpu: float = 0.0
    overhead_bytes_per_gpu: float = 0.0
    reserved_token_cache_bytes_per_gpu: float | None = None
    max_concurrency_at_ctx: float | None = None

    @property
    def total_bytes_per_gpu(self) -> float:
        return sum(c.bytes_per_gpu for c in self.components)

    @property
    def total_cluster_bytes(self) -> float:
        return sum(c.total_bytes for c in self.components)

    @property
    def fit_budget_bytes_per_gpu(self) -> float | None:
        if self.managed_budget_bytes_per_gpu is not None:
            return self.managed_budget_bytes_per_gpu
        return self.physical_budget_bytes_per_gpu

    @property
    def fits_per_gpu(self) -> bool | None:
        budget = self.fit_budget_bytes_per_gpu
        if budget is None:
            return None
        return self.total_bytes_per_gpu <= budget


def gib(x: float) -> float:
    return x / GiB


def _is_token_dependent_group(group: CacheGroupSpec) -> bool:
    return group.kind in {"full_kv", "sliding_kv"}


def _request_floor_cache_components(
    model: ModelMemorySpec,
    deployment: DeploymentSpec,
) -> tuple[list[MemoryComponent], float, float]:
    components: list[MemoryComponent] = []
    token_dependent_total = 0.0
    fixed_total = 0.0
    for group in model.cache_groups:
        total = group.total_bytes(deployment)
        per_gpu = total / deployment.tensor_parallel_size
        if _is_token_dependent_group(group):
            token_dependent_total += per_gpu
        else:
            fixed_total += per_gpu
        components.append(
            MemoryComponent(
                name=group.name,
                category="cache",
                total_bytes=total * deployment.data_parallel_size,
                bytes_per_gpu=per_gpu,
                exactness="modeled",
                notes=group.notes,
            )
        )
    return components, token_dependent_total, fixed_total


def _load_json(repo_id: str, filename: str, token: str | None = None) -> dict[str, Any]:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required to fetch configs")
    path = hf_hub_download(repo_id, filename=filename, token=token)
    return json.loads(Path(path).read_text())


def _model_info(repo_id: str, token: str | None = None) -> tuple[Any | None, Any | None]:
    if HfApi is None:
        return None, None
    api = HfApi(token=token)
    info_expand = None
    info_files = None
    try:
        info_expand = api.model_info(
            repo_id,
            expand=["config", "safetensors", "siblings", "tags", "pipeline_tag", "createdAt", "lastModified"],
        )
    except Exception:
        try:
            info_expand = api.model_info(repo_id)
        except Exception:
            info_expand = None
    try:
        info_files = api.model_info(repo_id, files_metadata=True)
    except Exception:
        info_files = None
    return info_expand, info_files


def _to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    out: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        value = getattr(obj, name)
        if callable(value):
            continue
        out[name] = value
    return out


def _listify(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _weight_footprint_from_hf(repo_id: str, token: str | None = None) -> WeightFootprint:
    info_expand, info_files = _model_info(repo_id, token=token)
    expand_dict = _to_dict(info_expand)
    candidates: list[tuple[int, str, str]] = []

    safetensors_info = expand_dict.get("safetensors")
    if isinstance(safetensors_info, dict):
        total = safetensors_info.get("total") or safetensors_info.get("total_size")
        if isinstance(total, (int, float)):
            candidates.append(
                (
                    int(total),
                    "huggingface model_info.safetensors.total",
                    "Checkpoint storage bytes reported by HF safetensors metadata.",
                )
            )

    if info_files is not None:
        siblings = _listify(getattr(info_files, "siblings", None))
        total = 0
        used = False
        for sibling in siblings:
            sdict = _to_dict(sibling)
            name = str(sdict.get("rfilename") or sdict.get("path") or "")
            size = sdict.get("size")
            if name.endswith((".safetensors", ".bin", ".pt")) and isinstance(size, (int, float)):
                total += int(size)
                used = True
        if used:
            candidates.append(
                (
                    total,
                    "sum(model_info(files_metadata=True).siblings[*].size)",
                    "Checkpoint storage bytes reconstructed from HF file metadata.",
                )
            )

    if not candidates:
        return WeightFootprint(
            total_bytes=None,
            source="unavailable",
            notes="Could not recover checkpoint byte size from HF metadata. Weight memory will be unavailable.",
        )

    best_total, best_source, best_note = max(candidates, key=lambda x: x[0])
    notes = [best_note]
    if len(candidates) > 1:
        rendered = ", ".join(f"{src_name}={total / GiB:.2f} GiB" for total, src_name, _ in candidates)
        notes.append(
            "Multiple HF byte signals were available; using the maximum to avoid static undercounting. "
            f"Candidates: {rendered}."
        )

    return WeightFootprint(
        total_bytes=best_total,
        source=best_source,
        notes=" ".join(notes),
    )


def _default_qwen_layer_types(num_hidden_layers: int, full_attention_interval: int = 4) -> list[str]:
    return ["linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention" for i in range(num_hidden_layers)]


def _default_gemma_layer_types(num_hidden_layers: int, sliding_pattern: int = 6) -> list[str]:
    layer_types = ["sliding_attention" if bool((i + 1) % sliding_pattern) else "full_attention" for i in range(num_hidden_layers)]
    if layer_types and layer_types[-1] != "full_attention":
        layer_types[-1] = "full_attention"
    return layer_types


def _build_qwen_spec(repo_id: str, raw_config: dict[str, Any], weight_footprint: WeightFootprint, family: Literal["qwen3.5", "qwen3.6"]) -> ModelMemorySpec:
    text = raw_config.get("text_config", raw_config)
    vision = raw_config.get("vision_config", {})
    num_hidden_layers = int(text["num_hidden_layers"])
    layer_types = tuple(text.get("layer_types") or _default_qwen_layer_types(num_hidden_layers))
    num_full = sum(1 for t in layer_types if t == "full_attention")
    num_linear = sum(1 for t in layer_types if t == "linear_attention")

    full_group = CacheGroupSpec(
        name="full_attention_kv",
        kind="full_kv",
        layer_type="full_attention",
        total_layers=num_full,
        unique_cache_layers=num_full,
        num_heads=int(text["num_key_value_heads"]),
        head_dim=int(text.get("head_dim") or (text["hidden_size"] // text["num_attention_heads"])),
        seq_len_mode="full",
        kv_copies=2,
        dtype_source="kv",
        notes="Standard KV cache for the full-attention layers only.",
    )

    linear_num_key_heads = int(text["linear_num_key_heads"])
    linear_num_value_heads = int(text["linear_num_value_heads"])
    linear_key_head_dim = int(text["linear_key_head_dim"])
    linear_value_head_dim = int(text["linear_value_head_dim"])
    conv_kernel = int(text["linear_conv_kernel_dim"])
    conv_width = 2 * linear_num_key_heads * linear_key_head_dim + linear_num_value_heads * linear_value_head_dim

    linear_recurrent = CacheGroupSpec(
        name="linear_attention_recurrent_state",
        kind="linear_recurrent",
        layer_type="linear_attention",
        total_layers=num_linear,
        unique_cache_layers=num_linear,
        fixed_elements_per_sequence=linear_num_value_heads * linear_key_head_dim * linear_value_head_dim,
        seq_len_mode="fixed",
        dtype_source="linear_state",
        notes=(
            "Fixed-size recurrent state for Qwen Gated DeltaNet layers; shape follows the Transformers implementation: "
            "(batch, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim)."
        ),
    )
    linear_conv = CacheGroupSpec(
        name="linear_attention_conv_state",
        kind="linear_conv",
        layer_type="linear_attention",
        total_layers=num_linear,
        unique_cache_layers=num_linear,
        fixed_elements_per_sequence=conv_width * conv_kernel,
        seq_len_mode="fixed",
        dtype_source="linear_state",
        notes=(
            "Fixed-size causal-convolution state for Qwen Gated DeltaNet layers; conv width is 2*key_dim + value_dim."
        ),
    )

    notes = [
        f"Parsed as {family} hybrid text+vision model.",
        f"Layer mix: {num_linear} linear-attention layers and {num_full} full-attention layers.",
    ]
    media_hint = raw_config.get("vision_soft_tokens_per_image")
    if media_hint is None and isinstance(vision, dict):
        media_hint = vision.get("default_output_length")

    has_vision = isinstance(vision, dict) and bool(vision)
    multimodal_margin_fraction = 0.20 if has_vision else 0.0
    multimodal_margin_note = (
        "Static uncertain multimodal resident-memory margin set to 20% of checkpoint bytes because this repo "
        "contains a vision encoder / multimodal path. This is a conservative hedge for encoder, projector, "
        "and other resident multimodal memory not captured by the text-side cache groups."
        if has_vision
        else ""
    )

    return ModelMemorySpec(
        repo_id=repo_id,
        family=family,
        architecture="hybrid_qwen_linear_plus_full_attention",
        max_position_embeddings=int(text["max_position_embeddings"]),
        text_hidden_size=int(text["hidden_size"]),
        layer_types=layer_types,
        weight_footprint=weight_footprint,
        cache_groups=(full_group, linear_recurrent, linear_conv),
        media_token_hint=int(media_hint) if isinstance(media_hint, int) else None,
        multimodal_resident_margin_fraction=multimodal_margin_fraction,
        multimodal_resident_margin_note=multimodal_margin_note,
        notes=tuple(notes),
    )


def _build_gemma4_spec(repo_id: str, raw_config: dict[str, Any], weight_footprint: WeightFootprint) -> ModelMemorySpec:
    text = raw_config.get("text_config", raw_config)
    layer_types = tuple(text.get("layer_types") or _default_gemma_layer_types(int(text["num_hidden_layers"])))
    shared_tail = int(text.get("num_kv_shared_layers", 0))
    shared_mask = [False] * len(layer_types)
    for i in range(max(0, len(layer_types) - shared_tail), len(layer_types)):
        shared_mask[i] = True

    num_heads_sliding = int(text["num_key_value_heads"])
    head_dim_sliding = int(text.get("head_dim") or (text["hidden_size"] // text["num_attention_heads"]))
    num_heads_full = int(text.get("num_global_key_value_heads") or text.get("num_key_value_heads"))
    head_dim_full = int(text.get("global_head_dim") or text.get("head_dim") or (text["hidden_size"] // text["num_attention_heads"]))
    kv_copies = 1 if bool(text.get("attention_k_eq_v", False)) else 2

    sliding_total = sum(1 for t in layer_types if t == "sliding_attention")
    full_total = sum(1 for t in layer_types if t == "full_attention")
    sliding_unique = sum(1 for i, t in enumerate(layer_types) if t == "sliding_attention" and not shared_mask[i])
    full_unique = sum(1 for i, t in enumerate(layer_types) if t == "full_attention" and not shared_mask[i])

    sliding_group = CacheGroupSpec(
        name="sliding_attention_kv",
        kind="sliding_kv",
        layer_type="sliding_attention",
        total_layers=sliding_total,
        unique_cache_layers=sliding_unique,
        num_heads=num_heads_sliding,
        head_dim=head_dim_sliding,
        seq_len_mode="sliding",
        sliding_window=int(text["sliding_window"]),
        kv_copies=kv_copies,
        dtype_source="kv",
        notes="Sliding-window KV cache. Effective per-layer sequence length is min(total_sequence_tokens, sliding_window).",
    )
    full_group = CacheGroupSpec(
        name="full_attention_kv",
        kind="full_kv",
        layer_type="full_attention",
        total_layers=full_total,
        unique_cache_layers=full_unique,
        num_heads=num_heads_full,
        head_dim=head_dim_full,
        seq_len_mode="full",
        kv_copies=kv_copies,
        dtype_source="kv",
        notes="Global/full-attention KV cache. Gemma 4 can use larger head_dim/global KV heads on these layers.",
    )

    notes = [
        "Parsed as Gemma 4 hybrid sliding/full-attention model.",
        f"Shared KV tail layers: {shared_tail}. Only non-shared layers allocate unique cache storage.",
        f"Layer mix: {sliding_total} sliding-attention layers and {full_total} full-attention layers.",
    ]
    media_hint = raw_config.get("vision_soft_tokens_per_image")

    vision = raw_config.get("vision_config", {})
    has_vision = isinstance(vision, dict) and bool(vision)
    multimodal_margin_fraction = 0.12 if has_vision else 0.0
    multimodal_margin_note = (
        "Static uncertain multimodal resident-memory margin set to 12% of checkpoint bytes because this repo "
        "contains a vision encoder / multimodal path. This is a conservative hedge for encoder, projector, "
        "and other resident multimodal memory not captured by the cache groups."
        if has_vision
        else ""
    )

    return ModelMemorySpec(
        repo_id=repo_id,
        family="gemma4",
        architecture="hybrid_gemma4_sliding_plus_full_attention",
        max_position_embeddings=int(text["max_position_embeddings"]),
        text_hidden_size=int(text["hidden_size"]),
        layer_types=layer_types,
        weight_footprint=weight_footprint,
        cache_groups=(sliding_group, full_group),
        media_token_hint=int(media_hint) if isinstance(media_hint, int) else None,
        multimodal_resident_margin_fraction=multimodal_margin_fraction,
        multimodal_resident_margin_note=multimodal_margin_note,
        notes=tuple(notes),
    )


def load_model_spec(repo_id: str, token: str | None = None) -> ModelMemorySpec:
    raw_config = _load_json(repo_id, "config.json", token=token)
    weight_footprint = _weight_footprint_from_hf(repo_id, token=token)

    model_type = str(raw_config.get("model_type") or "")
    text_config = raw_config.get("text_config") or raw_config
    text_model_type = str(text_config.get("model_type") or model_type)

    if text_model_type in {"qwen3_5_text", "qwen3_5"}:
        return _build_qwen_spec(repo_id, raw_config, weight_footprint, family="qwen3.5")
    if text_model_type in {"qwen3_5_moe_text", "qwen3_5_moe"}:
        # Qwen3.6 currently uses qwen3_5_moe family configs on HF.
        return _build_qwen_spec(repo_id, raw_config, weight_footprint, family="qwen3.6")
    if text_model_type in {"gemma4_text", "gemma4"}:
        return _build_gemma4_spec(repo_id, raw_config, weight_footprint)

    raise ValueError(f"Unsupported model_type/text_model_type for this estimator: {model_type} / {text_model_type}")



def estimate_memory(model: ModelMemorySpec, deployment: DeploymentSpec) -> MemoryEstimate:
    comps: list[MemoryComponent] = []

    weight_per_gpu = 0.0
    if model.weight_footprint.total_bytes is not None:
        total_weight_bytes = float(model.weight_footprint.total_bytes)
        weight_per_gpu = total_weight_bytes / deployment.tensor_parallel_size
        comps.append(
            MemoryComponent(
                name="checkpoint_weights",
                category="weights",
                total_bytes=total_weight_bytes * deployment.data_parallel_size,
                bytes_per_gpu=weight_per_gpu,
                exactness="exact-ish",
                notes=(
                    f"Checkpoint bytes from {model.weight_footprint.source}. Per-GPU estimate assumes even TP sharding across "
                    f"{deployment.tensor_parallel_size} ranks. {model.weight_footprint.notes}"
                ),
            )
        )
        if model.multimodal_resident_margin_fraction > 0.0 and not deployment.language_model_only:
            margin_total = total_weight_bytes * model.multimodal_resident_margin_fraction
            margin_per_gpu = margin_total / deployment.tensor_parallel_size
            comps.append(
                MemoryComponent(
                    name="multimodal_resident_margin",
                    category="overhead",
                    total_bytes=margin_total * deployment.data_parallel_size,
                    bytes_per_gpu=margin_per_gpu,
                    exactness="heuristic",
                    notes=model.multimodal_resident_margin_note,
                )
            )

    cache_components, token_cache_floor_per_gpu, fixed_cache_per_gpu = _request_floor_cache_components(model, deployment)
    comps.extend(cache_components)

    weight_like_per_gpu = sum(
        c.bytes_per_gpu for c in comps if c.name in {"checkpoint_weights", "multimodal_resident_margin"}
    )
    overhead_per_gpu = deployment.overhead_policy.estimate_bytes_per_gpu(weight_like_per_gpu + fixed_cache_per_gpu)
    comps.append(
        MemoryComponent(
            name="framework_overhead",
            category="overhead",
            total_bytes=overhead_per_gpu * deployment.tensor_parallel_size * deployment.data_parallel_size,
            bytes_per_gpu=overhead_per_gpu,
            exactness="heuristic",
            notes=deployment.overhead_policy.note,
        )
    )

    managed_budget = deployment.managed_budget_bytes_per_gpu
    physical_budget = float(deployment.gpu_memory_bytes) if deployment.gpu_memory_bytes is not None else None

    reserved_token_cache_per_gpu: float | None = None
    if deployment.use_vllm_preallocation:
        budget_for_reservation = managed_budget if managed_budget is not None else physical_budget
        if budget_for_reservation is not None:
            reserved_token_cache_per_gpu = max(
                0.0,
                budget_for_reservation - weight_like_per_gpu - fixed_cache_per_gpu - overhead_per_gpu,
            )

    max_concurrency_at_ctx: float | None = None
    if reserved_token_cache_per_gpu is not None and token_cache_floor_per_gpu > 0:
        per_request_token_cache = token_cache_floor_per_gpu / max(1, deployment.concurrent_sequences)
        if per_request_token_cache > 0:
            max_concurrency_at_ctx = reserved_token_cache_per_gpu / per_request_token_cache

    return MemoryEstimate(
        model=model,
        deployment=deployment,
        components=tuple(comps),
        physical_budget_bytes_per_gpu=physical_budget,
        managed_budget_bytes_per_gpu=managed_budget,
        request_floor_cache_bytes_per_gpu=token_cache_floor_per_gpu + fixed_cache_per_gpu,
        token_cache_floor_bytes_per_gpu=token_cache_floor_per_gpu,
        fixed_cache_bytes_per_gpu=fixed_cache_per_gpu,
        overhead_bytes_per_gpu=overhead_per_gpu,
        reserved_token_cache_bytes_per_gpu=reserved_token_cache_per_gpu,
        max_concurrency_at_ctx=max_concurrency_at_ctx,
    )


def standard_deployments(
    model: ModelMemorySpec,
    *,
    gpu_gib: int | None = 96,
    gpu_memory_utilization: float | None = 0.95,
    language_model_only: bool = True,
) -> list[DeploymentSpec]:
    native_tokens = model.max_position_embeddings
    gpu_bytes = (gpu_gib * GiB) if gpu_gib is not None else None
    return [
        DeploymentSpec(
            name="tp4_ctx_98k",
            tensor_parallel_size=4,
            prompt_text_tokens=min(98304, native_tokens),
            max_new_tokens=0,
            gpu_memory_bytes=gpu_bytes,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        ),
        DeploymentSpec(
            name="tp4_ctx_128k",
            tensor_parallel_size=4,
            prompt_text_tokens=min(131072, native_tokens),
            max_new_tokens=0,
            gpu_memory_bytes=gpu_bytes,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        ),
        DeploymentSpec(
            name="tp4_ctx_256k",
            tensor_parallel_size=4,
            prompt_text_tokens=min(131072 * 2, native_tokens),
            max_new_tokens=0,
            gpu_memory_bytes=gpu_bytes,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        ),
    ]



def render_estimate_table(estimates: Sequence[MemoryEstimate], console: Console | None = None) -> None:
    console = console or Console()
    for estimate in estimates:
        model = estimate.model
        dep = estimate.deployment
        title = f"{model.repo_id} — {dep.name}"
        meta_parts = [
            f"family={model.family}",
            f"arch={model.architecture}",
            f"seq={dep.total_sequence_tokens:,}",
            f"tp={dep.tensor_parallel_size}",
            f"kv_dtype={dep.kv_cache_dtype.name}",
        ]
        if dep.gpu_memory_utilization is not None:
            meta_parts.append(f"gpu_mem_util={dep.gpu_memory_utilization:.2f}")
        if dep.language_model_only:
            meta_parts.append("language_model_only=true")
        if dep.gpu_memory_gib is not None:
            meta_parts.append(f"physical_gpu={dep.gpu_memory_gib:.1f} GiB")
        if estimate.managed_budget_bytes_per_gpu is not None:
            meta_parts.append(f"managed_budget={gib(estimate.managed_budget_bytes_per_gpu):.1f} GiB")
        meta = " | ".join(meta_parts)
        console.print(Panel(meta, title=title, expand=False, border_style=fit_border_style(estimate.fits_per_gpu)))

        budget_gib = gib(estimate.fit_budget_bytes_per_gpu) if estimate.fit_budget_bytes_per_gpu is not None else dep.gpu_memory_gib

        table = Table(box=box.SIMPLE_HEAVY, row_styles=["", "dim"])
        table.add_column("component")
        table.add_column("category")
        table.add_column("exactness")
        table.add_column("cluster GiB", justify="right")
        table.add_column("per-GPU GiB", justify="right")
        table.add_column("notes")
        for comp in estimate.components:
            table.add_row(
                comp.name,
                category_markup(comp.category),
                exactness_markup(comp.exactness),
                gib_markup(gib(comp.total_bytes), category=comp.category),
                gib_markup(gib(comp.bytes_per_gpu), gpu_budget_gib=budget_gib, category=comp.category),
                comp.notes,
            )
        table.add_section()
        table.add_row(
            "[bold]REQUEST_FLOOR_TOTAL[/bold]",
            category_markup("all"),
            exactness_markup("mixed"),
            gib_markup(gib(estimate.total_cluster_bytes)),
            gib_markup(gib(estimate.total_bytes_per_gpu), gpu_budget_gib=budget_gib),
            "Minimum modeled runtime footprint for the configured active request shape.",
            style=fit_border_style(estimate.fits_per_gpu),
        )
        if estimate.managed_budget_bytes_per_gpu is not None:
            table.add_row(
                "[bold]MANAGED_BUDGET[/bold]",
                category_markup("cache"),
                exactness_markup("heuristic"),
                "-",
                gib_markup(gib(estimate.managed_budget_bytes_per_gpu), gpu_budget_gib=dep.gpu_memory_gib, category="cache"),
                "vLLM-style per-GPU managed-memory budget from physical_gpu * gpu_memory_utilization.",
            )
        if estimate.reserved_token_cache_bytes_per_gpu is not None:
            table.add_row(
                "[bold]RESERVED_TOKEN_CACHE_BUDGET[/bold]",
                category_markup("cache"),
                exactness_markup("heuristic"),
                "-",
                gib_markup(gib(estimate.reserved_token_cache_bytes_per_gpu), gpu_budget_gib=budget_gib, category="cache"),
                "Approximate token-dependent KV budget left after weights, fixed cache/state, and non-KV overhead.",
            )
        if estimate.max_concurrency_at_ctx is not None:
            table.add_row(
                "[bold]MAX_CONCURRENCY_AT_THIS_CTX[/bold]",
                category_markup("all"),
                exactness_markup("modeled"),
                "-",
                f"[bold]{estimate.max_concurrency_at_ctx:.2f}x[/bold]",
                "Approximate maximum number of concurrent requests of this same sequence length under the reserved KV budget.",
            )
        console.print(table)
        console.print()



def render_summary_matrix(estimates: Sequence[MemoryEstimate], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Deployment summary", box=box.MINIMAL_DOUBLE_HEAD, row_styles=["", "dim"])
    table.add_column("model")
    table.add_column("deployment")
    table.add_column("physical/GPU GiB", justify="right")
    table.add_column("managed/GPU GiB", justify="right")
    table.add_column("weights/GPU GiB", justify="right")
    table.add_column("req-cache/GPU GiB", justify="right")
    table.add_column("overhead/GPU GiB", justify="right")
    table.add_column("req-total/GPU GiB", justify="right")
    table.add_column("kv-budget/GPU GiB", justify="right")
    table.add_column("max-conc", justify="right")

    for est in estimates:
        weights = sum(c.bytes_per_gpu for c in est.components if c.category == "weights")
        req_cache = est.request_floor_cache_bytes_per_gpu
        overhead = est.overhead_bytes_per_gpu
        total_gib = gib(est.total_bytes_per_gpu)
        physical_budget_text = f"{gib(est.physical_budget_bytes_per_gpu):.1f}" if est.physical_budget_bytes_per_gpu is not None else "-"
        managed_budget_text = f"{gib(est.managed_budget_bytes_per_gpu):.1f}" if est.managed_budget_bytes_per_gpu is not None else "-"
        kv_budget_text = f"{gib(est.reserved_token_cache_bytes_per_gpu):.2f}" if est.reserved_token_cache_bytes_per_gpu is not None else "-"
        max_conc_text = f"{est.max_concurrency_at_ctx:.2f}x" if est.max_concurrency_at_ctx is not None else "-"
        fit_budget_gib = gib(est.fit_budget_bytes_per_gpu) if est.fit_budget_bytes_per_gpu is not None else est.deployment.gpu_memory_gib
        row_style = fit_border_style(est.fits_per_gpu) if est.fits_per_gpu is False else None
        table.add_row(
            est.model.repo_id,
            est.deployment.name,
            physical_budget_text,
            managed_budget_text,
            gib_markup(gib(weights), gpu_budget_gib=fit_budget_gib, category="weights"),
            gib_markup(gib(req_cache), gpu_budget_gib=fit_budget_gib, category="cache"),
            gib_markup(gib(overhead), gpu_budget_gib=fit_budget_gib, category="overhead"),
            gib_markup(total_gib, gpu_budget_gib=fit_budget_gib),
            kv_budget_text,
            max_conc_text,
            style=row_style,
        )
    console.print(table)


def parse_deployment_arg(
    spec: str,
    *,
    default_gpu_gib: int | None = None,
    default_gpu_memory_utilization: float | None = None,
    default_language_model_only: bool = False,
) -> DeploymentSpec:
    # Format: name,tp,prompt,max_new[,gpu_gib][,media][,kv_dtype][,gpu_mem_util]
    parts = spec.split(",")
    if len(parts) < 4:
        raise ValueError(
            "Deployment spec must be name,tp,prompt,max_new[,gpu_gib][,media][,kv_dtype][,gpu_mem_util] "
            "e.g. one24,1,8192,0,24,,fp16,0.95"
        )
    name = parts[0]
    tp = int(parts[1])
    prompt = int(parts[2])
    max_new = int(parts[3])

    gpu_gib: int | None = default_gpu_gib
    media = 0
    kv_dtype = DTYPE_F16
    gpu_mem_util = default_gpu_memory_utilization

    if len(parts) >= 5 and parts[4]:
        gpu_gib = int(parts[4])
    if len(parts) >= 6 and parts[5]:
        media = int(parts[5])
    if len(parts) >= 7 and parts[6]:
        kv_name = parts[6].lower()
        if kv_name == "fp8":
            kv_dtype = DTYPE_FP8
        elif kv_name == "fp32":
            kv_dtype = DTYPE_F32
    if len(parts) >= 8 and parts[7]:
        gpu_mem_util = float(parts[7])

    return DeploymentSpec(
        name=name,
        tensor_parallel_size=tp,
        prompt_text_tokens=prompt,
        max_new_tokens=max_new,
        media_soft_tokens=media,
        gpu_memory_bytes=(gpu_gib * GiB) if gpu_gib is not None else None,
        gpu_memory_utilization=gpu_mem_util,
        language_model_only=default_language_model_only,
        kv_cache_dtype=kv_dtype,
    )



def discover_default_repo_ids(token: str | None = None) -> list[str]:
    if HfApi is None:
        raise RuntimeError(
            "huggingface_hub is required for no-arg discovery. Install huggingface_hub or pass explicit repo IDs."
        )
    api = HfApi(token=token)
    repo_ids: set[str] = set()

    for rule in DEFAULT_MODEL_DISCOVERY_RULES:
        models = list(
            api.list_models(
                author=rule.author,
                search=rule.search,
                limit=DEFAULT_MODEL_DISCOVERY_LIMIT,
            )
        )
        for model in models:
            mdict = model if isinstance(model, dict) else getattr(model, "__dict__", {})
            repo_id = str(mdict.get("id") or mdict.get("modelId") or "")
            if not repo_id or "/" not in repo_id:
                continue
            _, repo_name = repo_id.split("/", 1)
            if rule.include.match(repo_name):
                repo_ids.add(repo_id)

    return sorted(repo_ids, key=lambda x: x.lower())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memory estimator for recent Qwen3.5/Qwen3.6/Gemma4 models.")
    p.add_argument("repo_ids", nargs="*", help="HF repo IDs, e.g. Qwen/Qwen3.6-35B-A3B. If omitted, run the built-in Qwen3.5/Qwen3.6/Gemma4 set.")
    p.add_argument("--list-default-models", action="store_true", help="Print the built-in no-arg model set and exit.")
    p.add_argument("--token", default=None, help="HF token if needed")
    p.add_argument("--standard", action="store_true", help="Run the built-in standardized deployment set")
    p.add_argument("--gpu-gib", type=int, default=96, help="Per-GPU memory budget to use for standardized deployments (default: 96)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="vLLM gpu_memory_utilization target to model (default: 0.95)")
    p.add_argument("--language-model-only", action="store_true", help="Assume --language-model-only so multimodal resident-memory margin is disabled")
    p.add_argument(
        "--deployment",
        action="append",
        default=[],
        help="Custom deployment spec: name,tp,prompt,max_new[,gpu_gib][,media][,kv_dtype][,gpu_mem_util]",
    )
    p.add_argument("--details", action="store_true", help="Show per-component detailed tables")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    console = Console()

    repo_ids = list(args.repo_ids)
    if not repo_ids or args.list_default_models:
        repo_ids = discover_default_repo_ids(token=args.token)
        if args.list_default_models:
            for repo_id in repo_ids:
                print(repo_id)
            return 0
        console.print(
            Panel(
                "\n".join(repo_ids) if repo_ids else "(no models discovered)",
                title="No-arg default model set",
                expand=False,
            )
        )

    explicit_deployments = [
        parse_deployment_arg(
            item,
            default_gpu_gib=args.gpu_gib,
            default_gpu_memory_utilization=args.gpu_memory_utilization,
            default_language_model_only=args.language_model_only,
        )
        for item in args.deployment
    ]

    estimates: list[MemoryEstimate] = []
    for repo_id in repo_ids:
        model = load_model_spec(repo_id, token=args.token)
        model_deployments: list[DeploymentSpec] = []
        if args.standard or not explicit_deployments:
            model_deployments.extend(
                standard_deployments(
                    model,
                    gpu_gib=args.gpu_gib,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    language_model_only=args.language_model_only,
                )
            )
        model_deployments.extend(explicit_deployments)
        for dep in model_deployments:
            estimates.append(estimate_memory(model, dep))

    render_summary_matrix(estimates, console=console)
    if args.details:
        console.print()
        render_estimate_table(estimates, console=console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


