from __future__ import annotations

from pathlib import Path

import yaml

from vllm_service.backends.compose_renderer import render_compose_artifacts
from vllm_service.backends.kubeai_renderer import render_kubeai_artifacts
from vllm_service.config import initial_config
from vllm_service.exporters import export_benchmark_bundle, export_helm_bundle
from vllm_service.hardware import simulate_inventory
from vllm_service.resolver import resolve
from vllm_service.validator import validate_resolved


def _cfg(tmp_path: Path, *, backend: str = "compose") -> dict:
    cfg = initial_config()
    cfg["backend"] = backend
    cfg["state"] = {
        "hf_cache": "state/hf-cache",
        "open_webui": "state/open-webui",
        "postgres": "state/postgres",
        "runtime": "state/runtime",
    }
    cfg["ports"] = {"litellm": 14000, "open_webui": 13000, "postgres": 15432}
    return cfg


def _plan(tmp_path: Path, profile_name: str, *, backend: str = "compose", inventory: str = "4x96") -> dict:
    cfg = _cfg(tmp_path, backend=backend)
    deployment = resolve(tmp_path, cfg, inventory=simulate_inventory(inventory), profile_name=profile_name)
    validated = validate_resolved(deployment)
    assert validated["ok"], validated
    return {"deployment": deployment, "validated": validated}


def test_profile_resolution_uses_named_serving_profile(tmp_path: Path) -> None:
    deployment = _plan(tmp_path, "qwen2-72b-instruct-tp2-balanced")["deployment"]
    assert deployment["serving_profile"]["public_name"] == "qwen2-72b-instruct-tp2-balanced"
    assert deployment["serving_profile"]["logical_model_name"] == "qwen/qwen2-72b-instruct"
    assert deployment["services"][0]["tensor_parallel_size"] == 2
    assert "qwen2-72b-instruct-tp2-balanced" in deployment["router"]["aliases"]


def test_legacy_profile_alias_resolves_to_canonical_profile(tmp_path: Path) -> None:
    deployment = _plan(tmp_path, "helm-qwen2-72b-instruct")["deployment"]
    assert deployment["serving_profile"]["name"] == "qwen2-72b-instruct-tp2-balanced"


def test_gpt_oss_legacy_alias_prefers_completions_profile(tmp_path: Path) -> None:
    deployment = _plan(tmp_path, "helm-gpt-oss-20b")["deployment"]
    assert deployment["serving_profile"]["name"] == "gpt-oss-20b-completions"


def test_kubeai_render_uses_profile_identity(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "qwen2-72b-instruct-tp2-balanced", backend="kubeai")
    render_kubeai_artifacts(tmp_path, plan)
    models = list(yaml.safe_load_all((tmp_path / "generated" / "kubeai" / "models.yaml").read_text()))
    assert models[0]["metadata"]["name"] == "qwen2-72b-instruct-tp2-balanced"
    assert models[0]["metadata"]["annotations"]["vllm-service/logical-model-name"] == "qwen/qwen2-72b-instruct"
    assert "--tensor-parallel-size=2" in models[0]["spec"]["args"]


def test_compose_render_includes_profile_labels_and_aliases(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, plan)
    compose_text = (tmp_path / "generated" / "docker-compose.yml").read_text()
    litellm_text = (tmp_path / "state" / "runtime" / "litellm_config.yaml").read_text()
    assert 'vllm_service.public_name: "gpt-oss-20b-chat"' in compose_text
    assert "openai/gpt-oss-20b" in litellm_text
    assert "gpt-oss-20b-chat" in litellm_text


def test_exported_model_deployments_shape_for_gpt_oss_completions(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "gpt-oss-20b-completions")
    result = export_benchmark_bundle(tmp_path, plan["deployment"], output_dir=tmp_path / "bundle")
    model_deployments = yaml.safe_load(result["model_deployments_path"].read_text())
    assert model_deployments == {
        "model_deployments": [
            {
                "name": "litellm/gpt-oss-20b-local",
                "model_name": "openai/gpt-oss-20b",
                "tokenizer_name": "openai/o200k_harmony",
                "max_sequence_length": 32768,
                "client_spec": {
                    "class_name": "helm.clients.openai_client.OpenAILegacyCompletionsClient",
                    "args": {
                        "base_url": "http://localhost:14000/v1",
                        "api_key": "SET_LITELLM_MASTER_KEY_IN_RUNTIME_BUNDLE",
                        "openai_model_name": "openai/gpt-oss-20b",
                    },
                },
            }
        ]
    }


def test_exported_model_deployments_shape_for_gpt_oss_chat(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "gpt-oss-20b-chat")
    result = export_benchmark_bundle(tmp_path, plan["deployment"], output_dir=tmp_path / "bundle")
    model_deployments = yaml.safe_load(result["model_deployments_path"].read_text())
    assert model_deployments == {
        "model_deployments": [
            {
                "name": "litellm/gpt-oss-20b-chat-local",
                "model_name": "openai/gpt-oss-20b",
                "tokenizer_name": "openai/o200k_harmony",
                "max_sequence_length": 32768,
                "client_spec": {
                    "class_name": "helm.clients.openai_client.OpenAIClient",
                    "args": {
                        "base_url": "http://localhost:14000/v1",
                        "api_key": "SET_LITELLM_MASTER_KEY_IN_RUNTIME_BUNDLE",
                        "openai_model_name": "openai/gpt-oss-20b",
                    },
                },
            }
        ]
    }


def test_exported_model_deployments_shape_for_qwen_profile(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "qwen2-72b-instruct-tp2-balanced")
    result = export_benchmark_bundle(tmp_path, plan["deployment"], output_dir=tmp_path / "bundle")
    model_deployments = yaml.safe_load(result["model_deployments_path"].read_text())
    assert model_deployments == {
        "model_deployments": [
            {
                "name": "vllm/qwen2-72b-instruct-local",
                "model_name": "qwen/qwen2-72b-instruct",
                "tokenizer_name": "qwen/qwen2-72b-instruct",
                "max_sequence_length": 32768,
                "client_spec": {
                    "class_name": "helm.clients.vllm_client.VLLMChatClient",
                    "args": {
                        "base_url": "http://localhost:8000/v1",
                        "vllm_model_name": "Qwen/Qwen2-72B-Instruct",
                    },
                },
            }
        ]
    }


def test_kubeai_export_uses_openai_compatible_endpoint_semantics(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "qwen2-72b-instruct-tp2-balanced", backend="kubeai")
    result = export_benchmark_bundle(tmp_path, plan["deployment"], output_dir=tmp_path / "bundle")
    model_deployments = yaml.safe_load(result["model_deployments_path"].read_text())
    deployment = model_deployments["model_deployments"][0]
    assert deployment["client_spec"]["class_name"] == "helm.clients.openai_client.OpenAIClient"
    assert deployment["client_spec"]["args"]["base_url"].endswith("/openai/v1")
    assert deployment["client_spec"]["args"]["openai_model_name"] == "qwen2-72b-instruct-tp2-balanced"


def test_repo_relative_and_machine_local_bundle_paths_are_sane(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "gpt-oss-20b-completions")
    repo_result = export_benchmark_bundle(tmp_path, plan["deployment"])
    repo_bundle = yaml.safe_load(repo_result["bundle_path"].read_text())
    repo_smoke = yaml.safe_load(repo_result["benchmark_smoke_manifest_path"].read_text())
    assert repo_bundle["benchmark"]["model_deployments_repo_relative"] == "generated/benchmark/gpt-oss-20b-completions/model_deployments.yaml"
    assert repo_bundle["helm"]["model_deployments_repo_relative"] == "generated/benchmark/gpt-oss-20b-completions/model_deployments.yaml"
    assert repo_smoke["model_deployments_fpath"] == "generated/benchmark/gpt-oss-20b-completions/model_deployments.yaml"

    machine_dir = Path("/tmp") / "machine-local-gpt-oss-bundle"
    result = export_benchmark_bundle(tmp_path, plan["deployment"], output_dir=machine_dir)
    smoke = yaml.safe_load(result["benchmark_smoke_manifest_path"].read_text())
    assert smoke["model_deployments_fpath"] == str((machine_dir / "model_deployments.yaml").resolve())


def test_export_helm_bundle_alias_still_works(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "gpt-oss-20b-completions")
    result = export_helm_bundle(tmp_path, plan["deployment"], output_dir=tmp_path / "bundle")
    assert result["bundle_path"].name == "bundle.yaml"
    assert result["benchmark_smoke_manifest_path"].name == "benchmark_smoke_manifest.yaml"
