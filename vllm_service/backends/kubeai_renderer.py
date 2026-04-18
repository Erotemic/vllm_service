from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _base_profile_name(resource_profile: str) -> str:
    return resource_profile.split(":", 1)[0]


def _vllm_args(service: dict[str, Any]) -> list[str]:
    args = [
        f"--served-model-name={service['served_model_name']}",
        f"--tensor-parallel-size={service['tensor_parallel_size']}",
        f"--data-parallel-size={service['data_parallel_size']}",
        f"--max-model-len={service['max_model_len']}",
        f"--gpu-memory-utilization={service['gpu_memory_utilization']}",
        f"--max-num-batched-tokens={service['max_num_batched_tokens']}",
        f"--max-num-seqs={service['max_num_seqs']}",
        "--disable-log-requests",
    ]
    if service.get("enable_prefix_caching"):
        args.append("--enable-prefix-caching")
    if service.get("enable_auto_tool_choice"):
        args.append("--enable-auto-tool-choice")
        if service.get("tool_call_parser"):
            args.append(f"--tool-call-parser={service['tool_call_parser']}")
    args.extend(service.get("extra_args", []))
    return args


def _resource_profile_values(plan: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {"resourceProfiles": {}}
    for name, spec in (plan.get("deployment", {}).get("resource_profiles", {}) or {}).items():
        item: dict[str, Any] = {}
        if spec.get("node_selector"):
            item["nodeSelector"] = spec["node_selector"]
        if spec.get("requests"):
            item["requests"] = spec["requests"]
        if spec.get("limits"):
            item["limits"] = spec["limits"]
        if spec.get("tolerations"):
            item["tolerations"] = spec["tolerations"]
        if spec.get("runtime_class_name"):
            item["runtimeClassName"] = spec["runtime_class_name"]
        if spec.get("scheduler_name"):
            item["schedulerName"] = spec["scheduler_name"]
        if spec.get("image_name"):
            item["imageName"] = spec["image_name"]
        values["resourceProfiles"][name] = item
    return values


def _model_doc(service: dict[str, Any]) -> dict[str, Any]:
    doc = {
        "apiVersion": "kubeai.org/v1",
        "kind": "Model",
        "metadata": {"name": service["service_name"]},
        "spec": {
            "features": service.get("features", ["TextGeneration"]),
            "url": service["model_url"],
            "engine": service.get("engine", "VLLM"),
            "resourceProfile": service["resource_profile"],
            "minReplicas": int(service.get("min_replicas", 0)),
            "maxReplicas": int(service.get("max_replicas", 1)),
            "args": _vllm_args(service),
        },
    }
    if service.get("priority_class_name"):
        doc["spec"]["priorityClassName"] = service["priority_class_name"]
    return doc


def render_kubeai_artifacts(root: Path, lock_data: dict) -> None:
    deployment = lock_data.get("deployment", {})
    cluster = deployment.get("cluster", {})
    namespace = cluster.get("namespace", "kubeai")
    generated = root / "generated" / "kubeai"
    generated.mkdir(parents=True, exist_ok=True)

    namespace_doc = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": namespace},
    }
    (generated / "namespace.yaml").write_text(yaml.safe_dump(namespace_doc, sort_keys=False), encoding="utf-8")

    values_doc = _resource_profile_values(lock_data)
    (generated / "kubeai-values.yaml").write_text(yaml.safe_dump(values_doc, sort_keys=False), encoding="utf-8")

    model_docs = [_model_doc(service) for service in deployment.get("services", [])]
    model_text = "---\n".join(yaml.safe_dump(doc, sort_keys=False) for doc in model_docs)
    (generated / "models.yaml").write_text(model_text, encoding="utf-8")

    ingress = cluster.get("ingress", {}) or {}
    ingress_path = generated / "ingress.yaml"
    if ingress.get("enabled"):
        path_prefix = ingress.get("path_prefix", "/") or "/"
        ingress_doc: dict[str, Any] = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": cluster.get("service_name", "kubeai"),
                "namespace": namespace,
            },
            "spec": {
                "ingressClassName": ingress.get("class_name", "traefik"),
                "rules": [
                    {
                        "http": {
                            "paths": [
                                {
                                    "path": path_prefix,
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": cluster.get("service_name", "kubeai"),
                                            "port": {"number": 80},
                                        }
                                    },
                                }
                            ]
                        }
                    }
                ],
            },
        }
        if ingress.get("host"):
            ingress_doc["spec"]["rules"][0]["host"] = ingress["host"]
        if ingress.get("tls_secret_name") and ingress.get("host"):
            ingress_doc["spec"]["tls"] = [{"hosts": [ingress["host"]], "secretName": ingress["tls_secret_name"]}]
        ingress_path.write_text(yaml.safe_dump(ingress_doc, sort_keys=False), encoding="utf-8")
    elif ingress_path.exists():
        ingress_path.unlink()

    readme = f"""# Generated KubeAI artifacts

Namespace: `{namespace}`
Release: `{cluster.get('kubeai_release_name', 'kubeai')}`
Chart: `{cluster.get('kubeai_chart', 'kubeai/kubeai')}`

Files:
- `namespace.yaml`: namespace to apply before the chart and models
- `kubeai-values.yaml`: custom resource profiles for the KubeAI chart
- `models.yaml`: KubeAI `Model` objects derived from the active profile
- `ingress.yaml`: optional ingress for one stable hostname

Typical flow:

```bash
kubectl apply -f generated/kubeai/namespace.yaml
helm repo add kubeai https://www.kubeai.org --force-update
helm repo update
helm upgrade --install {cluster.get('kubeai_release_name', 'kubeai')} {cluster.get('kubeai_chart', 'kubeai/kubeai')} \
  -n {namespace} --create-namespace \
  -f generated/kubeai/kubeai-values.yaml \
  --wait
kubectl apply -f generated/kubeai/models.yaml
```
"""
    (generated / "README.md").write_text(readme, encoding="utf-8")
