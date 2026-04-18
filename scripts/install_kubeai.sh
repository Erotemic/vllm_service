#!/usr/bin/env bash
set -euo pipefail

VALUES_FILE="${1:-generated/kubeai/kubeai-values.yaml}"
NAMESPACE="${2:-kubeai}"

if ! command -v helm >/dev/null 2>&1; then
  echo "helm is required" >&2
  exit 1
fi
if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required" >&2
  exit 1
fi
if [[ ! -f "$VALUES_FILE" ]]; then
  echo "missing values file: $VALUES_FILE" >&2
  exit 1
fi

helm repo add kubeai https://www.kubeai.org --force-update
helm repo update

CMD=(helm upgrade --install kubeai kubeai/kubeai -n "$NAMESPACE" --create-namespace -f "$VALUES_FILE" --wait)
if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--set "secrets.huggingface.token=${HF_TOKEN}")
fi
"${CMD[@]}"
