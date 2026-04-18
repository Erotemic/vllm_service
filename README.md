# vLLM Service

`vllm_service` manages **named serving profiles** for local and Kubernetes-backed inference.

A serving profile is a complete serving recipe for a model, including:

* which model to load
* what public name to serve it under
* how requests should reach it
* how it should use GPUs
* runtime settings such as tensor parallelism, context length, and batching

This repo can render and run those profiles through two backends:

* **Compose** for local single-host serving
* **KubeAI** for Kubernetes-backed serving

## Main idea

You do not work directly with raw model IDs most of the time.

You work with a **profile name**, for example:

* `gpt-oss-20b-completions`
* `gpt-oss-20b-chat`
* `qwen2-72b-instruct-tp2-balanced`

Each profile resolves to a concrete serving plan.

## Main commands

```bash id="jv5tgf"
python manage.py setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
python manage.py list-profiles
python manage.py describe-profile <profile>
python manage.py render
python manage.py up -d
python manage.py deploy
python manage.py switch <profile> --apply
python manage.py status
python manage.py smoke-test
```

## Inspect a profile before running it

This is the best way to understand what a profile will do:

```bash id="ecl6n7"
python manage.py describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
```

That command prints the resolved serving contract for the profile, including model identity, access shape, placement, and runtime settings.  

---

## Backend 1: Compose

Use the Compose backend when you want the simplest local path on one machine.

### What it gives you

* easy local startup
* inspectable generated files
* simple iteration on one host
* a stable local API front door through LiteLLM

### What it does not give you

* Kubernetes-style model objects
* Kubernetes routing and deployment behavior
* cluster-managed serving

### Getting started

Prerequisite: Docker and the `docker compose` plugin must be installed on the local machine.

Choose a profile, write a working config with one command, render it, and start it:

```bash id="6v6r5x"
python manage.py setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
python manage.py list-profiles
python manage.py describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
python manage.py render
python manage.py up -d
```

The Compose backend renders files such as:

* `generated/plan.yaml`
* `generated/docker-compose.yml`
* `generated/.env`
* runtime files under `state/runtime` 

### Test that it is responding

The built-in smoke test checks `/models` and then sends a chat request unless told not to. 

```bash id="e0if9v"
python manage.py smoke-test
```

You can also test it directly. The default Compose front door is:

```text
http://127.0.0.1:14000/v1
```

unless you changed the LiteLLM port in config. 

List models:

```bash id="t1q5yd"
curl http://127.0.0.1:14000/v1/models
```

Send a request:

```bash id="bby2mi"
curl http://127.0.0.1:14000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen/qwen2.5-7b-instruct-turbo",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

### Stop it

```bash id="gh9915"
python manage.py down
```

---

## Backend 2: KubeAI

Use the KubeAI backend when you want the same serving-profile model, but deployed through Kubernetes.

### What it adds beyond Compose

* profiles rendered as KubeAI/Kubernetes artifacts
* an OpenAI-compatible front door through KubeAI
* deployment through `kubectl` and Helm-backed KubeAI install flow
* profile switching through the Kubernetes path

### What changes compared to Compose

* you use `deploy` instead of `up`
* the default front door is `/openai/v1`
* generated artifacts go under `generated/kubeai/`
* you need a working Kubernetes cluster and KubeAI installed or installable

The current repo also includes a K3s bootstrap path, but the backend itself is `kubeai` on Kubernetes, not K3s-specific in concept. 


### Prerequisites

## Prerequisites

### Kubernetes (using K3s)

The KubeAI backend needs:

* a working Kubernetes cluster
* `kubectl`
* Helm, the Kubernetes package manager

K3s is the quickest way to get a single-node Kubernetes cluster for local use.

Install K3s:

```bash
# Latest
curl -sfL https://get.k3s.io | sh -

# Or pin a version
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='v1.34.3+k3s1' sh -
```

Make `kubectl` usable without `sudo`:

```bash
sudo mkdir -p /etc/rancher/k3s/config.yaml.d
printf 'write-kubeconfig-mode: "0644"\n' | \
  sudo tee /etc/rancher/k3s/config.yaml.d/10-kubeconfig-mode.yaml >/dev/null
sudo systemctl restart k3s
kubectl get nodes
```

Install Helm:

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/83a46119086589a593a62ca544982977a60318ca/scripts/get-helm-4
chmod 700 get_helm.sh
./get_helm.sh
helm version
```

### NVIDIA GPU support

If this machine has NVIDIA GPUs, install the NVIDIA device plugin and GPU feature discovery so Kubernetes can expose GPU resources and labels:

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --version 0.17.1 \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --set gfd.enabled=true \
  --set runtimeClassName=nvidia
```

Check that GPU support is working:

```bash
kubectl -n nvidia-device-plugin get pods
kubectl get node "$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
kubectl get nodes --show-labels | tr ',' '\n' | grep 'nvidia.com/' || true
```

You want to see:

* the NVIDIA plugin pods are `Running`
* the node reports a non-empty `nvidia.com/gpu` count
* the node has `nvidia.com/*` labels such as product and memory

### KubeAI

Add the KubeAI Helm repository:

```bash
helm repo add kubeai https://www.kubeai.org
helm repo update
```

Generate a local KubeAI GPU resource profile file from the labels on this machine:

```bash
NODE="$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')"
PRODUCT="$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.product}')"
MEMORY="$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.memory}')"
COUNT="$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.count}')"
SLUG="$(printf '%s' "$PRODUCT" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
PROFILE="nvidia-gpu-${SLUG}-${MEMORY}mb"

cat > values-kubeai-local-gpu.yaml <<EOF
# Autogenerated from Kubernetes node labels on ${NODE}
# nvidia.com/gpu.product=${PRODUCT}
# nvidia.com/gpu.memory=${MEMORY}
# nvidia.com/gpu.count=${COUNT}

resourceProfiles:
  ${PROFILE}:
    nodeSelector:
      nvidia.com/gpu.product: "${PRODUCT}"
      nvidia.com/gpu.memory: "${MEMORY}"
EOF

echo "Wrote values-kubeai-local-gpu.yaml"
cat values-kubeai-local-gpu.yaml
```



HACK: use the name the code expects. TODO: fix the code to be robust later

```
PRODUCT="$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.product}')"
MEMORY="$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.memory}')"

cat > values-kubeai-local-gpu.yaml <<EOF
resourceProfiles:
  gpu-single-default:
    nodeSelector:
      nvidia.com/gpu.product: "${PRODUCT}"
      nvidia.com/gpu.memory: "${MEMORY}"

  gpu-tp2-balanced:
    nodeSelector:
      nvidia.com/gpu.product: "${PRODUCT}"
      nvidia.com/gpu.memory: "${MEMORY}"

  gpu-tp2-maxctx:
    nodeSelector:
      nvidia.com/gpu.product: "${PRODUCT}"
      nvidia.com/gpu.memory: "${MEMORY}"
EOF

cat values-kubeai-local-gpu.yaml


KUBEAI_HELM_ARGS=(
  helm upgrade --install kubeai kubeai/kubeai
  -f values-kubeai-local-gpu.yaml
  --wait
)

if [ -n "${HF_TOKEN:-}" ]; then
  KUBEAI_HELM_ARGS+=(--set "secrets.huggingface.token=${HF_TOKEN}")
fi

"${KUBEAI_HELM_ARGS[@]}"
```

Install KubeAI. If `.env` exists, this will load `HF_TOKEN` from there. If `HF_TOKEN` is already exported in your shell, that works too. If you only use public models, `HF_TOKEN` is optional.

```bash
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

KUBEAI_HELM_ARGS=(
  helm upgrade --install kubeai kubeai/kubeai
  -f values-kubeai-local-gpu.yaml
  --wait
)

if [ -n "${HF_TOKEN:-}" ]; then
  KUBEAI_HELM_ARGS+=(--set "secrets.huggingface.token=${HF_TOKEN}")
fi

"${KUBEAI_HELM_ARGS[@]}"
```

If `HF_TOKEN` is already in your environment, this effectively resolves to:

```bash
helm upgrade --install kubeai kubeai/kubeai \
  -f values-kubeai-local-gpu.yaml \
  --set secrets.huggingface.token="$HF_TOKEN" \
  --wait
```

### Status check before using `python manage.py`

Before using the `kubeai` backend in this repo, make sure the cluster is healthy:

```bash
kubectl get nodes
kubectl get crd models.kubeai.org
kubectl get pods
kubectl get node "$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
```

You want to see:

* the node is `Ready`
* `models.kubeai.org` exists
* the `kubeai` pod is `Running`
* the node reports a non-empty `nvidia.com/gpu` count

After that, continue with the KubeAI getting-started commands below.


### Getting started

Prerequisite: `kubectl` and Helm must already work against a Kubernetes cluster where KubeAI is installed or installable.

Use a smaller first-run profile unless you already know the cluster can fit something larger. Start from the same repo, write a working KubeAI config with one command, then render and deploy:

```bash id="31n0f7"
python manage.py setup --backend kubeai --profile qwen2-5-7b-instruct-turbo-default --namespace kubeai
python manage.py list-profiles
python manage.py describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
python manage.py render
python manage.py deploy
python manage.py status
```

The KubeAI backend renders files such as:

* `generated/plan.yaml`
* `generated/kubeai/namespace.yaml`
* `generated/kubeai/kubeai-values.yaml`
* `generated/kubeai/models.yaml` 

### Test that it is responding

If you are not exposing ingress yet, port-forward the service:

```bash id="9c0hbv"
kubectl -n kubeai port-forward svc/kubeai 8000:80
```

Then run the smoke test against the KubeAI front door:

```bash id="l3rh5a"
python manage.py smoke-test --base-url http://127.0.0.1:8000/openai/v1
```

That `/openai/v1` path is the expected KubeAI OpenAI-compatible access shape in this repo.  

You can also test it directly.

List models:

```bash id="ce4zcf"
curl http://127.0.0.1:8000/openai/v1/models
```

Send a request:

```bash id="ajjlwm"
curl http://127.0.0.1:8000/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-5-7b-instruct-turbo-default",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

### Switch to another model profile

You can change the active serving profile and apply it through the backend:

```bash id="gdsq2p"
python manage.py switch gpt-oss-20b-chat --apply
python manage.py status
```

Then test again:

```bash id="ryav9p"
curl http://127.0.0.1:8000/openai/v1/models
```

This is a **profile switch and re-apply**, not a claim that requests automatically lazy-load arbitrary new models without redeployment. The CLI does support switching the active profile and applying it for both backends. 

---

## Which backend should I start with?

Start with **Compose** if you want:

* the fastest path to a working local server
* easy inspection of generated files
* simple single-host iteration

Move to **KubeAI** when you want:

* the same profile model on Kubernetes
* KubeAI’s OpenAI-compatible front door
* profile deployment through Kubernetes artifacts

The normal learning path is:

1. inspect a profile with `describe-profile`
2. run it with Compose
3. move to KubeAI when you want Kubernetes-backed serving

---

## Files you will look at most often

* `config.yaml`
* `models.yaml`
* `generated/plan.yaml`
* `generated/docker-compose.yml`
* `generated/kubeai/models.yaml`

## Environment variable fallbacks

The `setup` command is the recommended first-run path, but the same common settings can also come from environment variables when that is easier for local automation.

Recognized variables include:

* `VLLM_SERVICE_BACKEND`
* `VLLM_SERVICE_PROFILE`
* `VLLM_SERVICE_COMPOSE_CMD`
* `VLLM_SERVICE_LITELLM_PORT`
* `VLLM_SERVICE_OPEN_WEBUI_PORT`
* `VLLM_SERVICE_POSTGRES_PORT`
* `VLLM_SERVICE_STATE_ROOT`
* `VLLM_SERVICE_RUNTIME_DIR`
* `VLLM_SERVICE_NAMESPACE`
* `VLLM_SERVICE_INGRESS_ENABLED`
* `VLLM_SERVICE_INGRESS_HOST`

---

## Typical workflow

### Local Compose iteration

```bash id="hfc0r4"
python manage.py list-profiles
python manage.py describe-profile gpt-oss-20b-completions --format yaml
python manage.py render --profile gpt-oss-20b-completions
python manage.py up -d
python manage.py smoke-test
```

### Kubernetes-backed serving with KubeAI

```bash id="nqim4m"
python manage.py list-profiles
python manage.py describe-profile qwen2-72b-instruct-tp2-balanced --format yaml
python manage.py render --profile qwen2-72b-instruct-tp2-balanced
python manage.py deploy
python manage.py status
kubectl -n kubeai port-forward svc/kubeai 8000:80
python manage.py smoke-test --base-url http://127.0.0.1:8000/openai/v1
```

### Switch profiles

```bash id="qnhmbo"
python manage.py switch qwen2-72b-instruct-tp2-balanced --apply
python manage.py switch gpt-oss-20b-completions --apply
```

---

## Troubleshooting

If something is unclear, check in this order:

```bash id="ybx2fe"
python manage.py list-profiles
python manage.py describe-profile <profile> --format yaml
python manage.py render --profile <profile>
python manage.py status
python manage.py smoke-test
```
