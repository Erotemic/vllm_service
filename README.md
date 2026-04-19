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

You usually do not work directly with raw model IDs.

You work with a **profile name**, for example:

* `gpt-oss-20b-completions`
* `gpt-oss-20b-chat`
* `qwen2-72b-instruct-tp2-balanced`

Each profile resolves to a concrete serving plan.

## Main commands

```bash
python manage.py setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
python manage.py list-profiles
python manage.py describe-profile <profile>
python manage.py validate
python manage.py render
python manage.py up -d
python manage.py deploy
python manage.py switch <profile> --apply
python manage.py status
python manage.py smoke-test
```

## Inspect a profile before running it

```bash
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

```bash
python manage.py setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
python manage.py list-profiles
python manage.py describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
python manage.py validate
python manage.py render
python manage.py up -d
```

The Compose backend renders files such as:

* `generated/plan.yaml`
* `generated/docker-compose.yml`
* `generated/.env`
* runtime files under `state/runtime`

### Test that it is responding

```bash
python manage.py smoke-test
```

The default Compose front door is:

```text
http://127.0.0.1:14000/v1
```

unless you changed the LiteLLM port in config.

List models:

```bash
curl http://127.0.0.1:14000/v1/models
```

Send a request:

```bash
curl http://127.0.0.1:14000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen/qwen2.5-7b-instruct-turbo",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

### Stop it

```bash
python manage.py down
```

---

## Backend 2: KubeAI

Use the KubeAI backend when you want the same serving-profile model, but deployed through Kubernetes.

### What it adds beyond Compose

* profiles rendered as KubeAI and Kubernetes artifacts
* an OpenAI-compatible front door through KubeAI
* deployment through `kubectl` and Helm-backed KubeAI install flow
* profile switching through the Kubernetes path

### What changes compared to Compose

* you use `deploy` instead of `up`
* the default front door is `/openai/v1`
* generated artifacts go under `generated/kubeai/`
* you need a working Kubernetes cluster and KubeAI installed or installable

The backend itself is `kubeai` on Kubernetes. The repo may show K3s examples for single-node setup, but the serving backend is not K3s-specific in concept.

## KubeAI quick start: important rules

Before the detailed steps, keep these four rules in mind:

1. **Use the same namespace everywhere.**  
   The namespace in `python manage.py setup --namespace ...` must match the namespace where the KubeAI Helm release already exists, or the namespace where you want the repo to install it.

2. **Prefer the repo-driven path.**  
   The normal path in this repo is:
   `setup` → `validate` → `render` → `deploy` → `status`

3. **`kubectl port-forward` stays in the foreground.**  
   Leave it running in one terminal and run requests from another.

4. **The first request can take a while.**  
   `/openai/v1/models` may work before chat completions work. The first real completion request may trigger KubeAI to create the model-serving pod.

---

## KubeAI prerequisites

### Kubernetes

The KubeAI backend needs:

* a working Kubernetes cluster
* `kubectl`
* Helm

If you want a quick local single-node cluster, K3s is a good option.

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

If this machine has NVIDIA GPUs, install the NVIDIA device plugin and GPU Feature Discovery so Kubernetes can expose GPU resources and labels:

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

### KubeAI Helm repository

```bash
helm repo add kubeai https://www.kubeai.org
helm repo update
```

---

## Determine which namespace to use

Before doing anything else, discover whether a `kubeai` release already exists and which namespace it uses.

```bash
KUBEAI_NAMESPACE="$(helm list -A | awk '$1=="kubeai"{print $2; exit}')"
if [ -z "${KUBEAI_NAMESPACE}" ]; then
  KUBEAI_NAMESPACE=default
fi
echo "Using KubeAI namespace: ${KUBEAI_NAMESPACE}"
```

If a release already exists, **reuse that namespace**.  
If no release exists yet, the snippet above defaults to `default` so the rest of the commands stay consistent.

Sanity-check the cluster:

```bash
kubectl get nodes
kubectl get crd models.kubeai.org || true
helm list -A | grep kubeai || true
kubectl -n "${KUBEAI_NAMESPACE}" get pods || true
kubectl get node "$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
```

You want to see:

* the node is `Ready`
* GPU allocatable count is non-empty
* if KubeAI is already installed, the `kubeai` pod is `Running`
* if KubeAI is not installed yet, that is okay; `python manage.py deploy` can install or upgrade it

---

## Generate the local KubeAI resource-profile file

Generate a local KubeAI resource-profile file from the labels on this machine.

For the built-in serving profiles in this repo, keep these names aligned:

* `gpu-single-default`
* `gpu-tp2-balanced`
* `gpu-tp2-maxctx`

```bash
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
```

Sync that file so `validate` and `render` use the same local resource-profile data:

```bash
python manage.py kubeai-sync-resource-profiles --from-file values-kubeai-local-gpu.yaml
```

---

## Example 1: single-GPU system

Use this example on a 1-GPU workstation, or as the first sanity-check profile on a larger machine.

```bash
python manage.py setup \
  --backend kubeai \
  --profile qwen2-5-7b-instruct-turbo-default \
  --namespace "${KUBEAI_NAMESPACE}"

python manage.py list-profiles
python manage.py describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
python manage.py validate
python manage.py render
python manage.py deploy
python manage.py status
```

This is the normal repo-driven path. It renders artifacts under `generated/kubeai/` and then uses `deploy` to install or upgrade the KubeAI release and apply the rendered `Model` objects.

---

## Example 2: four-GPU system

On a 4-GPU system, start with the same small single-GPU example above to verify the plumbing first. After that works, switch to a larger multi-GPU profile.

```bash
python manage.py setup \
  --backend kubeai \
  --profile qwen2-5-7b-instruct-turbo-default \
  --namespace "${KUBEAI_NAMESPACE}"

python manage.py validate
python manage.py render
python manage.py deploy
python manage.py status
```

Then try a multi-GPU profile:

```bash
python manage.py switch qwen2-72b-instruct-tp2-balanced --apply
python manage.py status
```

If you want to inspect the multi-GPU profile before applying it:

```bash
python manage.py describe-profile qwen2-72b-instruct-tp2-balanced --format yaml
```

---

## Test that KubeAI is responding

If you are not exposing ingress yet, port-forward the service.

**This command stays in the foreground.**  
Run it in one terminal and leave it there:

```bash
kubectl -n "${KUBEAI_NAMESPACE}" port-forward svc/kubeai 8000:80
```

Then use another terminal for requests.

### First check: `/models`

```bash
curl http://127.0.0.1:8000/openai/v1/models
```

If that works, the KubeAI front door is alive.

### Then try the smoke test

```bash
python manage.py smoke-test --base-url http://127.0.0.1:8000/openai/v1
```

### Or test chat completions directly

```bash
curl http://127.0.0.1:8000/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-5-7b-instruct-turbo-default",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

### What to expect on the first request

The first request may be slower than later ones.

Common first-request behavior:

* `/openai/v1/models` works immediately
* a completion request causes KubeAI to create a model-serving pod
* that pod may spend time in `ContainerCreating` while the image is pulled
* chat completions may time out before the first serving pod is fully ready

That is not automatically a failure. Watch the system state while the first request is happening.

```bash
watch -n 1 'kubectl -n '"${KUBEAI_NAMESPACE}"' get pods; echo; kubectl -n '"${KUBEAI_NAMESPACE}"' get models'
```

---

## Temporary workaround for the current KubeAI resource-profile format issue

If KubeAI logs show an error like:

```text
invalid resource profile: "gpu-single-default", should match <name>:<multiple>
```

then the current rendered `Model` object needs a temporary patch before it will reconcile cleanly.

Check the rendered model file:

```bash
grep -n 'resourceProfile' generated/kubeai/models.yaml
```

If it shows bare values like:

```yaml
resourceProfile: gpu-single-default
```

patch them to append `:1`:

```bash
python - <<'PY'
from pathlib import Path
import yaml

p = Path("generated/kubeai/models.yaml")
docs = [d for d in yaml.safe_load_all(p.read_text()) if d]

for d in docs:
    spec = d.setdefault("spec", {})
    rp = str(spec.get("resourceProfile", "")).strip()
    if rp and ":" not in rp:
        spec["resourceProfile"] = f"{rp}:1"

p.write_text("---\n".join(yaml.safe_dump(d, sort_keys=False) for d in docs))
print(p.read_text())
PY
```

Then re-apply the rendered model objects:

```bash
kubectl -n "${KUBEAI_NAMESPACE}" apply -f generated/kubeai/models.yaml
```

This is a temporary workaround for the current renderer and KubeAI version mismatch. Once the renderer is fixed, this manual patch should no longer be necessary.

---

## Switching profiles

You can change the active serving profile and apply it through the backend:

```bash
python manage.py switch gpt-oss-20b-chat --apply
python manage.py status
```

Then test again:

```bash
curl http://127.0.0.1:8000/openai/v1/models
```

This is a profile switch and re-apply. It is not a claim that requests automatically lazy-load arbitrary new models without redeployment.

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

A practical learning path is:

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

---

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

## Troubleshooting

If something is unclear, check in this order:

```bash
python manage.py list-profiles
python manage.py describe-profile <profile> --format yaml
python manage.py validate
python manage.py render
python manage.py status
```

Then check Kubernetes state directly:

```bash
kubectl -n "${KUBEAI_NAMESPACE}" get pods
kubectl -n "${KUBEAI_NAMESPACE}" get svc
kubectl -n "${KUBEAI_NAMESPACE}" get ingress
kubectl -n "${KUBEAI_NAMESPACE}" get models
kubectl -n "${KUBEAI_NAMESPACE}" get events --sort-by=.lastTimestamp | tail -n 40
```

Check the KubeAI controller logs:

```bash
kubectl -n "${KUBEAI_NAMESPACE}" logs deploy/kubeai --tail=200 -f
```

Describe the `Model` object:

```bash
kubectl -n "${KUBEAI_NAMESPACE}" describe model qwen2-5-7b-instruct-turbo-default
```

Describe the model-serving pod after it appears:

```bash
kubectl -n "${KUBEAI_NAMESPACE}" describe pod <model-pod-name>
```

Useful signs while debugging:

* `kubectl port-forward` staying attached is normal
* `No resources found in <namespace>.` during `status` often just means there is no ingress yet
* `/openai/v1/models` can work before completions work
* the first request may create the model-serving pod
* `ContainerCreating` on the first serving pod often just means the image is still being pulled

