# vLLM deployment compiler

This repo keeps the source-of-truth config, catalogs, and deployment scripts for two backends:

- `compose`: the existing local vLLM + LiteLLM + Open WebUI stack
- `kubeai`: a K3s/Kubernetes + KubeAI deployment path for a single stable API endpoint and hardware-aware model placement

The intent is:

1. keep the current Compose path working
2. add a parallel KubeAI rendering path
3. store bootstrap and deployment scripts in the repo
4. check in example configs that can be copied into place

## Current workflow

### Compose backend

```bash
python manage.py init
# edit config.yaml and models.yaml
python manage.py render
python manage.py up
```

### KubeAI backend

```bash
cp examples/single-node/config.yaml ./config.yaml
cp examples/single-node/models.yaml ./models.yaml
# edit hostname / node selector / model ids
python manage.py render
bash scripts/bootstrap_k3s.sh
python manage.py deploy
python manage.py status
```

If you are not using ingress yet, port-forward the service and smoke test it:

```bash
kubectl -n kubeai port-forward svc/kubeai 8000:80
python manage.py smoke-test --base-url http://127.0.0.1:8000/openai/v1
```

## What render produces

### Compose

- `generated/plan.yaml`
- `generated/docker-compose.yml`
- `generated/.env`
- runtime files under `state.runtime`

### KubeAI

- `generated/plan.yaml`
- `generated/kubeai/namespace.yaml`
- `generated/kubeai/kubeai-values.yaml`
- `generated/kubeai/models.yaml`
- `generated/kubeai/ingress.yaml` if enabled

## Commands

```bash
python manage.py init
python manage.py render
python manage.py deploy
python manage.py status
python manage.py smoke-test
python manage.py switch <profile> --apply
python manage.py list-models
python manage.py list-profiles
```

## Repo additions for KubeAI

- `scripts/bootstrap_k3s.sh`: install single-node K3s and copy kubeconfig locally
- `scripts/install_kubeai.sh`: install or upgrade the KubeAI Helm release using generated values
- `scripts/join_agent.sh`: join another machine as a K3s agent later
- `examples/single-node/`: example config and models for a single big GPU server

## Notes

- `deploy` uses the backend in `config.yaml`
- `up` and `down` only apply to the Compose backend
- `deploy` for KubeAI applies the namespace, installs or upgrades the Helm chart, and applies the rendered `Model` objects
- the single-node example uses real Qwen2.5 model ids as defaults; adjust them if you want different variants
