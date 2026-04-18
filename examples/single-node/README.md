# Single-node KubeAI example

Copy these files into the repo root:

```bash
cp examples/single-node/config.yaml ./config.yaml
cp examples/single-node/models.yaml ./models.yaml
```

Then edit:

- `resource_profiles.blackwell-96gb.node_selector.kubernetes.io/hostname`
- any Hugging Face model ids you want to use
- `cluster.ingress.host` if you want an Ingress on your LAN

Typical flow:

```bash
python manage.py render
bash scripts/bootstrap_k3s.sh
python manage.py deploy
python manage.py status
python manage.py smoke-test --base-url http://127.0.0.1:8000/openai/v1
```

If you are not using ingress yet, port-forward first:

```bash
kubectl -n kubeai port-forward svc/kubeai 8000:80
```
