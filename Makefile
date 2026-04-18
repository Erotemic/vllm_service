PYTHON ?= python
EXAMPLE ?= single-node

init:
	$(PYTHON) manage.py init

render:
	$(PYTHON) manage.py render

deploy:
	$(PYTHON) manage.py deploy

status:
	$(PYTHON) manage.py status

smoke-test:
	$(PYTHON) manage.py smoke-test

example-single-node:
	cp examples/single-node/config.yaml ./config.yaml
	cp examples/single-node/models.yaml ./models.yaml
	@echo "Copied single-node example into repo root. Edit hostname/model ids as needed."

bootstrap-k3s:
	bash scripts/bootstrap_k3s.sh

install-kubeai:
	bash scripts/install_kubeai.sh generated/kubeai/kubeai-values.yaml kubeai
