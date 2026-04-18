#!/usr/bin/env bash
set -euo pipefail

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

if systemctl is-active --quiet k3s; then
  echo "k3s is already active"
else
  curl -sfL https://get.k3s.io | sh -
fi

sudo mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
chmod 600 "$HOME/.kube/config"

echo "K3s installed. Kubeconfig copied to $HOME/.kube/config"
echo "Node token: /var/lib/rancher/k3s/server/node-token"
