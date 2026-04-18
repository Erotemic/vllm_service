#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <server-url> <node-token> [node-name]" >&2
  exit 1
fi

SERVER_URL="$1"
NODE_TOKEN="$2"
NODE_NAME="${3:-}"

if [[ -n "$NODE_NAME" ]]; then
  curl -sfL https://get.k3s.io | K3S_URL="$SERVER_URL" K3S_TOKEN="$NODE_TOKEN" K3S_NODE_NAME="$NODE_NAME" sh -
else
  curl -sfL https://get.k3s.io | K3S_URL="$SERVER_URL" K3S_TOKEN="$NODE_TOKEN" sh -
fi
