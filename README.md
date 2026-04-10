# vLLM + LiteLLM stack generator

This project manages a local or remote-targeted multi-model stack using a
simple user-facing pipeline:

1. `config.yaml` — portable intent and profile selection
2. `generated/plan.yaml` — resolved deployment plan plus validation metadata
3. `generated/docker-compose.yml` + files under `state.runtime` — runtime artifacts

## Quick start

```bash
python manage.py init
# edit config.yaml
python manage.py render
python manage.py up
```

To switch to another profile:

```bash
python manage.py switch gemma-mixed --apply
python manage.py switch gemma4-26b-max-context --apply
python manage.py switch gemma4-26b-3090-workstation --apply
```

## Simplified pipeline

`render` is the main step. It does the internal work needed to go from source
config to runnable artifacts:

- resolve the active config and profile
- validate the resolved deployment
- write `generated/plan.yaml`
- render `generated/docker-compose.yml`
- write mounted runtime files under `state.runtime`

That means the primary path is:

```bash
python manage.py init
# edit config.yaml
python manage.py render
python manage.py up
```

## Advanced commands

These still exist for debugging and inspection, but they are not the normal path:

```bash
python manage.py resolve
python manage.py validate
python manage.py lock
python manage.py explain
```

`lock` is now mostly a compatibility/debug command. The main generated artifact
is `generated/plan.yaml`.

## Design goals

- portable source config
- a single explicit generated plan artifact
- validation before Compose rendering
- multiple pinned vLLM backends behind one LiteLLM endpoint when needed
- built-in model catalog plus user overrides in `models.yaml`

## Files

- `config.yaml`: editable portable config
- `models.yaml`: optional user model/profile overrides
- `generated/plan.yaml`: resolved deployment plan plus validation metadata
- `generated/docker-compose.yml`: rendered Compose stack
- `state.runtime/.env`: runtime env file consumed by Compose
- `state.runtime/litellm_config.yaml`: mounted LiteLLM routing config

## Pinned images

- vLLM: `vllm/vllm-openai:v0.19.0`
- LiteLLM: `ghcr.io/berriai/litellm:v1.81.3-stable`
- Open WebUI: `ghcr.io/open-webui/open-webui:v0.8.6`
- Postgres: `postgres:16.8`

## Notes

- `render` writes `generated/plan.yaml`, `generated/docker-compose.yml`, and the mounted runtime files under `state.runtime`.
- `state.runtime` defaults to `/data/service/docker/vllm-stack/runtime` when `/data/service/docker` exists, otherwise `./state/runtime`.
- `render` refuses to proceed if validation reports hard errors.
- Use `--allow-unsupported` when you intentionally want to generate a plan for
  another machine or a speculative configuration.
- `docker-compose.yml` is a concrete runtime artifact, but `plan.yaml` remains
  the better source of truth for understanding how the deployment was resolved.

## Check vLLM container

```bash
source .env || source generated/.env 

# Get the container name (which will be different based on the model)
VLLM_CONTAINER_NAME=$(docker ps --filter "name=^vllm" --format '{{json .Names}}' | jq -s '.' | jq -r '.[0]')
echo "VLLM_CONTAINER_NAME=$VLLM_CONTAINER_NAME"

# Health check
curl -w "\n%{http_code}\n" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_BACKEND_API_KEY" \
  0.0.0.0:18000/health

# List available models
docker exec -it "${VLLM_CONTAINER_NAME}" curl \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_BACKEND_API_KEY" \
  0.0.0.0:8000/v1/models | jq

# We do not have much freedom to choose a model; use whatever is currently being served.
MODEL_NAME=$(docker exec -it "${VLLM_CONTAINER_NAME}" curl -H "Content-Type: application/json" -H "Authorization: Bearer $VLLM_BACKEND_API_KEY" 0.0.0.0:8000/v1/models | jq -r ".data[0].id")

echo "=="
echo "MODEL_NAME=$MODEL_NAME"

curl -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_BACKEND_API_KEY" \
  http://127.0.0.1:18000/v1/chat/completions \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Tell me about yourself."}
    ],
    "max_tokens": 1024
  }' | jq
```

## Test whether the Responses API works

```bash
mkdir -p response-test
cd response-test

# 1) First response call
curl -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_BACKEND_API_KEY" \
  http://127.0.0.1:18000/v1/responses \
  -d '{
    "model": "'"$MODEL_NAME"'",
    "input": "Tell me a short joke about programming."
  }' \
  | tee 01-response-raw.json | jq | tee 01-response-pretty.json

# 2) Extract the response id
RESP_ID=$(jq -r '.id' 01-response-raw.json)
echo "RESP_ID=$RESP_ID" | tee 02-response-id.txt

# 3) Continue using previous_response_id
curl -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_BACKEND_API_KEY" \
  http://127.0.0.1:18000/v1/responses \
  -d '{
    "model": "'"$MODEL_NAME"'",
    "previous_response_id": "'"$RESP_ID"'",
    "input": "Explain why that joke is funny in one sentence."
  }' \
  | tee 03-followup-raw.json | jq | tee 03-followup-pretty.json
```

## Check LiteLLM container

```bash
source .env || source generated/.env 

# Get the container name (which will be different based on the model)
VLLM_CONTAINER_NAME=$(docker ps --filter "name=^vllm" --format '{{json .Names}}' | jq -s '.' | jq -r '.[0]')
echo "VLLM_CONTAINER_NAME=$VLLM_CONTAINER_NAME"

# Use the litellm to get the model name it uses
MODEL_NAME=$(curl -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  http://127.0.0.1:14000/v1/models | jq -r ".data[0].id")

echo "=="
echo "MODEL_NAME=$MODEL_NAME"


# Check litellm has the right key for vllm:
docker exec -it litellm printenv VLLM_BACKEND_API_KEY

curl -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  http://0.0.0.0:14000/v1/chat/completions \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Tell me about yourself."}
    ],
    "max_tokens": 1024
  }' | jq


curl -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  http://0.0.0.0:14000/v1/chat/completions \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Tell me a story."}
    ],
    "max_tokens": 1024
  }' | jq




curl -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  http://0.0.0.0:14000/v1/chat/completions \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Tell me a joke."}
    ],
    "max_tokens": 1024
  }' | jq





 
curl -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  http://127.0.0.1:14000/v1/chat/completions \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Tell me about yourself."}
    ],
    "max_tokens": 1024
  }' | jq


docker run --rm --entrypoint python $VLLM_CONTAINER_NAME -m vllm.entrypoints.openai.api_server --help | grep -A4 tool-call-parser
```
