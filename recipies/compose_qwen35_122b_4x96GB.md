# Compose guide: Qwen3.5-122B-A10B-FP8 on a 4x96GB host

This guide brings up a working **FP8** deployment of `Qwen/Qwen3.5-122B-A10B-FP8` on a machine with **4 x 96GB GPUs**.

This is the concise version for the configuration that worked:

- one model instance
- all 4 GPUs
- tensor parallel size 4
- native **262,144** context
- text-only mode
- OpenAI-compatible backend on `http://127.0.0.1:18000/v1`
- LiteLLM on `http://127.0.0.1:14000/v1`
- Open WebUI on `http://127.0.0.1:13000`

---

## 1. Go to the repo root

```bash
cd /path/to/vllm_service
```

Check the GPUs:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

You should see four GPUs with roughly 96GB each.

---

## 2. Initialize Compose config

```bash
python manage.py setup --backend compose
```

---

## 3. Write `models.yaml`

```bash
cat > models.yaml <<'EOF'
models:
  qwen3.5-122b-a10b-fp8-local:
    hf_model_id: Qwen/Qwen3.5-122B-A10B-FP8
    tokenizer_name: Qwen/Qwen3.5-122B-A10B-FP8
    served_model_name: qwen3.5-122b-a10b-fp8-262k
    family: qwen3.5
    modalities: [text]
    memory_class_gib: 80
    min_vram_gib_per_replica: 80
    preferred_gpu_count: 4
    context_window: 262144
    defaults:
      max_model_len: 262144
      gpu_memory_utilization: 0.95
      enable_prefix_caching: false
      max_num_batched_tokens: 1024
      max_num_seqs: 1

profiles:
  qwen3.5-122b-a10b-fp8-tp4-262k-local:
    description: "Single Qwen3.5-122B-A10B-FP8 service across all 4 GPUs at 262k context."
    vllm:
      enable_responses_api_store: false
      logging_level: INFO
    services:
      - service_name: qwen-122b-fp8
        model: qwen3.5-122b-a10b-fp8-local
        served_model_name: qwen3.5-122b-a10b-fp8-262k
        placement:
          strategy: exact
          gpu_indices: [0, 1, 2, 3]
        topology:
          tensor_parallel_size: 4
        runtime:
          max_model_len: 262144
          gpu_memory_utilization: 0.95
          max_num_batched_tokens: 1024
          max_num_seqs: 1
          enable_prefix_caching: false
        extra_args:
          - --language-model-only
          - --reasoning-parser
          - qwen3

    router:
      aliases:
        qwen3.5-122b-a10b-fp8-262k: qwen-122b-fp8
EOF
```

---

## 4. Switch to the profile

```bash
python manage.py switch qwen3.5-122b-a10b-fp8-tp4-262k-local
```

---

## 5. Validate and render

```bash
python manage.py validate
python manage.py render
```

---

## 6. Set the Hugging Face token

```bash
grep '^HF_TOKEN=' generated/.env
```

Edit `generated/.env` and set:

```text
HF_TOKEN=your_token_here
```

---

## 7. Increase the backend health-check warmup window

Edit `generated/docker-compose.yml` and change the `vllm-qwen-122b` health check to:

```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 || exit 1"]
  interval: 30s
  timeout: 10s
  retries: 60
  start_period: 1800s
```

---

## 8. Start only the backend first

```bash
cd generated
docker compose up -d postgres vllm-qwen-122b
```

Watch logs:

```bash
docker logs -f vllm-qwen-122b
```

Wait for:

- `Starting vLLM server on http://0.0.0.0:8000`
- `/health` returning `200 OK`

---

## 9. Verify the backend directly

Health:

```bash
curl http://127.0.0.1:18000/health
```

List models:

```bash
curl http://127.0.0.1:18000/v1/models   -H "Authorization: Bearer $(grep '^VLLM_BACKEND_API_KEY=' .env | cut -d= -f2-)"
```

You should see:

- `qwen3.5-122b-a10b-fp8-262k`

Direct test request:

```bash
curl http://127.0.0.1:18000/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer $(grep '^VLLM_BACKEND_API_KEY=' .env | cut -d= -f2-)"   -d '{
    "model": "qwen3.5-122b-a10b-fp8-262k",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

---

## 10. Start the full stack

```bash
docker compose up -d
```

Open:

- Open WebUI: `http://127.0.0.1:13000`
- LiteLLM API: `http://127.0.0.1:14000/v1`

If Open WebUI asks for an admin account on first run, create it.

---

## 11. Verify from the repo root

```bash
cd ..
python manage.py smoke-test   --model qwen3.5-122b-a10b-fp8-262k   --prompt "Say hello in one sentence."
```

---

## Notes

- This profile is for **one large long-context request at a time**.
- `max_num_seqs: 1` and `max_num_batched_tokens: 1024` are intentional.
- Thinking is enabled by default for Qwen3.5 unless the client disables it.
- The FP8 model name is exposed as `qwen3.5-122b-a10b-fp8-262k` so it is easy to distinguish in Open WebUI.

