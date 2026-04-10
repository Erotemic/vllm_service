# syntax=docker/dockerfile:1.6
FROM vllm/vllm-openai:v0.19.0
RUN pip install --no-cache-dir --upgrade transformers

################
### __DOCS__ ###
################
RUN <<EOF
echo '

docker build -t local/vllm-patched:latest -f vllm-hack.dockerfile .

' > /dev/null
EOF

