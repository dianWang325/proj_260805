#!/usr/bin/env bash

set -Eeuo pipefail

VLLM_BIN="/usr/local/python3.11.10/bin/vllm"
MODEL_PATH="/mnt/a800_weight/DeepSeek-V4-Flash-w8a8-mtp"
SERVER_URL="http://127.0.0.1:8013"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="/home/w00985415/proj_260805/artifacts/benchmarks/cpp_srf_${RUN_ID}"

mkdir -p "${RESULT_DIR}"

echo "Checking service health..."
curl --noproxy '*' -fsS "${SERVER_URL}/health" >/dev/null

echo "Service health check passed."
echo "Benchmark result directory: ${RESULT_DIR}"

"${VLLM_BIN}" bench serve \
  --backend openai \
  --base-url "${SERVER_URL}" \
  --endpoint /v1/completions \
  --model dsv4 \
  --tokenizer "${MODEL_PATH}" \
  --tokenizer-mode deepseek_v4 \
  --dataset-name random \
  --num-prompts 100 \
  --random-input-len 4096 \
  --random-output-len 128 \
  --random-range-ratio '{"input":0.75,"output":0.0}' \
  --request-rate inf \
  --max-concurrency 16 \
  --num-warmups 4 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result \
  --save-detailed \
  --result-dir "${RESULT_DIR}" \
  --result-filename cpp_srf_mixed_c16.json \
  --metadata \
    variant=cpp_srf \
    threshold=4096 \
    tp=4 \
    pp=2 \
    max_num_seqs=4 \
  2>&1 | tee "${RESULT_DIR}/benchmark.log"

echo "Benchmark completed."
echo "Metrics JSON: ${RESULT_DIR}/cpp_srf_mixed_c16.json"
echo "Console log: ${RESULT_DIR}/benchmark.log"