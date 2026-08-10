#!/usr/bin/env bash

# DeepSeek V4 Flash W8A8 CPP + ShortRequestFirst smoke-test service.
# Target stack:
#   vLLM 0.25.1+empty
#   vLLM Ascend 0.19.1rc2.dev1256+g804317471
#
# Parallel layout:
#   TP=4, PP=2, DP=1
#   Total NPUs: 4 * 2 * 1 = 8

set -Eeuo pipefail

BASE_DIR="${BASE_DIR:-/home/w00985415/proj_260805}"
MODEL_PATH="${MODEL_PATH:-/mnt/a800_weight/DeepSeek-V4-Flash-w8a8-mtp}"
MODEL_NAME="${MODEL_NAME:-dsv4}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8013}"

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-2}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

CPP_ENABLED="${CPP_ENABLED:-true}"
CPP_SMOOTH_FACTOR="${CPP_SMOOTH_FACTOR:-1.0}"
CPP_MIN_CHUNK="${CPP_MIN_CHUNK:-4096}"
CPP_NEED_TIMING="${CPP_NEED_TIMING:-true}"
CPP_MAX_FIT_CHUNK="${CPP_MAX_FIT_CHUNK:-30}"

SRF_ENABLED="${SRF_ENABLED:-true}"
SRF_THRESHOLD="${SRF_THRESHOLD:-4096}"
SRF_LONG_MAX_WAIT_MS="${SRF_LONG_MAX_WAIT_MS:-2000}"

RUN_VARIANT="${RUN_VARIANT:-dsv4_cpp_srf}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${BASE_DIR}/artifacts/${RUN_VARIANT}}"

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${ARTIFACT_ROOT}/run_${RUN_ID}"
SERVER_LOG="${RUN_DIR}/server.log"
RUN_CONFIG="${RUN_DIR}/run-config.txt"

check_boolean() {
    local name="$1"
    local value="$2"

    if [[ ! "${value}" =~ ^(true|false)$ ]]; then
        echo "ERROR: ${name} must be true or false, got: ${value}" >&2
        exit 2
    fi
}

check_non_negative_integer() {
    local name="$1"
    local value="$2"

    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
}

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm command was not found in the current environment." >&2
    exit 127
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2
    exit 2
fi

check_boolean "CPP_ENABLED" "${CPP_ENABLED}"
check_boolean "CPP_NEED_TIMING" "${CPP_NEED_TIMING}"
check_boolean "SRF_ENABLED" "${SRF_ENABLED}"

check_non_negative_integer "CPP_MIN_CHUNK" "${CPP_MIN_CHUNK}"
check_non_negative_integer "CPP_MAX_FIT_CHUNK" "${CPP_MAX_FIT_CHUNK}"
check_non_negative_integer "SRF_THRESHOLD" "${SRF_THRESHOLD}"

if [[ ! "${SRF_LONG_MAX_WAIT_MS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: SRF_LONG_MAX_WAIT_MS must be non-negative, got: ${SRF_LONG_MAX_WAIT_MS}" >&2
    exit 2
fi

if (( PIPELINE_PARALLEL_SIZE <= 1 )); then
    echo "ERROR: CPP requires PIPELINE_PARALLEL_SIZE > 1." >&2
    exit 2
fi

if (( CPP_MIN_CHUNK <= 0 || CPP_MIN_CHUNK > MAX_NUM_BATCHED_TOKENS )); then
    echo "ERROR: CPP_MIN_CHUNK must be greater than 0 and no larger than MAX_NUM_BATCHED_TOKENS." >&2
    exit 2
fi

if (( CPP_MAX_FIT_CHUNK <= 5 )); then
    echo "ERROR: CPP_MAX_FIT_CHUNK must be greater than 5; 30 or more is recommended." >&2
    exit 2
fi

IFS=',' read -r -a VISIBLE_DEVICE_LIST <<< "${ASCEND_RT_VISIBLE_DEVICES}"

REQUIRED_DEVICE_COUNT=$((
    TENSOR_PARALLEL_SIZE
    * PIPELINE_PARALLEL_SIZE
    * DATA_PARALLEL_SIZE
))

if (( ${#VISIBLE_DEVICE_LIST[@]} < REQUIRED_DEVICE_COUNT )); then
    echo "ERROR: TP*PP*DP=${REQUIRED_DEVICE_COUNT}, but only ${#VISIBLE_DEVICE_LIST[@]} devices were configured." >&2
    exit 2
fi

if command -v ss >/dev/null 2>&1; then
    if ss -ltn | grep -q ":${VLLM_PORT} "; then
        echo "ERROR: port ${VLLM_PORT} is already in use." >&2
        exit 2
    fi
fi

mkdir -p "${RUN_DIR}"

ADDITIONAL_CONFIG="$(
    printf \
        '{"scheduler_config":{"enable_balance_scheduling":false,"recompute_scheduler_enable":false,"short_request_first_config":{"enabled":%s,"threshold":%s,"long_max_wait_ms":%s},"batch_job_sched_config":{"enabled":false},"profiling_chunk_config":{"enabled":%s,"smooth_factor":%s,"min_chunk":%s,"need_timing":%s,"max_fit_chunk":%s}}}' \
        "${SRF_ENABLED}" \
        "${SRF_THRESHOLD}" \
        "${SRF_LONG_MAX_WAIT_MS}" \
        "${CPP_ENABLED}" \
        "${CPP_SMOOTH_FACTOR}" \
        "${CPP_MIN_CHUNK}" \
        "${CPP_NEED_TIMING}" \
        "${CPP_MAX_FIT_CHUNK}"
)"

export ASCEND_RT_VISIBLE_DEVICES
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"

# Keep unrelated scheduler/communication optimizations disabled during
# the first CPP+SRF correctness validation.

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-30000}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-204}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-120}"

# Dynamic CPP performs its own device synchronization during online timing.
unset ASCEND_LAUNCH_BLOCKING || true

JEMALLOC_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
if [[ -f "${JEMALLOC_PATH}" ]]; then
    export LD_PRELOAD="${JEMALLOC_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

VLLM_COMMAND=(
    vllm serve "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"

    --data-parallel-size "${DATA_PARALLEL_SIZE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
    --enable-expert-parallel

    --quantization ascend
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --trust-remote-code

    --model-loader-extra-config
    '{"enable_multithread_load":true,"num_threads":128}'

    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --block-size 128

    --enable-chunked-prefill
    --scheduling-policy fcfs
    --no-async-scheduling
    --no-enable-prefix-caching
    --enforce-eager

    --additional-config "${ADDITIONAL_CONFIG}"
    --enable-request-id-headers
    --uvicorn-log-level info
)

{
    echo "run_id=${RUN_ID}"
    echo "start_time=$(date --iso-8601=seconds)"
    echo "run_variant=${RUN_VARIANT}"
    echo "model_path=${MODEL_PATH}"
    echo "model_name=${MODEL_NAME}"
    echo "listen=${VLLM_HOST}:${VLLM_PORT}"
    echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
    echo "tp=${TENSOR_PARALLEL_SIZE}"
    echo "pp=${PIPELINE_PARALLEL_SIZE}"
    echo "dp=${DATA_PARALLEL_SIZE}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    echo "max_num_seqs=${MAX_NUM_SEQS}"
    echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    echo "cpp_enabled=${CPP_ENABLED}"
    echo "cpp_smooth_factor=${CPP_SMOOTH_FACTOR}"
    echo "cpp_min_chunk=${CPP_MIN_CHUNK}"
    echo "cpp_need_timing=${CPP_NEED_TIMING}"
    echo "cpp_max_fit_chunk=${CPP_MAX_FIT_CHUNK}"
    echo "srf_enabled=${SRF_ENABLED}"
    echo "srf_threshold=${SRF_THRESHOLD}"
    echo "srf_long_max_wait_ms=${SRF_LONG_MAX_WAIT_MS}"
    echo "additional_config=${ADDITIONAL_CONFIG}"

    python -m pip show vllm vllm-ascend 2>/dev/null |
        grep -E '^(Name|Version|Editable project location):' || true

    printf 'command='
    printf '%q ' "${VLLM_COMMAND[@]}"
    printf '\n'
} | tee "${RUN_CONFIG}"

echo "Run directory: ${RUN_DIR}"
echo "Configuration: ${RUN_CONFIG}"
echo "Server log: ${SERVER_LOG}"
echo "Health check: curl --noproxy '*' -sS http://127.0.0.1:${VLLM_PORT}/health"

"${VLLM_COMMAND[@]}" 2>&1 | tee "${SERVER_LOG}"