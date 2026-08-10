#!/usr/bin/env bash

set -Eeuo pipefail

BASE_DIR="${BASE_DIR:-/home/w00985415/proj_260805}"
MODEL_NAME="${MODEL_NAME:-dsv4}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/a800_weight/DeepSeek-V4-Flash-w8a8-mtp}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8013}"
VARIANT="${VARIANT:?VARIANT must be set}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR must be set}"

WORKLOADS="${WORKLOADS:-short,long,mixed}"
REPEATS="${REPEATS:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-0}"
SRF_THRESHOLD="${SRF_THRESHOLD:-4096}"

SHORT_PROMPTS="${SHORT_PROMPTS:-20}"
SHORT_INPUT_LEN="${SHORT_INPUT_LEN:-1024}"
SHORT_OUTPUT_LEN="${SHORT_OUTPUT_LEN:-128}"
SHORT_CONCURRENCY="${SHORT_CONCURRENCY:-16}"

LONG_PROMPTS="${LONG_PROMPTS:-8}"
LONG_INPUT_LEN="${LONG_INPUT_LEN:-16384}"
LONG_OUTPUT_LEN="${LONG_OUTPUT_LEN:-128}"
LONG_CONCURRENCY="${LONG_CONCURRENCY:-8}"

MIXED_PROMPTS="${MIXED_PROMPTS:-20}"
MIXED_INPUT_LEN="${MIXED_INPUT_LEN:-4096}"
MIXED_OUTPUT_LEN="${MIXED_OUTPUT_LEN:-128}"
MIXED_RANGE_RATIO="${MIXED_RANGE_RATIO:-}"
if [[ -z "${MIXED_RANGE_RATIO}" ]]; then
    MIXED_RANGE_RATIO='{"input":0.75,"output":0.0}'
fi
MIXED_CONCURRENCY="${MIXED_CONCURRENCY:-16}"

VLLM_BIN="${VLLM_BIN:-/usr/local/python3.11.10/bin/vllm}"
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"

check_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
}

check_positive_integer "REPEATS" "${REPEATS}"
check_positive_integer "NUM_WARMUPS" "${NUM_WARMUPS}"

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "ERROR: vllm executable was not found: ${VLLM_BIN}" >&2
    exit 127
fi
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
    echo "ERROR: tokenizer directory does not exist: ${TOKENIZER_PATH}" >&2
    exit 2
fi

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

if ! curl --noproxy '*' --fail --silent --show-error \
    --max-time 5 "${BASE_URL}/health" >/dev/null; then
    echo "ERROR: vLLM health check failed: ${BASE_URL}/health" >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"

run_one() {
    local workload="$1"
    local repeat="$2"
    local num_prompts="$3"
    local input_len="$4"
    local output_len="$5"
    local range_ratio="$6"
    local max_concurrency="$7"

    local stem="${VARIANT}_${workload}_r${repeat}"
    local result_file="${RESULT_DIR}/${stem}.json"
    local client_log="${RESULT_DIR}/${stem}.log"
    local command_file="${RESULT_DIR}/${stem}.command.txt"

    if [[ -s "${result_file}" ]]; then
        echo "ERROR: refusing to overwrite existing result: ${result_file}" >&2
        exit 2
    fi

    local -a command=(
        "${VLLM_BIN}" bench serve
        --backend openai
        --base-url "${BASE_URL}"
        --endpoint /v1/completions
        --model "${MODEL_NAME}"
        --tokenizer "${TOKENIZER_PATH}"
        --tokenizer-mode deepseek_v4
        --dataset-name random
        --num-prompts "${num_prompts}"
        --random-input-len "${input_len}"
        --random-output-len "${output_len}"
        --random-range-ratio "${range_ratio}"
        --request-rate "${REQUEST_RATE}"
        --max-concurrency "${max_concurrency}"
        --num-warmups "${NUM_WARMUPS}"
        --seed "${SEED}"
        --ignore-eos
        --percentile-metrics ttft,tpot,itl,e2el
        --metric-percentiles 50,90,95,99
        --save-result
        --save-detailed
        --result-dir "${RESULT_DIR}"
        --result-filename "${stem}.json"
        --metadata
        "variant=${VARIANT}"
        "workload=${workload}"
        "repeat=${repeat}"
        "threshold=${SRF_THRESHOLD}"
        "input_len=${input_len}"
        "output_len=${output_len}"
        "range_ratio=${range_ratio}"
        "max_concurrency=${max_concurrency}"
        "seed=${SEED}"
    )

    {
        printf 'command='
        printf '%q ' "${command[@]}"
        printf '\n'
    } >"${command_file}"

    echo "[$(date --iso-8601=seconds)] Starting ${stem}"
    "${command[@]}" 2>&1 | tee "${client_log}"
    if [[ ! -s "${result_file}" ]]; then
        echo "ERROR: benchmark did not create result: ${result_file}" >&2
        exit 1
    fi
    echo "[$(date --iso-8601=seconds)] Completed ${stem}"
}

run_workload() {
    local workload="$1"
    local repeat
    case "${workload}" in
        short)
            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
                run_one short "${repeat}" "${SHORT_PROMPTS}" \
                    "${SHORT_INPUT_LEN}" "${SHORT_OUTPUT_LEN}" \
                    0.0 "${SHORT_CONCURRENCY}"
            done
            ;;
        long)
            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
                run_one long "${repeat}" "${LONG_PROMPTS}" \
                    "${LONG_INPUT_LEN}" "${LONG_OUTPUT_LEN}" \
                    0.0 "${LONG_CONCURRENCY}"
            done
            ;;
        mixed)
            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
                run_one mixed "${repeat}" "${MIXED_PROMPTS}" \
                    "${MIXED_INPUT_LEN}" "${MIXED_OUTPUT_LEN}" \
                    "${MIXED_RANGE_RATIO}" "${MIXED_CONCURRENCY}"
            done
            ;;
        *)
            echo "ERROR: unsupported workload: ${workload}" >&2
            exit 2
            ;;
    esac
}

IFS=',' read -r -a WORKLOAD_LIST <<<"${WORKLOADS}"
for workload in "${WORKLOAD_LIST[@]}"; do
    run_workload "${workload}"
done

echo "All requested workloads completed. Results: ${RESULT_DIR}"
