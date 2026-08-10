#!/usr/bin/env bash

set -Eeuo pipefail

BASE_DIR="${BASE_DIR:-/home/w00985415/proj_260805}"
MODEL_NAME="${MODEL_NAME:-dsv4}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/a800_weight/DeepSeek-V4-Flash-w8a8-mtp}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8013}"
VARIANT="${VARIANT:?VARIANT must be set}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR must be set}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR must be set}"

WORKLOADS="${WORKLOADS:-fixed_long,variable_long,srf_mixed}"
REPEATS="${REPEATS:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-0}"
SRF_THRESHOLD="${SRF_THRESHOLD:-4096}"

FIXED_PROMPTS="${FIXED_PROMPTS:-2}"
FIXED_CONCURRENCY="${FIXED_CONCURRENCY:-4}"
VARIABLE_PROMPTS="${VARIABLE_PROMPTS:-4}"
VARIABLE_CONCURRENCY="${VARIABLE_CONCURRENCY:-4}"
SRF_MIXED_PROMPTS="${SRF_MIXED_PROMPTS:-8}"
SRF_MIXED_CONCURRENCY="${SRF_MIXED_CONCURRENCY:-16}"

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

check_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
}

check_positive_integer "REPEATS" "${REPEATS}"
check_nonnegative_integer "NUM_WARMUPS" "${NUM_WARMUPS}"

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "ERROR: vllm executable was not found: ${VLLM_BIN}" >&2
    exit 127
fi
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
    echo "ERROR: tokenizer directory does not exist: ${TOKENIZER_PATH}" >&2
    exit 2
fi
if [[ ! -f "${DATASET_DIR}/dataset_manifest.json" ]]; then
    echo "ERROR: dataset manifest was not found: ${DATASET_DIR}" >&2
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
    local dataset_path="$4"
    local max_concurrency="$5"

    local stem="${VARIANT}_${workload}_r${repeat}"
    local result_file="${RESULT_DIR}/${stem}.json"
    local client_log="${RESULT_DIR}/${stem}.log"
    local command_file="${RESULT_DIR}/${stem}.command.txt"

    if [[ -s "${result_file}" ]]; then
        echo "ERROR: refusing to overwrite existing result: ${result_file}" >&2
        exit 2
    fi
    if [[ ! -s "${dataset_path}" ]]; then
        echo "ERROR: dataset does not exist or is empty: ${dataset_path}" >&2
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
        --dataset-name custom
        --dataset-path "${dataset_path}"
        --skip-chat-template
        --disable-shuffle
        --num-prompts "${num_prompts}"
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
        "dataset_path=${dataset_path}"
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
        fixed_long)
            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
                run_one fixed_long "${repeat}" "${FIXED_PROMPTS}" \
                    "${DATASET_DIR}/fixed_64k.jsonl" "${FIXED_CONCURRENCY}"
            done
            ;;
        variable_long)
            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
                run_one variable_long "${repeat}" "${VARIABLE_PROMPTS}" \
                    "${DATASET_DIR}/variable_40k_80k.jsonl" \
                    "${VARIABLE_CONCURRENCY}"
            done
            ;;
        srf_mixed)
            for ((repeat = 1; repeat <= REPEATS; repeat++)); do
                run_one srf_mixed "${repeat}" "${SRF_MIXED_PROMPTS}" \
                    "${DATASET_DIR}/srf_mixed.jsonl" "${SRF_MIXED_CONCURRENCY}"
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
