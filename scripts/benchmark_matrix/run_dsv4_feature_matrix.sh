#!/usr/bin/env bash

set -Eeuo pipefail

BASE_DIR="${BASE_DIR:-/home/w00985415/proj_260805}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="${START_SCRIPT:-${BASE_DIR}/scripts/start_dsv4_cpp_srf.sh}"
WORKLOAD_SCRIPT="${WORKLOAD_SCRIPT:-${SCRIPT_DIR}/run_serving_workloads.sh}"
GENERATOR_SCRIPT="${GENERATOR_SCRIPT:-${SCRIPT_DIR}/generate_long_sequence_datasets.py}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-${SCRIPT_DIR}/analyze_serving_results.py}"

MATRIX_MODE="${MATRIX_MODE:-smoke}"
VARIANTS="${VARIANTS:-baseline,cpp_only,srf_only,cpp_srf}"
WORKLOADS="${WORKLOADS:-fixed_long,variable_long,srf_mixed}"
VLLM_PORT="${VLLM_PORT:-8013}"
START_TIMEOUT_S="${START_TIMEOUT_S:-1800}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-120}"
STABILIZE_SECONDS="${STABILIZE_SECONDS:-10}"
SRF_THRESHOLD="${SRF_THRESHOLD:-4096}"
DEVICE_COUNT="${DEVICE_COUNT:-8}"

FIXED_INPUT_LEN="${FIXED_INPUT_LEN:-65536}"
VARIABLE_MIN_LEN="${VARIABLE_MIN_LEN:-40960}"
VARIABLE_MAX_LEN="${VARIABLE_MAX_LEN:-81920}"
VARIABLE_MEAN_LEN="${VARIABLE_MEAN_LEN:-65536}"
VARIABLE_SIGMA="${VARIABLE_SIGMA:-10240}"
LONG_OUTPUT_LEN="${LONG_OUTPUT_LEN:-2560}"
SRF_SHORT_INPUT_LEN="${SRF_SHORT_INPUT_LEN:-1024}"
SRF_LONG_INPUT_LEN="${SRF_LONG_INPUT_LEN:-65536}"
SRF_OUTPUT_LEN="${SRF_OUTPUT_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-90112}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

case "${MATRIX_MODE}" in
    smoke)
        REPEATS="${REPEATS:-1}"
        NUM_WARMUPS="${NUM_WARMUPS:-0}"
        FIXED_PROMPTS="${FIXED_PROMPTS:-2}"
        VARIABLE_PROMPTS="${VARIABLE_PROMPTS:-4}"
        SRF_MIXED_PROMPTS="${SRF_MIXED_PROMPTS:-8}"
        ;;
    formal)
        REPEATS="${REPEATS:-3}"
        NUM_WARMUPS="${NUM_WARMUPS:-1}"
        FIXED_PROMPTS="${FIXED_PROMPTS:-12}"
        VARIABLE_PROMPTS="${VARIABLE_PROMPTS:-24}"
        SRF_MIXED_PROMPTS="${SRF_MIXED_PROMPTS:-32}"
        ;;
    *)
        echo "ERROR: MATRIX_MODE must be smoke or formal, got: ${MATRIX_MODE}" >&2
        exit 2
        ;;
esac

FIXED_CONCURRENCY="${FIXED_CONCURRENCY:-4}"
VARIABLE_CONCURRENCY="${VARIABLE_CONCURRENCY:-4}"
SRF_MIXED_CONCURRENCY="${SRF_MIXED_CONCURRENCY:-16}"

required_max_model_len=$((VARIABLE_MAX_LEN + LONG_OUTPUT_LEN))
if ((FIXED_INPUT_LEN + LONG_OUTPUT_LEN > required_max_model_len)); then
    required_max_model_len=$((FIXED_INPUT_LEN + LONG_OUTPUT_LEN))
fi
if ((SRF_LONG_INPUT_LEN + SRF_OUTPUT_LEN > required_max_model_len)); then
    required_max_model_len=$((SRF_LONG_INPUT_LEN + SRF_OUTPUT_LEN))
fi
if ((MAX_MODEL_LEN < required_max_model_len)); then
    echo "ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} is below required ${required_max_model_len}." >&2
    exit 2
fi

MATRIX_ID="${MATRIX_ID:-$(date '+%Y%m%d_%H%M%S')}"
MATRIX_ROOT="${MATRIX_ROOT:-${BASE_DIR}/artifacts/benchmark_matrix/matrix_${MATRIX_ID}}"
MATRIX_CONFIG="${MATRIX_ROOT}/matrix-config.txt"
DATASET_DIR="${MATRIX_ROOT}/datasets"

SERVICE_PID=""
SERVICE_PGID=""
CURRENT_VARIANT=""

health_check() {
    curl --noproxy '*' --fail --silent --show-error --max-time 3 \
        "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1
}

service_group_is_alive() {
    [[ -n "${SERVICE_PGID}" ]] && kill -0 -- "-${SERVICE_PGID}" 2>/dev/null
}

stop_service() {
    if [[ -z "${SERVICE_PGID}" ]]; then
        return
    fi
    echo "[$(date --iso-8601=seconds)] Stopping ${CURRENT_VARIANT} (process group ${SERVICE_PGID})"
    kill -TERM -- "-${SERVICE_PGID}" 2>/dev/null || true
    local waited=0
    while service_group_is_alive && ((waited < STOP_TIMEOUT_S)); do
        sleep 2
        waited=$((waited + 2))
    done
    if service_group_is_alive; then
        echo "WARNING: service did not stop after ${STOP_TIMEOUT_S}s; sending SIGKILL" >&2
        kill -KILL -- "-${SERVICE_PGID}" 2>/dev/null || true
        sleep 5
    fi
    wait "${SERVICE_PID}" 2>/dev/null || true
    SERVICE_PID=""
    SERVICE_PGID=""
    CURRENT_VARIANT=""
    local port_waited=0
    while health_check && ((port_waited < 60)); do
        sleep 2
        port_waited=$((port_waited + 2))
    done
}

on_exit() {
    local status=$?
    trap - EXIT INT TERM
    stop_service
    exit "${status}"
}
trap on_exit EXIT INT TERM

variant_flags() {
    case "$1" in
        baseline) printf 'false false\n' ;;
        cpp_only) printf 'true false\n' ;;
        srf_only) printf 'false true\n' ;;
        cpp_srf) printf 'true true\n' ;;
        *) echo "ERROR: unsupported variant: $1" >&2; exit 2 ;;
    esac
}

wait_for_service() {
    local launch_log="$1"
    local waited=0
    while ((waited < START_TIMEOUT_S)); do
        if health_check; then
            echo "[$(date --iso-8601=seconds)] ${CURRENT_VARIANT} is healthy"
            sleep "${STABILIZE_SECONDS}"
            return 0
        fi
        if ! service_group_is_alive; then
            echo "ERROR: ${CURRENT_VARIANT} exited before becoming healthy" >&2
            tail -n 160 "${launch_log}" >&2 || true
            return 1
        fi
        if ((waited % 30 == 0)); then
            echo "[$(date --iso-8601=seconds)] Waiting for ${CURRENT_VARIANT}: ${waited}/${START_TIMEOUT_S}s"
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "ERROR: ${CURRENT_VARIANT} did not become healthy within ${START_TIMEOUT_S}s" >&2
    tail -n 160 "${launch_log}" >&2 || true
    return 1
}

start_variant() {
    local variant="$1"
    local cpp_enabled="$2"
    local srf_enabled="$3"
    local variant_dir="$4"
    local launch_log="${variant_dir}/service-launch.log"
    local service_artifacts="${variant_dir}/service"

    CURRENT_VARIANT="${variant}"
    mkdir -p "${variant_dir}/results" "${service_artifacts}"
    echo "[$(date --iso-8601=seconds)] Starting ${variant}: CPP=${cpp_enabled}, SRF=${srf_enabled}"
    setsid env \
        BASE_DIR="${BASE_DIR}" \
        VLLM_PORT="${VLLM_PORT}" \
        CPP_ENABLED="${cpp_enabled}" \
        SRF_ENABLED="${srf_enabled}" \
        SRF_THRESHOLD="${SRF_THRESHOLD}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
        RUN_VARIANT="matrix_${MATRIX_ID}_${variant}" \
        ARTIFACT_ROOT="${service_artifacts}" \
        bash "${START_SCRIPT}" >"${launch_log}" 2>&1 &
    SERVICE_PID=$!
    SERVICE_PGID="$(ps -o pgid= -p "${SERVICE_PID}" | tr -d '[:space:]')"
    if [[ -z "${SERVICE_PGID}" ]]; then
        echo "ERROR: failed to determine process group for ${variant}" >&2
        return 1
    fi
    wait_for_service "${launch_log}"
}

for path in "${START_SCRIPT}" "${WORKLOAD_SCRIPT}" "${GENERATOR_SCRIPT}"; do
    if [[ ! -x "${path}" ]]; then
        echo "ERROR: required executable was not found: ${path}" >&2
        exit 2
    fi
done
if [[ ! -f "${ANALYZER_SCRIPT}" ]]; then
    echo "ERROR: analyzer was not found: ${ANALYZER_SCRIPT}" >&2
    exit 2
fi
if health_check; then
    echo "ERROR: port ${VLLM_PORT} already has a healthy service." >&2
    echo "Stop the manually-started service before running the matrix." >&2
    exit 2
fi

mkdir -p "${MATRIX_ROOT}"
{
    echo "matrix_id=${MATRIX_ID}"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "matrix_mode=${MATRIX_MODE}"
    echo "variants=${VARIANTS}"
    echo "workloads=${WORKLOADS}"
    echo "repeats=${REPEATS}"
    echo "num_warmups=${NUM_WARMUPS}"
    echo "fixed_prompts=${FIXED_PROMPTS}"
    echo "variable_prompts=${VARIABLE_PROMPTS}"
    echo "srf_mixed_prompts=${SRF_MIXED_PROMPTS}"
    echo "fixed_input_len=${FIXED_INPUT_LEN}"
    echo "variable_min_len=${VARIABLE_MIN_LEN}"
    echo "variable_max_len=${VARIABLE_MAX_LEN}"
    echo "variable_mean_len=${VARIABLE_MEAN_LEN}"
    echo "variable_sigma=${VARIABLE_SIGMA}"
    echo "long_output_len=${LONG_OUTPUT_LEN}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "max_num_seqs=${MAX_NUM_SEQS}"
    echo "device_count=${DEVICE_COUNT}"
    echo "srf_threshold=${SRF_THRESHOLD}"
    git -C "${BASE_DIR}/vllm" rev-parse HEAD 2>/dev/null | sed 's/^/vllm_commit=/' || true
    git -C "${BASE_DIR}/vllm-ascend-804317471" rev-parse HEAD 2>/dev/null | sed 's/^/vllm_ascend_commit=/' || true
    /usr/local/python3.11.10/bin/python3.11 -m pip show \
        vllm vllm-ascend transformers 2>/dev/null \
        | grep -E '^(Name|Version|Editable project location):' || true
} | tee "${MATRIX_CONFIG}"

echo "[$(date --iso-8601=seconds)] Generating exact-token datasets"
/usr/local/python3.11.10/bin/python3.11 "${GENERATOR_SCRIPT}" \
    --tokenizer /mnt/a800_weight/DeepSeek-V4-Flash-w8a8-mtp \
    --output-dir "${DATASET_DIR}" \
    --fixed-count "${FIXED_PROMPTS}" \
    --variable-count "${VARIABLE_PROMPTS}" \
    --srf-mixed-count "${SRF_MIXED_PROMPTS}" \
    --fixed-input-len "${FIXED_INPUT_LEN}" \
    --variable-min-len "${VARIABLE_MIN_LEN}" \
    --variable-max-len "${VARIABLE_MAX_LEN}" \
    --variable-mean-len "${VARIABLE_MEAN_LEN}" \
    --variable-sigma "${VARIABLE_SIGMA}" \
    --output-len "${LONG_OUTPUT_LEN}" \
    --srf-short-input-len "${SRF_SHORT_INPUT_LEN}" \
    --srf-long-input-len "${SRF_LONG_INPUT_LEN}" \
    --srf-output-len "${SRF_OUTPUT_LEN}" \
    --seed 0 | tee "${MATRIX_ROOT}/dataset-generation.log"

IFS=',' read -r -a VARIANT_LIST <<<"${VARIANTS}"
for variant in "${VARIANT_LIST[@]}"; do
    read -r cpp_enabled srf_enabled < <(variant_flags "${variant}")
    variant_dir="${MATRIX_ROOT}/${variant}"
    start_variant "${variant}" "${cpp_enabled}" "${srf_enabled}" "${variant_dir}"
    env \
        BASE_DIR="${BASE_DIR}" \
        VLLM_PORT="${VLLM_PORT}" \
        VARIANT="${variant}" \
        RESULT_DIR="${variant_dir}/results" \
        DATASET_DIR="${DATASET_DIR}" \
        WORKLOADS="${WORKLOADS}" \
        REPEATS="${REPEATS}" \
        NUM_WARMUPS="${NUM_WARMUPS}" \
        FIXED_PROMPTS="${FIXED_PROMPTS}" \
        VARIABLE_PROMPTS="${VARIABLE_PROMPTS}" \
        SRF_MIXED_PROMPTS="${SRF_MIXED_PROMPTS}" \
        FIXED_CONCURRENCY="${FIXED_CONCURRENCY}" \
        VARIABLE_CONCURRENCY="${VARIABLE_CONCURRENCY}" \
        SRF_MIXED_CONCURRENCY="${SRF_MIXED_CONCURRENCY}" \
        SRF_THRESHOLD="${SRF_THRESHOLD}" \
        bash "${WORKLOAD_SCRIPT}" 2>&1 | tee "${variant_dir}/benchmark-suite.log"
    stop_service
done

/usr/local/python3.11.10/bin/python3.11 \
    "${ANALYZER_SCRIPT}" "${MATRIX_ROOT}" \
    --threshold "${SRF_THRESHOLD}" --device-count "${DEVICE_COUNT}"

echo "Matrix completed: ${MATRIX_ROOT}"
echo "Summary: ${MATRIX_ROOT}/matrix_summary.md"
echo "Acceptance report: ${MATRIX_ROOT}/acceptance_report.md"
