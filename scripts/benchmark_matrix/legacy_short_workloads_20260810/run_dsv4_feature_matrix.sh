#!/usr/bin/env bash

set -Eeuo pipefail

BASE_DIR="${BASE_DIR:-/home/w00985415/proj_260805}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="${START_SCRIPT:-${BASE_DIR}/scripts/start_dsv4_cpp_srf.sh}"
WORKLOAD_SCRIPT="${WORKLOAD_SCRIPT:-${SCRIPT_DIR}/run_serving_workloads.sh}"
ANALYZER_SCRIPT="${ANALYZER_SCRIPT:-${SCRIPT_DIR}/analyze_serving_results.py}"

MATRIX_MODE="${MATRIX_MODE:-smoke}"
VARIANTS="${VARIANTS:-baseline,cpp_only,srf_only,cpp_srf}"
WORKLOADS="${WORKLOADS:-short,long,mixed}"
VLLM_PORT="${VLLM_PORT:-8013}"
START_TIMEOUT_S="${START_TIMEOUT_S:-1800}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-120}"
STABILIZE_SECONDS="${STABILIZE_SECONDS:-10}"
SRF_THRESHOLD="${SRF_THRESHOLD:-4096}"

case "${MATRIX_MODE}" in
    smoke)
        REPEATS="${REPEATS:-1}"
        NUM_WARMUPS="${NUM_WARMUPS:-2}"
        SHORT_PROMPTS="${SHORT_PROMPTS:-20}"
        LONG_PROMPTS="${LONG_PROMPTS:-8}"
        MIXED_PROMPTS="${MIXED_PROMPTS:-20}"
        ;;
    formal)
        REPEATS="${REPEATS:-3}"
        NUM_WARMUPS="${NUM_WARMUPS:-4}"
        SHORT_PROMPTS="${SHORT_PROMPTS:-100}"
        LONG_PROMPTS="${LONG_PROMPTS:-40}"
        MIXED_PROMPTS="${MIXED_PROMPTS:-100}"
        ;;
    *)
        echo "ERROR: MATRIX_MODE must be smoke or formal, got: ${MATRIX_MODE}" >&2
        exit 2
        ;;
esac

MATRIX_ID="${MATRIX_ID:-$(date '+%Y%m%d_%H%M%S')}"
MATRIX_ROOT="${MATRIX_ROOT:-${BASE_DIR}/artifacts/benchmark_matrix/matrix_${MATRIX_ID}}"
MATRIX_CONFIG="${MATRIX_ROOT}/matrix-config.txt"

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
        echo "WARNING: process group ${SERVICE_PGID} did not stop after ${STOP_TIMEOUT_S}s; sending SIGKILL" >&2
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
    local variant="$1"
    case "${variant}" in
        baseline) printf 'false false\n' ;;
        cpp_only) printf 'true false\n' ;;
        srf_only) printf 'false true\n' ;;
        cpp_srf) printf 'true true\n' ;;
        *)
            echo "ERROR: unsupported variant: ${variant}" >&2
            exit 2
            ;;
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

if [[ ! -x "${START_SCRIPT}" ]]; then
    echo "ERROR: start script is not executable: ${START_SCRIPT}" >&2
    exit 2
fi
if [[ ! -x "${WORKLOAD_SCRIPT}" ]]; then
    echo "ERROR: workload script is not executable: ${WORKLOAD_SCRIPT}" >&2
    exit 2
fi
if [[ ! -f "${ANALYZER_SCRIPT}" ]]; then
    echo "ERROR: analyzer script was not found: ${ANALYZER_SCRIPT}" >&2
    exit 2
fi

if health_check; then
    echo "ERROR: port ${VLLM_PORT} already has a healthy service." >&2
    echo "Stop the current manually-started service before running the matrix." >&2
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
    echo "short_prompts=${SHORT_PROMPTS}"
    echo "long_prompts=${LONG_PROMPTS}"
    echo "mixed_prompts=${MIXED_PROMPTS}"
    echo "srf_threshold=${SRF_THRESHOLD}"
    echo "vllm_port=${VLLM_PORT}"
    echo "start_script=${START_SCRIPT}"
    echo "workload_script=${WORKLOAD_SCRIPT}"
    echo "analyzer_script=${ANALYZER_SCRIPT}"
    git -C "${BASE_DIR}/vllm" rev-parse HEAD 2>/dev/null | sed 's/^/vllm_commit=/' || true
    git -C "${BASE_DIR}/vllm-ascend-804317471" rev-parse HEAD 2>/dev/null | sed 's/^/vllm_ascend_commit=/' || true
    /usr/local/python3.11.10/bin/python3.11 -m pip show \
        vllm vllm-ascend transformers 2>/dev/null \
        | grep -E '^(Name|Version|Editable project location):' || true
} | tee "${MATRIX_CONFIG}"

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
        WORKLOADS="${WORKLOADS}" \
        REPEATS="${REPEATS}" \
        NUM_WARMUPS="${NUM_WARMUPS}" \
        SHORT_PROMPTS="${SHORT_PROMPTS}" \
        LONG_PROMPTS="${LONG_PROMPTS}" \
        MIXED_PROMPTS="${MIXED_PROMPTS}" \
        SRF_THRESHOLD="${SRF_THRESHOLD}" \
        bash "${WORKLOAD_SCRIPT}" 2>&1 | tee "${variant_dir}/benchmark-suite.log"

    stop_service
done

/usr/local/python3.11.10/bin/python3.11 \
    "${ANALYZER_SCRIPT}" "${MATRIX_ROOT}" --threshold "${SRF_THRESHOLD}"

echo "Matrix completed: ${MATRIX_ROOT}"
echo "Summary: ${MATRIX_ROOT}/matrix_summary.md"
