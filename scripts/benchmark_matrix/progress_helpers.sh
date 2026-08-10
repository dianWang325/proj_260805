#!/usr/bin/env bash

progress_format_seconds() {
    local total_seconds="${1:-0}"
    if ((total_seconds < 0)); then
        total_seconds=0
    fi
    printf '%02d:%02d:%02d' \
        $((total_seconds / 3600)) \
        $(((total_seconds % 3600) / 60)) \
        $((total_seconds % 60))
}

progress_emit() {
    local completed="$1"
    local total="$2"
    local start_epoch="$3"
    local label="$4"
    local log_path="${5:-}"
    local width=30
    local now elapsed filled empty percentage_tenths eta_seconds
    local filled_bar empty_bar elapsed_text eta_text line

    now="$(date +%s)"
    elapsed=$((now - start_epoch))
    if ((total <= 0)); then
        total=1
    fi
    if ((completed < 0)); then
        completed=0
    elif ((completed > total)); then
        completed="${total}"
    fi

    filled=$((completed * width / total))
    empty=$((width - filled))
    percentage_tenths=$((completed * 1000 / total))
    printf -v filled_bar '%*s' "${filled}" ''
    printf -v empty_bar '%*s' "${empty}" ''
    filled_bar="${filled_bar// /#}"
    empty_bar="${empty_bar// /-}"
    elapsed_text="$(progress_format_seconds "${elapsed}")"

    if ((completed > 0)); then
        eta_seconds=$((elapsed * (total - completed) / completed))
        eta_text="$(progress_format_seconds "${eta_seconds}")"
    else
        eta_text="estimating"
    fi

    printf -v line \
        '[%s%s] %3d.%d%% | %d/%d | elapsed %s | rough ETA %s | %s' \
        "${filled_bar}" "${empty_bar}" \
        $((percentage_tenths / 10)) $((percentage_tenths % 10)) \
        "${completed}" "${total}" "${elapsed_text}" "${eta_text}" "${label}"
    printf '%s\n' "${line}"
    if [[ -n "${log_path}" ]]; then
        printf '[%s] %s\n' "$(date --iso-8601=seconds)" "${line}" >>"${log_path}"
    fi
}
