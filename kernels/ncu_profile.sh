#!/usr/bin/env bash
# ncu_profile.sh — profile a practice session's CUDA kernels with Nsight Compute.
#
# Usage:
#   ./kernels/ncu_profile.sh --session ./sessions/2026-04-02
#   ./kernels/ncu_profile.sh --session ./sessions/2026-04-02 --op gemm
#   ./kernels/ncu_profile.sh --session ./sessions/2026-04-02 --op gemm --backend std
#   ./kernels/ncu_profile.sh --session ./sessions/2026-04-02 --op gemm --metrics roofline
#
# Output:   <session>/profiles/<op>.ncu-rep   (or all.ncu-rep when no --op)
#
# View report:
#   ncu --import <report>.ncu-rep                   # Nsight Compute GUI
#   ncu --csv -i <report>.ncu-rep --page details    # CLI summary
#
# Metrics sets (--metrics):
#   full      — all counters: memory, compute, occupancy, roofline  [default]
#   roofline  — roofline analysis only (faster)
#   default   — basic SM/memory throughput (fastest)
#
# Permissions (WSL2 / Linux):
#   If you see "ERR_NVGPUCTRPERM", lower the perf paranoid level once per boot:
#     sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
#   Or run this script with sudo:
#     sudo env PATH="$PATH" ./kernels/ncu_profile.sh --session ...
set -euo pipefail

KERNELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$KERNELS_DIR/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

# ── Defaults ───────────────────────────────────────────────────────────────────
SESSION=""
OP=""
BACKEND="cuda"
METRICS_SET="full"
EXTRA_NCU_ARGS=()

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session)  SESSION="$2";      shift 2 ;;
        --op)       OP="$2";           shift 2 ;;
        --backend)  BACKEND="$2";      shift 2 ;;
        --metrics)  METRICS_SET="$2";  shift 2 ;;
        --)         shift; EXTRA_NCU_ARGS+=("$@"); break ;;
        -*)         echo "Unknown option: $1" >&2; exit 1 ;;
        *)          echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SESSION" ]]; then
    echo "Usage: $0 --session <path> [--op <op>] [--backend cuda|std|torch] [--metrics full|roofline|default]" >&2
    exit 1
fi

# ── Resolve paths ──────────────────────────────────────────────────────────────
SESSION_DIR="$(cd "$SESSION" && pwd)"
SESSION_NAME="$(basename "$SESSION_DIR")"
PROFILES_DIR="$SESSION_DIR/profiles"
mkdir -p "$PROFILES_DIR"

REPORT="$PROFILES_DIR/${OP:-all}"

# ── Check prerequisites ────────────────────────────────────────────────────────
if ! command -v ncu &>/dev/null; then
    echo "ERROR: 'ncu' not found in PATH." >&2
    echo "  Common locations:" >&2
    echo "    /usr/local/cuda/bin/ncu" >&2
    echo "    /opt/nvidia/nsight-compute/*/ncu" >&2
    echo "  Add the correct path to PATH and retry." >&2
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python venv not found at $PYTHON" >&2
    echo "  Run 'uv sync' in the repo root first." >&2
    exit 1
fi

# ── Print plan ─────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ncu profile"
echo "  session  : $SESSION_NAME"
echo "  op       : ${OP:-all}"
echo "  backend  : $BACKEND"
echo "  metrics  : $METRICS_SET"
echo "  output   : $REPORT.ncu-rep"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Build ncu command ──────────────────────────────────────────────────────────
NCU_CMD=(
    ncu
    --target-processes all       # capture child processes too
    --nvtx                       # show NVTX ranges in GUI timeline
    --profile-from-start no      # only profile inside cudaProfilerStart/Stop markers
    --set "$METRICS_SET"
    --replay-mode kernel         # replay each kernel independently (most accurate)
    --import-source yes          # embed source lines in report
    -o "$REPORT"
    --force-overwrite
)

# Pass-through args, e.g.: -- --kernel-name my_kernel
if [[ ${#EXTRA_NCU_ARGS[@]} -gt 0 ]]; then
    NCU_CMD+=("${EXTRA_NCU_ARGS[@]}")
fi

# ── Build profile.py command ───────────────────────────────────────────────────
PROFILE_CMD=(
    "$PYTHON"
    "$KERNELS_DIR/profile.py"
    --session "$SESSION_DIR"
    --backend "$BACKEND"
)
if [[ -n "$OP" ]]; then
    PROFILE_CMD+=(--op "$OP")
fi

# ── Run ────────────────────────────────────────────────────────────────────────
echo "$ ${NCU_CMD[*]} \\"
echo "    ${PROFILE_CMD[*]}"
echo ""

"${NCU_CMD[@]}" "${PROFILE_CMD[@]}"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Report saved: $REPORT.ncu-rep"
echo ""
echo "  View options:"
echo "    GUI : ncu --import $REPORT.ncu-rep"
echo "    CLI : ncu --csv -i $REPORT.ncu-rep --page details | head -60"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
