#!/usr/bin/env bash
# post_run_backup.sh - Copy a pipeline run from outputs/ to a backup directory.

set -euo pipefail

usage() {
    cat <<'USAGE_EOF'
Usage: scripts/post_run_backup.sh [<run_name>|<abs_path>]

Copies a run directory from outputs/ to the backup dir, preserving
timestamps and permissions. Verifies the file count matches after copy.

Arguments:
  <run_name>     Run directory name, relative to outputs/.
  <abs_path>     Absolute path to a run directory inside outputs/.
  (no args)      Uses the most recent run_* under outputs/.

Environment:
  BACKUP_DIR     Destination directory. Default: ~/backup_outputs.

Exit codes:
  0  success
  1  invalid arguments (--help also exits 0)
  2  run not found
  3  BACKUP_DIR not writable
  4  post-copy integrity check failed
USAGE_EOF
}

RUN_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "[post_run_backup] FAIL: unknown flag '$1'" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$RUN_ARG" ]]; then
                echo "[post_run_backup] FAIL: too many positional arguments" >&2
                usage >&2
                exit 1
            fi
            RUN_ARG="$1"
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"

if [[ ! -d "$OUTPUTS_DIR" ]]; then
    echo "[post_run_backup] FAIL: outputs/ directory not found at ${OUTPUTS_DIR}" >&2
    exit 2
fi

resolve_run_path() {
    local arg="$1"
    if [[ -z "$arg" ]]; then
        local latest
        latest="$(find "$OUTPUTS_DIR" -mindepth 1 -maxdepth 1 -type d -name 'run_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
        if [[ -z "$latest" ]]; then
            echo ""
            return
        fi
        echo "$latest"
        return
    fi
    if [[ "$arg" = /* ]]; then
        local abs_arg
        if [[ -d "$arg" ]]; then
            abs_arg="$(cd "$arg" 2>/dev/null && pwd)"
        else
            abs_arg="$arg"
        fi
        case "$abs_arg" in
            "$OUTPUTS_DIR"/*)
                echo "$abs_arg"
                return
                ;;
            *)
                echo ""
                return
                ;;
        esac
    fi
    echo "${OUTPUTS_DIR}/${arg}"
}

RUN_PATH="$(resolve_run_path "$RUN_ARG")"

if [[ -z "$RUN_PATH" || ! -d "$RUN_PATH" ]]; then
    if [[ -z "$RUN_ARG" ]]; then
        echo "[post_run_backup] FAIL: no run_* directories found under ${OUTPUTS_DIR}" >&2
    else
        echo "[post_run_backup] FAIL: run not found: ${RUN_ARG}" >&2
    fi
    exit 2
fi

RUN_NAME="$(basename "$RUN_PATH")"

BACKUP_DIR="${BACKUP_DIR:-${HOME}/backup_outputs}"

if [[ ! -d "$BACKUP_DIR" ]]; then
    if ! mkdir -p "$BACKUP_DIR" 2>/dev/null; then
        echo "[post_run_backup] FAIL: cannot create BACKUP_DIR '${BACKUP_DIR}'" >&2
        exit 3
    fi
fi

if [[ ! -w "$BACKUP_DIR" ]]; then
    echo "[post_run_backup] FAIL: BACKUP_DIR '${BACKUP_DIR}' is not writable" >&2
    exit 3
fi

DST_PATH="${BACKUP_DIR}/${RUN_NAME}"

if [[ -e "$DST_PATH" ]]; then
    rm -rf "$DST_PATH"
fi

if ! cp -a "$RUN_PATH" "$DST_PATH"; then
    echo "[post_run_backup] FAIL: cp -a '${RUN_PATH}' '${DST_PATH}' returned non-zero" >&2
    exit 4
fi

SRC_COUNT="$(find "$RUN_PATH" -type f 2>/dev/null | wc -l | tr -d ' ')"
DST_COUNT="$(find "$DST_PATH" -type f 2>/dev/null | wc -l | tr -d ' ')"

if [[ "$SRC_COUNT" != "$DST_COUNT" ]]; then
    echo "[post_run_backup] FAIL: integrity check failed (src=${SRC_COUNT} dst=${DST_COUNT})" >&2
    exit 4
fi

DST_SIZE_KB="$(du -sk "$DST_PATH" 2>/dev/null | awk '{print $1}')"
TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

echo "[post_run_backup] OK ${TS} src=${RUN_PATH} dst=${DST_PATH} files=${SRC_COUNT} size=${DST_SIZE_KB}KB"
exit 0
