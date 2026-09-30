#!/bin/sh
# Start the validated local r8 + v4 selector using existing, offline model files.
set -eu
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
EXP_DIR=${JEV_EXPERIMENTS_DIR:-"$REPO_DIR/../open-jev-experiments/exp"}
if [ -z "${OPEN_JEV_API_KEY:-}" ]; then
    KEY_DIR=${XDG_STATE_HOME:-"$HOME/.local/state"}/open-jev
    mkdir -p "$KEY_DIR"
    chmod 700 "$KEY_DIR"
    if [ ! -s "$KEY_DIR/api-key" ]; then
        (umask 077; "$REPO_DIR/.venv/bin/python" -c 'import secrets; print(secrets.token_hex(32))' > "$KEY_DIR/api-key")
    fi
    OPEN_JEV_API_KEY=$(cat "$KEY_DIR/api-key")
    export OPEN_JEV_API_KEY
    echo "Local API key stored at $KEY_DIR/api-key"
fi
export PYTHONPATH="$REPO_DIR/vendor:$REPO_DIR"
export MODEL_DIR="$REPO_DIR/model"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ENABLE_EXPERIMENTAL_SELECTOR=1 OPEN_JEV_PARSER=learned
export LEARNED_PARSER_DIR="$EXP_DIR/jevparse_r8a,$EXP_DIR/jevparse_r8b"
export LEARNED_PARSER_MIN_CONFIDENCE=0.9
export LEARNED_VERIFIER_DIR="$EXP_DIR/verifier_v4" LEARNED_VERIFIER_MIN=0.0
cd "$REPO_DIR"
exec "$REPO_DIR/.venv/bin/uvicorn" server:create_app --factory --host 127.0.0.1 --port "${PORT:-18080}" --no-access-log
