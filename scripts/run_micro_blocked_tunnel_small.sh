#!/usr/bin/env bash
set -euo pipefail

MAZE="micro_blocked_tunnel_small"
EXPDIR="experiments/micro_blocked_tunnel_small_20251116T000000Z"
SEEDS=(0 1 2 3 4)
BASELINES_JSON="$EXPDIR/baselines.json"
MAX_CONCURRENCY=${MAX_CONCURRENCY:-6}  # default smaller to avoid duplicate wave; override via env

# Fail fast if no Azure key present
check_keys() {
  set -a && [ -f "$HOME/.env" ] && source "$HOME/.env" && set +a
  if [ -z "${AZURE_API_KEY:-}" ] && [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
    echo "ERROR: AZURE_API_KEY (or AZURE_OPENAI_API_KEY) missing. Add it to ~/.env before running." >&2
    exit 1
  fi
}

find_latest_run_dir() {
  local pattern="$1"
  local latest=""
  local candidate
  shopt -s nullglob
  for candidate in $pattern; do
    if [ -d "$candidate" ]; then
      latest="$candidate"
    fi
  done
  shopt -u nullglob
  echo "$latest"
}

run_one() {
  local name="$1" comm="$2" loop="$3" rules="$4" mapshare="$5" seed="$6"

  # Prefer resuming an existing run (runs/ then runs_inflight/) if a checkpoint is present.
  local existing
  existing=$(find_latest_run_dir "$EXPDIR/runs/${name}_seed${seed}_*" )
  if [ -z "$existing" ]; then
    existing=$(find_latest_run_dir "$EXPDIR/runs_inflight/${name}_seed${seed}_*" )
  fi

  local run_dir="$existing"
  local resume_from=""
  if [ -n "$existing" ] && [ -f "$existing/results/checkpoint.json" ]; then
    local turn_next turns_total
    turn_next=$(jq -r '.turn_next // 0' "$existing/results/checkpoint.json")
    turns_total=$(jq -r '.turns_total // 0' "$existing/results/checkpoint.json")
    if [ "$turn_next" -ge "$turns_total" ]; then
      echo "SKIP completed: $existing (turn $turn_next/$turns_total)"
      return
    fi
    resume_from="$existing/results/checkpoint.json"
    run_dir="$existing"
    echo "RESUME $name seed=$seed at turn $turn_next/$turns_total from $resume_from"
  fi

  if [ -z "$run_dir" ]; then
    run_dir="$EXPDIR/runs/${name}_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)"
  fi

  mkdir -p "$run_dir"
  mkdir -p "$run_dir/results"

  # Map rules_id to env var content
  export LLMGRID_STRUCTURED_EXTRA_RULES=""
  export LLMGRID_FREEFORM_EXTRA_RULES=""
  case "$rules" in
    none_default) : ;;
    safety_bias) export LLMGRID_STRUCTURED_EXTRA_RULES="- If last_move_outcome != OK, STAY for one turn before any new MOVE.\n- Prefer unexplored cells over backtracking when safe; if unsure, STAY.";\
                  export LLMGRID_FREEFORM_EXTRA_RULES="$LLMGRID_STRUCTURED_EXTRA_RULES" ;;
    ff_default) : ;;
    ff_cautious) export LLMGRID_FREEFORM_EXTRA_RULES="- Send CHAT only if contended_neighbors != 0 or last_move_outcome != OK.\n- If you cannot argue a move is safe, STAY and explain.";;
    ff_tag_light) export LLMGRID_FREEFORM_EXTRA_RULES="- CHAT format: [TAG payload] short note. TAG in {INTENT, YIELD, INFO}.\n- If you plan to enter a choke or contested cell, send [INTENT MOVE_dir].\n- If you choose to STAY in a choke, send [YIELD cell=(x,y)].\n- Receivers: honor YIELD for 1 turn; if lower id sent INTENT, yield.";;
    ff_tag_strict) export LLMGRID_FREEFORM_EXTRA_RULES="- CHAT format: [TAG payload] short note. TAG in {INTENT, YIELD, INFO, ROUTE}.\n- Always send [INTENT MOVE_dir] before entering the main corridor.\n- If two agents target same corridor, lower id sends INTENT, higher id STAYs 2 turns.\n- Honor any YIELD for 2 turns unless goal is adjacent and free.";;
    ff_kv_light) export LLMGRID_FREEFORM_EXTRA_RULES="- CHAT key=value: KIND=INTENT|YIELD|INFO; ACTION=MOVE_X or STAY; CELL=(x,y); NOTE=free text.\n- Send only when contended_neighbors != 0 or after a block.\n- Receivers: if KIND=YIELD and CELL matches your target, STAY 1 turn.";;
    ff_kv_strict) export LLMGRID_FREEFORM_EXTRA_RULES="- CHAT key=value required before moving into any one-cell choke.\n- If you send KIND=INTENT, include CELL target; lower id gets priority.\n- Receivers: if higher id sees lower id INTENT for same CELL, STAY 2 turns.";;
    struct_intent_only) export LLMGRID_STRUCTURED_EXTRA_RULES="- Use INTENT only; no REQUEST. Send INTENT before entering corridor cells.\n- If last_move_outcome != OK, STAY then resend INTENT with updated target.";;
    struct_yield_rule) export LLMGRID_STRUCTURED_EXTRA_RULES="- Lowest agent_id has right of way in chokes; higher ids send REQUEST:YIELD and STAY 1 turn.\n- If you receive REQUEST:YIELD, STAY unless you are lower id and already in the cell.";;
    struct_sparse_cautious) export LLMGRID_STRUCTURED_EXTRA_RULES="- Send INTENT or REQUEST only after a block or contended_neighbors != 0.\n- After sending a REQUEST, STAY until the corridor is free or 2 turns have passed.";;
    ff_priority_lowest_goes) export LLMGRID_FREEFORM_EXTRA_RULES="- Lower id always proceeds in a choke; higher id yields without sending unless ambiguous.\n- Only CHAT when divergence from this rule is necessary; keep messages ≤80 chars.";;
  esac

  (set -a && [ -f "$HOME/.env" ] && source "$HOME/.env" && set +a; \
   if [ -n "${AZURE_API_KEY:-}" ]; then export AZURE_OPENAI_API_KEY="$AZURE_API_KEY"; fi; \
   echo "DEBUG AZURE_OPENAI_API_KEY len=${#AZURE_OPENAI_API_KEY}"; \
   PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
    --model azure:gpt-5-mini \
    --maze-preset "$MAZE" \
    --agents 2 \
    --turns 50 \
    --seed "$seed" \
    --comm-strategy "$comm" \
    --loop-guidance "$loop" \
    --log-movements \
    --emit-config "$run_dir/config.yaml" \
    --episode-json "$run_dir/results/episode.json" \
    --transcript-jsonl "$run_dir/results/transcript.jsonl" \
    --checkpoint-json "$run_dir/results/checkpoint.json" \
    --map-sharing "$mapshare" \
    ${resume_from:+--resume-from "$resume_from"} \
    ) > "$run_dir/run.log" 2>&1
}

main() {
  check_keys
  local i=0
  jq -c '.[]' "$BASELINES_JSON" | while read -r cfg; do
    name=$(echo "$cfg" | jq -r '.name')
    comm=$(echo "$cfg" | jq -r '.comm_strategy')
    loop=$(echo "$cfg" | jq -r '.loop_guidance')
    rules=$(echo "$cfg" | jq -r '.rules_id')
    mapshare=$(echo "$cfg" | jq -r '.map_sharing // "none"')
    for seed in "${SEEDS[@]}"; do
      run_one "$name" "$comm" "$loop" "$rules" "$mapshare" "$seed" &
      ((i++))
      if (( i % MAX_CONCURRENCY == 0 )); then
        wait
      fi
    done
  done
  wait
}

main "$@"
