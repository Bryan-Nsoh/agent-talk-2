#!/bin/bash
# Run 3 parallel experiments (structured, freeform, none) for a given seed
# Usage: ./scripts/run_seed_batch.sh <seed>
# Example: ./scripts/run_seed_batch.sh 14

set -euo pipefail

# Check argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <seed>"
  echo "Example: $0 14"
  exit 1
fi

SEED=$1

# Load environment variables
set -a && source ~/.env && set +a

EXPERIMENT_DIR="experiments/robustness_study_20251112T143355Z"
mkdir -p "${EXPERIMENT_DIR}/runs"

echo "========================================"
echo "Launching seed=${SEED} batch (3 parallel runs)"
echo "========================================"
echo ""

# Generate unique timestamps for each run (2s spacing)
TIMESTAMP_STRUCTURED="$(date -u +%Y%m%dT%H%M%SZ)"
sleep 2
TIMESTAMP_FREEFORM="$(date -u +%Y%m%dT%H%M%SZ)"
sleep 2
TIMESTAMP_NONE="$(date -u +%Y%m%dT%H%M%SZ)"

# Define run directories
RUN_DIR_STRUCTURED="${EXPERIMENT_DIR}/runs/seed${SEED}_structured_${TIMESTAMP_STRUCTURED}"
RUN_DIR_FREEFORM="${EXPERIMENT_DIR}/runs/seed${SEED}_freeform_${TIMESTAMP_FREEFORM}"
RUN_DIR_NONE="${EXPERIMENT_DIR}/runs/seed${SEED}_none_${TIMESTAMP_NONE}"

# Create directories
mkdir -p "${RUN_DIR_STRUCTURED}"
mkdir -p "${RUN_DIR_FREEFORM}"
mkdir -p "${RUN_DIR_NONE}"

echo "Run directories:"
echo "  structured -> ${RUN_DIR_STRUCTURED}/"
echo "  freeform   -> ${RUN_DIR_FREEFORM}/"
echo "  none       -> ${RUN_DIR_NONE}/"
echo ""

# EXACT parameters from validated baseline (long_corridor_final config.yaml)
# Only --seed and --comm-strategy vary

# Structured run
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model azure:gpt-5-mini \
  --maze-preset long_corridor \
  --agents 5 \
  --turns 100 \
  --seed ${SEED} \
  --obstacle-seed 606 \
  --radio-range 2 \
  --visibility 1 \
  --history-limit 5 \
  --loop-guidance explore \
  --bearing-flip-p 0.0 \
  --bearing-drop-p 0.0 \
  --bearing-bias-p 0.0 \
  --bearing-bias-wall-bonus 0.0 \
  --maze-extra-connection 0.2 \
  --comm-strategy structured \
  --emit-config "${RUN_DIR_STRUCTURED}/config.yaml" \
  > "${RUN_DIR_STRUCTURED}/stdout.log" 2>&1 &

STRUCTURED_PID=$!
echo "✓ Launched structured (PID: ${STRUCTURED_PID})"

# Freeform run
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model azure:gpt-5-mini \
  --maze-preset long_corridor \
  --agents 5 \
  --turns 100 \
  --seed ${SEED} \
  --obstacle-seed 606 \
  --radio-range 2 \
  --visibility 1 \
  --history-limit 5 \
  --loop-guidance explore \
  --bearing-flip-p 0.0 \
  --bearing-drop-p 0.0 \
  --bearing-bias-p 0.0 \
  --bearing-bias-wall-bonus 0.0 \
  --maze-extra-connection 0.2 \
  --comm-strategy freeform \
  --emit-config "${RUN_DIR_FREEFORM}/config.yaml" \
  > "${RUN_DIR_FREEFORM}/stdout.log" 2>&1 &

FREEFORM_PID=$!
echo "✓ Launched freeform (PID: ${FREEFORM_PID})"

# None run (radio-range 0)
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --model azure:gpt-5-mini \
  --maze-preset long_corridor \
  --agents 5 \
  --turns 100 \
  --seed ${SEED} \
  --obstacle-seed 606 \
  --radio-range 0 \
  --visibility 1 \
  --history-limit 5 \
  --loop-guidance explore \
  --bearing-flip-p 0.0 \
  --bearing-drop-p 0.0 \
  --bearing-bias-p 0.0 \
  --bearing-bias-wall-bonus 0.0 \
  --maze-extra-connection 0.2 \
  --comm-strategy none \
  --emit-config "${RUN_DIR_NONE}/config.yaml" \
  > "${RUN_DIR_NONE}/stdout.log" 2>&1 &

NONE_PID=$!
echo "✓ Launched none (PID: ${NONE_PID})"

echo ""
echo "========================================"
echo "All 3 runs launched for seed=${SEED}"
echo "PIDs: ${STRUCTURED_PID} ${FREEFORM_PID} ${NONE_PID}"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  tail -f ${RUN_DIR_STRUCTURED}/stdout.log"
echo "  tail -f ${RUN_DIR_FREEFORM}/stdout.log"
echo "  tail -f ${RUN_DIR_NONE}/stdout.log"
echo ""
echo "Check if still running:"
echo "  ps -p ${STRUCTURED_PID} ${FREEFORM_PID} ${NONE_PID}"
echo ""
echo "Estimated time: 50-67 minutes"
echo ""
echo "Waiting for completion..."

# Wait for all jobs
wait

echo ""
echo "========================================"
echo "✓ Seed ${SEED} batch complete!"
echo "========================================"
echo ""

# Show results summary
echo "Results summary:"
for dir in "${RUN_DIR_STRUCTURED}" "${RUN_DIR_FREEFORM}" "${RUN_DIR_NONE}"; do
  if [ -f "${dir}/results/metrics.json" ]; then
    STRATEGY=$(basename "$dir" | cut -d'_' -f2)
    RESULT=$(jq -r '"Strategy: \(.comm_strategy) | Success: \(.success) | Agents: \(.agents) | Finished: (if .success then "ALL" else "PARTIAL" end) | Turns: \(.turns) | Messages: \(.messages_sent) | Collisions: \(.collisions)"' "${dir}/results/metrics.json")
    echo "  ${STRATEGY}: ${RESULT}"
  else
    echo "  $(basename "$dir"): NO RESULTS (check ${dir}/stdout.log for errors)"
  fi
done

echo ""
echo "Full results:"
echo "  ls ${RUN_DIR_STRUCTURED}/results/"
echo "  ls ${RUN_DIR_FREEFORM}/results/"
echo "  ls ${RUN_DIR_NONE}/results/"
echo ""
echo "View metrics:"
echo "  jq . ${RUN_DIR_STRUCTURED}/results/metrics.json"
echo "  jq . ${RUN_DIR_FREEFORM}/results/metrics.json"
echo "  jq . ${RUN_DIR_NONE}/results/metrics.json"
