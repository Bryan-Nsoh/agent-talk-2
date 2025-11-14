#!/bin/bash
# Fixed parallel batch script - each run gets isolated directory
# Bug fix: Pass config.yaml FILE path, not directory path to --emit-config

set -euo pipefail

# Load environment variables
set -a && source ~/.env && set +a

EXPERIMENT_DIR="experiments/cross_seed_baseline_20251112T143355Z"
mkdir -p "${EXPERIMENT_DIR}/runs"

echo "Launching seed=14 batch with FIXED isolated directories..."
echo ""

# Generate unique timestamps for each run (with 2s spacing to ensure uniqueness)
TIMESTAMP_STRUCTURED="$(date -u +%Y%m%dT%H%M%SZ)"
sleep 2
TIMESTAMP_FREEFORM="$(date -u +%Y%m%dT%H%M%SZ)"
sleep 2
TIMESTAMP_NONE="$(date -u +%Y%m%dT%H%M%SZ)"

# Define run directories
RUN_DIR_STRUCTURED="${EXPERIMENT_DIR}/runs/seed14_structured_${TIMESTAMP_STRUCTURED}"
RUN_DIR_FREEFORM="${EXPERIMENT_DIR}/runs/seed14_freeform_${TIMESTAMP_FREEFORM}"
RUN_DIR_NONE="${EXPERIMENT_DIR}/runs/seed14_none_${TIMESTAMP_NONE}"

# Create run directories ahead of time
mkdir -p "${RUN_DIR_STRUCTURED}"
mkdir -p "${RUN_DIR_FREEFORM}"
mkdir -p "${RUN_DIR_NONE}"

echo "Will launch to:"
echo "  structured -> ${RUN_DIR_STRUCTURED}/"
echo "  freeform   -> ${RUN_DIR_FREEFORM}/"
echo "  none       -> ${RUN_DIR_NONE}/"
echo ""

# Structured run - FIXED: Pass config.yaml FILE not directory
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --maze-preset long_corridor \
  --comm-strategy structured \
  --model azure:gpt-5-mini \
  --agents 5 \
  --turns 100 \
  --seed 14 \
  --obstacle-seed 606 \
  --radio-range 2 \
  --visibility 1 \
  --history-limit 5 \
  --loop-guidance explore \
  --bearing-flip-p 0.0 \
  --bearing-drop-p 0.0 \
  --maze-extra-connection 0.2 \
  --emit-config "${RUN_DIR_STRUCTURED}/config.yaml" \
  > "${RUN_DIR_STRUCTURED}/stdout.log" 2>&1 &

STRUCTURED_PID=$!
echo "Launched structured (PID: ${STRUCTURED_PID})"

# Freeform run - FIXED: Pass config.yaml FILE not directory
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --maze-preset long_corridor \
  --comm-strategy freeform \
  --model azure:gpt-5-mini \
  --agents 5 \
  --turns 100 \
  --seed 14 \
  --obstacle-seed 606 \
  --radio-range 2 \
  --visibility 1 \
  --history-limit 5 \
  --loop-guidance explore \
  --bearing-flip-p 0.0 \
  --bearing-drop-p 0.0 \
  --maze-extra-connection 0.2 \
  --emit-config "${RUN_DIR_FREEFORM}/config.yaml" \
  > "${RUN_DIR_FREEFORM}/stdout.log" 2>&1 &

FREEFORM_PID=$!
echo "Launched freeform (PID: ${FREEFORM_PID})"

# None run - FIXED: Pass config.yaml FILE not directory
PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents \
  --maze-preset long_corridor \
  --comm-strategy none \
  --model azure:gpt-5-mini \
  --agents 5 \
  --turns 100 \
  --seed 14 \
  --obstacle-seed 606 \
  --radio-range 0 \
  --visibility 1 \
  --history-limit 5 \
  --loop-guidance explore \
  --bearing-flip-p 0.0 \
  --bearing-drop-p 0.0 \
  --maze-extra-connection 0.2 \
  --emit-config "${RUN_DIR_NONE}/config.yaml" \
  > "${RUN_DIR_NONE}/stdout.log" 2>&1 &

NONE_PID=$!
echo "Launched none (PID: ${NONE_PID})"

echo ""
echo "All 3 runs launched successfully!"
echo ""
echo "PIDs: structured=${STRUCTURED_PID} freeform=${FREEFORM_PID} none=${NONE_PID}"
echo ""
echo "Monitor progress:"
echo "  tail -f ${RUN_DIR_STRUCTURED}/stdout.log"
echo "  tail -f ${RUN_DIR_FREEFORM}/stdout.log"
echo "  tail -f ${RUN_DIR_NONE}/stdout.log"
echo ""
echo "Check status:"
echo "  ps -p ${STRUCTURED_PID} ${FREEFORM_PID} ${NONE_PID}"
echo ""
echo "Waiting for all runs to complete..."

# Wait for all background jobs
wait

echo ""
echo "======================================"
echo "All runs completed!"
echo "======================================"
echo ""
echo "Check results:"
echo "  ls ${RUN_DIR_STRUCTURED}/results/"
echo "  ls ${RUN_DIR_FREEFORM}/results/"
echo "  ls ${RUN_DIR_NONE}/results/"
echo ""
echo "View metrics:"
echo "  jq . ${RUN_DIR_STRUCTURED}/results/metrics.json"
echo "  jq . ${RUN_DIR_FREEFORM}/results/metrics.json"
echo "  jq . ${RUN_DIR_NONE}/results/metrics.json"
