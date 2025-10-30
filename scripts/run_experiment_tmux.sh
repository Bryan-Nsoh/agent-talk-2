#!/usr/bin/env bash
# Template for launching long-running experiments in tmux with proper error capture
#
# Usage:
#   ./scripts/run_experiment_tmux.sh \
#     --model azure:gpt-5-mini \
#     --maze-preset long_corridor \
#     --agents 5 \
#     --turns 50 \
#     --log-prompts --log-movements
#
# This script:
# - Creates a detached tmux session
# - Loads environment variables from ~/.env
# - Captures both stdout and stderr to a log file
# - Records exit code for debugging
# - Allows reattachment with: tmux attach -t <session>

set -euo pipefail

# Generate unique session and run IDs
SESSION="run_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="experiments/two-agents-bearing-r1_20251028T120000Z/runs/${SESSION}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${SESSION}.log"
PID_FILE="${LOG_DIR}/${SESSION}.pid"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RUN_DIR/results"

# Echo run info
echo "======================================"
echo "Starting experiment in tmux session: $SESSION"
echo "Run directory: $RUN_DIR"
echo "Log file: $LOG_FILE"
echo "======================================"
echo ""
echo "To watch progress:"
echo "  tmux attach -t $SESSION"
echo "  # or"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  tmux list-sessions"
echo "  # or"
echo "  tail -n 50 $LOG_FILE"
echo ""
echo "======================================"

# Build the command with all arguments passed to this script
PYTHON_CMD="PYTHONPATH=src uv run python -m llmgrid.cli.poc_two_agents"
PYTHON_CMD="${PYTHON_CMD} $*"  # Pass through all CLI arguments
PYTHON_CMD="${PYTHON_CMD} --emit-config ${RUN_DIR}/config.yaml"

# Create tmux session that:
# 1. Sources environment variables
# 2. Echoes run metadata
# 3. Runs the experiment
# 4. Captures exit code
# 5. Uses 'tee' to show output AND log it
tmux new-session -d -s "$SESSION" bash -c "
set -euo pipefail
export PYTHONUNBUFFERED=1

# Load environment variables
if [ -f \"\$HOME/.env\" ]; then
  set -a
  source \"\$HOME/.env\"
  set +a
fi

# Record PID
echo \$\$ > \"$PID_FILE\"

# Echo run start info
echo \"run_start session=$SESSION\" | tee -a \"$LOG_FILE\"
echo \"command: $PYTHON_CMD\" | tee -a \"$LOG_FILE\"
echo \"run_dir=$RUN_DIR\" | tee -a \"$LOG_FILE\"
echo \"\" | tee -a \"$LOG_FILE\"

# Run the experiment, capturing both stdout and stderr
set +e  # Don't exit on error, capture exit code
$PYTHON_CMD 2>&1 | tee -a \"$LOG_FILE\"
EXIT_CODE=\${PIPESTATUS[0]}
set -e

# Record completion
echo \"\" | tee -a \"$LOG_FILE\"
echo \"run_complete exit_code=\$EXIT_CODE\" | tee -a \"$LOG_FILE\"

if [ \$EXIT_CODE -eq 0 ]; then
  echo \"Success! Results in: $RUN_DIR\" | tee -a \"$LOG_FILE\"
else
  echo \"Failed with exit code \$EXIT_CODE\" | tee -a \"$LOG_FILE\"
  echo \"Check log: $LOG_FILE\" | tee -a \"$LOG_FILE\"
fi

# Keep the window open for 5 seconds so you can see the result if attached
sleep 5
"

echo ""
echo "Session $SESSION started successfully!"
echo ""
echo "Quick check (wait 5s for initialization):"
echo "  sleep 5 && tail -n 30 $LOG_FILE"
