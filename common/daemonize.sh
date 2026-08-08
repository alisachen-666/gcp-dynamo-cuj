#!/usr/bin/env bash
# Run a long orchestration script detached from the Claude Code session so it survives
# session restarts. Logs to ~/DynamoBench/.runs/<name>.log (stable path, re-attachable).
# Usage: daemonize.sh <name> <script> [args...]
set -euo pipefail
NAME=$1; shift
RUNS=$HOME/DynamoBench/.runs
mkdir -p "$RUNS"
LOG="$RUNS/$NAME.log"
setsid nohup "$@" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUNS/$NAME.pid"
echo "daemonized '$NAME' pid=$PID log=$LOG"
