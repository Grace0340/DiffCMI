#!/bin/bash
# Stop the background run.
if [ -f outputs/run.pid ]; then
  PID=$(cat outputs/run.pid)
  kill "$PID" 2>/dev/null && echo "Stopped PID ${PID}" || echo "Process ${PID} not running"
  rm -f outputs/run.pid
else
  echo "No PID file found."
fi
