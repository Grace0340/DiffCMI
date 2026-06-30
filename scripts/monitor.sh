#!/bin/bash
# Show training progress.
LATEST=$(ls -t logs/run_*.log 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then echo "No log found."; exit 1; fi
echo "=== Tailing ${LATEST} ==="
tail -n 40 "$LATEST"
echo ""
if [ -f outputs/results.json ]; then
  python3 -c "import json; print(len(json.load(open('outputs/results.json'))), 'experiments completed')"
fi
