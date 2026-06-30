#!/bin/bash
# Run the full experiment matrix in the background.
# Trains all models (zero/mean/MMIN/DiffCMI) across three datasets,
# the MOSEI missing-rate sweep, and the structured-missing study,
# all under the fixed-mask evaluation protocol, with uncertainty.
set -e

DATA_ROOT="${1:-./data}"
EPOCHS="${2:-50}"
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs outputs

echo "Data root : ${DATA_ROOT}"
echo "Epochs    : ${EPOCHS}"
echo "Log file  : ${LOG}"

setsid nohup python3 -u diffcmi_experiment.py \
  --run_all \
  --data_root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --with_baselines \
  --eval_uncertainty \
  --out_dir ./outputs \
  > "${LOG}" 2>&1 < /dev/null &

echo $! > outputs/run.pid
echo "Started (PID $(cat outputs/run.pid)). Monitor with: bash scripts/monitor.sh"
