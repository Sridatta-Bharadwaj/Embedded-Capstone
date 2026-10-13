#!/bin/bash
# Full E0 run on the local GPU: 5 CV folds (each in its own process), the pooled summary, then the deployable final model.
# Start from Windows PowerShell:
#   wsl bash /mnt/d/Sridatta/Embedded_Systems_project/E0_baseline/run_e0.sh
# Everything is printed live and also appended to outputs/run.log. Finished folds are skipped on a re-run.
set -u
HERE=/mnt/d/Sridatta/Embedded_Systems_project/E0_baseline
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=1 PYTHONWARNINGS=ignore
source ~/ovenv/bin/activate || { echo "Could not activate ~/ovenv"; exit 1; }
cd "$HERE"
mkdir -p outputs
LOG=outputs/run.log
nvidia-smi || { echo "nvidia-smi failed: no GPU visible in WSL. Aborting."; exit 1; }
echo "===== E0 run started $(date) =====" | tee -a "$LOG"
for k in 0 1 2 3 4; do
  python -u train_e0.py --mode cv --fold "$k" "$@" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!!! fold $k failed, stopping" | tee -a "$LOG"; exit 1; }
done
python -u summarize_cv.py 2>&1 | tee -a "$LOG"
python -u train_e0.py --mode final "$@" 2>&1 | tee -a "$LOG"
echo "===== E0 run finished $(date) =====" | tee -a "$LOG"
