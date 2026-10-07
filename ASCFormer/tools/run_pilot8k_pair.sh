#!/usr/bin/env bash
# Run sequentially to keep the single GPU within its memory budget.
set -euo pipefail
cd /home/duy/DACN/RTM/ASCFormer
export TORCH_HOME=/home/duy/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin=/home/duy/miniconda3/envs/rtm-gpu/bin/python
for variant in baseline dwt; do
    run_dir="work_dirs/pilot8k_${variant}"
    if [[ -e "$run_dir" ]]; then
        echo "Refusing to overwrite existing run: $run_dir"
        exit 1
    fi
done
for variant in baseline dwt; do
    config=configs/ascformer/ascformer_rtm_pilot8k.py
    if [[ "$variant" == dwt ]]; then
        config=configs/ascformer/ascformer_rtm_dwt_pilot8k.py
    fi
    run_dir="work_dirs/pilot8k_${variant}"
    mkdir -p "$run_dir"
    echo "Starting $variant at $(date -Is)"
    "$python_bin" -u tools/train.py "$config" --work-dir "$run_dir" > "$run_dir/console.log" 2>&1
    echo "Finished $variant at $(date -Is)"
done
echo "Both pilots completed. Compare each val_metrics/binary_metrics.json."
