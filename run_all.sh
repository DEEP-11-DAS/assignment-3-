#!/usr/bin/env bash
# Full pipeline, in the required order. Test data is only touched by the `final` phases.
# Before step 3: commit q2_predictions.md.   After everything: git rev-parse HEAD  -> report page 1.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # GPU 1 on the shared machine is busy
export NJOBS=${NJOBS:-8}                                 # CPU processes for the Q4 SVM sweep

# ---- Q1
python q1a.py tune
python q1b.py 200
python q1a.py final
python make_figs_q1.py
# ---- Q2 (predictions must already be committed)
python q2.py perm_tune
python q2.py perm_final
python q2.py conditions
python make_figs_q2.py
# ---- Q3
python q3.py ae
python q3.py dae
python q3.py final
python make_figs_q3.py
# ---- Q4 (representation = Q3 selected AE bottleneck; set Q4_REP=cnn for the Q1 CNN features instead)
python q4.py features
python q4.py sweep
python q4.py cluster_select
python q4.py final
python make_figs_q4.py
