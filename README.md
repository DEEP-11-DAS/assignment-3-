# CIFAR10 assignment code (roll m26mac004, seed 26004)

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # Blackwell needs CUDA >= 12.8
    pip install -r requirements.txt
    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

GPU 1 on the shared machine is busy, so always pin GPU 0:  `export CUDA_VISIBLE_DEVICES=0`

## Q1
    python q1a.py tune         # 5 models x 3 LRs, validation only (test set not loaded)
    python q1a.py final        # best LR per model, 3 seeds, single test evaluation -> results/q1a_table.md
    python q1b.py 200          # memorisation runs p = 0, 0.4, 1.0 -> runs/q1b/*.json
    python make_figs_q1.py     # figs/q1a_curves.pdf, figs/q1b_memorisation.pdf, results/q1b_summary.md

Split is saved at splits/split_26004.npz on first run. Do NOT run `q1a.py final` until you are happy with the design:
it is the only place the test set is touched.

## Q1 update (after first results)
    python q1a.py tune         # LR grid is now 1e-4..1e-2 (only the 2 new LRs are run; old logs are kept)
    python q1a.py final        # re-selects LR on validation, 3 seeds, test once
    python q1b.py 200          # adds seeds 26005 and 26006 (existing seed-26004 files are kept)
    python make_figs_q1.py     # also writes results/q1a_gap.md (train-vs-val gap) and mean±std q1b summary

## Q2  (commit q2_predictions.md first!)
    python q2.py perm_tune     # (b) MLP & CNN on permuted pixels, validation only
    python q2.py perm_final    # (b) 3 seeds, test once
    python q2.py conditions    # (c) needs q1a final; interventions + gray-trained control + confidence/error stats
    python make_figs_q2.py

## Q3
    python q3.py ae && python q3.py dae && python q3.py final && python make_figs_q3.py
Needs ckpt/q1a/CNN_s26004.pt (fixed classifier, from Q1(a)). Writes results/q3.md (all three deliverable tables).

## Q4 (bonus)
    python q4.py features        # representation + the ONE 10k stratified SVM subset (splits/q4_subset_idx.npy)
    NJOBS=8 python q4.py sweep   # 3 kernels x 6 values of C (1e-2..1e3); validation only
    python q4.py cluster_select  # KMeans / diagonal GMM, k in {5..30}; label-free fit, selection by val ARI
    python q4.py final           # test: SVM (once), hard pairs + figures, clustering ARI/NMI/contingency, disagreement
    python make_figs_q4.py
`Q4_REP=ae` (default, Q3 bottleneck, label-free) or `Q4_REP=cnn` (Q1 CNN features). Decide BEFORE running `sweep`.

## Everything
    bash run_all.sh      # then:  pip freeze > requirements-lock.txt ;  git rev-parse HEAD
