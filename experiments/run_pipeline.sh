#!/usr/bin/env bash
# Orchestrate the remaining experiment stages after model training.
set -e
cd /home/user/sbi_spatstat/experiments
export PYTHONPATH=/home/user/sbi_spatstat
LOG=/home/user/sbi_spatstat/results

# 1) wait for the training process to finish writing reports
echo "[pipeline] waiting for training PID ${TRAIN_PID:-none} ..."
if [ -n "$TRAIN_PID" ]; then
  while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 10; done
fi
until [ -f "$LOG/train_reports.json" ]; do sleep 5; done
echo "[pipeline] training done."

# 2) calibration / coverage / recovery / misspecification
echo "[pipeline] eval_calibration ..."
python eval_calibration.py > "$LOG/eval_calib_log.txt" 2>&1
echo "[pipeline] calibration done."

# 3) gold-standard MCMC (the expensive stage)
echo "[pipeline] run_mcmc ..."
python run_mcmc.py > "$LOG/mcmc_log.txt" 2>&1
echo "[pipeline] mcmc done."

# 4) NPE vs MCMC comparison
echo "[pipeline] eval_mcmc ..."
python eval_mcmc.py > "$LOG/eval_mcmc_log.txt" 2>&1
echo "[pipeline] mcmc comparison done."

# 5) macros + figures + tables
echo "[pipeline] figures & macros ..."
python gen_macros.py > "$LOG/macros_log.txt" 2>&1
python make_figures.py > "$LOG/figures_log.txt" 2>&1
echo "[pipeline] figures done."

# 6) compile paper
echo "[pipeline] compiling paper ..."
cd /home/user/sbi_spatstat/paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex > "$LOG/tex1.log" 2>&1 || true
bibtex paper > "$LOG/bib.log" 2>&1 || true
pdflatex -interaction=nonstopmode -halt-on-error paper.tex > "$LOG/tex2.log" 2>&1 || true
pdflatex -interaction=nonstopmode -halt-on-error paper.tex > "$LOG/tex3.log" 2>&1 || true
echo "[pipeline] ALL DONE. pdf: $(ls -la paper.pdf 2>/dev/null)"
