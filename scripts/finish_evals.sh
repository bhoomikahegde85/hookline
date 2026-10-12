#!/bin/bash
# Waits for the two prediction runs, retries whatever failed (a Space that was still loading), then scores everything.
cd "$(dirname "$0")/.."
while pgrep -f "eval_snapugc.py predict" >/dev/null || pgrep -f "calibrate_ttcc.py predict" >/dev/null; do sleep 20; done
set -a; . ./.env; set +a
for i in 1 2; do
  until curl -s -m 15 -H "Authorization: Bearer $HF_TOKEN" "${TTCC_ENDPOINT_URL%/}/health" | grep -q '"checkpoint_ready": *true'; do sleep 20; done
  .venv/bin/python scripts/eval_snapugc.py predict >> artifacts/snapugc_eval/predict.log 2>&1 &
  .venv/bin/python scripts/calibrate_ttcc.py predict >> artifacts/ttcc_eval/hook_predict.log 2>&1 &
  wait
done
.venv/bin/python scripts/eval_snapugc.py score
.venv/bin/python scripts/train_snapugc_head.py
.venv/bin/python scripts/calibrate_ttcc.py fit
echo "ALL DONE"
