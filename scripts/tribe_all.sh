#!/bin/bash
# Waits for the TRIBE Space, then runs TRIBE on every reel without output (finished reels are skipped). Keeps the Mac awake meanwhile.
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
until [ "$(curl -s -m 15 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" "${TRIBE_ENDPOINT_URL%/}/health")" = "200" ]; do sleep 20; done
caffeinate -i .venv/bin/python scripts/run_tribe_batch.py --all --seconds 10 --artifacts artifacts/tribe --allow-remote --approve-all
