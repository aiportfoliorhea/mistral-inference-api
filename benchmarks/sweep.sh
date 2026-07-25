
# Concurrency sweep to saturation for Mistral-7B-v0.3 (BASE model) served by vLLM.
# Pinned target: vllm==0.25.1 


set -euo pipefail

MODEL="mistralai/Mistral-7B-v0.3"
OUTDIR="results/sweep"
FLOOR=100
mkdir -p "$OUTDIR"

for C in 1 2 4 8 16 32 64 128; do
  N=$(( C * 10 ))
  (( N < FLOOR )) && N=FLOOR
  echo "=== concurrency=$C ==="
  vllm bench serve \
    --backend openai \
    --model "$MODEL" \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts $N \
    --max-concurrency "$C" \
    --request-rate inf \
    --ignore-eos \
    --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 95,99 \
    --save-result \
    --result-dir "$OUTDIR" \
    --metadata concurrency="$C"
done

echo "Sweep complete. Results in $OUTDIR/"