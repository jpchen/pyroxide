#!/usr/bin/env bash
# Run every (model, algo) pair in both libraries and write JSONL results.
#
#   benchmarks/run_all.sh [conda-env] [warmup] [samples] [repeat]
#
# Results land in benchmarks/results/{pyroxide,numpyro}.jsonl; render the table with
#   python benchmarks/report.py
set -euo pipefail
ENV=${1:-atlco}
WARMUP=${2:-1000}
SAMPLES=${3:-1000}
REPEAT=${4:-3}
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
: > benchmarks/results/pyroxide.jsonl
: > benchmarks/results/numpyro.jsonl
cargo build --release --example bench 2>&1 | tail -1

MODELS="gauss eight_schools funnel baseball logreg hier"
for algo in nuts hmc mh; do
  for model in $MODELS; do
    w=$WARMUP; s=$SAMPLES
    if [ "$algo" = "mh" ]; then w=$((WARMUP * 10)); s=$((SAMPLES * 20)); fi
    echo "== pyroxide $model $algo (warmup $w, samples $s)"
    ./target/release/examples/bench --model "$model" --algo "$algo" --warmup "$w" --samples "$s" --repeat "$REPEAT" \
      | tee -a benchmarks/results/pyroxide.jsonl
    echo "== numpyro $model $algo"
    conda run --no-capture-output -n "$ENV" python benchmarks/numpyro_bench.py \
      --model "$model" --algo "$algo" --warmup "$w" --samples "$s" --repeat "$REPEAT" \
      | tee -a benchmarks/results/numpyro.jsonl
  done
done
python3 benchmarks/report.py
