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
# numpyro's MAMS lives on a not-yet-upstreamed branch; point NUMPYRO_SRC at a
# checkout that has numpyro/contrib/microcanonical to include it.
NUMPYRO_SRC=${NUMPYRO_SRC:-/Users/jpchen/research/numpyro}
if [ -d "$NUMPYRO_SRC/numpyro/contrib/microcanonical" ]; then export PYTHONPATH="$NUMPYRO_SRC${PYTHONPATH:+:$PYTHONPATH}"; fi
mkdir -p benchmarks/results
: > benchmarks/results/pyroxide.jsonl
: > benchmarks/results/numpyro.jsonl
cargo build --release --example bench 2>&1 | tail -1

MODELS="gauss eight_schools funnel baseball logreg hier"
for algo in nuts mams barker hmc mh aies ess; do
  for model in $MODELS; do
    w=$WARMUP; s=$SAMPLES; chains=1
    case "$algo" in
      mh) w=$((WARMUP * 10)); s=$((SAMPLES * 20)) ;;
      barker) w=$((WARMUP * 2)); s=$((SAMPLES * 4)) ;;
      aies|ess) w=$((WARMUP * 2)); s=$SAMPLES; chains=20 ;;
    esac
    # ensemble samplers need >= 2 * dim walkers; hier has 107 latents, gauss 100
    if [ "$chains" = 20 ] && { [ "$model" = "hier" ] || [ "$model" = "gauss" ]; }; then chains=220; fi
    echo "== pyroxide $model $algo (warmup $w, samples $s, chains $chains)"
    ./target/release/examples/bench --model "$model" --algo "$algo" --warmup "$w" --samples "$s" --chains "$chains" --repeat "$REPEAT" \
      | tee -a benchmarks/results/pyroxide.jsonl
    echo "== numpyro $model $algo"
    conda run --no-capture-output -n "$ENV" python benchmarks/numpyro_bench.py \
      --model "$model" --algo "$algo" --warmup "$w" --samples "$s" --chains "$chains" --repeat "$REPEAT" \
      | tee -a benchmarks/results/numpyro.jsonl || true
  done
done
python3 benchmarks/report.py
