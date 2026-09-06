{{#include ../../benchmarks/RESULTS.md}}

## Reproducing

```
benchmarks/run_all.sh <conda-env-with-numpyro> 1000 1000 3
python benchmarks/report.py
```

`examples/bench.rs` (pyroxide) and `benchmarks/numpyro_bench.py` (numpyro)
implement the same six models with the same kernels and print one JSON line per
run; the report script merges them.
