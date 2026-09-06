# Benchmarks: pyroxide vs numpyro

Machine: Apple M4 Max, Darwin 25.5.0. Single chain, CPU, float64 in both libraries.

`time` is wall-clock for warmup + sampling, median over repeats. numpyro `time` includes XLA compilation of the
sampler for that model; `cached` is a second run reusing the compiled program. `ESS/s` uses the minimum effective
sample size over all parameters divided by the (uncached) wall time.

| model | algo | warmup/samples | pyroxide time (s) | numpyro time (s) | numpyro cached (s) | speedup vs numpyro | speedup vs cached | pyroxide min ESS | numpyro min ESS | pyroxide ESS/s | numpyro ESS/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseball | nuts | 1000/1000 | 0.045 | 1.207 | 0.708 | 27.0x | 15.8x | 51 | 65 | 834 | 38 |
| eight_schools | nuts | 1000/1000 | 0.016 | 0.627 | 0.620 | 39.2x | 38.7x | 747 | 633 | 42454 | 971 |
| funnel | nuts | 1000/1000 | 0.008 | 0.535 | 0.531 | 67.7x | 67.2x | 539 | 443 | 69066 | 730 |
| gauss | nuts | 1000/1000 | 0.045 | 0.475 | 0.470 | 10.6x | 10.5x | 833 | 892 | 18578 | 1685 |
| hier | nuts | 1000/1000 | 0.811 | 1.207 | 1.157 | 1.5x | 1.4x | 758 | 786 | 942 | 651 |
| logreg | nuts | 1000/1000 | 0.423 | 0.640 | 0.622 | 1.5x | 1.5x | 291 | 246 | 644 | 296 |
| baseball | hmc | 1000/1000 | 0.138 | 0.387 | 0.386 | 2.8x | 2.8x | 29 | 40 | 220 | 27 |
| eight_schools | hmc | 1000/1000 | 0.036 | 0.324 | 0.312 | 8.9x | 8.6x | 17 | 29 | 447 | 69 |
| funnel | hmc | 1000/1000 | 0.008 | 0.267 | 0.266 | 34.3x | 34.1x | 5 | 6 | 543 | 20 |
| gauss | hmc | 1000/1000 | 0.033 | 0.294 | 0.274 | 8.9x | 8.3x | 3 | 3 | 108 | 7 |
| hier | hmc | 1000/1000 | 6.939 | 1.217 | 1.194 | 0.2x | 0.2x | 3 | 3 | 1 | 3 |
| logreg | hmc | 1000/1000 | 1.069 | 0.655 | 0.640 | 0.6x | 0.6x | 1101 | 19 | 1102 | 30 |
| baseball | mh | 10000/20000 | 0.033 | 0.502 | 0.503 | 15.4x | 15.5x | 11 | 19 | 342 | 38 |
| eight_schools | mh | 10000/20000 | 0.014 | 0.449 | 0.436 | 31.1x | 30.2x | 389 | 419 | 26960 | 237 |
| funnel | mh | 10000/20000 | 0.010 | 0.421 | 0.411 | 40.5x | 39.5x | 484 | 488 | 45645 | 1113 |
| gauss | mh | 10000/20000 | 0.079 | 1.268 | 1.239 | 16.1x | 15.8x | 5 | 7 | 68 | 5 |
| hier | mh | 10000/20000 | 0.428 | 1.538 | 1.539 | 3.6x | 3.6x | 4 | 4 | 8 | 2 |
| logreg | mh | 10000/20000 | 0.427 | 0.722 | 0.718 | 1.7x | 1.7x | 1031 | 827 | 2413 | 1116 |

## Sampler diagnostics (median over repeats)

| model | algo | lib | leapfrog steps | mean accept | divergences | max R-hat |
|---|---|---|---:|---:|---:|---:|
| baseball | nuts | pyroxide | 13291 | 0.838 | 2 | 1.011 |
| baseball | nuts | numpyro | 12289 | 0.785 | 14 | 1.002 |
| eight_schools | nuts | pyroxide | 6892 | 0.850 | 0 | 1.002 |
| eight_schools | nuts | numpyro | 7698 | 0.875 | 0 | 1.004 |
| funnel | nuts | pyroxide | 6744 | 0.903 | 0 | 1.006 |
| funnel | nuts | numpyro | 6568 | 0.886 | 0 | 1.005 |
| gauss | nuts | pyroxide | 7000 | 0.847 | 0 | 1.003 |
| gauss | nuts | numpyro | 7000 | 0.844 | 0 | 1.005 |
| hier | nuts | pyroxide | 7432 | 0.842 | 0 | 1.006 |
| hier | nuts | numpyro | 16424 | 0.857 | 0 | 1.005 |
| logreg | nuts | pyroxide | 7644 | 0.908 | 0 | 1.001 |
| logreg | nuts | numpyro | 7836 | 0.920 | 0 | 1.008 |
| baseball | hmc | pyroxide | 41000 | 0.905 | 3 | 1.026 |
| baseball | hmc | numpyro | 28000 | 0.787 | 6 | 1.135 |
| eight_schools | hmc | pyroxide | 23000 | 0.953 | 0 | 1.028 |
| eight_schools | hmc | numpyro | 19000 | 0.917 | 0 | 1.040 |
| funnel | hmc | pyroxide | 9000 | 0.920 | 0 | 1.665 |
| funnel | hmc | numpyro | 9000 | 0.919 | 0 | 1.404 |
| gauss | hmc | pyroxide | 13000 | 0.899 | 0 | 2.283 |
| gauss | hmc | numpyro | 14000 | 0.915 | 0 | 2.438 |
| hier | hmc | pyroxide | 15000 | 0.895 | 0 | 2.417 |
| hier | hmc | numpyro | 14000 | 0.878 | 0 | 2.217 |
| logreg | hmc | pyroxide | 17000 | 0.939 | 0 | 1.000 |
| logreg | hmc | numpyro | 14000 | 0.934 | 0 | 1.172 |
| baseball | mh | pyroxide | 0 | 0.225 | 0 | 1.023 |
| baseball | mh | numpyro | 0 | 0.222 | 0 | 1.062 |
| eight_schools | mh | pyroxide | 0 | 0.220 | 0 | 1.006 |
| eight_schools | mh | numpyro | 0 | 0.225 | 0 | 1.003 |
| funnel | mh | pyroxide | 0 | 0.197 | 0 | 1.004 |
| funnel | mh | numpyro | 0 | 0.231 | 0 | 1.006 |
| gauss | mh | pyroxide | 0 | 0.163 | 0 | 1.589 |
| gauss | mh | numpyro | 0 | 0.227 | 0 | 1.509 |
| hier | mh | pyroxide | 0 | 0.111 | 0 | 2.136 |
| hier | mh | numpyro | 0 | 0.227 | 0 | 2.005 |
| logreg | mh | pyroxide | 0 | 0.223 | 0 | 1.001 |
| logreg | mh | numpyro | 0 | 0.245 | 0 | 1.001 |
