# Benchmarks: pyroxide vs numpyro

Machine: Apple M4 Max, Darwin 25.5.0. Single chain, CPU, float64 in both libraries.

`time` is wall-clock for warmup + sampling, median over repeats. numpyro `time` includes XLA compilation of the
sampler for that model; `cached` is a second run reusing the compiled program. `ESS/s` uses the minimum effective
sample size over all parameters divided by the (uncached) wall time.

| model | algo | warmup/samples | pyroxide time (s) | numpyro time (s) | numpyro cached (s) | speedup vs numpyro | speedup vs cached | pyroxide min ESS | numpyro min ESS | pyroxide ESS/s | numpyro ESS/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseball | nuts | 1000/1000 | 0.045 | 1.202 | 0.723 | 26.8x | 16.1x | 51 | 65 | 832 | 41 |
| eight_schools | nuts | 1000/1000 | 0.016 | 0.595 | 0.616 | 36.5x | 37.8x | 747 | 633 | 42573 | 1024 |
| funnel | nuts | 1000/1000 | 0.008 | 0.598 | 0.514 | 74.7x | 64.3x | 539 | 443 | 68166 | 760 |
| gauss | nuts | 1000/1000 | 0.047 | 0.486 | 0.449 | 10.4x | 9.6x | 833 | 892 | 16992 | 1707 |
| hier | nuts | 1000/1000 | 0.639 | 1.242 | 1.134 | 1.9x | 1.8x | 758 | 786 | 1191 | 633 |
| logreg | nuts | 1000/1000 | 0.427 | 0.628 | 0.625 | 1.5x | 1.5x | 291 | 246 | 639 | 297 |
| baseball | hmc | 1000/1000 | 0.139 | 0.382 | 0.387 | 2.7x | 2.8x | 29 | 40 | 219 | 27 |
| eight_schools | hmc | 1000/1000 | 0.037 | 0.346 | 0.318 | 9.4x | 8.7x | 17 | 29 | 449 | 69 |
| funnel | hmc | 1000/1000 | 0.008 | 0.275 | 0.271 | 36.1x | 35.6x | 5 | 6 | 555 | 19 |
| gauss | hmc | 1000/1000 | 0.032 | 0.302 | 0.260 | 9.5x | 8.2x | 3 | 3 | 108 | 7 |
| hier | hmc | 1000/1000 | 5.436 | 1.157 | 1.185 | 0.2x | 0.2x | 3 | 3 | 1 | 3 |
| logreg | hmc | 1000/1000 | 1.069 | 0.651 | 0.645 | 0.6x | 0.6x | 1101 | 19 | 1097 | 30 |
| baseball | mh | 10000/20000 | 0.032 | 0.497 | 0.509 | 15.5x | 15.9x | 11 | 19 | 352 | 39 |
| eight_schools | mh | 10000/20000 | 0.013 | 0.456 | 0.433 | 33.8x | 32.1x | 389 | 419 | 28856 | 232 |
| funnel | mh | 10000/20000 | 0.010 | 0.405 | 0.400 | 39.3x | 38.8x | 484 | 488 | 46071 | 1166 |
| gauss | mh | 10000/20000 | 0.077 | 1.276 | 1.267 | 16.5x | 16.3x | 5 | 7 | 70 | 5 |
| hier | mh | 10000/20000 | 0.530 | 1.552 | 1.553 | 2.9x | 2.9x | 4 | 4 | 7 | 2 |
| logreg | mh | 10000/20000 | 0.428 | 0.717 | 0.712 | 1.7x | 1.7x | 1031 | 827 | 2418 | 1125 |

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
