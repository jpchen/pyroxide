# Benchmarks: pyroxide vs numpyro

Machine: Apple M4 Max, Darwin 25.5.0. Single chain, CPU, float64 in both libraries.

`time` is wall-clock for warmup + sampling, median over repeats. numpyro `time` includes XLA compilation of the
sampler for that model; `cached` is a second run reusing the compiled program. `ESS/s` uses the minimum effective
sample size over all parameters divided by the (uncached) wall time.

| model | algo | warmup/samples (×walkers) | pyroxide time (s) | numpyro time (s) | numpyro cached (s) | speedup vs numpyro | speedup vs cached | pyroxide min ESS | numpyro min ESS | pyroxide ESS/s | numpyro ESS/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseball | nuts | 1000/1000 | 0.045 | 0.713 | 0.704 | 15.7x | 15.5x | 51 | 72 | 823 | 45 |
| eight_schools | nuts | 1000/1000 | 0.016 | 0.608 | 0.610 | 37.7x | 37.9x | 747 | 806 | 43793 | 1138 |
| funnel | nuts | 1000/1000 | 0.008 | 0.529 | 0.531 | 66.1x | 66.4x | 539 | 443 | 68799 | 742 |
| gauss | nuts | 1000/1000 | 0.046 | 0.465 | 0.426 | 10.2x | 9.3x | 833 | 892 | 17987 | 1732 |
| hier | nuts | 1000/1000 | 0.632 | 1.189 | 1.171 | 1.9x | 1.9x | 758 | 1101 | 1199 | 702 |
| logreg | nuts | 1000/1000 | 0.427 | 0.623 | 0.621 | 1.5x | 1.5x | 291 | 246 | 647 | 298 |
| baseball | mams | 1000/1000 | 0.127 | 1.542 | 1.537 | 12.1x | 12.1x | 80 | 69 | 700 | 45 |
| eight_schools | mams | 1000/1000 | 0.007 | 1.167 | 1.166 | 162.0x | 161.9x | 67 | 80 | 8932 | 43 |
| funnel | mams | 1000/1000 | 0.003 | 0.993 | 0.999 | 354.5x | 356.9x | 76 | 60 | 27301 | 51 |
| gauss | mams | 1000/1000 | 0.007 | 0.935 | 0.969 | 128.1x | 132.7x | 294 | 297 | 40500 | 322 |
| hier | mams | 1000/1000 | 3.288 | 3.770 | 3.794 | 1.1x | 1.2x | 408 | 470 | 123 | 119 |
| logreg | mams | 1000/1000 | 0.431 | 1.119 | 1.117 | 2.6x | 2.6x | 478 | 548 | 1112 | 368 |
| baseball | barker | 2000/4000 | 0.014 | 0.442 | 0.440 | 30.7x | 30.5x | 32 | 29 | 2258 | 19 |
| eight_schools | barker | 2000/4000 | 0.010 | 0.391 | 0.374 | 40.7x | 38.9x | 219 | 193 | 23244 | 235 |
| funnel | barker | 2000/4000 | 0.007 | 0.311 | 0.315 | 45.8x | 46.4x | 416 | 350 | 56113 | 579 |
| gauss | barker | 2000/4000 | 0.032 | 0.297 | 0.305 | 9.4x | 9.7x | 84 | 52 | 2675 | 176 |
| hier | barker | 2000/4000 | 0.229 | 0.707 | 0.721 | 3.1x | 3.2x | 72 | 46 | 317 | 65 |
| logreg | barker | 2000/4000 | 0.178 | 0.343 | 0.335 | 1.9x | 1.9x | 119 | 83 | 662 | 192 |
| baseball | hmc | 1000/1000 | 0.141 | 0.425 | 0.411 | 3.0x | 2.9x | 29 | 46 | 213 | 108 |
| eight_schools | hmc | 1000/1000 | 0.037 | 0.333 | 0.338 | 9.1x | 9.2x | 17 | 27 | 454 | 52 |
| funnel | hmc | 1000/1000 | 0.008 | 0.306 | 0.281 | 40.3x | 36.9x | 5 | 6 | 549 | 19 |
| gauss | hmc | 1000/1000 | 0.030 | 0.274 | 0.273 | 9.2x | 9.2x | 3 | 3 | 110 | 7 |
| hier | hmc | 1000/1000 | 5.432 | 1.228 | 1.214 | 0.2x | 0.2x | 3 | 4 | 1 | 3 |
| logreg | hmc | 1000/1000 | 1.070 | 0.662 | 0.627 | 0.6x | 0.6x | 1101 | 19 | 1114 | 29 |
| baseball | mh | 10000/20000 | 0.032 | 0.514 | 0.507 | 16.1x | 15.9x | 11 | 36 | 345 | 18 |
| eight_schools | mh | 10000/20000 | 0.014 | 0.430 | 0.432 | 30.7x | 30.9x | 389 | 527 | 27797 | 1227 |
| funnel | mh | 10000/20000 | 0.010 | 0.420 | 0.396 | 41.6x | 39.2x | 484 | 488 | 47160 | 1116 |
| gauss | mh | 10000/20000 | 0.079 | 1.214 | 1.208 | 15.4x | 15.3x | 5 | 7 | 68 | 5 |
| hier | mh | 10000/20000 | 0.533 | 1.546 | 1.532 | 2.9x | 2.9x | 4 | 4 | 7 | 2 |
| logreg | mh | 10000/20000 | 0.424 | 0.732 | 0.738 | 1.7x | 1.7x | 1031 | 827 | 2425 | 1101 |
| baseball | aies | 2000/1000 ×20 | 0.060 | 0.593 | 0.603 | 10.0x | 10.1x | 10 | 10 | 171 | 18 |
| eight_schools | aies | 2000/1000 ×20 | 0.020 | 0.516 | 0.493 | 25.4x | 24.3x | 323 | 273 | 16034 | 549 |
| funnel | aies | 2000/1000 ×20 | 0.013 | 0.428 | 0.430 | 32.4x | 32.5x | 392 | 366 | 29645 | 769 |
| gauss | aies | 2000/1000 ×220 | 0.218 | 0.652 | 0.626 | 3.0x | 2.9x | 339 | 333 | 1522 | 495 |
| hier | aies | 2000/1000 ×220 | 8.786 | 2.158 | 2.137 | 0.2x | 0.2x | 117 | 121 | 13 | 54 |
| logreg | aies | 2000/1000 ×20 | 0.796 | 1.614 | 1.607 | 2.0x | 2.0x | 1015 | 1024 | 1275 | 637 |
| baseball | ess | 2000/1000 ×20 | 0.261 | 0.987 | 0.991 | 3.8x | 3.8x | 10 | 10 | 40 | 11 |
| eight_schools | ess | 2000/1000 ×20 | 0.075 | 0.675 | 0.670 | 9.0x | 9.0x | 748 | 716 | 9714 | 1061 |
| funnel | ess | 2000/1000 ×20 | 0.045 | 0.583 | 0.562 | 13.0x | 12.6x | 778 | 691 | 17448 | 1133 |
| gauss | ess | 2000/1000 ×220 | 0.830 | 1.750 | 1.737 | 2.1x | 2.1x | 533 | 473 | 659 | 270 |
| hier | ess | 2000/1000 ×220 | 47.614 | 60.905 | 59.921 | 1.3x | 1.3x | 112 | 110 | 2 | 2 |
| logreg | ess | 2000/1000 ×20 | 3.725 | 6.183 | 6.197 | 1.7x | 1.7x | 1984 | 1917 | 532 | 291 |

## Sampler diagnostics (median over repeats)

| model | algo | lib | leapfrog steps | mean accept | divergences | max R-hat |
|---|---|---|---:|---:|---:|---:|
| baseball | nuts | pyroxide | 13291 | 0.838 | 2 | 1.011 |
| baseball | nuts | numpyro | 14222 | 0.858 | 2 | 1.019 |
| eight_schools | nuts | pyroxide | 6892 | 0.850 | 0 | 1.002 |
| eight_schools | nuts | numpyro | 7756 | 0.894 | 0 | 1.002 |
| funnel | nuts | pyroxide | 6744 | 0.903 | 0 | 1.006 |
| funnel | nuts | numpyro | 6568 | 0.886 | 0 | 1.005 |
| gauss | nuts | pyroxide | 7000 | 0.847 | 0 | 1.003 |
| gauss | nuts | numpyro | 7000 | 0.844 | 0 | 1.005 |
| hier | nuts | pyroxide | 7432 | 0.842 | 0 | 1.006 |
| hier | nuts | numpyro | 14352 | 0.876 | 0 | 1.003 |
| logreg | nuts | pyroxide | 7644 | 0.908 | 0 | 1.001 |
| logreg | nuts | numpyro | 7836 | 0.920 | 0 | 1.008 |
| baseball | mams | pyroxide | 12907 | 0.823 | 0 | 1.014 |
| baseball | mams | numpyro | 10606 | 0.720 | 0 | 1.010 |
| eight_schools | mams | pyroxide | 1831 | 0.933 | 0 | 1.010 |
| eight_schools | mams | numpyro | 1691 | 0.897 | 0 | 1.012 |
| funnel | mams | pyroxide | 1000 | 0.926 | 0 | 1.007 |
| funnel | mams | numpyro | 1000 | 0.923 | 0 | 1.015 |
| gauss | mams | pyroxide | 1000 | 0.897 | 0 | 1.016 |
| gauss | mams | numpyro | 1000 | 0.888 | 0 | 1.016 |
| hier | mams | pyroxide | 17692 | 0.972 | 0 | 1.011 |
| hier | mams | numpyro | 17567 | 0.975 | 0 | 1.010 |
| logreg | mams | pyroxide | 2680 | 0.910 | 0 | 1.002 |
| logreg | mams | numpyro | 2758 | 0.910 | 0 | 1.002 |
| baseball | barker | pyroxide | 0 | 0.272 | 0 | 1.049 |
| baseball | barker | numpyro | 0 | 0.461 | 0 | 1.115 |
| eight_schools | barker | pyroxide | 0 | 0.369 | 0 | 1.013 |
| eight_schools | barker | numpyro | 0 | 0.248 | 0 | 1.016 |
| funnel | barker | pyroxide | 0 | 0.394 | 0 | 1.006 |
| funnel | barker | numpyro | 0 | 0.304 | 0 | 1.016 |
| gauss | barker | pyroxide | 0 | 0.348 | 0 | 1.043 |
| gauss | barker | numpyro | 0 | 0.210 | 0 | 1.069 |
| hier | barker | pyroxide | 0 | 0.383 | 0 | 1.046 |
| hier | barker | numpyro | 0 | 0.208 | 0 | 1.067 |
| logreg | barker | pyroxide | 0 | 0.425 | 0 | 1.006 |
| logreg | barker | numpyro | 0 | 0.247 | 0 | 1.002 |
| baseball | hmc | pyroxide | 41000 | 0.905 | 3 | 1.026 |
| baseball | hmc | numpyro | 41000 | 0.896 | 4 | 1.015 |
| eight_schools | hmc | pyroxide | 23000 | 0.953 | 0 | 1.028 |
| eight_schools | hmc | numpyro | 24000 | 0.972 | 0 | 1.067 |
| funnel | hmc | pyroxide | 9000 | 0.920 | 0 | 1.665 |
| funnel | hmc | numpyro | 9000 | 0.919 | 0 | 1.404 |
| gauss | hmc | pyroxide | 13000 | 0.899 | 0 | 2.283 |
| gauss | hmc | numpyro | 14000 | 0.915 | 0 | 2.438 |
| hier | hmc | pyroxide | 15000 | 0.895 | 0 | 2.417 |
| hier | hmc | numpyro | 15000 | 0.892 | 0 | 1.992 |
| logreg | hmc | pyroxide | 17000 | 0.939 | 0 | 1.000 |
| logreg | hmc | numpyro | 14000 | 0.934 | 0 | 1.172 |
| baseball | mh | pyroxide | 0 | 0.225 | 0 | 1.023 |
| baseball | mh | numpyro | 0 | 0.270 | 0 | 1.064 |
| eight_schools | mh | pyroxide | 0 | 0.220 | 0 | 1.006 |
| eight_schools | mh | numpyro | 0 | 0.226 | 0 | 1.005 |
| funnel | mh | pyroxide | 0 | 0.197 | 0 | 1.004 |
| funnel | mh | numpyro | 0 | 0.231 | 0 | 1.006 |
| gauss | mh | pyroxide | 0 | 0.163 | 0 | 1.589 |
| gauss | mh | numpyro | 0 | 0.227 | 0 | 1.509 |
| hier | mh | pyroxide | 0 | 0.111 | 0 | 2.136 |
| hier | mh | numpyro | 0 | 0.213 | 0 | 1.984 |
| logreg | mh | pyroxide | 0 | 0.223 | 0 | 1.001 |
| logreg | mh | numpyro | 0 | 0.245 | 0 | 1.001 |
| baseball | aies | pyroxide | 0 | 0.008 | 0 | 9.287 |
| baseball | aies | numpyro | 0 | nan | 0 | 9.464 |
| eight_schools | aies | pyroxide | 0 | 0.245 | 0 | 1.066 |
| eight_schools | aies | numpyro | 0 | nan | 0 | 1.071 |
| funnel | aies | pyroxide | 0 | 0.257 | 0 | 1.058 |
| funnel | aies | numpyro | 0 | nan | 0 | 1.068 |
| gauss | aies | pyroxide | 0 | 0.237 | 0 | 1.440 |
| gauss | aies | numpyro | 0 | nan | 0 | 1.436 |
| hier | aies | pyroxide | 0 | 0.080 | 0 | 4.877 |
| hier | aies | numpyro | 0 | nan | 0 | 3.993 |
| logreg | aies | pyroxide | 0 | 0.291 | 0 | 1.018 |
| logreg | aies | numpyro | 0 | nan | 0 | 1.024 |
| baseball | ess | pyroxide | 0 | nan | 0 | 5.411 |
| baseball | ess | numpyro | 0 | nan | 0 | 4.927 |
| eight_schools | ess | pyroxide | 0 | nan | 0 | 1.030 |
| eight_schools | ess | numpyro | 0 | nan | 0 | 1.037 |
| funnel | ess | pyroxide | 0 | nan | 0 | 1.030 |
| funnel | ess | numpyro | 0 | nan | 0 | 1.031 |
| gauss | ess | pyroxide | 0 | nan | 0 | 1.272 |
| gauss | ess | numpyro | 0 | nan | 0 | 1.296 |
| hier | ess | pyroxide | 0 | nan | 0 | 14.258 |
| hier | ess | numpyro | 0 | nan | 0 | 32.637 |
| logreg | ess | pyroxide | 0 | nan | 0 | 1.014 |
| logreg | ess | numpyro | 0 | nan | 0 | 1.011 |
