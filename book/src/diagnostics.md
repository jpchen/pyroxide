# Diagnostics and predictive checks

## The summary table

```rust
samples.print_summary();
```

prints, for every element of every site: mean, standard deviation, median, the
90% highest-posterior-density interval, the effective sample size (`n_eff`) and
the split R-hat, followed by the number of divergent transitions. The same
information is available programmatically:

```rust
for site in samples.summary(0.9) {
    for (i, e) in site.elements.iter().enumerate() {
        println!("{}[{}] n_eff = {:.0}, r_hat = {:.3}", site.name, i, e.n_eff, e.r_hat);
    }
}
```

## What to look for

* **R-hat** compares within-chain and between-chain variance; values above
  ~1.01–1.05 mean the chains have not mixed. Run more chains and longer warmup,
  or reparameterize.
* **n_eff** is the number of independent draws the chain is worth. Below a few
  hundred, posterior summaries are noisy. NUTS typically achieves `n_eff` close
  to the number of draws on well-conditioned posteriors.
* **Divergences** (`samples.num_divergences()`) signal that the integrator
  failed in a region of high curvature — the funnel geometry of hierarchical
  models is the classic cause. Raise `target_accept_prob`, or non-center the
  model.
* **Mean acceptance probability** (`samples.extra("mean_accept_prob")`, last
  draw of each chain) should be near the target (0.8 for NUTS, 0.234 for MH).

## Functions

`pyroxide::diagnostics` exposes the building blocks, each taking one vector of
draws per chain:

| function | description |
|---|---|
| `effective_sample_size(&chains, bias)` | Geyer's initial monotone sequence estimator across chains |
| `gelman_rubin(&chains)` | R-hat |
| `split_gelman_rubin(&chains)` | split R-hat (each chain halved) |
| `autocorrelation(&x, bias)` / `autocovariance(&x, bias)` | FFT-based |
| `hpdi(&x, prob)` | highest posterior density interval |

These are ports of `numpyro.diagnostics` and reproduce its reference values.

## Posterior predictive checks

`Predictive` re-runs the model with latent sites replayed from posterior draws
and observed sites *sampled* instead of conditioned on:

```rust
let pred = Predictive::new(&model).posterior(&samples).run(1);
let y_rep = pred.get("y");     // chains × draws × len: replicated data sets

// compare a statistic of the data with its predictive distribution
let observed_mean = data.iter().sum::<f64>() / data.len() as f64;
let rep_means: Vec<f64> = (0..y_rep.draws)
    .map(|d| y_rep.draw(0, d).iter().sum::<f64>() / y_rep.len as f64)
    .collect();
```

Deterministic sites are recomputed for every draw, which is the easiest way to
get derived quantities you did not record during sampling. Use
`return_sites(&["y"])` to keep only some sites.

Without `posterior`, `Predictive` draws from the prior — a prior predictive
check:

```rust
let prior = Predictive::new(&model).num_samples(2000).run(0);
```
