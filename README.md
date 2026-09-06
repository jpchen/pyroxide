# pyroxide

Fast, decoupled probabilistic programming in Rust: write a generative model once,
run it under NUTS, HMC, or Metropolis–Hastings.

pyroxide re-creates the core of [NumPyro](https://github.com/pyro-ppl/numpyro)
without JAX: a Stan-style reverse-mode autodiff tape, distributions with
analytic gradients, Pyro-style effect handlers, and a line-by-line port of
NumPyro's NUTS (iterative tree building, Stan windowed adaptation, dense/diagonal
mass matrices). No JIT compilation step, no Python, no runtime dependencies
beyond `rand`, `rayon` and `rustfft`.

* **Book**: user guide, design notes and API reference: `mdbook build book`
  (then open `target/book/index.html`). `.github/workflows/docs.yml` deploys it
  with the rustdoc reference to GitHub Pages; run it once the repository is
  public (GitHub Pages is not available on private repositories under the free
  plan).
* **Design**: [`docs/DESIGN.md`](docs/DESIGN.md) — goals, the model-syntax
  decision, and every layer explained.
* **Benchmarks**: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) — pyroxide vs
  NumPyro on the same models. On an M4 Max, single chain, float64, 1000 warmup +
  1000 samples: NUTS is 10–68× faster than NumPyro's compiled sampler on eight
  schools, Neal's funnel, baseball and a 100-d Gaussian (same ESS), 1.5× faster
  on logistic regression (1000 rows) and a 2000-row hierarchical regression;
  Metropolis–Hastings is 2–40× faster. The warmup adaptation is a numerically
  exact port (identical step-size trajectories to five digits).

## Example

```rust
use pyroxide::prelude::*;

/// Eight schools (Rubin 1981), non-centered.
struct EightSchools { y: Vec<f64>, sigma: Vec<f64> }

impl Model for EightSchools {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let mu = h.sample("mu", Normal::new(0.0, 5.0));
        let tau = h.sample("tau", HalfCauchy::new(5.0));
        let eta = h.sample_vec("eta", Normal::new(0.0, 1.0).expand(8));
        let theta: Vec<R> = eta.iter().map(|e| mu + tau * *e).collect();
        h.deterministic("theta", &theta);
        h.observe("y", Normal::new(&theta, &self.sigma), &self.y);
    }
}

fn main() {
    let model = EightSchools {
        y: vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        sigma: vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    };
    let kernel = HmcKernel::nuts(ModelPotential::new(&model));
    let samples = MCMC::new(kernel, 1000, 1000).num_chains(4).run(0);
    samples.print_summary();
    println!("tau = {:.2}", samples.get("tau").scalar_mean());
}
```

```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.39      3.31      4.40     -1.03      9.73   4021.61      1.00
       tau      3.61      3.14      2.81      0.00      7.88   2569.83      1.00
    eta[0]      0.39      0.94      0.42     -1.14      1.94   4230.13      1.00
       ...
Number of divergences: 0
```

Swap the kernel and nothing about the model changes:

```rust
let mh = MetropolisHastings::new(ModelPotential::new(&model)).dense_mass(true);
let hmc = HmcKernel::hmc(ModelPotential::new(&model)).trajectory_length(4.0);
```

Or skip models entirely and hand the sampler a density:

```rust
let banana = FnPotential::new(2, |z: &[Var]| z[0] * z[0] * 0.5 + (z[1] - z[0] * z[0]).square() * 50.0);
let samples = MCMC::new(HmcKernel::nuts(banana), 1000, 2000).run(1);
```

## What is here

| module | contents |
|---|---|
| `ad` | thread-local flat tape, `Real` trait for `f64` / `Var`, n-ary nodes with analytic partials |
| `dist` | Normal, LogNormal, HalfNormal, Cauchy, HalfCauchy, StudentT, Uniform, Exponential, Gamma, InverseGamma, Beta, Pareto, Laplace, ImproperUniform, Bernoulli, Binomial, Poisson, Categorical, Dirichlet, MultivariateNormal, LKJCholesky, `Ordered<D>` |
| `model` | `Model` / `Handler` traits, constraint transforms (positive, interval, simplex, ordered, correlation Cholesky), layout discovery, log-density / tracing / postprocessing handlers, `Predictive` |
| `infer` | `Potential`, `HmcKernel` (NUTS + HMC), `MetropolisHastings`, warmup adaptation, `MCMC` driver with parallel chains, `Samples` |
| `diagnostics` | ESS, R-hat, split R-hat, HPDI, autocorrelation, summary table |

## Tests

```
cargo test            # 61 unit tests + 20 end-to-end posterior tests + 8 doctests
```

The end-to-end tests are ports of NumPyro's `test_mcmc.py`: they run full
warmup and sampling and check posterior moments for every kernel.

## Benchmarks

```
benchmarks/run_all.sh <conda-env-with-numpyro> 1000 1000 3
```

runs six models (100-d Gaussian, eight schools, Neal's funnel, baseball partial
pooling, logistic regression, hierarchical regression) under NUTS, HMC and MH in
both libraries and writes `benchmarks/RESULTS.md`.

## Status

Early but complete for its scope: the two MCMC algorithms, 21 distributions, the
model layer and predictive simulation are implemented and tested. Not yet: variational inference,
discrete latent enumeration, a broadcasting shape system, GPU. See the roadmap
in `docs/DESIGN.md`.

## License

Apache-2.0.
