# pyroxide

**Fast, decoupled probabilistic programming in Rust.**

pyroxide lets you write a Bayesian model as an ordinary Rust generative
program and fit it with the No-U-Turn Sampler (NUTS), Hamiltonian Monte Carlo,
or adaptive Metropolis–Hastings. It re-creates the core of
[NumPyro](https://github.com/pyro-ppl/numpyro) without JAX:

* a Stan-style reverse-mode autodiff tape, with one tape node per distribution
  site regardless of plate size;
* 20 distributions with analytic gradients, checked against SciPy and finite
  differences;
* Pyro-style effect handlers, so the same model code is traced, differentiated,
  and used for predictive simulation;
* a line-by-line port of NumPyro's NUTS: iterative tree building, Stan's
  windowed warmup adaptation, dense or diagonal mass matrices;
* parallel chains, ESS / R-hat diagnostics, and a NumPyro-style summary table.

There is no JIT compilation step and no Python: models compile with your
program, and the first gradient costs microseconds rather than seconds.

```rust
use pyroxide::prelude::*;

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

let model = EightSchools { /* data */ };
let samples = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 1000, 1000)
    .num_chains(4)
    .run(0);
samples.print_summary();
```

## How this book is organized

* **User guide** — how to install pyroxide, write models, choose distributions,
  run inference, and check results.
* **Design** — why pyroxide looks the way it does: the model-syntax decision,
  the autodiff strategy, and how models stay decoupled from inference.
* **Benchmarks** — pyroxide against NumPyro on identical models.
* **API documentation** — the full `rustdoc` reference.

## Status

The two MCMC algorithms, the distribution library and the model layer are
complete and tested (unit tests, end-to-end posterior tests ported from
NumPyro, doctests). Not yet available: variational inference, discrete latent
enumeration, a broadcasting shape system, GPU execution.

pyroxide is Apache-2.0 licensed. Source: <https://github.com/jpchen/pyroxide>.
