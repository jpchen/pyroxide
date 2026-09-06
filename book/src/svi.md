# Variational inference

Stochastic variational inference (SVI) replaces sampling with optimization:
a *guide* `q(z; θ)` from a tractable family is fitted to the posterior by
maximizing the evidence lower bound

```
ELBO(θ) = E_q[ log p(x, z) − log q(z; θ) ]
```

with stochastic gradients from reparameterized samples. It scales to large
data through minibatching and gives a fast, if approximate, posterior.

## Fitting an autoguide

```rust
use pyroxide::prelude::*;

let model = EightSchools { /* data */ };
let guide = AutoDiagonalNormal::new(&model);          // mean-field Gaussian
let mut svi = SVI::new(&model, &guide, Adam::new(0.01)).num_particles(2);
let losses = svi.run(0, 5000);                         // -ELBO per step
println!("final loss {:.2}", losses.last().unwrap());

let posterior = svi.sample_posterior(1, 2000);         // Samples, like MCMC
posterior.print_summary();
let params = svi.params();                             // HashMap<String, Vec<f64>>
```

Available autoguides:

| guide | family | parameters |
|---|---|---|
| `AutoDelta` | point mass (MAP estimate) | `{site}_auto_loc` |
| `AutoDiagonalNormal` | independent Normal per unconstrained coordinate | `{site}_auto_loc`, `{site}_auto_scale` |
| `AutoMultivariateNormal` | full-rank Normal over all unconstrained latents | `auto_loc`, `auto_scale_tril` |

Every autoguide samples in unconstrained space and pushes the draw through the
site's constraining transform, so positive scales, probabilities, simplexes and
Cholesky factors are handled automatically (the log-Jacobian is part of
`log q`).

Optimizers: `Adam` (default choice), `ClippedAdam` (gradient-norm clipping),
`Sgd` (optionally with momentum). All operate on the flat unconstrained
parameter vector; implement the `Optimizer` trait for anything else.

## Custom guides

A guide is a `Model` that declares parameters with `param` and samples the
model's latent sites (same names) from distributions with a reparameterized
sampler:

```rust
struct Guide;
impl Model for Guide {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let loc = h.param("mu_loc", &[0.0], Support::Real);
        let scale = h.param("mu_scale", &[1.0], Support::Positive);
        h.sample("mu", Normal::new(loc[0], scale[0]));
        // a positive site: sample in log space and transform
        let tl = h.param("tau_loc", &[0.0], Support::Real);
        let ts = h.param("tau_scale", &[0.5], Support::Positive);
        h.sample("tau", Transformed::new(Normal::new(tl[0], ts[0]), Support::Positive, 1));
    }
}
```

`param(name, init, support)` returns the current value (constrained through
`support`). Distributions with `rsample`: `Normal`, `LogNormal`, `HalfNormal`,
`Cauchy`, `Laplace`, `Uniform`, `Exponential`, `MultivariateNormal`, `Delta`,
and `Transformed` of any of these. Guides that draw several sites jointly use
`h.draw(dist)` for an auxiliary reparameterized draw and `h.sample_given(name,
value, log_q)` to register each site — see `AutoMultivariateNormal`.

## Minibatching

Wrap the data-dependent part of the model in `push_scale(N / n)` /
`pop_scale()` and evaluate it on a batch of indices:

```rust
use std::sync::Mutex;

struct Regression { x: Vec<f64>, y: Vec<f64>, batch: Mutex<Option<Vec<usize>>> }

impl Model for Regression {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let w = h.sample("w", Normal::new(0.0, 2.0));
        let b = h.sample("b", Normal::new(0.0, 2.0));
        let all: Vec<usize> = (0..self.x.len()).collect();
        let batch = self.batch.lock().unwrap();
        let idx = batch.as_ref().unwrap_or(&all);
        h.push_scale(self.x.len() as f64 / idx.len() as f64);
        let xb: Vec<f64> = idx.iter().map(|&i| self.x[i]).collect();
        let yb: Vec<f64> = idx.iter().map(|&i| self.y[i]).collect();
        let mean: Vec<R> = xb.iter().map(|xi| w * *xi + b).collect();
        h.observe("y", Normal::new(&mean, 0.5), &yb);
        h.pop_scale();
    }
}

let mut svi = SVI::new(&model, &guide, Adam::new(0.02));
svi.run_with(0, 4000, |_step, rng| {
    *model.batch.lock().unwrap() = Some(subsample(rng, model.x.len(), 40));
});
*model.batch.lock().unwrap() = None;   // full data for evaluation / MCMC
```

The scaled likelihood makes each minibatch gradient an unbiased estimate of the
full-data ELBO gradient (this is what the test suite checks), so nothing else
changes. The same `push_scale` is honoured by the MCMC log density, which is the
hook for stochastic-gradient MCMC kernels.

## Checking a fit

* Plot or smooth `losses`; it should decrease and plateau. Increase
  `num_particles` or lower the learning rate if it is noisy.
* Compare `svi.sample_posterior` moments with a short NUTS run on the same
  model; mean-field guides typically match means well and underestimate
  variances, as expected.
* `svi.evaluate(seed, n)` gives a lower-variance loss estimate without
  updating parameters.
