# Getting started

## Installation

Add pyroxide to your `Cargo.toml`:

```toml
[dependencies]
pyroxide = { git = "https://github.com/jpchen/pyroxide" }
```

Build in release mode when sampling — the autodiff tape and the NUTS tree
builder are tight numeric loops and debug builds are 10–50× slower:

```
cargo run --release
```

## A first model

We estimate a coin's bias from a handful of flips.

```rust
use pyroxide::prelude::*;

/// A model is a struct holding its data ...
struct Coin {
    flips: Vec<f64>, // 1.0 = heads, 0.0 = tails
}

/// ... plus a generic `run` method describing how the data were generated.
impl Model for Coin {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        // a latent variable with a prior
        let p = h.sample("p", Beta::new(1.0, 1.0));
        // a derived quantity we want in the output
        h.deterministic("odds", &[p / (-p + 1.0)]);
        // the likelihood: condition the plate of flips on the data
        h.observe("flips", Bernoulli::new(p).expand(self.flips.len()), &self.flips);
    }
}

fn main() {
    let model = Coin { flips: vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0] };

    // Turn the model into a potential energy and hand it to a kernel.
    let kernel = HmcKernel::nuts(ModelPotential::new(&model));

    // 500 warmup iterations, 1000 posterior draws, 4 chains in parallel.
    let samples = MCMC::new(kernel, 500, 1000).num_chains(4).run(0);

    samples.print_summary();
    println!("P(heads) ≈ {:.3}", samples.get("p").scalar_mean());
}
```

Running it prints a table with the posterior mean, standard deviation, median,
90% highest-density interval, effective sample size and R-hat for every site:

```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
         p      0.70      0.14      0.71      0.48      0.92   1531.70      1.00
      odds      3.08      3.53      2.45      0.51      6.71   1349.85      1.00

Number of divergences: 0
```

## What just happened

1. `ModelPotential::new(&model)` ran the model once to discover its latent
   sites (`p`, on the unit interval) and built a differentiable log density on
   unconstrained space, applying the logit transform and its Jacobian.
2. `HmcKernel::nuts` wrapped that density in a NUTS kernel with default
   settings: step size and diagonal mass matrix adapted during warmup, target
   acceptance 0.8, maximum tree depth 10.
3. `MCMC::run` found a starting point with finite density, ran warmup and
   sampling for each chain on its own thread, mapped every draw back to
   constrained space, and evaluated the deterministic site.

The next chapters cover each of these pieces in more depth.
