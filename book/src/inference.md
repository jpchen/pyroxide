# Inference

Inference never sees your model. It sees a `Potential` — a differentiable
function `U(z) = -log p(z)` on unconstrained space — and `ModelPotential::new(&model)`
turns any model into one. That separation is what lets you swap kernels without
touching the model.

```rust
let potential = ModelPotential::new(&model);
```

## Kernels

### NUTS

```rust
let kernel = HmcKernel::nuts(potential)
    .target_accept_prob(0.8)   // default; raise toward 0.95 if you see divergences
    .max_tree_depth(10)        // default; 2^10 leapfrog steps at most per iteration
    .dense_mass(false)         // true adapts a full covariance (good for correlated posteriors)
    .step_size(1.0)            // initial step size (adapted during warmup)
    .find_heuristic_step_size(false);
```

NUTS adapts the step size by dual averaging and the (inverse) mass matrix by
Welford estimation in Stan's windowed schedule (75 / 25·2^k / 50 iterations).
Use `max_tree_depths(warmup, sampling)` to cap tree depth separately during
warmup, and `inverse_mass_matrix(vec)` to supply a starting metric.

### HMC

```rust
let kernel = HmcKernel::hmc(potential).trajectory_length(4.0);   // or .num_steps(20)
```

Fixed-length HMC shares NUTS's integrator and adaptation. With the default
`2π` trajectory it is a poor sampler on many targets (a standard normal has
period `2π`), so prefer NUTS unless you have a reason.

### Metropolis–Hastings

```rust
let kernel = MetropolisHastings::new(potential)
    .dense_mass(true)            // full proposal covariance
    .target_accept_prob(0.234);  // default
```

Random-walk MH with the proposal covariance estimated during warmup and the
scale tuned toward the target acceptance rate. It uses only `Potential::value`,
so it works for any density, including ones without gradients.

## Running

```rust
let mcmc = MCMC::new(kernel, num_warmup, num_samples)
    .num_chains(4)                         // run on separate threads (rayon)
    .init_strategy(InitStrategy::Uniform { radius: 2.0 })
    .collect_warmup(false)
    .progress(true);                       // one line per chain per phase
let samples = mcmc.run(seed);
```

Chains are seeded from `seed` with independent Xoshiro256++ streams, so runs
are reproducible and give identical results whether chains run in parallel or
sequentially.

### Initialization

By default each chain starts uniformly in `(-2, 2)` on every unconstrained
coordinate, retrying until the density and gradient are finite (up to 100
times). To start from chosen constrained values:

```rust
let mut init = HashMap::new();
init.insert("lambda1".to_string(), vec![1.0]);
init.insert("lambda2".to_string(), vec![72.0]);
let mcmc = MCMC::new(kernel, 1000, 3000).init_to_value(init);
```

Sites not listed are initialized uniformly.

## Working with samples

`run` returns `Samples`:

```rust
let theta = samples.get("theta");          // Array: chains × draws × len
theta.mean();                              // Vec<f64>, one per element
theta.std();
theta.draw(chain, i);                      // &[f64] of length len
theta.column(j);                           // all draws of element j, chains concatenated
theta.column_per_chain(j);                 // Vec<Vec<f64>>, one per chain
theta.covariance();                        // len × len sample covariance

samples.get("tau").scalar_mean();          // for length-1 sites
samples.extra("accept_prob");              // per-iteration kernel diagnostics
samples.num_divergences();
samples.print_summary();
```

Kernel diagnostics available through `extra`: `accept_prob`, `step_size`,
`num_steps`, `diverging`, `energy`, `potential_energy`, `mean_accept_prob`
(NUTS/HMC), or `accept_prob`, `step_size`, `potential_energy`,
`mean_accept_prob` (MH).

## Writing a kernel

A kernel implements the `Kernel` trait: `init` builds a per-chain `State`,
`step` advances it, and `stats` reports diagnostics. The kernel itself is
immutable and shared across threads; every mutable buffer lives in the state,
which is what makes parallel chains free. See `src/infer/mh.rs` for a compact
example (about 150 lines including adaptation).
