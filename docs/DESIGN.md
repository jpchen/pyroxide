# pyroxide: technical design

*A fast, decoupled probabilistic programming library for Rust.*

This document explains what pyroxide is, why it is built the way it is, and what
writing a model and running inference looks like. It is written as a design
talk: goals first, then the landscape and the central syntax decision, then the
system layer by layer, then testing and benchmarking.

---

## 1. Goals and non-goals

pyroxide re-creates the core of [NumPyro](https://github.com/pyro-ppl/numpyro)
end to end in Rust:

1. **The two workhorse MCMC algorithms, as fast as possible on a CPU.**
   NUTS (HMC with the No-U-Turn kernel and Stan-style warmup adaptation) and
   random-walk Metropolis–Hastings.
2. **Models and inference that know nothing about each other.** Any model can be
   run under any kernel; new kernels need no changes to models; models can be
   evaluated, sampled from, and differentiated by anything that speaks the small
   `Potential` interface.
3. **Trustworthy.** The function-level tests and end-to-end posterior tests of
   NumPyro are ported, plus finite-difference gradient checks for every
   distribution.
4. **Measurable.** A benchmark harness runs identical models in both libraries.

Explicit non-goals for now: GPUs / XLA, vectorized (`vmap`) chains, variational
inference, discrete latent enumeration, and a full broadcasting shape system.
The design leaves room for all of these (see §11).

---

## 2. The landscape and the syntax question

### 2.1 How existing PPLs define models

| System | Model syntax | Inference sees |
|---|---|---|
| Stan | declarative blocks in a DSL, compiled to C++ | a `log_prob(unconstrained)` with gradient |
| PyMC | Python objects registered on a graph via context managers | a graph → compiled logp |
| Pyro / NumPyro | **imperative generative function**; `sample` statements intercepted by *effect handlers* | a `potential_fn` built by running the function under handlers |
| Turing.jl / Gen.jl | imperative function under a macro (`@model`, `@gen`) | traces / logpdf via multiple dispatch |
| Infer.NET | declarative factor graph | message passing on the graph |
| nuts-rs (PyMC's Rust sampler) | none: takes a `logp(&x, &mut grad)` trait | that trait |

Two things stand out. First, every gradient-based system funnels the model into
the same tiny interface: *a differentiable log density on unconstrained
`R^d`*. That interface is the true decoupling point and it is what pyroxide's
inference layer targets. Second, the *pleasant* way to write a model is
imperative and generative (Pyro, Turing): you write the forward simulation and
the system figures out the rest. Declarative graphs are nicer for the compiler
than for the person.

### 2.2 Rust PPL prior art

Rust has good building blocks but no established PPL:

* `nuts-rs` is the strongest piece: a production NUTS sampler used by PyMC's
  `nutpie`. It deliberately has **no model language** — it consumes a
  `logp`/gradient trait. pyroxide's `Potential` is the same idea, and pyroxide's
  model layer is what nuts-rs leaves to the caller.
* `rv` and `statrs` provide distributions (sampling, densities), but no
  gradients and no notion of latent sites.
* Assorted experiments (`ferric`, `probabilistic`, monadic DSLs) demonstrate
  that macro-heavy or monadic encodings fight the borrow checker and produce
  opaque error messages.

### 2.3 Decision: generative functions, encoded as a trait with a generic method

The Pyro style is the right one, but it cannot be transplanted literally:

* Pyro's handlers are a *global stack* mutated by `with` blocks. Rust has no
  dynamic scoping, and global mutable state defeats multi-threaded chains.
* Pyro's model is a closure. A Rust closure cannot be generic over the scalar
  type, and we need the *same model code* to run with plain `f64` (evaluation,
  prior sampling) and with a differentiable `Var` (gradients).

So a model is a **struct holding its data plus one generic method**:

```rust
pub trait Model: Send + Sync {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H);
}
```

The handler is passed explicitly (no global stack) and the scalar type `R` is a
generic parameter (so one definition serves evaluation and differentiation with
zero runtime dispatch). Inside `run`, the model calls `h.sample`, `h.observe`,
`h.factor`, `h.deterministic` — exactly Pyro's primitives. This is *imperative*
in the sense that matters (you write the forward process in ordinary Rust with
loops and arithmetic) while being *declarative* in the sense that matters for
inference (the handler owns the interpretation of every random choice).

Compared with alternatives considered:

* **Declarative graph builder.** Rejected: loses ordinary control flow and
  arithmetic, needs its own expression type, and error messages become graph
  errors instead of type errors.
* **Procedural macro `#[model]`.** Rejected for v1: it hides the mechanism,
  breaks IDE tooling, and the trait form is already short. A macro can be
  layered on later purely as sugar.
* **Model as closure `Fn(&mut dyn Handler)`.** Rejected: not generic over `R`,
  so gradients would need dynamic dispatch on every arithmetic operation.

The cost is that `Model` is not object-safe (it has a generic method). That is
fine: inference is generic over `M: Model`, and the trait object we *do* need
for decoupling is `dyn Potential`, which is object-safe.

---

## 3. Architecture

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  user model:  struct EightSchools { y, sigma }   impl Model for …       │
 └──────────────────────────────────────────────────────────────────────────┘
        │ run::<R, H>()                     ▲ postprocess (f64)
        ▼                                   │
 ┌──────────────── model layer (src/model) ─────────────────────────────────┐
 │ Handler<R> implementations:                                              │
 │   LayoutDiscovery  → SiteLayout (names, sizes, supports, offsets)        │
 │   LogDensity<R>    → log p(z) + log|J|   (R = f64 or Var)                │
 │   Tracer           → prior samples / replay, records a Trace             │
 │   Postprocess      → unconstrained z → named constrained values          │
 │ transforms: Positive, Interval, Simplex … (bijections + log-Jacobian)    │
 └──────────────────────────────────────────────────────────────────────────┘
        │ uses                                          │ uses
        ▼                                               ▼
 ┌──── distributions (src/dist) ─────┐   ┌──── autodiff (src/ad) ──────────┐
 │ Distribution<R>: len, support,     │   │ Real trait: f64 | Var           │
 │   log_prob (one tape node with     │──▶│ thread-local flat tape,         │
 │   analytic partials), sample       │   │ n-ary nodes, one backward sweep │
 └────────────────────────────────────┘   └─────────────────────────────────┘
        │
        ▼  ModelPotential<M> implements
 ┌──────────────── inference (src/infer) ───────────────────────────────────┐
 │ trait Potential { dim, value, value_and_grad, postprocess }              │
 │ trait Kernel   { init, step, position, stats }                           │
 │   HmcKernel (NUTS | HMC)  ·  MetropolisHastings                          │
 │   WarmupAdapter: dual averaging + Welford + Stan windows, MassMatrix     │
 │ MCMC driver: init search, warmup, sampling, rayon-parallel chains        │
 │   → Samples { sites: name → chains×draws×len, extras }                   │
 └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──── diagnostics ───┐   ESS (Geyer/Stan), R-hat, split R-hat, HPDI, summary
 └────────────────────┘
```

The arrows only point downward. Models depend on distributions and the scalar
abstraction; inference depends only on `Potential`; nothing depends on
inference. A model written today will run unchanged under a kernel written next
year, and a kernel can be tested on a closed-form `FnPotential` with no model at
all (that is how the sampler unit tests work).

---

## 4. Automatic differentiation: Stan's approach, not JAX's

NumPyro gets gradients from JAX tracing and XLA compilation. The Rust analogue
would be building an expression graph and JIT-compiling it. We chose the other
proven design — **Stan's reverse-mode tape** — for three reasons:

1. **No compile step.** JIT compilation costs seconds per model (the eight
   schools benchmark: 2.0 s total in NumPyro, 1.4 s of it compilation).
   Rust is already compiled; a model's first gradient costs microseconds.
2. **It matches how the model is written.** Ordinary Rust arithmetic on a
   scalar type records the tape. No tracer, no restrictions on control flow, no
   "abstract values".
3. **It is fast when done right.** The tape is *flat* (structure-of-arrays:
   `starts`, `parents`, `weights`), reused between gradients, and — crucially —
   **distributions push one node per site, not one node per scalar
   operation**, with partial derivatives computed in closed form.

That last point is the whole game. A naive tape over
`Normal(mu, sigma).log_prob(x)` for a plate of 3000 observations records ~5
nodes per element: 15 000 nodes, 15 000 heap-adjacent pushes, and a 15 000-step
backward sweep. pyroxide records **one** node whose parents are `mu`, `sigma`
(and the 3000 `x`s only if they are latent), with partials `Σ (x-μ)/σ²` etc.
computed in the same loop that computes the value. The backward sweep is then
proportional to the number of *sites*, not the number of *data points*.

```rust
pub trait Real: Copy + Add + Sub + Mul + Div + … {
    type Node: NodeBuilder<Self>;
    fn constant(v: f64) -> Self;
    fn value(self) -> f64;
    const DIFFERENTIABLE: bool;          // false for f64: partial computations compile away
    fn begin_node(capacity: usize) -> Self::Node;   // add(input, partial)…finish(value)
    fn exp(self) -> Self; fn ln(self) -> Self; …     // scalar ops for model code
}
```

`Var` is a 16-byte `Copy` value (value + tape index). The tape is thread-local,
so parallel chains never contend, and `reset()` keeps the allocated capacity so
steady-state sampling does not allocate for autodiff at all.

*Why thread-local rather than passing a tape handle?* Because then `Var`
implements `Add`, `Mul`, … directly and model code reads like math. Stan made
the same trade.

---

## 5. Distributions

Each family implements `Distribution<R>` for both `f64` and `Var`:

```rust
pub trait Distribution<R: Real> {
    fn len(&self) -> usize;             // batch × event scalars per draw
    fn event_len(&self) -> usize;       // 1 for univariate, k for Dirichlet/MVN
    fn support(&self) -> Support;       // drives the unconstraining transform
    fn log_prob(&self, x: &[R]) -> R;         // latent x (differentiable)
    fn log_prob_data(&self, x: &[f64]) -> R;  // observed x
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]);
}
```

Univariate families share one helper (`univariate`) that loops over the batch,
calls a per-element closure returning `(log_prob, ∂/∂params, ∂/∂x)`, and builds
the single tape node — including *broadcast* parameters (a scalar `mu` receives
the sum of per-element partials). Multivariate families (Dirichlet,
MultivariateNormal) are hand-written.

### 5.1 Parameter passing

Parameters accept scalars or slices, constant or differentiable, through
`IntoParam<'a, R>`:

```rust
Normal::new(0.0, 5.0)              // constants
Normal::new(mu, tau)               // differentiable scalars
Normal::new(&theta, &self.sigma)   // &Vec<R> and &Vec<f64>, broadcast against each other
Normal::new(0.0, 1.0).expand(8)    // an iid plate
```

Making all of these work *in generic code* (`fn run<R: Real, …>`) required care
with Rust's coherence rules: a blanket `impl IntoParam<R> for &[R]` overlaps
with a blanket `impl IntoParam<R> for &[f64]` at `R = f64`. The solution is a
small element-type trait `ParamElem<R>` (implemented for `f64` for every `R`,
and for `Var` when `R = Var`) with a single slice impl generic over the element,
and `Real: ParamElem<Self> + IntoParam<Self>` as supertraits. Users never see
this machinery; they just pass `&vec`.

### 5.2 Shapes

NumPyro has full NumPy broadcasting with batch/event shape semantics. pyroxide
deliberately has a flat model: a site is a vector of `len` scalars, univariate
families broadcast scalar parameters against vector parameters and the
`expand(n)` batch size, and multivariate families expose `event_len`. This
covers plates, hierarchical models, and regressions without a shape system, and
it keeps every hot path a plain loop over slices. Matrices are row-major
slices (see `MultivariateNormal::new(loc, scale_tril)`).

### 5.3 Families implemented

Normal, LogNormal, HalfNormal, Cauchy, HalfCauchy, StudentT, Uniform,
Exponential, Gamma, InverseGamma, Beta, Pareto, Laplace, ImproperUniform;
Bernoulli, Binomial (probs or logits), Poisson, Categorical (probs or logits);
Dirichlet, MultivariateNormal. Every log density is checked against SciPy and
every gradient against central finite differences.

---

## 6. The model layer

### 6.1 Handlers

```rust
pub trait Handler<R: Real> {
    fn sample_vec<D: Distribution<R>>(&mut self, name: &str, dist: D) -> Vec<R>;
    fn sample<D: Distribution<R>>(&mut self, name: &str, dist: D) -> R;   // scalar sugar
    fn observe<D: Distribution<R>>(&mut self, name: &str, dist: D, value: &[f64]);
    fn factor(&mut self, name: &str, log_factor: R);
    fn deterministic(&mut self, name: &str, value: &[R]);
}
```

Four handlers cover everything inference needs:

| Handler | `R` | What `sample` does |
|---|---|---|
| `LayoutDiscovery` | f64 | records name, length, event length, support; assigns offsets in the flat unconstrained vector |
| `LogDensity` | f64 or Var | reads the next slice of `z`, applies the constraining transform, adds `log_prob + log|J|` |
| `Tracer` | f64 | draws from the prior (or replays given values), records a `Trace` |
| `Postprocess` | f64 | maps `z` to constrained named values and records deterministic sites |

`observe` adds `log_prob_data` in `LogDensity` and is a no-op in `Postprocess`
(which is why postprocessing is cheap). The model runs once under
`LayoutDiscovery` when a `ModelPotential` is built; afterwards every gradient is
one run under `LogDensity<Var>` plus one backward sweep.

### 6.2 Constraints

Latent supports map to `R^d` by standard bijections: `Positive` → `exp`,
`Interval(a,b)` → affine sigmoid, `Simplex` → stick-breaking (k−1 parameters),
`GreaterThan`/`LessThan` → shifted exp. The log-Jacobian is added by the
handler, so the potential is the density of the *unconstrained* variable, as in
Stan and NumPyro. Discrete distributions can be observed but not latent (the
layout discovery panics with a clear message).

### 6.3 Static structure

Like NumPyro under `jit`, pyroxide assumes the set of sample sites and their
sizes do not change between executions. Data-dependent control flow that
*doesn't* change site structure (e.g. the Poisson change-point model, which
branches on a latent's value) is fine and tested.

---

## 7. Inference

### 7.1 The decoupling point

```rust
pub trait Potential: Send + Sync {
    fn dim(&self) -> usize;
    fn value(&self, z: &[f64]) -> f64;                          // U(z) = -log p(z)
    fn value_and_grad(&self, z: &[f64], grad: &mut [f64]) -> f64;
    fn postprocess(&self, z: &[f64]) -> Vec<(String, Vec<f64>)>; // default: {"z": z}
}
```

`ModelPotential<M>` implements it for any model. `FnPotential` implements it for
a closure over `Var`s. Kernels are generic over `P: Potential`; nothing in
`src/infer` imports `Model`.

### 7.2 Kernels

```rust
pub trait Kernel: Sync {
    type State: Send + Clone;
    fn potential(&self) -> &dyn Potential;
    fn init(&self, z: Vec<f64>, num_warmup: usize, rng: &mut ChainRng) -> Self::State;
    fn step(&self, state: &mut Self::State, rng: &mut ChainRng);
    fn position(state: &Self::State) -> &[f64];
    fn stat_names(&self) -> &'static [&'static str];
    fn stats(&self, state: &Self::State, out: &mut [f64]);
}
```

The kernel is immutable and shared across threads; *all* mutable state — the
position, the adaptation state, and every scratch buffer — lives in `State`.
This is what makes `MCMC` embarrassingly parallel over chains with rayon and
what makes a step allocation-free.

**NUTS / HMC (`HmcKernel`)** is a line-by-line port of `numpyro.infer.hmc_util`:

* velocity-Verlet integrator with a diagonal or dense Euclidean metric;
* the *iterative* tree builder (Phan, Pradhan & Jankowiak 2019) with
  `O(log depth)` momentum checkpoints and the generalized U-turn check at each
  subtree boundary, so memory is `O(max_depth × d)` rather than `O(2^depth × d)`;
* biased progressive sampling (Betancourt 2017) at the top level, uniform
  within subtrees; divergence detection at `ΔE > 1000`;
* warmup: dual averaging on `log ε` toward a target acceptance (0.8), Welford
  (co)variance for the inverse mass matrix updated at the end of each of Stan's
  doubling windows (75 / 25·2^k / 50), regularized as in Stan, optional
  Hoffman–Gelman step-size heuristic at window starts;
* fixed-length HMC shares everything but the trajectory logic.

Tree buffers are four `TreeInfo` structs (main, subtree, leaf, tmp) that are
swapped by pointer; combining trees copies `O(d)` floats, which is negligible
next to the gradient evaluation each leaf requires.

**Metropolis–Hastings (`MetropolisHastings`)** is random-walk with proposal
`z + s·L·ε`, `LLᵀ` the Welford covariance estimate (diagonal or dense, same
windows as NUTS) and `s` tuned by dual averaging toward 0.234. Two differences
from the HMC adapter were needed and are worth recording: the prox center is
`log s` (not `log 10s` — that heuristic exists to favour large early HMC steps),
and dual averaging is *not* restarted at window ends, because the final
50-iteration window is too short for a random walk's noisy acceptance signal to
re-converge (we measured the frozen step landing 2× too large; see the `git
log`). MH uses `Potential::value` only, so models are evaluated with `f64` and
the partial-derivative code compiles away (`Real::DIFFERENTIABLE == false`).

### 7.3 Driver, RNG, initialization

`MCMC::new(kernel, num_warmup, num_samples).num_chains(4).run(seed)`:

* Chains get independent Xoshiro256++ streams by `long_jump` from one seeded
  generator, so results are reproducible and identical whether chains run in
  parallel or sequentially (tested).
* Initialization follows NumPyro: uniform in `(−2, 2)` in unconstrained space,
  retried up to 100 times until the potential and gradient are finite;
  `init_to_value` maps constrained values through the inverse transforms.
* Output is `Samples`: per site a `chains × draws × len` array, plus kernel
  diagnostics per iteration (`accept_prob`, `step_size`, `num_steps`,
  `diverging`, `energy`, …) and `print_summary()`.

---

## 8. Diagnostics

`diagnostics.rs` ports NumPyro's: FFT-based autocorrelation (with
`next_fast_len`), Geyer's initial monotone sequence ESS across chains, Gelman–
Rubin and split R-hat, HPDI, and the summary table. The numerical tests
(`autocorrelation(arange(10))`, `ESS(arange(1000).reshape(100,10)) == 52.64`,
`R-hat == 0.98`) are copied verbatim.

---

## 9. Testing strategy

Stochastic software needs layered tests:

1. **Deterministic unit tests** (`cargo test --lib`, 56 tests): AD gradients vs
   finite differences; every distribution's log density vs SciPy reference
   values and its gradient vs finite differences; sampling moments; transform
   round-trips and log-Jacobians vs numerical Jacobian determinants; dual
   averaging on a quadratic; Welford vs sample covariance; the adaptation
   schedule table; the NUTS checkpoint index arithmetic and turning checks;
   velocity-Verlet on the harmonic, planetary and quartic oscillators
   (accuracy, energy conservation, reversibility); `build_tree` invariants.
2. **End-to-end posterior tests** (`tests/mcmc.rs`, 18 tests, ported from
   `test_mcmc.py`): unnormalized normal (all kernels, diagonal and dense),
   correlated 5-d Gaussian with dense mass, logistic regression (NUTS, HMC, MH),
   uniform/improper reparameterized normal, Beta–Bernoulli, Dirichlet–
   Categorical (simplex transform), dense-mass covariance recovery, Poisson
   change point (`init_to_value`), numerically stable Binomial with 5M trials,
   improper priors, eight schools with 4 chains (R-hat, ESS, parallel ≡
   sequential), fixed-step HMC, extra fields.
3. **Doctests** for the public API.

Where NumPyro's tolerance relied on its data seed (the improper-normal test
compares to 0.9 at `atol=0.007` while the posterior mean is the data mean), the
port compares to the quantity the posterior actually identifies.

---

## 10. Benchmarks

`examples/bench.rs` and `benchmarks/numpyro_bench.py` implement the same six
models — 100-d Gaussian, eight schools (non-centered), Neal's funnel
(non-centered), baseball partial pooling with logit link (NumPyro's
`examples/baseball.py`), logistic regression (N=1000, D=5, shared CSV data), and
a 50-group hierarchical regression (107 latents, 2000 observations) — with the
same kernels and settings, and print JSON. `benchmarks/run_all.sh` runs the
matrix; `benchmarks/report.py` renders `benchmarks/RESULTS.md`.

Methodology notes:

* Both sides use float64 on CPU, one chain, same warmup/sample counts. NumPyro
  times are reported both for a fresh `MCMC.run` (includes XLA compilation) and
  a second run reusing the compiled program.
* NumPyro has no built-in random-walk MH; the harness implements one as an
  `MCMCKernel` (following the sketch in NumPyro's own docstring) with the same
  adaptation as pyroxide, and it is JIT-compiled like any NumPyro kernel.
* Efficiency is reported as minimum ESS per second, since raw wall time can be
  gamed by a poorly adapted sampler that takes shorter trajectories.

See `benchmarks/RESULTS.md` for the numbers on this machine.

---

## 11. Roadmap

* **Documentation site**: mdBook (narrative docs + tutorials) with `cargo doc`
  API reference, deployed by GitHub Actions to GitHub Pages — free, fast, and
  Rust-native (this is what the Rust project itself uses).
* **Performance**: SIMD-vectorized `exp`/`log` for the big plates (XLA's main
  remaining advantage), a fused linear-predictor node for regressions, and
  `f32` support via the `Real` trait.
* **Modeling**: LKJ / correlation Cholesky and ordered-vector transforms,
  truncated distributions, mixtures, a `Predictive` utility over `Tracer`,
  `Samples` export to ArviZ's InferenceData (NetCDF/zarr).
* **Inference**: vectorized chains (an `R = [f64; N]` SIMD scalar would give
  `vmap` for free), SVI with autoguides (the handler layer already supports
  it), discrete-site enumeration (needs a shape system).

---

## 12. Worked example

```rust
use pyroxide::prelude::*;

/// Eight schools (Rubin 1981), non-centered parameterization.
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

    // NUTS, 4 chains in parallel
    let nuts = HmcKernel::nuts(ModelPotential::new(&model)).target_accept_prob(0.9);
    let samples = MCMC::new(nuts, 1000, 1000).num_chains(4).run(0);
    samples.print_summary();
    println!("mean tau = {:.2}", samples.get("tau").scalar_mean());
    println!("theta[0] draws in chain 2: {:?}", &samples.get("theta").column_per_chain(0)[2][..5]);

    // The same model under a different kernel: nothing about the model changes.
    let mh = MetropolisHastings::new(ModelPotential::new(&model)).dense_mass(true);
    let samples = MCMC::new(mh, 10_000, 50_000).run(0);
    samples.print_summary();

    // Inference on a closed-form density, no model at all.
    let banana = FnPotential::new(2, |z: &[Var]| {
        let a = z[0] * z[0] * 0.5;
        let b = (z[1] - z[0] * z[0]).square() * 0.5 / 0.01;
        a + b
    });
    let samples = MCMC::new(HmcKernel::nuts(banana), 1000, 2000).run(1);
    println!("{:?}", samples.get("z").mean());
}
```

The output of `print_summary()` is the NumPyro table:

```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.39      3.31      4.40     -1.03      9.73   4021.61      1.00
       tau      3.61      3.14      2.81      0.00      7.88   2569.83      1.00
    eta[0]      0.39      0.94      0.42     -1.14      1.94   4230.13      1.00
    …
Number of divergences: 0
```
