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

Two smaller tricks matter for regression-style models where the linear
predictor, not the likelihood, dominates: `matvec_const` pushes all `n` rows of
`X·β` under one tape borrow (one node per row with `k` parents), and
`mul_add` / `fma` fuse `a + b·x` into one node instead of two. The first
benchmark round showed exactly this: the 2000-observation hierarchical
regression was on par with XLA until the per-observation `mul` + `add` pair was
fused and `log σ` was hoisted out of the plate loop for scalar scales.

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
Dirichlet, MultivariateNormal, LKJCholesky; and the `Ordered<D>` wrapper that
restricts any vector distribution to increasing vectors (Stan's `ordered`).
Every log density is checked against SciPy or numpyro reference values and
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
`GreaterThan`/`LessThan` → shifted exp, `OrderedVector` → cumulative exp,
`CorrCholesky` → tanh partial correlations with signed stick-breaking (Stan's
`cholesky_corr_constrain`; `k(k-1)/2` parameters for a `k × k` factor). The
log-Jacobian is added by the handler, so the potential is the density of the
*unconstrained* variable, as in Stan and NumPyro. Transforms are checked by
round trip, by comparing log-Jacobians against numerical Jacobian determinants,
and against numpyro's `CorrCholeskyTransform` / `OrderedTransform` values. Discrete distributions can be observed but not latent (the
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

**More kernels (round three).** All follow the same `Kernel` contract and
were ported from numpyro with their tests:

* `BarkerMH` — the Barker proposal (Livingstone & Zanella 2022): one gradient
  per step, sign-flipped Gaussian increments with `P(keep) = σ(z ∂ log π)`,
  and the skew-symmetric acceptance correction. Uses the HMC warmup adapter
  with target acceptance 0.4.
* `AIES` / `ESS` — ensemble samplers (`emcee` / `zeus`). These required one
  extension of the `Kernel` trait: `is_ensemble`, `init_ensemble` and
  `position_of(state, walker)`, so a state can hold a whole population and
  the driver reports each walker as a chain. Both use only `Potential::value`.
* `MAMS` — Metropolis-adjusted microcanonical sampling (Robnik et al. 2025),
  ported from the `numpyro.contrib.microcanonical` branch (itself from
  BlackJAX). Isokinetic dynamics with the closed-form ESH momentum rotation,
  McLachlan two-stage splitting, Halton-jittered trajectory length, two-phase
  dual averaging with the trajectory length set from warmup variances.
  This is the current "state of the art" contender to NUTS: exact, no tree
  building, and often more effective samples per gradient. The integrator is
  unit-tested for energy conservation and sphere constraint; the kernel passes
  the same posterior tests as NUTS.

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

### 8.1 Predictive simulation

`Predictive::new(&model).posterior(&samples).run(seed)` replays each posterior
draw through the model under `Tracer` in *predictive mode*: latent sites take
the stored values, observed sites are **sampled** from their distributions, and
deterministic sites are recomputed. Without `posterior` it draws from the prior
(prior predictive checks). The result is a `Samples` with the same
chains × draws structure, so the diagnostics and summaries apply unchanged.

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

### 10.1 Validation of the port

Before comparing speed we checked that the two implementations do the *same
thing*. Running numpyro's HMC on the hierarchical-regression benchmark with
`collect_warmup=True` and comparing the adapted step size after each of the
first warmup iterations against pyroxide's gives identical trajectories to five
significant digits (2.33506, 0.230240, 0.0166900, 0.00107, …), and both
libraries take ~80 000 leapfrog steps during HMC warmup on that model (the
fixed-trajectory-length pathology when the early step size is tiny — the reason
NUTS exists). The dual-averaging, windowing and integrator arithmetic are
therefore bit-for-bit the same algorithm; remaining differences are RNG streams.

One artifact to know about when reading numpyro's ensemble numbers: its `ESS`
kernel permutes the walker array in place every iteration when
`randomize_split=True` (the default), so "chain k" is a different walker at
every step. That destroys the per-chain autocorrelation the ESS estimator
measures — on a 10-d standard normal numpyro reports lag-1 autocorrelation 0.08
for a sampler that moves along one direction per step (the exact value is
`1 - 1/d ≈ 0.9`, which pyroxide reproduces). The benchmark therefore runs
numpyro's `ESS` with `randomize_split=False`; pyroxide randomizes only the
active/inactive split and keeps walker identity.

### 10.2 What the numbers say

`benchmarks/RESULTS.md` holds the table for this machine (Apple M4 Max). In
summary:

* On small and medium models (eight schools, Neal's funnel, baseball, a 100-d
  Gaussian) pyroxide's NUTS is **10–70× faster** than numpyro's compiled sampler
  and produces the same effective sample sizes; Metropolis–Hastings is 15–40×
  faster. Here XLA's per-iteration control-flow overhead dominates numpyro, and
  pyroxide's allocation-free tree builder has essentially none.
* On models whose cost is a large vectorized likelihood (logistic regression
  with 1000 rows, hierarchical regression with 2000 rows) the two are within a
  small factor of each other. XLA vectorizes the per-observation arithmetic with
  SIMD; pyroxide evaluates it in a scalar loop with one tape node per row of the
  linear predictor. Fusing `a + b·x` into one node and hoisting `log σ` out of
  the plate loop (first benchmark round → second, both kept under
  `benchmarks/results/`) moved NUTS on these two models from parity to 1.5–1.9×
  faster than numpyro (the third round's fused `mul_add_vec` linear predictor
  took the hierarchical regression from 1.5× to 1.9×). Fixed-length HMC on the hierarchical regression is the
  one row where numpyro wins (≈5×): both libraries spend ~80 000 leapfrog steps
  in warmup there, so the comparison is purely per-gradient cost — about 40 µs
  in pyroxide's scalar loop against ~12 µs for XLA's SIMD-vectorized likelihood.
  SIMD kernels for the plate loops are the next step (§11).
* The third round added four kernels to the matrix. MAMS in pyroxide runs
  12–350× faster than the numpyro `contrib.microcanonical` port on the small
  models with matching effective sample sizes and acceptance rates (both ports
  take the same number of integrator steps — 1000 on the funnel and the 100-d
  Gaussian, ~1800 on eight schools — so this is per-step overhead), and 1.1×
  on the hierarchical regression where the gradient dominates. Barker MH is
  2–46× faster with better-adapted acceptance (0.35–0.43 against a 0.4 target;
  numpyro's restarts leave it at 0.21–0.30). The ensemble samplers are 2–32×
  faster with 20 walkers; with 220 walkers on the 2000-row hierarchical model
  numpyro's vectorized evaluation of all walkers in one XLA call wins (0.2× for
  AIES, 1.3× for ESS) — the one place a `vmap`-style batched potential would
  pay off. Both libraries agree closely on the sampler statistics themselves
  (leapfrog counts, acceptance, R-hat), which is the port-fidelity check.
* HMC with a fixed `2π` trajectory is a poor sampler on several of these
  targets in *both* libraries (ESS of 3–30, R-hat > 2 on the Gaussian, whose
  period is exactly 2π). It is included because it is the same algorithm with
  the same defaults on both sides, not because anyone should use it that way.

---

## 11. Roadmap

* **Documentation site**: mdBook (narrative docs + tutorials) with `cargo doc`
  API reference, deployed by GitHub Actions to GitHub Pages — free, fast, and
  Rust-native (this is what the Rust project itself uses).
* **Performance**: we tried SIMD-vectorized `exp`/`log` for the big plates (a
  branch-free fdlibm port that LLVM auto-vectorizes). On Apple Silicon it was a
  wash: the system libm already evaluates `exp` in ~4.6 ns and `ln` in ~2.5 ns,
  and the fused `softplus`/`sigmoid` pair took 7 ns scalar against 11 ns
  vectorized, so the code was removed. Profiling the 1000-row logistic
  regression gradient (28 µs) shows the time is tape traffic — 10 µs to push the
  1000 linear-predictor nodes, 4 µs to build the likelihood node, 6.5 µs for the
  backward sweep — not transcendentals. The next lever is *range parents*: a
  node whose parents are a contiguous index range (as `matvec_const` outputs
  are) needs no per-parent storage at all. `f32` support via the `Real` trait
  is straightforward.
* **Modeling**: truncated distributions, mixture distributions (mixtures are
  already expressible with `factor` + `logsumexp`), `Samples` export to
  ArviZ's InferenceData (NetCDF/zarr). LKJ / correlation-Cholesky and
  ordered-vector transforms and the `Predictive` utility landed in the second
  round (§5.3, §6.2, §8.1).
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

---

## 12. Tensors, autodiff backends, and deep learning

**What pyroxide uses today.** There is no tensor library and no third-party
autodiff. Everything is `f64` slices and the scalar tape in `src/ad.rs`:
`Var` is a value plus a `u32` index, arithmetic on `Var` records a node,
distributions record one node per site with hand-derived partials, and
`gradient_into` runs one reverse sweep. It is "vectorized" only in the sense
that a whole plate is one node; the arithmetic inside a plate is a scalar loop
that LLVM auto-vectorizes where it can (the fused helpers `matvec_const`,
`mul_add_vec` exist so that loop is tight). We measured this against XLA in
§10: faster on everything except very large plates, where XLA's fused,
SIMD-vectorized likelihood evaluates a gradient in ~12 µs against our ~40 µs.

**Why not build on a tensor library from the start?** For the models MCMC is
used on — tens to a few thousand parameters, likelihoods that are sums over
data — a tensor library's strengths (large dense matmuls, GPU dispatch,
broadcasting) are irrelevant and its costs (per-op dispatch of ~1–10 µs,
allocation per intermediate, dynamic shapes) dominate. A NUTS leaf on eight
schools costs pyroxide ~2 µs total; a single `candle` or `tch` op costs more
than that. Stan reached the same conclusion two decades ago. The scalar tape is
the right tool for the core.

**How the Rust options compare** (for the case where you *do* want tensors):

| library | what it is | autodiff | GPU | fit for pyroxide |
|---|---|---|---|---|
| `candle` (HF) | PyTorch-like tensors, minimal, fast CPU/CUDA/Metal kernels; runs LLMs | yes, dynamic tape over tensor ops | CUDA, Metal | best candidate for a *tensor backend*: mature, actively maintained, `Tensor::backward()` gives gradients w.r.t. `Var` leaves |
| `burn` | framework with pluggable backends (ndarray, wgpu, candle, tch, CUDA) | yes, via `Autodiff<B>` backend decorator | via backends | heavier abstraction; good if portability to wgpu matters |
| `tch` | bindings to libtorch | yes | CUDA/MPS | brings the whole PyTorch runtime; pragmatic if you already have torch models |
| `dfdx` | shape-typed tensors, autodiff | yes | CUDA | compile-time shapes make dynamic plates awkward |
| `ndarray` + `faer` / `nalgebra` | arrays and linear algebra, no autodiff | no | no | useful for dense mass matrices and Laplace approximations, not for gradients |
| `enzyme` (LLVM plugin, `rustc` `-Zautodiff`) | source-level AD of Rust code | yes | n/a | nightly-only today; would make hand-written partials unnecessary |

**The integration point is `Potential`, not `Real`.** Inference needs one
thing from a model: `value_and_grad(z, grad)`. That contract is satisfied by
any system that can differentiate a scalar with respect to a flat vector:

```rust
struct CandlePotential { model: MyCandleNet, data: Tensor, dim: usize }

impl Potential for CandlePotential {
    fn dim(&self) -> usize { self.dim }
    fn value(&self, z: &[f64]) -> f64 { self.log_joint(Tensor::new(z, &Device::Cpu)?)... }
    fn value_and_grad(&self, z: &[f64], grad: &mut [f64]) -> f64 {
        let zt = Var::from_slice(z, (self.dim,), &Device::Cpu)?;      // candle Var = leaf
        let u = -self.log_joint(zt.as_tensor());                       // candle graph
        let grads = u.backward()?;                                     // candle reverse mode
        grad.copy_from_slice(&grads.get(&zt)?.to_vec1::<f64>()?);
        u.to_scalar::<f64>()?
    }
}
let samples = MCMC::new(HmcKernel::nuts(CandlePotential { .. }), 1000, 1000).run(0);
```

Nothing in `infer/` changes: NUTS, MAMS, Barker and the ensemble samplers
run unmodified on a candle-backed density, on a GPU if the tensors live there.
A `pyroxide-candle` companion crate would provide (a) that adapter, (b)
tensor-valued distributions (`Normal<Tensor>` etc.) implementing a
`TensorDistribution` trait with `log_prob(&Tensor) -> Tensor`, and (c) a
`Handler` whose scalar type is a candle `Tensor` of shape `()` — the `Real`
trait as written is `Copy`, so a tensor handle would need a small wrapper, or
the handler layer generalizes `R` to `Clone`. That is a contained change.

**"Sampling from LLMs."** Two different things hide behind that phrase, and
they need different machinery:

1. *Bayesian inference over network weights* (Bayesian neural nets, last-layer
   Bayes, LoRA-adapter posteriors). The parameter vector is `10^5`–`10^7`
   dimensional and the likelihood is a forward pass over a minibatch. HMC is
   possible only with stochastic gradients (SGHMC, SGLD) or subsampled
   energy-conserving variants (HMCECS); exact NUTS is not. This is the
   candle-`Potential` route plus a stochastic-gradient kernel — a natural
   follow-on once minibatched SVI (§13) exists, since both share the
   subsampled-likelihood plumbing. Last-layer or adapter posteriors with a few
   thousand parameters are very much in range of MAMS/NUTS today via the
   adapter above.

2. *Sampling sequences from a distribution defined through an LLM* — e.g.
   `p(x) ∝ p_LM(x) · exp(r(x))` for a reward or constraint, or conditioning a
   generation on a downstream classifier. Here the state is discrete (tokens),
   there are no gradients, and the log density is one forward pass per
   evaluation (expensive). The relevant machinery is Metropolis–Hastings with
   proposals from the LM itself (independence or block-resampling proposals),
   sequential Monte Carlo / twisted SMC over the token prefix, and
   importance weighting — all *gradient-free* kernels over an opaque
   `Potential::value`. pyroxide's `MetropolisHastings`, `AIES` and `ESS`
   already only need `value`; a `TokenPotential` wrapping a candle LLM plus a
   proposal kernel that resamples a span of tokens from the LM is the missing
   piece, and SMC (§14) is the scalable version.

Neither requires changing the scalar tape; both are additive crates.

---

## 13. Variational inference

*Status: implemented as planned below (`src/infer/svi/`): `SVI`, `Trace_ELBO`
with `num_particles`, `AutoDelta`, `AutoDiagonalNormal`,
`AutoMultivariateNormal`, `Adam`/`ClippedAdam`/`Sgd`, `push_scale`/`pop_scale`
minibatching with the `subsample` helper, and the tests of §13.5. The one
deviation from the sketch: the parameter method is `param(name, init,
support)` and the joint-draw idiom is `draw` + `sample_given` rather than a
separate `rsample` handler method.*

SVI reuses everything above except the kernel. The pieces:

### 13.1 Parameters and guides

A *guide* is a `Model` that may also declare learnable parameters:

```rust
pub trait Handler<R: Real> {
    // existing: sample_vec / sample / observe / factor / deterministic
    /// Declare a learnable parameter (returns its current value). Unconstrained
    /// by default; `support` maps it through the constraining transform.
    fn param(&mut self, name: &str, init: &[f64], support: Support) -> Vec<R>;
    /// Scale factor applied to log-density terms declared until `pop_scale`
    /// (minibatch upweighting). Nested scales multiply.
    fn push_scale(&mut self, scale: f64);
    fn pop_scale(&mut self);
}
```

`param` on the existing handlers: `LogDensity` and `Tracer` return the stored
value (guides are traced like models); `LayoutDiscovery` records the parameter
layout in a `ParamStore` (name → offset, length, support), analogous to
`SiteLayout` for latents.

Autoguides are plain `Model` implementations generated from a `SiteLayout`:

* `AutoDelta` — one `param` per latent site (MAP).
* `AutoDiagonalNormal` — `loc` and `scale` params per latent, `z = loc +
  scale · ε`, mapped through the site's constraining transform, with
  `log q` = Normal log density minus the log-Jacobian.
* `AutoMultivariateNormal` — one `loc` vector and a `scale_tril`
  (`LowerCholesky` support) over the flat unconstrained vector.
* `AutoLowRankMultivariateNormal`, `AutoLaplaceApproximation` afterwards.

### 13.2 The ELBO

`Trace_ELBO` with one reparameterized sample:

1. Run the guide under a `GuideTrace<Var>` handler: parameters are `Var`
   leaves (read from a flat parameter vector), noise `ε` is drawn with the RNG,
   each `sample` returns a *differentiable* `z` (the reparameterization
   trick: `z = loc + scale·ε` is a function of `Var`s), and `log q(z)` is
   accumulated.
2. Run the model under a `Replay<Var>` handler that returns the guide's `z` for
   each latent site and accumulates `log p(x, z)` (respecting `push_scale`).
3. `elbo = log p − log q`; one backward sweep gives `∂elbo/∂params`.

This is exactly numpyro's `Trace_ELBO` with `num_particles = 1`; averaging over
particles is a loop. `TraceMeanField_ELBO` swaps the sampled `log q − log p_z`
for analytic KL terms where both sides are in the same family (`kl_divergence`
table on distribution pairs). Non-reparameterizable guides (discrete latents)
are out of scope, as in NumPyro's default ELBO.

### 13.3 Optimizers and the loop

`Adam`, `ClippedAdam`, `SGD`, `RMSProp` as small structs over flat `Vec<f64>`
with a `step(&mut params, &grad)` method (~30 lines each; Adam with bias
correction is the default, lr `1e-3`).

```rust
let guide = AutoDiagonalNormal::new(&model);
let mut svi = SVI::new(&model, &guide, Adam::new(0.01), TraceELBO::new(1));
let state = svi.run(seed, 5000);          // prints loss periodically
let posterior = guide.sample_posterior(&state.params, 1000, seed);  // Samples
let loss = svi.evaluate(&state, 100);      // Monte Carlo ELBO estimate
```

`svi.run` supports a callback per step for logging / early stopping, and
`svi.step(&mut state)` for custom loops.

### 13.4 Minibatching

NumPyro's `plate(..., subsample_size=n)` does two things: it draws a random
subset of indices and it scales the log density of everything inside by
`N / n`. pyroxide makes both explicit and both trivial to test:

```rust
struct Regression { x: Vec<f64>, y: Vec<f64>, batch: Mutex<Option<Vec<usize>>> }   // batch set per step

impl Model for Regression {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let w = h.sample("w", Normal::new(0.0, 1.0));
        let b = h.sample("b", Normal::new(0.0, 1.0));
        let scale = self.y.len() as f64 / self.batch.len() as f64;
        h.push_scale(scale);
        let xb: Vec<f64> = self.batch.iter().map(|&i| self.x[i]).collect();
        let yb: Vec<f64> = self.batch.iter().map(|&i| self.y[i]).collect();
        let mean: Vec<R> = xb.iter().map(|xi| w * *xi + b).collect();
        h.observe("y", Normal::new(&mean, 0.5), &yb);
        h.pop_scale();
    }
}
```

`SVI::run_minibatch(seed, steps, |step, rng| model.set_batch(...))` (or the
user resampling `batch` in the callback) is the loop; a `Subsample::draw(rng,
N, n)` helper provides the index sampling. Because the model owns its data and
`run` takes `&self` and models are `Sync`, the batch is stored behind a `Mutex`
or the model is rebuilt per step — both are cheap. Since minibatch gradients are unbiased
for the full ELBO gradient, the same `Trace_ELBO` code applies unchanged; the
scaled `observe` is the only new handler semantics, and `LogDensity` honours it
too, so MCMC kernels get **stochastic-gradient variants for free** (SGLD /
SGHMC as `Kernel`s over a `Potential` whose `value_and_grad` is noisy).

### 13.5 Tests

Ported from `test_svi.py` / `test_autoguide.py`: Beta–Bernoulli with
`AutoDiagonalNormal` recovers the analytic posterior mean; logistic regression
with `AutoMultivariateNormal` matches NUTS moments; `AutoDelta` on a Gaussian
recovers the MAP exactly; minibatch ELBO gradient is an unbiased estimate of
the full-data gradient (compare averages over many batches to the exact
gradient); Adam on a quadratic converges. Loss curves must decrease
monotonically on the smoothed average.

---

## 14. Sequential Monte Carlo without coroutines

Pyro's `SMCFilter` relies on Python generators: the model is a coroutine that
yields at each time step so the filter can reweight and resample between
steps. Rust has no stable coroutines, but SMC does not need them — the
coroutine is only a convenient way to *pause a program at a site boundary*. Two
equivalent formulations fit pyroxide directly.

**1. Explicit state-space interface (recommended first).** A particle filter
needs three things from the user: an initial distribution, a transition, and
an observation likelihood. Make that a trait:

```rust
pub trait StateSpaceModel {
    type Obs;
    fn init(&self, h: &mut impl Handler<f64>) -> Vec<f64>;                   // sample x_0
    fn transition(&self, t: usize, x: &[f64], h: &mut impl Handler<f64>) -> Vec<f64>;   // sample x_t | x_{t-1}
    fn observe(&self, t: usize, x: &[f64], y: &Self::Obs) -> f64;             // log p(y_t | x_t)
}
```

The bootstrap filter is then a plain loop over `t`: propagate every particle
with `transition` under a `Tracer`, weight by `observe`, resample
(systematic / multinomial / stratified) when the ESS drops below a threshold,
and record the log normalizing-constant increments. Guided proposals are one
more trait method with a default. This is how every Rust or C++ SMC library
(e.g. `SMCTC`, `libbi`) is written, and it maps to pyroxide's handlers with no
new language features. Particles are rows of an `Array`, so diagnostics and
`Samples` reuse applies.

**2. Handler-driven SMC for arbitrary generative programs** (Pyro's
generality). The trick is that we do not need to *suspend* the program; we
can *re-run it under a handler that replays the past*. An `SmcHandler` runs
the whole model once per particle per time step, replays the already-decided
latent values for sites with time index `< t` (from the particle's trace),
samples sites at time `t`, and stops accumulating log-weight at the first
`observe` of time `t+1`. Cost is `O(T)` re-execution per step (`O(T²)`
total), which is acceptable for the `T ≲ 10³` regime where general-purpose
SMC is used and can be avoided entirely with formulation 1. Site naming
carries the time index (`format!("x_{t}")`), as in Pyro. This is also the
mechanism for *twisted SMC over token sequences* (§12): sites are tokens,
the "observe" is the twist/reward.

**Where SMC sits in the architecture.** Neither formulation touches `ad`,
`dist` or `infer::Kernel`. `SmcFilter` is a new driver alongside `MCMC` that
consumes handlers; particle MCMC (PMMH) then falls out by using the filter's
log-normalizing-constant estimate as a `Potential::value` inside
`MetropolisHastings` — another demonstration that `Potential` is the right
seam.
