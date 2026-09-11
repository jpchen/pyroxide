# pyroxide vs NumPyro, Pyro, and Stan

A feature-level comparison of pyroxide against the three systems it borrows
most from. [`PARITY.md`](PARITY.md) tracks the fine-grained NumPyro checklist
and is the authority on what is implemented today; this document is the wider
view: what each system is *for*, where the designs actually differ, and when
you would pick one over another.

Versions referenced: pyroxide 0.1, NumPyro 0.21, Pyro 1.9, Stan 2.36
(CmdStan / cmdstanpy).

---

## 1. One-paragraph summary of each

| System | Model language | Execution | Inference core |
|---|---|---|---|
| **pyroxide** | Rust trait with one generic method; Pyro-style `sample`/`observe`/`factor`/`deterministic` primitives | native Rust, no compile step at run time | reverse-mode tape autodiff (Stan-style), NUTS ported line-by-line from NumPyro |
| **NumPyro** | Python generative function, effect handlers | JAX; XLA-compiled, JIT per model | JAX autodiff, `lax.scan`-based NUTS |
| **Pyro** | Python generative function, effect handlers | PyTorch eager (optionally `torch.compile`) | PyTorch autograd; SVI-first |
| **Stan** | declarative `.stan` DSL, transpiled to C++ and compiled | native C++, ahead-of-time compile per model | Stan Math reverse-mode tape, NUTS |

pyroxide is deliberately the NumPyro design with the Stan execution model: the
model API is Pyro's, the sampler is NumPyro's algorithm, and the numerics are a
Stan-shaped autodiff tape running on the CPU with no tracing, no JIT, and no
Python.

---

## 2. Modeling API

```rust
// pyroxide
impl Model for EightSchools {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let mu  = h.sample("mu", Normal::new(0.0, 5.0));
        let tau = h.sample("tau", HalfCauchy::new(5.0));
        let eta = h.sample_vec("eta", Normal::new(0.0, 1.0).expand(8));
        let theta: Vec<R> = eta.iter().map(|e| mu + tau * *e).collect();
        h.observe("y", Normal::new(&theta, &self.sigma), &self.y);
    }
}
```

```python
# NumPyro / Pyro
def eight_schools(sigma, y=None):
    mu  = sample("mu", dist.Normal(0., 5.))
    tau = sample("tau", dist.HalfCauchy(5.))
    with plate("j", 8):
        eta = sample("eta", dist.Normal(0., 1.))
        theta = mu + tau * eta
        sample("y", dist.Normal(theta, sigma), obs=y)
```

```stan
// Stan
parameters { real mu; real<lower=0> tau; vector[8] eta; }
model {
  mu ~ normal(0, 5); tau ~ cauchy(0, 5); eta ~ std_normal();
  y ~ normal(mu + tau * eta, sigma);
}
```

| Dimension | pyroxide | NumPyro | Pyro | Stan |
|---|---|---|---|---|
| Paradigm | imperative generative | imperative generative | imperative generative | declarative blocks |
| Ordinary control flow in the model | ✅ Rust `if`/`for`/recursion | 🟡 Python control flow is traced; data-dependent branching needs `lax.cond`/`scan` | ✅ eager Python | 🟡 DSL control flow only |
| Model is a first-class value | ✅ a struct (data lives in it, typed) | ✅ a function + closed-over args | ✅ | ✖ a file |
| Effect handlers | ✅ explicit `&mut H`, no global stack | ✅ global handler stack (`with` blocks) | ✅ global handler stack | ✖ |
| Reusable one model → many algorithms | ✅ `Potential` trait | ✅ `potential_fn` | ✅ | ✅ `log_prob` |
| Shape algebra (`batch_shape`/`event_shape`, broadcasting) | ⬜ flat `len`/`event_len` instead | ✅ full | ✅ full | 🟡 typed containers, no broadcasting |
| `plate` | 🟡 independence via `expand`; subsampling via `push_scale`/`subsample` | ✅ | ✅ | ✖ (vectorized statements) |
| Errors surface as | Rust type errors at compile time | Python/JAX tracer errors at run time | Python errors at run time | DSL compile errors |
| Neural networks in the model | ✖ | ✅ (flax/haiku) | ✅ (torch.nn — the main use case) | ✖ |

The trade is legibility of mechanism against expressiveness of shapes. pyroxide
gets compile-time checking, no tracing surprises, and models that are plain Rust
you can step through in a debugger; it gives up broadcasting, which is a real
cost for models that lean on batch/event semantics.

---

## 3. Inference algorithms

### MCMC

| Kernel | pyroxide | NumPyro | Pyro | Stan |
|---|---|---|---|---|
| NUTS | ✅ (port of NumPyro's, incl. iterative tree building) | ✅ | ✅ | ✅ (the original) |
| HMC (fixed trajectory) | ✅ | ✅ | ✅ | ✅ |
| Windowed (Stan) warmup, dense & diagonal mass | ✅ | ✅ | ✅ | ✅ |
| Barker MH | ✅ | ✅ | ✖ | ✖ |
| MAMS (adjusted microcanonical) | ✅ | 🟡 contrib | ✖ | ✖ |
| MCLMC (unadjusted microcanonical) | ⬜ | 🟡 contrib | ✖ | ✖ |
| AIES / ESS (ensemble) | ✅ | ✅ | ✖ | ✖ |
| Random-walk Metropolis | ✅ | ✖ | ✖ | ✖ |
| Sample-adaptive (SA) | ⬜ | ✅ | ✖ | ✖ |
| HMC-within-Gibbs, discrete Gibbs, MixedHMC | ⬜ | ✅ | 🟡 | ✖ |
| HMCECS (subsampling MCMC) | ⬜ | ✅ | ✖ | ✖ |
| Discrete latent enumeration | ✖ out of scope | ✅ (funsor) | ✅ (funsor) | 🟡 marginalize by hand |
| Parallel chains | ✅ rayon threads | ✅ `pmap`/`vectorized` | ✅ multiprocessing | ✅ processes/threads |
| Vectorized chains (one kernel, many chains in lockstep) | ⬜ | ✅ | ✖ | ✖ |

### Variational and point inference

| Feature | pyroxide | NumPyro | Pyro | Stan |
|---|---|---|---|---|
| SVI loop, reparameterized ELBO | ✅ | ✅ | ✅ | ✅ ADVI (deprecated in favor of Pathfinder) |
| Minibatching / subsampling | ✅ | ✅ | ✅ | ✖ |
| AutoDelta (MAP) | ✅ | ✅ | ✅ | ✅ `optimize` |
| AutoDiagonalNormal / AutoNormal | ✅ | ✅ | ✅ | ✅ (ADVI meanfield) |
| AutoMultivariateNormal | ✅ | ✅ | ✅ | ✅ (ADVI fullrank) |
| AutoLowRank / AutoLaplace | ⬜ | ✅ | ✅ | 🟡 Laplace ✅ |
| Normalizing-flow guides (IAF/BNAF/DAIS) | ✖ | ✅ | ✅ | ✖ |
| TraceGraph / TraceEnum ELBO (non-reparam, enumeration) | ✖ | 🟡 | ✅ | ✖ |
| Amortized inference / deep generative models | ✖ | ✅ | ✅ (the flagship use case) | ✖ |
| Custom guides | ✅ (a guide is a `Model` that calls `param`) | ✅ | ✅ | ✖ |
| Optimizers | Adam, ClippedAdam, SGD | full optax | full torch.optim | LBFGS/Adam internally |
| Pathfinder | ⬜ | ✅ | ✖ | ✅ |
| Laplace approximation | ⬜ | ✅ | ✅ | ✅ |
| SMC / particle filtering | ⬜ (design in DESIGN §14) | ✖ | ✅ `SMCFilter` | ✖ |

Pyro remains the clear choice for deep probabilistic modeling — amortized
inference, normalizing flows, and enumeration over discrete latents are its
reason to exist and are explicitly out of pyroxide's scope. NumPyro is the
broadest MCMC toolbox. Stan is the narrowest and the most battle-tested.

---

## 4. Distributions and transforms

| | pyroxide | NumPyro | Pyro | Stan |
|---|---|---|---|---|
| Count of distributions | 22 | ~110 | ~100 | ~50 built-in `_lpdf`s |
| Continuous univariate | 12 core (Normal, LogNormal, HalfNormal, Cauchy, HalfCauchy, StudentT, Uniform, Exponential, Gamma, InverseGamma, Beta, Pareto, Laplace) | full | full | full |
| Discrete | Bernoulli, Binomial, Poisson, Categorical (observed sites only) | full incl. zero-inflated, negative binomial, ordered logistic | full | full |
| Multivariate | Dirichlet, MultivariateNormal, LKJCholesky | + LKJ, MVStudentT, Wishart, CAR, GaussianRandomWalk, ZeroSumNormal, copulas | similar | + Wishart, GP covariances |
| Truncated / censored | ⬜ | ✅ | ✅ | ✅ (`T[a,b]` syntax) |
| Mixtures | 🟡 expressible with `factor` + logsumexp | ✅ | ✅ | 🟡 by hand |
| Directional (VonMises etc.) | ⬜ | ✅ | ✅ | ✅ |
| Constraint transforms | real, positive, interval, simplex, ordered, corr_cholesky, lower_cholesky | full | full | full |
| Arbitrary user transforms / TransformedDistribution | 🟡 `Transformed<D>` over built-in supports | ✅ | ✅ | ✅ (Jacobian by hand) |
| Analytic KL divergences | ⬜ | ✅ | ✅ | ✖ |
| Gradients | hand-written analytic partials per distribution | JAX autodiff of the lpdf | autograd of the lpdf | hand-written analytic (Stan Math) |

pyroxide and Stan share the hand-written-gradient approach: more work per
distribution, fewer nodes on the tape, and no dependence on a general-purpose
AD system to find the efficient form.

---

## 5. Performance and execution model

| | pyroxide | NumPyro | Pyro | Stan |
|---|---|---|---|---|
| Startup cost per model | none | XLA compile, ~0.3–2 s (cached per shape) | none (eager) | C++ compile, ~30–60 s (cached per file) |
| Per-gradient overhead | Rust tape, no interpreter | fused XLA kernel | Python/ATen dispatch per op | C++ tape |
| Small models (≤ ~100 params) | fastest of the four | dominated by dispatch/compile | slowest | fast |
| Large / wide models | competitive (1.5–1.9× NumPyro on 1000–2000 row regressions) | scales best on CPU vector ops, best on GPU | moderate | good |
| GPU / TPU | ✖ out of scope | ✅ | ✅ | 🟡 limited (OpenCL for some GLMs) |
| float32 | ⬜ P1 | ✅ | ✅ | ✖ (double only) |
| Parallel chains | rayon threads, shared model, no process overhead | `pmap` devices or vectorized | processes | processes |
| Deployment artifact | a Rust binary or `cdylib`, no runtime | Python + JAX + XLA | Python + PyTorch | a compiled binary + CmdStan |

Measured on an M4 Max, single chain, float64, 1000 warmup + 1000 samples
(`benchmarks/RESULTS.md`, same models and same ESS in both libraries):

| model | NUTS speedup vs NumPyro | MAMS | Barker | MH |
|---|---:|---:|---:|---:|
| eight schools | 37.7× | 162× | 40.7× | 30.7× |
| Neal's funnel | 66.1× | 355× | 45.8× | 41.6× |
| baseball | 15.7× | 12.1× | 30.7× | 16.1× |
| 100-d Gaussian | 10.2× | 128× | 9.4× | 15.4× |
| hierarchical regression (2000 rows) | 1.9× | 1.1× | 3.1× | 2.9× |
| logistic regression (1000 rows) | 1.5× | 1.9× | 1.9× | 1.7× |

The shape of the result is the point: the advantage is largest where per-step
overhead dominates (small models, many cheap gradient evaluations) and narrows
to ~1.5–2× where the work is real linear algebra that XLA vectorizes well. Two
cases are currently *slower* than NumPyro — plain HMC and AIES on the 2000-row
hierarchical model — because those kernels take many more gradient evaluations
per sample and pyroxide's tape does not vectorize the per-row work.

No comparable head-to-head against Stan or Pyro has been run; the table above is
NumPyro only.

---

## 6. Diagnostics, tooling, ecosystem

| | pyroxide | NumPyro | Pyro | Stan |
|---|---|---|---|---|
| ESS, R-hat, split R-hat, HPDI, autocorrelation, summary table | ✅ | ✅ | ✅ | ✅ |
| Divergence / tree-depth / energy stats | ✅ | ✅ | ✅ | ✅ (+ E-BFMI) |
| ArviZ interop | ⬜ P1 | ✅ | ✅ | ✅ |
| Plotting | ✖ (bring your own) | via ArviZ | via ArviZ | via ArviZ / bayesplot |
| Posterior predictive | ✅ `Predictive` | ✅ | ✅ | ✅ `generated quantities` |
| `log_likelihood` for PSIS-LOO/WAIC | ⬜ P1 | ✅ | ✅ | ✅ (loo package) |
| Model rendering (graphviz) | ⬜ | ✅ | ✅ | ✖ |
| Serialization of draws | ⬜ P1 (serde) | ✅ | ✅ | ✅ (CSV) |
| Formula interface (brms/bambi style) | ✖ | ✖ | ✖ | ✅ via brms/rstanarm |
| Community / packages | new, single crate | large | large | largest in applied Bayesian stats |
| Language | Rust | Python | Python | DSL + R/Python/Julia wrappers |

Ecosystem is where pyroxide is furthest behind and where the gap is structural
rather than a matter of writing more code: Stan's twenty years of applied
literature, brms, and loo, or Pyro's PyTorch interop, are not things a young
crate closes.

---

## 7. When to use which

**pyroxide** — the model is small-to-medium and CPU-bound, per-step overhead
dominates, and you want the sampler embedded in a Rust service or CLI with no
Python runtime, no JIT warmup, and a single static binary. Also when you want to
read and modify the sampler itself: it is ~11k lines of ordinary Rust.

**NumPyro** — the broadest algorithm coverage, GPU/TPU, discrete enumeration,
vectorized chains, and large models where XLA's vectorization is the whole game.
The default choice for research-grade Bayesian modeling in Python today.

**Pyro** — deep probabilistic programming: amortized inference, variational
autoencoders, normalizing-flow guides, neural networks inside the model,
enumeration over discrete latents, and SMC. If the model contains a `torch.nn`
module, this is the answer.

**Stan** — applied statistics with a need for maximal trust: the reference NUTS
implementation, the deepest diagnostic culture, and an ecosystem (brms, loo,
bayesplot, posteriordb) that no other system matches. Best when the model is
expressible in the DSL and the up-front C++ compile is amortized over a long
analysis.

---

## 8. Honest list of what pyroxide does not have

Beyond the per-row gaps above, the load-bearing omissions:

1. **No broadcasting / shape algebra.** Models are written over flat vectors.
   Anything that leans on `batch_shape`/`event_shape` has to be rewritten.
2. **No GPU.** Explicitly out of scope (DESIGN §12).
3. **No discrete latent variables.** Continuous latents only; discrete sites are
   observe-only. No enumeration, no HMCGibbs, no MixedHMC.
4. **22 distributions,** against ~100 in NumPyro and Pyro.
5. **No ArviZ / serialization.** Draws live in a `Samples` struct; getting them
   into the standard diagnostic ecosystem means writing the bridge yourself.
6. **No neural-network integration.** No autodiff interop with a DL framework.
7. **New and unvalidated by use.** The kernels are tested against NumPyro's own
   posterior tests and the warmup is a numerically exact port, but nothing here
   has the mileage of Stan or NumPyro.
