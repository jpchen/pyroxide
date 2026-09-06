# Feature parity with NumPyro

Status of pyroxide against NumPyro 0.21 (the checkout at `~/research/numpyro`,
which also carries the not-yet-upstreamed microcanonical kernels). Legend:
✅ done · 🟡 partial · ⬜ not started · ✖ out of scope for now.

Priority reflects the plan: **P0** this round, **P1** next, **P2** later.

## MCMC kernels

| NumPyro | pyroxide | priority | notes |
|---|---|---|---|
| `NUTS` | ✅ `HmcKernel::nuts` | | line-by-line port incl. iterative tree, Stan windows, dense mass |
| `HMC` | ✅ `HmcKernel::hmc` | | fixed trajectory or fixed `num_steps` |
| `BarkerMH` | ✅ `BarkerMH` | P0 | gradient-based skew-symmetric proposal, adaptive step + mass |
| `AIES` (affine-invariant ensemble) | ✅ `AIES` | P0 | DE and stretch moves, gradient-free, ensemble of walkers |
| `ESS` (ensemble slice) | ✅ `ESS` | P0 | differential move + stepping out / shrinking, `mu` tuning |
| `MAMS` (adjusted microcanonical, contrib) | ✅ `MAMS` | P0 | isokinetic McLachlan integrator, Halton-jittered length, dual averaging + variance-based `L` |
| `MCLMC` (unadjusted, contrib) | ⬜ | P1 | shares integrator with MAMS; needs energy-variance tuner |
| random-walk Metropolis (no NumPyro built-in) | ✅ `MetropolisHastings` | | adaptive scale + covariance |
| `SA` (sample adaptive) | ⬜ | P2 | gradient-free; needs 10^5 warmup, low value vs ESS/AIES |
| `HMCGibbs` / `DiscreteHMCGibbs` | ⬜ | P1 | needs discrete latent sites + Gibbs conditionals |
| `MixedHMC` | ⬜ | P2 | discrete + continuous |
| `HMCECS` (energy-conserving subsampling) | ⬜ | P2 | needs subsample plates + control variates |
| chain methods | 🟡 | | `parallel` (rayon) and `sequential`; no `vectorized` (see SIMD scalar idea in DESIGN §11) |
| `MCMC.warmup` / `post_warmup_state` reuse | ⬜ | P1 | driver refactor: expose `Kernel::State` between runs |
| `thinning` | ⬜ | P1 | trivial in the driver |
| progress bar | 🟡 | | one line per phase; no live bar |
| `extra_fields` | ✅ | | all kernel stats always collected |
| `init_to_uniform` / `init_to_value` | ✅ | | |
| `init_to_median` / `init_to_mean` / `init_to_sample` / `init_to_feasible` | 🟡 | P1 | `Tracer` makes sample/median easy |

## Variational inference

| NumPyro | pyroxide | priority | notes |
|---|---|---|---|
| `SVI` loop | ✅ `SVI` | | `step`, `run`, `run_with` (minibatch hook), `evaluate`, `sample_posterior` |
| `Trace_ELBO` | ✅ | | reparameterized, `num_particles` samples |
| `TraceMeanField_ELBO` | ⬜ | P1 | analytic KL where available |
| `TraceGraph_ELBO` / `TraceEnum_ELBO` | ✖ | | non-reparameterized / enumerated sites |
| `RenyiELBO` | ⬜ | P2 | |
| `AutoNormal` / `AutoDiagonalNormal` | ✅ `AutoDiagonalNormal` | | |
| `AutoMultivariateNormal` / `AutoLowRankMultivariateNormal` | 🟡 | P1 | full-rank ✅; low-rank ⬜ |
| `AutoDelta` (MAP) | ✅ | | |
| `AutoLaplaceApproximation` | ⬜ | P1 | Hessian via finite differences of the gradient |
| `AutoIAFNormal` / `AutoBNAFNormal` / `AutoDAIS` | ✖ | | normalizing-flow guides |
| custom guides | ✅ | | a guide is a `Model` that calls `param`; `draw` + `sample_given` for joint draws |
| optimizers (`Adam`, `ClippedAdam`, `SGD`, `RMSProp`, ...) | 🟡 | P1 | Adam, ClippedAdam, Sgd ✅ |
| minibatching (`plate(subsample_size)`) | ✅ | | `push_scale`/`pop_scale` + `subsample`; unbiasedness tested |
| `SteinVI` (contrib.einstein) | ✖ | | |

## Sequential Monte Carlo

Not in NumPyro either (it is in Pyro as `pyro.infer.SMCFilter`). See DESIGN §14
for how it maps onto pyroxide's handler model without coroutines.

## Distributions

| group | NumPyro | pyroxide | priority |
|---|---|---|---|
| continuous univariate | Normal, LogNormal, HalfNormal, Cauchy, HalfCauchy, StudentT, Uniform, Exponential, Gamma, InverseGamma, Beta, Pareto, Laplace | ✅ | |
| | Chi2, Gumbel, Logistic, LogUniform, Weibull, Kumaraswamy, AsymmetricLaplace, SoftLaplace, Gompertz, BetaProportion, Dagum, Levy | ⬜ | P1 (each ~40 lines with the `univariate` helper) |
| truncated / censored | LeftTruncated, RightTruncated, TwoSidedTruncated, TruncatedNormal, TruncatedCauchy, Censored | ⬜ | P1 |
| discrete | Bernoulli, Binomial, Poisson, Categorical | ✅ (observe only) | |
| | Geometric, NegativeBinomial, DiscreteUniform, Multinomial, OrderedLogistic, BetaBinomial, DirichletMultinomial, GammaPoisson | ⬜ | P1 |
| zero-inflated / hurdle | ZeroInflatedPoisson, ZeroInflated*, Hurdle* | ⬜ | P1 |
| multivariate | Dirichlet, MultivariateNormal, LKJCholesky | ✅ | |
| | LKJ (full matrix), MultivariateStudentT, LowRankMultivariateNormal, MatrixNormal, Wishart, InverseWishart, GaussianRandomWalk, GaussianStateSpace, CAR, ZeroSumNormal, GaussianCopula | ⬜ | P1–P2 |
| directional | VonMises, ProjectedNormal, SineBivariateVonMises | ⬜ | P2 |
| mixtures | MixtureSameFamily, MixtureGeneral | ⬜ | P1 (expressible today with `factor` + `logsumexp`) |
| `ImproperUniform`, `Unit`, `Delta` | 🟡 | | ImproperUniform ✅, Delta ✅ |
| `Ordered` restriction | ✅ (`Ordered<D>`, not in NumPyro) | | |
| `Independent` / `expand` / `to_event` shape algebra | 🟡 | | flat `len`/`event_len` model instead of batch/event shapes |
| KL divergences (`kl_divergence`) | ⬜ | P1 | needed for `TraceMeanField_ELBO` |
| conjugate updates (`BetaBinomial` ...) | ⬜ | P2 | |

## Constraints and transforms

| NumPyro | pyroxide | priority |
|---|---|---|
| real, positive, unit interval, interval, greater/less than | ✅ | |
| simplex (stick-breaking) | ✅ | |
| ordered vector | ✅ | |
| corr_cholesky | ✅ | |
| lower_cholesky, scaled_unit_lower_cholesky, softplus variants | 🟡 (`LowerCholesky` ✅) | P1 |
| positive_definite (via Cholesky) | ⬜ | P1 |
| l1_ball, sphere, circular, zero_sum | ⬜ | P2 |
| user-defined `TransformedDistribution` | 🟡 `Transformed<D>` over the built-in supports | P1 — arbitrary `Transform<R>` trait |
| `Affine`, `Exp`, `Sigmoid`, `Power`, `Compose` transforms | ⬜ | P1 (with the above) |

## Handlers and model utilities

| NumPyro | pyroxide | priority | notes |
|---|---|---|---|
| `trace` | ✅ `Tracer` | | |
| `condition` / `substitute` | ✅ `Tracer::with_values` | | |
| `seed` | ✅ (explicit RNG everywhere) | | |
| `Predictive` | ✅ | | |
| `log_likelihood` | ⬜ | P1 | one more handler: per-site observed log_prob per draw |
| `deterministic` | ✅ | | |
| `factor` | ✅ | | |
| `plate` (independence + subsampling) | ✅ | | plates via `expand`; subsampling via `push_scale` + `subsample` |
| `scale`, `mask` | 🟡 | P1 | `push_scale` ✅; `mask` ⬜ |
| `block`, `reparam`, `lift`, `do`, `collapse`, `infer_config` | ⬜ | P2 | reparam is a modeling idiom here (write the non-centered form) |
| `render_model` | ⬜ | P2 | graphviz from a `Trace` |
| `format_shapes` | ⬜ | P2 | |
| discrete enumeration (funsor) | ✖ | | |

## Diagnostics

| NumPyro | pyroxide |
|---|---|
| `effective_sample_size`, `gelman_rubin`, `split_gelman_rubin`, `autocorrelation`, `autocovariance`, `hpdi`, `summary`, `print_summary` | ✅ |
| ArviZ interop (`az.from_numpyro`) | ⬜ P1 — write `InferenceData`-compatible NetCDF/zarr, or a simple JSON/Arrow dump |

## Infrastructure

| NumPyro | pyroxide |
|---|---|
| GPU / TPU | ✖ (see DESIGN §12 for the tensor-backend discussion) |
| float32 | ⬜ P1 (`Real for f32`) |
| pickling / serialization of samples | ⬜ P1 (serde feature) |
| documentation site | 🟡 book builds; Pages needs a public repo |
