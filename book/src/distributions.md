# Distributions

All distributions live in `pyroxide::dist` and implement one trait:

```rust
pub trait Distribution<R: Real> {
    fn len(&self) -> usize;                       // scalars per draw (batch × event)
    fn event_len(&self) -> usize;                 // 1 for univariate families
    fn support(&self) -> Support;                 // drives the unconstraining transform
    fn log_prob(&self, x: &[R]) -> R;             // latent (differentiable) values
    fn log_prob_data(&self, x: &[f64]) -> R;      // observed data
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]);
}
```

Log densities are summed over the batch and pushed onto the autodiff tape as a
single node with analytic partial derivatives — the reason pyroxide's gradients
are cheap. You can use distributions outside of models too:

```rust
use pyroxide::dist::{Normal, Distribution};

let d = Normal::<f64>::new(0.0, 1.0).expand(3);
let lp = d.log_prob_data(&[0.1, -0.2, 0.3]);
let mut rng = rand::rng();
let draw = d.sample_vec(&mut rng);
```

## Parameters and broadcasting

Every parameter accepts a scalar or a vector, constant or differentiable:

| argument | meaning |
|---|---|
| `2.0` | constant scalar |
| `mu` (an `R`) | differentiable scalar |
| `&self.sigma` (`&Vec<f64>`, `&[f64]`, `&[f64; N]`) | constant per-element values |
| `&theta` (`&Vec<R>`, `&[R]`) | differentiable per-element values |

Vector parameters must all have the same length, which becomes the batch size;
scalars broadcast. `expand(n)` sets the batch size when all parameters are
scalars.

## Continuous, univariate

| distribution | parameters | support |
|---|---|---|
| `Normal` | `loc, scale` | real line |
| `LogNormal` | `loc, scale` (of the log) | positive |
| `HalfNormal` | `scale` | positive |
| `Cauchy` | `loc, scale` | real line |
| `HalfCauchy` | `scale` | positive |
| `StudentT` | `df, loc, scale` | real line |
| `Uniform` | `low, high` | interval |
| `Exponential` | `rate` | positive |
| `Gamma` | `concentration, rate` | positive |
| `InverseGamma` | `concentration, rate` | positive |
| `Beta` | `alpha, beta` | unit interval |
| `Pareto` | `scale, alpha` | `(scale, ∞)` |
| `Laplace` | `loc, scale` | real line |
| `ImproperUniform` | `support, len` | as given (density 0) |

## Discrete (observe only)

| distribution | parameters |
|---|---|
| `Bernoulli::new(probs)` / `Bernoulli::logits(l)` | success probability or logit |
| `Binomial::new(total_count, probs)` / `Binomial::logits(total_count, l)` | trials (data) and probability or logit |
| `Poisson` | `rate` |
| `Categorical::new(probs)` / `Categorical::logits(l)` | a `k` vector shared across the batch (`expand(n)`), or an `n × k` row-major matrix (`with_k(k)`) |

The logit parameterizations are numerically stable for extreme logits and huge
counts (a Binomial with five million trials is in the test suite).

## Multivariate

| distribution | event | notes |
|---|---|---|
| `Dirichlet::new(concentration)` | simplex of size `k` | `expand(n)` or `with_k(k)` for batches |
| `MultivariateNormal::new(loc, scale_tril)` | `k` vector | `scale_tril` row-major lower Cholesky factor, may be differentiable |
| `MultivariateNormal::from_covariance(loc, cov)` | `k` vector | constant covariance |
| `LKJCholesky::new(k, eta)` | `k × k` row-major lower Cholesky factor of a correlation matrix | `eta = 1` is uniform over correlation matrices |
| `Ordered::new(base, k)` | increasing `k` vector | restricts any vector distribution to the ordered region |

A covariance prior in the LKJ style (numpyro's `LKJCholesky` docstring model):

```rust
let sigma = h.sample_vec("sigma", HalfCauchy::new(1.0).expand(d));
let l_omega = h.sample_vec("L_omega", LKJCholesky::new(d, 1.0));
let l_cov: Vec<R> = (0..d * d).map(|i| sigma[i / d] * l_omega[i]).collect();
h.observe("obs", MultivariateNormal::new(&vec![0.0; d], &l_cov).expand(n), &self.y);
```

## Supports

`Support` tells inference how to unconstrain a latent site:

`Real`, `Positive`, `UnitInterval`, `Interval(lo, hi)`, `GreaterThan(lo)`,
`LessThan(hi)`, `Simplex`, `OrderedVector`, `CorrCholesky`, and the discrete
`Boolean`, `NonNegativeInteger`, `IntegerInterval(lo, hi)`.

## Adding a distribution

Implement `Distribution<R>` for your type. For a univariate family, use the
`univariate` helper pattern from `src/dist/continuous.rs`: a closure computes
`(log_prob, ∂/∂params, ∂/∂x)` per element and the helper assembles the tape
node, handling scalar broadcasting. Then add three tests: a reference log
density value, a `check_grad` finite-difference comparison, and sampling
moments.
