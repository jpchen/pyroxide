# Writing models

A pyroxide model is a Rust type that implements the `Model` trait:

```rust
pub trait Model: Send + Sync {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H);
}
```

The struct holds the data; `run` describes the generative process by calling
four handler methods. `R` is the scalar type — `f64` when the model is simply
evaluated or simulated, `Var` when it is differentiated — and you never need to
name it beyond the signature.

## The four primitives

| call | meaning |
|---|---|
| `h.sample("name", dist)` | a scalar latent variable with prior `dist`; returns its value as `R` |
| `h.sample_vec("name", dist)` | a vector latent variable (a plate or a multivariate event); returns `Vec<R>` |
| `h.observe("name", dist, &data)` | condition on data: adds `log p(data | dist)` to the joint |
| `h.factor("name", term)` | add an arbitrary term to the log joint density |
| `h.deterministic("name", &values)` | record derived quantities so they appear in the samples |

Sites must have unique names, and the set and sizes of sample sites must be the
same on every execution (data-dependent control flow that *doesn't* change the
site structure is fine).

## Plates and vector parameters

Distributions broadcast scalar parameters against vector parameters, and
`expand(n)` declares a plate of iid draws:

```rust
// 8 iid draws sharing scalar parameters
let eta = h.sample_vec("eta", Normal::new(0.0, 1.0).expand(8));

// per-element locations (a Vec<R>) with a shared scale (an R)
let theta = h.sample_vec("theta", Normal::new(&eta, tau));

// per-element scales from data (a Vec<f64>) with per-element locations
h.observe("y", Normal::new(&theta, &self.sigma), &self.y);
```

Parameters accept `f64`, `R`, `&Vec<f64>`, `&Vec<R>`, `&[f64]`, `&[R]`, and
fixed-size arrays. Everything is a flat vector: a matrix parameter such as a
Cholesky factor is passed row-major.

## Arithmetic

Latent values are ordinary numbers. `+ - * /` work between `R`s and with `f64`
on either side, and the `Real` trait provides `exp`, `ln`, `sqrt`, `powf`,
`sigmoid`, `softplus`, `ln_gamma`, `tanh`, and so on:

```rust
let log_rate = alpha + beta * self.x[i];
let rate = log_rate.exp();
```

For vector arithmetic, prefer the fused helpers in `pyroxide::ad` — each is a
single autodiff node:

| helper | computes |
|---|---|
| `ad::dot_const(&v, &w)` | `Σ v_i w_i` with constant weights |
| `ad::matvec_const(&x, n, k, &beta)` | `X β` for a row-major `n × k` design matrix |
| `ad::dot(&v, &u)`, `ad::sum(&v)`, `ad::logsumexp(&v)` | as named |
| `b.mul_add(x, a)` | `b * x + a` with constant `x` |
| `b.fma(m, a)` | `b * m + a`, all differentiable |

A logistic regression, for example:

```rust
impl Model for LogisticRegression {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let coefs = h.sample_vec("coefs", Normal::new(0.0, 1.0).expand(self.d));
        let logits = ad::matvec_const(&self.x, self.n, self.d, &coefs);
        h.observe("obs", Bernoulli::logits(&logits), &self.labels);
    }
}
```

## Constrained variables

Latent sites are mapped to unconstrained space automatically, based on the
distribution's support: positive scales through `exp`, probabilities through the
logit, simplexes through stick-breaking, correlation Cholesky factors through
signed stick-breaking, and ordered vectors through cumulative `exp`. The
log-Jacobian is added for you. This is why you can write

```rust
let tau = h.sample("tau", HalfCauchy::new(5.0));
let p = h.sample_vec("p", Dirichlet::new(&[1.0, 1.0, 1.0]));
let l = h.sample_vec("L", LKJCholesky::new(3, 2.0));                 // 3x3 row-major
let mu = h.sample_vec("mu", Ordered::new(Normal::new(0.0, 5.0).expand(2), 2));
```

and the sampler sees only unconstrained real numbers.

Discrete distributions (`Bernoulli`, `Binomial`, `Poisson`, `Categorical`) can be
observed but not used as latent sites; marginalize them out with `factor` and
`logsumexp` instead (see the mixture example in the test suite).

## Hierarchical models

The eight-schools model with a non-centered parameterization:

```rust
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
```

Non-centering is a modeling choice you make in code, exactly as in NumPyro with
`LocScaleReparam`, and it changes nothing about how inference is invoked.

## Improper priors

`ImproperUniform::new(support, len)` contributes zero to the log density but
declares a support, so flat priors on a scale or a location are one line each:

```rust
let mean = h.sample("mean", ImproperUniform::new(Support::Real, 1));
let std = h.sample("std", ImproperUniform::new(Support::Positive, 1));
```

## Working with a density directly

If you already have a log density and no generative story, skip models
entirely:

```rust
let banana = FnPotential::new(2, |z: &[Var]| {
    z[0] * z[0] * 0.5 + (z[1] - z[0] * z[0]).square() * 50.0
});
let samples = MCMC::new(HmcKernel::nuts(banana), 1000, 2000).run(1);
```

`FnPotential` returns the raw unconstrained vector under the site name `"z"`.
