//! Stochastic variational inference.
//!
//! A *guide* is a [`Model`] that declares learnable parameters with
//! [`Handler::param`] and samples the model's latent sites from a tractable
//! family. [`SVI`] maximizes the evidence lower bound
//! `ELBO = E_q[log p(x, z) - log q(z)]` by stochastic gradient ascent, using
//! one (or a few) reparameterized samples per step:
//!
//! 1. the guide runs under [`GuideTrace`] — parameters are `Var` leaves read
//!    from a flat vector, `sample` draws `z = T(loc + scale·ε)` differentiably
//!    and accumulates `log q`;
//! 2. the model runs under [`Replay`](crate::model::Replay) with those values,
//!    accumulating `log p(x, z)` (respecting `push_scale` for minibatches);
//! 3. one backward sweep gives `∂ELBO/∂θ` and the optimizer takes a step.
//!
//! ```no_run
//! # use pyroxide::prelude::*;
//! # use pyroxide::infer::svi::*;
//! # struct M; impl Model for M { fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {} }
//! # let model = M;
//! let guide = AutoDiagonalNormal::new(&model);
//! let mut svi = SVI::new(&model, &guide, Adam::new(0.01));
//! let losses = svi.run(0, 2000);
//! let posterior = svi.sample_posterior(1, 1000);   // Samples from the fitted guide
//! posterior.print_summary();
//! ```

pub mod autoguide;
pub mod optim;

pub use autoguide::{AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal};
pub use optim::{Adam, ClippedAdam, Optimizer, Sgd};

use std::collections::HashMap;

use rand::{RngCore, SeedableRng};

use crate::ad::{self, Real, Var};
use crate::dist::{Distribution, Support};
use crate::infer::{Array, ChainRng, Samples};
use crate::model::{transform, Handler, Model, ParamLayout, Replay, SiteLayout, Tracer};

/// Handler for running a guide during the ELBO computation.
///
/// Parameters come from the flat unconstrained vector `theta` (as `R`, so they
/// are differentiable when `R = Var`), sample sites are drawn with the
/// reparameterization trick, and `log q` is accumulated.
pub struct GuideTrace<'a, R: Real> {
    theta: &'a [R],
    layout: &'a ParamLayout,
    rng: &'a mut dyn RngCore,
    /// `(site name, value)` for every latent site the guide declared.
    pub values: Vec<(String, Vec<R>)>,
    /// Accumulated `log q(z)`.
    pub log_q: R,
    scale: f64,
    scale_stack: Vec<f64>,
}

impl<'a, R: Real> GuideTrace<'a, R> {
    pub fn new(theta: &'a [R], layout: &'a ParamLayout, rng: &'a mut dyn RngCore) -> Self {
        GuideTrace {
            theta,
            layout,
            rng,
            values: Vec::new(),
            log_q: R::zero(),
            scale: 1.0,
            scale_stack: Vec::new(),
        }
    }

    #[inline]
    fn add(&mut self, term: R) {
        self.log_q += if self.scale == 1.0 {
            term
        } else {
            term * self.scale
        };
    }
}

impl<'a, R: Real> Handler<R> for GuideTrace<'a, R> {
    fn sample_vec<D: Distribution<R>>(&mut self, name: &str, dist: D) -> Vec<R> {
        let (z, lq) = dist.rsample_log_prob(self.rng);
        self.add(lq);
        self.values.push((name.to_string(), z.clone()));
        z
    }
    fn observe<D: Distribution<R>>(&mut self, name: &str, _dist: D, _value: &[f64]) {
        panic!("guides cannot observe data (site '{name}')");
    }
    fn factor(&mut self, _name: &str, log_factor: R) {
        self.add(log_factor);
    }
    fn deterministic(&mut self, _name: &str, _value: &[R]) {}
    fn param(&mut self, name: &str, _init: &[f64], support: Support) -> Vec<R> {
        let info = self
            .layout
            .param(name)
            .unwrap_or_else(|| panic!("param '{name}' not in the guide's parameter layout"));
        let u = &self.theta[info.offset..info.offset + info.unconstrained_len];
        let mut out = Vec::with_capacity(info.len);
        let event_len = match support {
            Support::Simplex | Support::CorrCholesky | Support::LowerCholesky => info.len,
            _ => 1,
        };
        transform::to_constrained(support, event_len, u, &mut out);
        out
    }
    fn push_scale(&mut self, scale: f64) {
        self.scale_stack.push(self.scale);
        self.scale *= scale;
    }
    fn pop_scale(&mut self) {
        self.scale = self.scale_stack.pop().expect("pop_scale without push_scale");
    }
    fn sample_given(&mut self, name: &str, value: Vec<R>, log_prob: R) -> Vec<R> {
        self.add(log_prob);
        self.values.push((name.to_string(), value.clone()));
        value
    }
    fn draw<D: Distribution<R>>(&mut self, dist: D) -> Vec<R> {
        dist.rsample(self.rng)
    }
}

/// Stochastic variational inference driver.
pub struct SVI<'a, M: Model, G: Model, O: Optimizer> {
    pub model: &'a M,
    pub guide: &'a G,
    pub optimizer: O,
    /// Number of reparameterized samples per gradient estimate.
    pub num_particles: usize,
    /// Parameter layout discovered from the guide.
    pub layout: ParamLayout,
    /// Current unconstrained parameter vector.
    pub theta: Vec<f64>,
    grad: Vec<f64>,
    pub step_count: usize,
}

impl<'a, M: Model, G: Model, O: Optimizer> SVI<'a, M, G, O> {
    pub fn new(model: &'a M, guide: &'a G, optimizer: O) -> Self {
        let layout = SiteLayout::discover(guide).params;
        let theta = layout.initial_vector();
        let mut optimizer = optimizer;
        optimizer.init(theta.len());
        SVI {
            model,
            guide,
            optimizer,
            num_particles: 1,
            grad: vec![0.0; theta.len()],
            layout,
            theta,
            step_count: 0,
        }
    }

    pub fn num_particles(mut self, n: usize) -> Self {
        self.num_particles = n.max(1);
        self
    }

    /// Start from given constrained parameter values (others keep their init).
    pub fn init_params(mut self, values: &HashMap<String, Vec<f64>>) -> Self {
        for p in &self.layout.params {
            if let Some(v) = values.get(&p.name) {
                let u = transform::to_unconstrained(p.support, p.len, v);
                self.theta[p.offset..p.offset + p.unconstrained_len].copy_from_slice(&u);
            }
        }
        self
    }

    /// One stochastic ELBO estimate and its gradient with respect to `theta`
    /// (written to `grad`). Returns the ELBO.
    fn elbo_and_grad(&self, rng: &mut dyn RngCore, grad: &mut [f64]) -> f64 {
        ad::reset();
        let theta = Var::leaves(&self.theta);
        let mut total = Var::constant(0.0);
        for _ in 0..self.num_particles {
            let mut g = GuideTrace::new(&theta, &self.layout, rng);
            self.guide.run(&mut g);
            let mut r = Replay::new(&g.values);
            self.model.run(&mut r);
            total += r.log_prob - g.log_q;
        }
        let elbo = total / self.num_particles as f64;
        ad::gradient_into(elbo, &theta, grad);
        elbo.value()
    }

    /// Take one optimization step. Returns the loss (`-ELBO`) at the current
    /// parameters (before the step).
    pub fn step(&mut self, rng: &mut dyn RngCore) -> f64 {
        let mut grad = std::mem::take(&mut self.grad);
        let elbo = self.elbo_and_grad(rng, &mut grad);
        // minimize -ELBO
        for g in grad.iter_mut() {
            *g = -*g;
        }
        if grad.iter().all(|g| g.is_finite()) {
            self.optimizer.step(&mut self.theta, &grad);
        }
        self.grad = grad;
        self.step_count += 1;
        -elbo
    }

    /// Run `steps` optimization steps and return the loss at each step.
    pub fn run(&mut self, seed: u64, steps: usize) -> Vec<f64> {
        let mut rng = ChainRng::seed_from_u64(seed);
        (0..steps).map(|_| self.step(&mut rng)).collect()
    }

    /// Like [`run`](Self::run) but calls `before_step(step, rng)` first — the
    /// hook for drawing a new minibatch.
    pub fn run_with<F: FnMut(usize, &mut ChainRng)>(
        &mut self,
        seed: u64,
        steps: usize,
        mut before_step: F,
    ) -> Vec<f64> {
        let mut rng = ChainRng::seed_from_u64(seed);
        (0..steps)
            .map(|i| {
                before_step(i, &mut rng);
                self.step(&mut rng)
            })
            .collect()
    }

    /// Monte Carlo estimate of the loss (`-ELBO`) with `num_particles` samples,
    /// without updating parameters.
    pub fn evaluate(&self, seed: u64, num_particles: usize) -> f64 {
        let mut rng = ChainRng::seed_from_u64(seed);
        let mut grad = vec![0.0; self.theta.len()];
        let mut total = 0.0;
        for _ in 0..num_particles {
            total += self.elbo_and_grad(&mut rng, &mut grad);
        }
        -total / num_particles as f64
    }

    /// Current parameter values in constrained space.
    pub fn params(&self) -> HashMap<String, Vec<f64>> {
        self.layout.constrain(&self.theta)
    }

    /// Draw `n` samples of the latent sites (and deterministic sites) from the
    /// fitted guide, run through the model to recover deterministic values.
    pub fn sample_posterior(&self, seed: u64, n: usize) -> Samples {
        let params = self.params();
        let mut rng = ChainRng::seed_from_u64(seed);
        let mut sites: Vec<(String, Array)> = Vec::new();
        for d in 0..n {
            let trace = Tracer::new(&mut rng).with_params(&params).trace(self.guide);
            let latent = trace.latent_values();
            // replay through the model to evaluate deterministic sites
            let mtrace = Tracer::with_values(&mut rng, &latent).trace(self.model);
            for site in mtrace.sites.iter().filter(|s| {
                matches!(
                    s.kind,
                    crate::model::SiteKind::Latent | crate::model::SiteKind::Deterministic
                )
            }) {
                let idx = match sites.iter().position(|(nm, _)| nm == &site.name) {
                    Some(i) => i,
                    None => {
                        sites.push((site.name.clone(), Array::new(1, n, site.value.len())));
                        sites.len() - 1
                    }
                };
                sites[idx].1.set_draw(0, d, &site.value);
            }
        }
        Samples {
            sites,
            extras: Vec::new(),
            num_chains: 1,
            num_samples: n,
        }
    }
}

/// Draw `n` distinct indices from `0..total` without replacement (a minibatch).
pub fn subsample(rng: &mut dyn RngCore, total: usize, n: usize) -> Vec<usize> {
    assert!(n <= total);
    // partial Fisher–Yates
    let mut idx: Vec<usize> = (0..total).collect();
    for i in 0..n {
        let j = i + (rand::Rng::random_range(&mut *rng, 0..(total - i)));
        idx.swap(i, j);
    }
    idx.truncate(n);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::*;
    use crate::infer::{HmcKernel, ModelPotential, MCMC};
    use std::sync::Mutex;

    /// Beta(2, 3) prior, 7 heads of 10: posterior Beta(9, 6), mean 0.6.
    struct Coin;
    impl Model for Coin {
        fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
            let p = h.sample("p", Beta::new(2.0, 3.0));
            h.observe(
                "obs",
                Bernoulli::new(p).expand(10),
                &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            );
        }
    }

    #[test]
    fn auto_diagonal_normal_beta_bernoulli() {
        let model = Coin;
        let guide = AutoDiagonalNormal::new(&model);
        let mut svi = SVI::new(&model, &guide, Adam::new(0.05)).num_particles(4);
        let losses = svi.run(0, 3000);
        // loss decreases on average
        let early: f64 = losses[..100].iter().sum::<f64>() / 100.0;
        let late: f64 = losses[losses.len() - 300..].iter().sum::<f64>() / 300.0;
        assert!(late < early, "loss did not decrease: {early} -> {late}");
        let post = svi.sample_posterior(1, 20000);
        let p = post.get("p");
        assert!((p.scalar_mean() - 0.6).abs() < 0.03, "mean {}", p.scalar_mean());
        // Beta(9,6) sd = sqrt(9*6/(15^2*16)) = 0.1225
        assert!((p.scalar_std() - 0.1225).abs() < 0.03, "std {}", p.scalar_std());
    }

    #[test]
    fn auto_delta_is_map() {
        // N(0, 1) prior, observations with sigma 1: MAP = posterior mean = n xbar / (n + 1)
        struct G;
        impl Model for G {
            fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
                let mu = h.sample("mu", Normal::new(0.0, 1.0));
                h.observe("y", Normal::new(mu, 1.0).expand(4), &[1.0, 2.0, 3.0, 2.0]);
            }
        }
        let model = G;
        let guide = AutoDelta::new(&model);
        let mut svi = SVI::new(&model, &guide, Adam::new(0.05));
        svi.run(0, 2000);
        let mu = svi.params()["mu_auto_loc"][0];
        assert!((mu - 8.0 / 5.0).abs() < 1e-3, "MAP {mu}");
    }

    #[test]
    fn auto_multivariate_normal_matches_gaussian_posterior() {
        // Correlated Gaussian target: the optimal full-rank Gaussian guide is exact.
        struct Corr;
        impl Model for Corr {
            fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
                let cov = [1.0, 0.8, 0.8, 1.0];
                h.sample_vec("x", MultivariateNormal::from_covariance(&[1.0, -1.0], &cov));
            }
        }
        let model = Corr;
        let guide = AutoMultivariateNormal::new(&model);
        let mut svi = SVI::new(&model, &guide, Adam::new(0.02)).num_particles(8);
        svi.run(0, 4000);
        let post = svi.sample_posterior(2, 20000);
        let x = post.get("x");
        let m = x.mean();
        assert!((m[0] - 1.0).abs() < 0.05 && (m[1] + 1.0).abs() < 0.05, "{m:?}");
        let c = x.covariance();
        assert!((c[0] - 1.0).abs() < 0.1 && (c[3] - 1.0).abs() < 0.1, "{c:?}");
        assert!((c[1] - 0.8).abs() < 0.1, "correlation {}", c[1]);
    }

    /// Linear regression with minibatches: the minibatch ELBO gradient is an
    /// unbiased estimate of the full-data gradient, and SVI recovers the NUTS
    /// posterior.
    struct Regression {
        x: Vec<f64>,
        y: Vec<f64>,
        batch: Mutex<Option<Vec<usize>>>,
    }
    impl Model for Regression {
        fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
            let w = h.sample("w", Normal::new(0.0, 2.0));
            let b = h.sample("b", Normal::new(0.0, 2.0));
            let all: Vec<usize> = (0..self.x.len()).collect();
            let batch = self.batch.lock().unwrap();
            let idx = batch.as_ref().unwrap_or(&all);
            let scale = self.x.len() as f64 / idx.len() as f64;
            h.push_scale(scale);
            let xb: Vec<f64> = idx.iter().map(|&i| self.x[i]).collect();
            let yb: Vec<f64> = idx.iter().map(|&i| self.y[i]).collect();
            let mean: Vec<R> = xb.iter().map(|xi| w * *xi + b).collect();
            h.observe("y", Normal::new(&mean, 0.5), &yb);
            h.pop_scale();
        }
    }

    #[test]
    fn minibatch_gradient_is_unbiased_and_svi_matches_nuts() {
        let n = 400;
        let mut rng = ChainRng::seed_from_u64(3);
        let x: Vec<f64> = (0..n)
            .map(|_| rand::Rng::random_range(&mut rng, -2.0..2.0))
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|xi| {
                1.5 * xi - 0.7 + 0.5 * rand::Rng::sample::<f64, _>(&mut rng, rand_distr::StandardNormal)
            })
            .collect();
        let model = Regression {
            x,
            y,
            batch: Mutex::new(None),
        };
        let guide = AutoDiagonalNormal::new(&model);

        // unbiasedness: average minibatch gradient over many batches ≈ full gradient
        let svi = SVI::new(&model, &guide, Adam::new(0.01));
        let mut full = vec![0.0; svi.theta.len()];
        let mut rng = ChainRng::seed_from_u64(0);
        // fix the guide noise by using num_particles large -> instead compare expectations
        let mut full_acc = vec![0.0; full.len()];
        for _ in 0..500 {
            svi.elbo_and_grad(&mut rng, &mut full);
            for (a, g) in full_acc.iter_mut().zip(&full) {
                *a += g / 500.0;
            }
        }
        let mut mb_acc = vec![0.0; full.len()];
        for _ in 0..4000 {
            *model.batch.lock().unwrap() = Some(subsample(&mut rng, n, 40));
            svi.elbo_and_grad(&mut rng, &mut full);
            for (a, g) in mb_acc.iter_mut().zip(&full) {
                *a += g / 4000.0;
            }
        }
        *model.batch.lock().unwrap() = None;
        // both are Monte Carlo estimates of the same vector; the scale-parameter
        // components are small (~10) and noisy next to the location ones (~3000),
        // so compare against the gradient's overall magnitude
        let gmax = full_acc.iter().map(|g| g.abs()).fold(0.0, f64::max);
        for (i, (a, b)) in full_acc.iter().zip(&mb_acc).enumerate() {
            assert!(
                (a - b).abs() < 0.01 * gmax,
                "grad {i}: full {a} vs minibatch {b} (gmax {gmax})"
            );
        }

        // fit with minibatches of 40
        let mut svi = SVI::new(&model, &guide, Adam::new(0.02));
        svi.run_with(0, 4000, |_, rng| {
            *model.batch.lock().unwrap() = Some(subsample(rng, n, 40));
        });
        *model.batch.lock().unwrap() = None;
        let post = svi.sample_posterior(1, 10000);
        let nuts = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 500, 2000).run(0);
        for name in ["w", "b"] {
            let (a, b) = (post.get(name).scalar_mean(), nuts.get(name).scalar_mean());
            assert!((a - b).abs() < 0.05, "{name}: svi {a} vs nuts {b}");
            let (sa, sb) = (post.get(name).scalar_std(), nuts.get(name).scalar_std());
            assert!((sa - sb).abs() < 0.4 * sb, "{name} std: svi {sa} vs nuts {sb}");
        }
    }

    #[test]
    fn subsample_is_uniform_without_replacement() {
        let mut rng = ChainRng::seed_from_u64(0);
        let mut counts = vec![0usize; 10];
        for _ in 0..5000 {
            let s = subsample(&mut rng, 10, 3);
            assert_eq!(s.len(), 3);
            let mut u = s.clone();
            u.sort();
            u.dedup();
            assert_eq!(u.len(), 3);
            for i in s {
                counts[i] += 1;
            }
        }
        for c in counts {
            assert!((c as f64 - 1500.0).abs() < 150.0, "{c}");
        }
    }
}
