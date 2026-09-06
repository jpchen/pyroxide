//! End-to-end posterior tests, ported from numpyro's `test/infer/test_mcmc.py`.
//!
//! Each test runs a full warmup + sampling and checks posterior moments (or
//! other identifiable quantities) against known truth, for NUTS, HMC and
//! Metropolis–Hastings.

use std::collections::HashMap;

use pyroxide::ad::Real;
use pyroxide::dist::*;
use pyroxide::infer::*;
use pyroxide::model::{Handler, Model};
use pyroxide::Var;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: got {actual}, expected {expected} (tol {tol})"
    );
}

fn assert_rel(actual: f64, expected: f64, rtol: f64, what: &str) {
    assert_close(actual, expected, rtol * expected.abs(), what);
}

// ---------------------------------------------------------------------------
// test_unnormalized_normal: sample from an unnormalized N(1, 0.5) density given
// directly as a potential function, with every kernel and both mass matrices.
// ---------------------------------------------------------------------------

fn unnormalized_normal_potential() -> FnPotential<impl Fn(&[Var]) -> Var + Send + Sync> {
    let (true_mean, true_std) = (1.0, 0.5);
    FnPotential::new(1, move |z: &[Var]| ((z[0] - true_mean) / true_std).square() * 0.5)
}

fn check_unnormalized_normal(samples: &Samples, rtol: f64) {
    let z = samples.get("z");
    assert_rel(z.scalar_mean(), 1.0, rtol, "mean");
    assert_rel(z.scalar_std(), 0.5, rtol, "std");
}

#[test]
fn unnormalized_normal_nuts() {
    for dense in [false, true] {
        let kernel = HmcKernel::nuts(unnormalized_normal_potential()).dense_mass(dense);
        let samples = MCMC::new(kernel, 1000, 8000).run(0);
        check_unnormalized_normal(&samples, 0.07);
    }
}

#[test]
fn unnormalized_normal_hmc() {
    for dense in [false, true] {
        let kernel = HmcKernel::hmc(unnormalized_normal_potential())
            .trajectory_length(8.0)
            .dense_mass(dense);
        let samples = MCMC::new(kernel, 1000, 8000).run(0);
        check_unnormalized_normal(&samples, 0.07);
    }
}

#[test]
fn unnormalized_normal_metropolis() {
    for dense in [false, true] {
        let kernel = MetropolisHastings::new(unnormalized_normal_potential()).dense_mass(dense);
        let samples = MCMC::new(kernel, 5000, 100_000).run(0);
        check_unnormalized_normal(&samples, 0.07);
        let accept = samples.extra("mean_accept_prob");
        let last = accept.draw(0, accept.draws - 1)[0];
        // 1-D random walk tuned toward 0.234 (dual averaging is approximate)
        assert!(last > 0.15 && last < 0.5, "mean accept prob {last}");
    }
}

// ---------------------------------------------------------------------------
// test_correlated_mvn: 5-d correlated Gaussian needs dense mass adaptation.
// ---------------------------------------------------------------------------

#[test]
fn correlated_mvn_dense_mass() {
    let d = 5;
    // a = tril(0.5 * fliplr(eye) + 0.1 * exp(noise)); cov = a a^T
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    let mut a = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let anti = if i + j == d - 1 { 0.5 } else { 0.0 };
            let e: f64 = rand::Rng::sample(&mut rng, rand_distr::StandardNormal);
            a[i * d + j] = anti + 0.1 * e.exp();
        }
    }
    let mut cov = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            cov[i * d + j] = (0..d).map(|k| a[i * d + k] * a[j * d + k]).sum();
        }
    }
    let prec = pyroxide::linalg::spd_inverse(&cov, d).unwrap();
    for regularize in [true, false] {
        let prec2 = prec.clone();
        let pot = FnPotential::new(d, move |z: &[Var]| {
            let mut q = Var::constant(0.0);
            for i in 0..d {
                for j in 0..d {
                    q = q + z[i] * z[j] * prec2[i * d + j];
                }
            }
            q * 0.5
        });
        let kernel = HmcKernel::nuts(pot).dense_mass(true).regularize_mass_matrix(regularize);
        let samples = MCMC::new(kernel, 5000, 8000)
            .init_strategy(InitStrategy::Unconstrained(vec![0.0; d]))
            .run(0);
        let z = samples.get("z");
        let mean = z.mean();
        for m in &mean {
            assert_close(*m, 0.0, 0.02 * 5.0, "mean");
        }
        let est = z.covariance();
        let err: f64 = est.iter().zip(&cov).map(|(a, b)| (a - b).abs()).sum::<f64>() / (d * d) as f64;
        assert!(err < 0.02, "mean abs covariance error {err}");
    }
}

// ---------------------------------------------------------------------------
// test_logistic_regression
// ---------------------------------------------------------------------------

struct LogisticRegression {
    x: Vec<f64>, // N x D row-major
    n: usize,
    d: usize,
    labels: Vec<f64>,
}

impl Model for LogisticRegression {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let coefs = h.sample_vec("coefs", Normal::new(0.0, 1.0).expand(self.d));
        let logits = pyroxide::ad::matvec_const(&self.x, self.n, self.d, &coefs);
        h.deterministic("logits", &logits);
        h.observe("obs", Bernoulli::logits(&logits), &self.labels);
    }
}

fn logistic_data(n: usize, d: usize) -> LogisticRegression {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    let x: Vec<f64> = (0..n * d)
        .map(|_| rand::Rng::sample(&mut rng, rand_distr::StandardNormal))
        .collect();
    let true_coefs: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    let labels: Vec<f64> = (0..n)
        .map(|i| {
            let logit: f64 = (0..d).map(|j| x[i * d + j] * true_coefs[j]).sum();
            let p = pyroxide::ad::sigmoid_f64(logit);
            if rand::Rng::random::<f64>(&mut rng) < p { 1.0 } else { 0.0 }
        })
        .collect();
    LogisticRegression { x, n, d, labels }
}

#[test]
fn logistic_regression_nuts_and_hmc() {
    let model = logistic_data(3000, 3);
    for algo in ["nuts", "hmc"] {
        let pot = ModelPotential::new(&model);
        let kernel = match algo {
            "nuts" => HmcKernel::nuts(pot).find_heuristic_step_size(true),
            _ => HmcKernel::hmc(pot).trajectory_length(8.0).find_heuristic_step_size(true),
        };
        let samples = MCMC::new(kernel, 1000, 8000).run(0);
        let coefs = samples.get("coefs");
        assert_eq!(samples.get("logits").len, 3000);
        assert_eq!(samples.get("logits").draws, 8000);
        for (i, m) in coefs.mean().iter().enumerate() {
            assert_close(*m, (i + 1) as f64, 0.4, &format!("{algo} coef {i}"));
        }
    }
}

#[test]
fn logistic_regression_metropolis() {
    let model = logistic_data(500, 2);
    let pot = ModelPotential::new(&model);
    let kernel = MetropolisHastings::new(pot).dense_mass(true);
    let samples = MCMC::new(kernel, 20_000, 100_000).run(0);
    let coefs = samples.get("coefs");
    // compare against NUTS on the same model rather than the true coefficients
    let nuts = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 1000, 8000).run(1);
    let ref_mean = nuts.get("coefs").mean();
    let ref_std = nuts.get("coefs").std();
    for i in 0..2 {
        assert_close(coefs.mean()[i], ref_mean[i], 0.1 * ref_std[i].max(0.5), &format!("mh coef {i}"));
        assert_rel(coefs.std()[i], ref_std[i], 0.15, &format!("mh std {i}"));
    }
}

// ---------------------------------------------------------------------------
// test_uniform_normal / test_improper_normal: latent with a transformed
// (interval) support depending on another latent.
// ---------------------------------------------------------------------------

struct UniformNormal {
    data: Vec<f64>,
    improper: bool,
}

impl Model for UniformNormal {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let alpha = h.sample("alpha", Uniform::new(0.0, 1.0));
        // loc = alpha * u, u ~ Uniform(0, 1) (a TransformReparam of Uniform(0, alpha))
        let u = if self.improper {
            h.sample("u", ImproperUniform::new(Support::UnitInterval, 1))
        } else {
            h.sample("u", Uniform::new(0.0, 1.0))
        };
        let loc = alpha * u;
        h.deterministic("loc", &[loc]);
        h.observe("obs", Normal::new(loc, 0.1).expand(self.data.len()), &self.data);
    }
}

#[test]
fn uniform_normal() {
    let true_coef = 0.9;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    let data: Vec<f64> = (0..1000)
        .map(|_| true_coef + rand::Rng::sample::<f64, _>(&mut rng, rand_distr::StandardNormal))
        .collect();
    let model = UniformNormal { data, improper: false };
    let mcmc = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 1000, 1000).collect_warmup(true);
    let samples = mcmc.run(2);
    assert_eq!(samples.get("loc").draws, 2000);
    let post: Vec<f64> = samples.get("loc").column(0)[1000..].to_vec();
    let mean = post.iter().sum::<f64>() / post.len() as f64;
    assert_close(mean, true_coef, 0.05, "loc");
}

#[test]
fn improper_normal() {
    let true_coef = 0.9;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    let data: Vec<f64> = (0..1000)
        .map(|_| true_coef + rand::Rng::sample::<f64, _>(&mut rng, rand_distr::StandardNormal))
        .collect();
    // with a flat prior the posterior mean of loc is the data mean (numpyro's
    // test compares to 0.9 with atol 0.007, which relies on its data seed)
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;
    let model = UniformNormal { data, improper: true };
    for depths in [(10, 10), (5, 10)] {
        let kernel = HmcKernel::nuts(ModelPotential::new(&model)).max_tree_depths(depths.0, depths.1);
        let samples = MCMC::new(kernel, 1000, 1000).run(0);
        assert_close(samples.get("loc").scalar_mean(), data_mean, 0.007, "loc");
        assert_close(samples.get("loc").scalar_mean(), true_coef, 0.05, "loc vs truth");
    }
}

// ---------------------------------------------------------------------------
// test_beta_bernoulli
// ---------------------------------------------------------------------------

struct BetaBernoulli {
    data: Vec<f64>, // 1000 x 2 row-major
}

impl Model for BetaBernoulli {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let p = h.sample_vec("p_latent", Beta::new(&[1.1, 1.1], &[1.1, 1.1]));
        // each row of the data is a pair (one Bernoulli per component)
        let n = self.data.len() / 2;
        let probs: Vec<R> = (0..n).flat_map(|_| [p[0], p[1]]).collect();
        h.observe("obs", Bernoulli::new(&probs), &self.data);
    }
}

fn beta_bernoulli_data() -> BetaBernoulli {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    let true_probs = [0.9, 0.1];
    let data = (0..1000)
        .flat_map(|_| {
            let u0: f64 = rand::Rng::random(&mut rng);
            let u1: f64 = rand::Rng::random(&mut rng);
            [(u0 < true_probs[0]) as u8 as f64, (u1 < true_probs[1]) as u8 as f64]
        })
        .collect();
    BetaBernoulli { data }
}

#[test]
fn beta_bernoulli_nuts_hmc_mh() {
    let model = beta_bernoulli_data();
    let check = |samples: &Samples, what: &str| {
        let m = samples.get("p_latent").mean();
        assert_close(m[0], 0.9, 0.05, &format!("{what} p0"));
        assert_close(m[1], 0.1, 0.05, &format!("{what} p1"));
    };
    let s = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 500, 20000).run(2);
    check(&s, "nuts");
    let s = MCMC::new(
        HmcKernel::hmc(ModelPotential::new(&model)).trajectory_length(0.1),
        500,
        20000,
    )
    .run(2);
    check(&s, "hmc");
    let s = MCMC::new(MetropolisHastings::new(ModelPotential::new(&model)), 5000, 100_000).run(2);
    check(&s, "mh");
}

// ---------------------------------------------------------------------------
// test_dirichlet_categorical (simplex transform)
// ---------------------------------------------------------------------------

struct DirichletCategorical {
    data: Vec<f64>,
}

impl Model for DirichletCategorical {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let p = h.sample_vec("p_latent", Dirichlet::new(&[1.0, 1.0, 1.0]));
        h.observe("obs", Categorical::new(&p).expand(self.data.len()), &self.data);
    }
}

#[test]
fn dirichlet_categorical() {
    let true_probs = [0.1, 0.6, 0.3];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    let data: Vec<f64> = (0..2000)
        .map(|_| Categorical::<f64>::new(&true_probs[..]).sample_vec(&mut rng)[0])
        .collect();
    let model = DirichletCategorical { data };
    for dense in [false, true] {
        for algo in ["nuts", "hmc"] {
            let pot = ModelPotential::new(&model);
            let kernel = match algo {
                "nuts" => HmcKernel::nuts(pot).dense_mass(dense),
                _ => HmcKernel::hmc(pot).trajectory_length(1.0).dense_mass(dense),
            };
            let samples = MCMC::new(kernel, 100, 20000).run(2);
            let m = samples.get("p_latent").mean();
            for i in 0..3 {
                assert_close(m[i], true_probs[i], 0.02, &format!("{algo} dense={dense} p{i}"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// test_dense_mass: the adapted dense inverse mass matrix estimates the
// posterior covariance.
// ---------------------------------------------------------------------------

struct Mvn2 {
    cov: Vec<f64>,
}

impl Model for Mvn2 {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        h.sample_vec("x", MultivariateNormal::from_covariance(&[0.0, 0.0], &self.cov));
    }
}

#[test]
fn dense_mass_estimates_covariance() {
    for rho in [-0.7, 0.8] {
        let cov = vec![10.0, rho, rho, 0.1];
        let model = Mvn2 { cov: cov.clone() };
        for algo in ["nuts", "hmc"] {
            let pot = ModelPotential::new(&model);
            let kernel = match algo {
                "nuts" => HmcKernel::nuts(pot).dense_mass(true),
                _ => HmcKernel::hmc(pot).trajectory_length(2.0).dense_mass(true),
            };
            let mcmc = MCMC::new(kernel, 20000, 10000);
            // run one chain manually to inspect the adapted mass matrix
            let mut rng = ChainRng::seed_from_u64(0);
            let mut st = mcmc.kernel.init(vec![0.0, 0.0], 20000, &mut rng);
            for _ in 0..20000 {
                mcmc.kernel.step(&mut st, &mut rng);
            }
            let est = st.adapt.mass.inverse();
            for k in 0..4 {
                assert_rel(est[k], cov[k], 0.10, &format!("{algo} rho={rho} inv mass {k}"));
            }
            let samples = mcmc.run(0);
            let x = samples.get("x");
            let mean = x.mean();
            assert_close(mean[0], 0.0, 0.5, "mean0");
            assert_close(mean[1], 0.0, 0.05, "mean1");
            let c = x.covariance();
            assert_close(c[1], rho, 0.2, "cov");
            assert_rel(c[0], 10.0, 0.2, "var0");
            assert_rel(c[3], 0.1, 0.2, "var1");
        }
    }
}

// ---------------------------------------------------------------------------
// test_change_point (Poisson change point, init_to_value)
// ---------------------------------------------------------------------------

struct ChangePoint {
    counts: Vec<f64>,
}

impl Model for ChangePoint {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let n = self.counts.len();
        let alpha = 1.0 / (self.counts.iter().sum::<f64>() / n as f64);
        let lambda1 = h.sample("lambda1", Exponential::new(alpha));
        let lambda2 = h.sample("lambda2", Exponential::new(alpha));
        let tau = h.sample("tau", Uniform::new(0.0, 1.0));
        let cut = tau.value() * n as f64;
        let rates: Vec<R> = (0..n)
            .map(|i| if (i as f64) < cut { lambda1 } else { lambda2 })
            .collect();
        h.observe("obs", Poisson::new(&rates), &self.counts);
    }
}

#[test]
fn change_point() {
    let counts: Vec<f64> = [
        13, 24, 8, 24, 7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29, 6, 19, 12, 22, 12, 18, 72,
        32, 9, 7, 13, 19, 23, 27, 20, 6, 17, 13, 10, 14, 6, 16, 15, 7, 2, 15, 15, 19, 70, 49, 7, 53,
        22, 21, 31, 19, 11, 1, 20, 12, 35, 17, 23, 17, 4, 2, 31, 30, 13, 27, 0, 39, 37, 5, 14, 13, 22,
    ]
    .iter()
    .map(|&c| c as f64)
    .collect();
    let n = counts.len();
    let model = ChangePoint { counts };
    let mut init = HashMap::new();
    init.insert("lambda1".to_string(), vec![1.0]);
    init.insert("lambda2".to_string(), vec![72.0]);
    let samples = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 1000, 3000)
        .init_to_value(init)
        .run(4);
    let tau = samples.get("tau").column(0);
    let mut hist = vec![0usize; n + 1];
    for t in tau {
        hist[(t * n as f64) as usize] += 1;
    }
    let mode = hist.iter().enumerate().max_by_key(|(_, c)| **c).unwrap().0;
    assert_eq!(mode, 44);
}

// ---------------------------------------------------------------------------
// test_binomial_stable (huge counts, logits parameterization)
// ---------------------------------------------------------------------------

struct BinomialStable {
    n: f64,
    x: f64,
    with_logits: bool,
}

impl Model for BinomialStable {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let p = h.sample("p", Beta::new(1.0, 1.0));
        if self.with_logits {
            let logit = p.ln() - (-p + 1.0).ln();
            h.observe("obs", Binomial::logits(self.n, logit), &[self.x]);
        } else {
            h.observe("obs", Binomial::new(self.n, p), &[self.x]);
        }
    }
}

#[test]
fn binomial_stable() {
    for with_logits in [true, false] {
        let model = BinomialStable { n: 5_000_000.0, x: 3849.0, with_logits };
        let samples = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 200, 200).run(2);
        assert_rel(samples.get("p").scalar_mean(), 3849.0 / 5e6, 0.05, "p");
    }
}

// ---------------------------------------------------------------------------
// test_improper_prior
// ---------------------------------------------------------------------------

struct ImproperPrior {
    data: Vec<f64>,
}

impl Model for ImproperPrior {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let mean = h.sample("mean", ImproperUniform::new(Support::Real, 1));
        let std = h.sample("std", ImproperUniform::new(Support::Positive, 1));
        h.observe("obs", Normal::new(mean, std).expand(self.data.len()), &self.data);
    }
}

#[test]
fn improper_prior() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    let data: Vec<f64> = Normal::<f64>::new(1.0, 2.0).expand(2000).sample_vec(&mut rng);
    let model = ImproperPrior { data };
    let samples = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 1000, 8000).run(2);
    assert_rel(samples.get("mean").scalar_mean(), 1.0, 0.05, "mean");
    assert_rel(samples.get("std").scalar_mean(), 2.0, 0.05, "std");
}

// ---------------------------------------------------------------------------
// test_chain: multiple chains agree and R-hat is ~1
// ---------------------------------------------------------------------------

struct EightSchools {
    y: Vec<f64>,
    sigma: Vec<f64>,
}

impl Model for EightSchools {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let mu = h.sample("mu", Normal::new(0.0, 5.0));
        let tau = h.sample("tau", HalfCauchy::new(5.0));
        // non-centered parameterization
        let eta = h.sample_vec("eta", Normal::new(0.0, 1.0).expand(8));
        let theta: Vec<R> = eta.iter().map(|e| mu + tau * *e).collect();
        h.deterministic("theta", &theta);
        h.observe("y", Normal::new(&theta, &self.sigma), &self.y);
    }
}

fn eight_schools() -> EightSchools {
    EightSchools {
        y: vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        sigma: vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    }
}

#[test]
fn multiple_chains_eight_schools() {
    let model = eight_schools();
    for parallel in [true, false] {
        let samples = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 1000, 2000)
            .num_chains(4)
            .parallel(parallel)
            .run(0);
        assert_eq!(samples.num_chains, 4);
        assert_eq!(samples.get("theta").len, 8);
        let summary = samples.summary(0.9);
        for s in &summary {
            for e in &s.elements {
                assert!(e.r_hat < 1.05, "{}: r_hat {}", s.name, e.r_hat);
                assert!(e.n_eff > 500.0, "{}: n_eff {}", s.name, e.n_eff);
            }
        }
        // posterior mean of mu for eight schools is about 4.4; tau about 3.6
        assert_close(samples.get("mu").scalar_mean(), 4.4, 0.6, "mu");
        assert_close(samples.get("tau").scalar_mean(), 3.6, 0.8, "tau");
        assert!(samples.num_divergences() < 40, "divergences {}", samples.num_divergences());
    }
    // sequential and parallel runs with the same seed are identical
    let a = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 200, 200)
        .num_chains(2)
        .parallel(true)
        .run(7);
    let b = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 200, 200)
        .num_chains(2)
        .parallel(false)
        .run(7);
    assert_eq!(a.get("mu").data(), b.get("mu").data());
}

// ---------------------------------------------------------------------------
// test_extra_fields / diagnostics fields
// ---------------------------------------------------------------------------

#[test]
fn extra_fields() {
    let model = eight_schools();
    let samples = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 100, 100).run(0);
    for name in ["accept_prob", "step_size", "num_steps", "diverging", "energy", "potential_energy", "mean_accept_prob"] {
        let a = samples.extra(name);
        assert_eq!(a.draws, 100);
    }
    let steps = samples.extra("num_steps");
    assert!(steps.data().iter().all(|&s| s >= 1.0 && s <= 1023.0));
    let ss = samples.extra("step_size");
    // step size is frozen after warmup
    assert!(ss.data().iter().all(|&s| s == ss.data()[0]));
}

// ---------------------------------------------------------------------------
// test_fixed_num_steps / test_prior_with_sample_shape style smoke tests
// ---------------------------------------------------------------------------

#[test]
fn fixed_num_steps_hmc() {
    let model = eight_schools();
    let kernel = HmcKernel::hmc(ModelPotential::new(&model)).num_steps(7);
    let samples = MCMC::new(kernel, 300, 300).run(0);
    let steps = samples.extra("num_steps");
    assert!(steps.data().iter().all(|&s| s == 7.0));
}

#[test]
fn init_to_value_starts_there() {
    let model = eight_schools();
    let mut init = HashMap::new();
    init.insert("mu".to_string(), vec![3.0]);
    init.insert("tau".to_string(), vec![2.0]);
    let kernel = HmcKernel::nuts(ModelPotential::new(&model)).adapt_step_size(false).step_size(1e-12);
    // with a tiny fixed step size the chain barely moves from its start
    let samples = MCMC::new(kernel, 0, 5).init_to_value(init).run(0);
    assert_close(samples.get("mu").scalar_mean(), 3.0, 1e-6, "mu");
    assert_close(samples.get("tau").scalar_mean(), 2.0, 1e-6, "tau");
}
