//! The probabilistic-programming layer: models as generative programs, run
//! under interchangeable *handlers*.
//!
//! A model is any type implementing [`Model`]: a struct holding the data plus a
//! generic `run` method that describes the generative process by calling
//! [`Handler::sample`], [`Handler::observe`], [`Handler::factor`] and
//! [`Handler::deterministic`]. The handler decides what those calls *do*:
//!
//! * [`Tracer`] draws latent values from their priors (or replays supplied
//!   values) and records every site — used for initialization and prediction.
//! * [`LogDensity`] reads latent values from a flat unconstrained vector,
//!   applies the constraining transforms, and accumulates the log joint density
//!   plus log-Jacobian — this is what inference differentiates.
//! * [`Postprocess`] maps unconstrained samples back to constrained,
//!   named values and evaluates deterministic sites.
//!
//! Because `run` is generic over the scalar type [`Real`], the same model code
//! is evaluated with `f64` for plain evaluation and with [`Var`](crate::ad::Var)
//! for gradients. Inference algorithms never see models directly; they see a
//! [`Potential`](crate::infer::Potential), which is what decouples the two.
//!
//! ```
//! use pyroxide::prelude::*;
//!
//! struct EightSchools { y: Vec<f64>, sigma: Vec<f64> }
//!
//! impl Model for EightSchools {
//!     fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
//!         let mu = h.sample("mu", Normal::new(0.0, 5.0));
//!         let tau = h.sample("tau", HalfCauchy::new(5.0));
//!         let theta = h.sample_vec("theta", Normal::new(mu, tau).expand(8));
//!         h.observe("y", Normal::new(&theta, &self.sigma), &self.y);
//!     }
//! }
//! ```

use std::collections::HashMap;

use rand::RngCore;

use crate::ad::Real;
use crate::dist::{Distribution, Support};

pub mod predictive;
pub mod transform;

pub use predictive::Predictive;

/// The interface a model program uses to declare random variables.
///
/// All methods take the distribution by value; distributions are small structs
/// that borrow their parameters.
pub trait Handler<R: Real> {
    /// Declare a latent random variable and return its (constrained) value(s).
    fn sample_vec<D: Distribution<R>>(&mut self, name: &str, dist: D) -> Vec<R>;

    /// Declare a scalar latent random variable.
    fn sample<D: Distribution<R>>(&mut self, name: &str, dist: D) -> R {
        let v = self.sample_vec(name, dist);
        assert_eq!(v.len(), 1, "site '{name}' is not scalar; use sample_vec");
        v[0]
    }

    /// Condition on observed data. `value` has length `dist.len()`.
    fn observe<D: Distribution<R>>(&mut self, name: &str, dist: D, value: &[f64]);

    /// Add an arbitrary term to the log joint density.
    fn factor(&mut self, name: &str, log_factor: R);

    /// Record a derived quantity so it appears in the posterior samples.
    fn deterministic(&mut self, name: &str, value: &[R]);
}

/// A generative model. Implement this for a struct holding your data.
pub trait Model: Send + Sync {
    fn run<R: Real, H: Handler<R>>(&self, handler: &mut H);
}

/// Static description of one latent site, discovered by running the model.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteInfo {
    pub name: String,
    /// Number of constrained scalars.
    pub len: usize,
    pub event_len: usize,
    pub support: Support,
    /// Number of unconstrained scalars.
    pub unconstrained_len: usize,
    /// Offset into the flat unconstrained vector.
    pub offset: usize,
}

/// The layout of a model's latent sites in the flat unconstrained vector.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SiteLayout {
    pub sites: Vec<SiteInfo>,
    /// Names of deterministic sites, in program order.
    pub deterministic: Vec<String>,
    pub dim: usize,
}

impl SiteLayout {
    /// Discover the layout by running the model once (with all unconstrained
    /// values at zero).
    pub fn discover<M: Model>(model: &M) -> SiteLayout {
        let mut h = LayoutDiscovery::default();
        model.run(&mut h);
        SiteLayout {
            sites: h.sites,
            deterministic: h.deterministic,
            dim: h.dim,
        }
    }

    pub fn site(&self, name: &str) -> Option<&SiteInfo> {
        self.sites.iter().find(|s| s.name == name)
    }

    /// Map named constrained values to the flat unconstrained vector.
    /// Sites missing from `values` are filled with `default` (unconstrained).
    pub fn unconstrain(&self, values: &HashMap<String, Vec<f64>>, default: &[f64]) -> Vec<f64> {
        let mut z = default.to_vec();
        for s in &self.sites {
            if let Some(v) = values.get(&s.name) {
                assert_eq!(
                    v.len(),
                    s.len,
                    "site '{}' has length {}, expected {}",
                    s.name,
                    v.len(),
                    s.len
                );
                let u = transform::to_unconstrained(s.support, s.event_len, v);
                z[s.offset..s.offset + s.unconstrained_len].copy_from_slice(&u);
            }
        }
        z
    }

    /// Map a flat unconstrained vector to named constrained values.
    pub fn constrain(&self, z: &[f64]) -> Vec<(String, Vec<f64>)> {
        self.sites
            .iter()
            .map(|s| {
                let mut out = Vec::with_capacity(s.len);
                transform::to_constrained::<f64>(
                    s.support,
                    s.event_len,
                    &z[s.offset..s.offset + s.unconstrained_len],
                    &mut out,
                );
                (s.name.clone(), out)
            })
            .collect()
    }
}

#[derive(Default)]
struct LayoutDiscovery {
    sites: Vec<SiteInfo>,
    deterministic: Vec<String>,
    dim: usize,
}

impl Handler<f64> for LayoutDiscovery {
    fn sample_vec<D: Distribution<f64>>(&mut self, name: &str, dist: D) -> Vec<f64> {
        assert!(
            !self.sites.iter().any(|s| s.name == name),
            "duplicate sample site '{name}'"
        );
        let support = dist.support();
        assert!(
            !support.is_discrete(),
            "latent site '{name}' is discrete; gradient-based kernels need continuous latents (observe it or marginalize)"
        );
        let len = dist.len();
        let event_len = dist.event_len();
        let ulen = transform::unconstrained_len(support, len, event_len);
        let zeros = vec![0.0; ulen];
        let mut out = Vec::with_capacity(len);
        transform::to_constrained::<f64>(support, event_len, &zeros, &mut out);
        self.sites.push(SiteInfo {
            name: name.to_string(),
            len,
            event_len,
            support,
            unconstrained_len: ulen,
            offset: self.dim,
        });
        self.dim += ulen;
        out
    }
    fn observe<D: Distribution<f64>>(&mut self, name: &str, dist: D, value: &[f64]) {
        assert_eq!(
            value.len(),
            dist.len(),
            "observed site '{name}': value length {} != distribution length {}",
            value.len(),
            dist.len()
        );
    }
    fn factor(&mut self, _name: &str, _log_factor: f64) {}
    fn deterministic(&mut self, name: &str, _value: &[f64]) {
        self.deterministic.push(name.to_string());
    }
}

/// Computes the log joint density (plus log-Jacobian) of a model at a point in
/// unconstrained space. This is the handler inference differentiates.
pub struct LogDensity<'a, R: Real> {
    z: &'a [R],
    layout: &'a SiteLayout,
    site_idx: usize,
    /// Accumulated log joint density.
    pub log_prob: R,
    scratch: Vec<R>,
}

impl<'a, R: Real> LogDensity<'a, R> {
    pub fn new(z: &'a [R], layout: &'a SiteLayout) -> Self {
        assert_eq!(z.len(), layout.dim, "unconstrained vector has wrong length");
        LogDensity {
            z,
            layout,
            site_idx: 0,
            log_prob: R::zero(),
            scratch: Vec::new(),
        }
    }

    /// Evaluate the log joint density of `model` at `z`.
    pub fn eval<M: Model>(model: &M, z: &'a [R], layout: &'a SiteLayout) -> R {
        let mut h = LogDensity::new(z, layout);
        model.run(&mut h);
        h.log_prob
    }
}

impl<'a, R: Real> Handler<R> for LogDensity<'a, R> {
    #[inline]
    fn sample_vec<D: Distribution<R>>(&mut self, name: &str, dist: D) -> Vec<R> {
        let info = &self.layout.sites[self.site_idx];
        debug_assert_eq!(info.name, name, "site order changed between model executions");
        debug_assert_eq!(
            info.len,
            dist.len(),
            "site '{name}' changed size between executions"
        );
        self.site_idx += 1;
        let u = &self.z[info.offset..info.offset + info.unconstrained_len];
        let mut x = std::mem::take(&mut self.scratch);
        x.clear();
        let logdet = transform::to_constrained(info.support, info.event_len, u, &mut x);
        self.log_prob += dist.log_prob(&x) + logdet;
        // hand back an owned vector; keep the scratch allocation for reuse
        let out = x.clone();
        self.scratch = x;
        out
    }
    #[inline]
    fn observe<D: Distribution<R>>(&mut self, _name: &str, dist: D, value: &[f64]) {
        self.log_prob += dist.log_prob_data(value);
    }
    #[inline]
    fn factor(&mut self, _name: &str, log_factor: R) {
        self.log_prob += log_factor;
    }
    #[inline]
    fn deterministic(&mut self, _name: &str, _value: &[R]) {}
}

/// What kind of site a [`Trace`] entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteKind {
    Latent,
    Observed,
    Deterministic,
    Factor,
}

/// One recorded site.
#[derive(Clone, Debug)]
pub struct TraceSite {
    pub name: String,
    pub kind: SiteKind,
    pub value: Vec<f64>,
    pub log_prob: f64,
    pub support: Option<Support>,
    pub event_len: usize,
}

/// An execution trace: every site in program order.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub sites: Vec<TraceSite>,
}

impl Trace {
    pub fn get(&self, name: &str) -> Option<&TraceSite> {
        self.sites.iter().find(|s| s.name == name)
    }
    /// Sum of log probabilities of all latent and observed sites and factors.
    pub fn log_joint(&self) -> f64 {
        self.sites.iter().map(|s| s.log_prob).sum()
    }
    /// Latent site values as a map (constrained space).
    pub fn latent_values(&self) -> HashMap<String, Vec<f64>> {
        self.sites
            .iter()
            .filter(|s| s.kind == SiteKind::Latent)
            .map(|s| (s.name.clone(), s.value.clone()))
            .collect()
    }
}

/// Runs a model forward with `f64`, drawing latent sites from their priors
/// unless a value is supplied in `values`, and records a [`Trace`].
pub struct Tracer<'a> {
    rng: &'a mut dyn RngCore,
    values: Option<&'a HashMap<String, Vec<f64>>>,
    /// Sample observed sites instead of conditioning on the given data.
    predictive: bool,
    pub trace: Trace,
}

impl<'a> Tracer<'a> {
    /// Sample every latent site from its prior.
    pub fn new(rng: &'a mut dyn RngCore) -> Self {
        Tracer {
            rng,
            values: None,
            predictive: false,
            trace: Trace::default(),
        }
    }

    /// Predictive mode: latent sites are replayed from `values` (or drawn from
    /// the prior) and observed sites are *sampled* from their distributions.
    pub fn predictive(rng: &'a mut dyn RngCore, values: &'a HashMap<String, Vec<f64>>) -> Self {
        Tracer {
            rng,
            values: Some(values),
            predictive: true,
            trace: Trace::default(),
        }
    }

    /// Replay the given latent values (sampling any site not present).
    pub fn with_values(rng: &'a mut dyn RngCore, values: &'a HashMap<String, Vec<f64>>) -> Self {
        Tracer {
            rng,
            values: Some(values),
            predictive: false,
            trace: Trace::default(),
        }
    }

    /// Run `model` and return the trace.
    pub fn trace<M: Model>(mut self, model: &M) -> Trace {
        model.run(&mut self);
        self.trace
    }
}

impl<'a> Handler<f64> for Tracer<'a> {
    fn sample_vec<D: Distribution<f64>>(&mut self, name: &str, dist: D) -> Vec<f64> {
        let value = match self.values.and_then(|v| v.get(name)) {
            Some(v) => {
                assert_eq!(
                    v.len(),
                    dist.len(),
                    "replayed value for '{name}' has wrong length"
                );
                v.clone()
            }
            None => dist.sample_vec(self.rng),
        };
        let log_prob = dist.log_prob_data(&value);
        self.trace.sites.push(TraceSite {
            name: name.to_string(),
            kind: SiteKind::Latent,
            value: value.clone(),
            log_prob,
            support: Some(dist.support()),
            event_len: dist.event_len(),
        });
        value
    }
    fn observe<D: Distribution<f64>>(&mut self, name: &str, dist: D, value: &[f64]) {
        let value = if self.predictive {
            dist.sample_vec(self.rng)
        } else {
            value.to_vec()
        };
        let log_prob = dist.log_prob_data(&value);
        self.trace.sites.push(TraceSite {
            name: name.to_string(),
            kind: SiteKind::Observed,
            value,
            log_prob,
            support: Some(dist.support()),
            event_len: dist.event_len(),
        });
    }
    fn factor(&mut self, name: &str, log_factor: f64) {
        self.trace.sites.push(TraceSite {
            name: name.to_string(),
            kind: SiteKind::Factor,
            value: vec![],
            log_prob: log_factor,
            support: None,
            event_len: 1,
        });
    }
    fn deterministic(&mut self, name: &str, value: &[f64]) {
        self.trace.sites.push(TraceSite {
            name: name.to_string(),
            kind: SiteKind::Deterministic,
            value: value.to_vec(),
            log_prob: 0.0,
            support: None,
            event_len: 1,
        });
    }
}

/// Runs the model forward with `f64` from an unconstrained vector, recording
/// constrained latent values and deterministic sites (observed sites are
/// skipped). Used to turn raw MCMC states into named samples.
pub struct Postprocess<'a> {
    z: &'a [f64],
    layout: &'a SiteLayout,
    site_idx: usize,
    /// `(name, value)` for latent (constrained) and deterministic sites.
    pub values: Vec<(String, Vec<f64>)>,
}

impl<'a> Postprocess<'a> {
    pub fn new(z: &'a [f64], layout: &'a SiteLayout) -> Self {
        Postprocess {
            z,
            layout,
            site_idx: 0,
            values: Vec::with_capacity(layout.sites.len() + layout.deterministic.len()),
        }
    }

    pub fn run<M: Model>(model: &M, z: &'a [f64], layout: &'a SiteLayout) -> Vec<(String, Vec<f64>)> {
        let mut h = Postprocess::new(z, layout);
        model.run(&mut h);
        h.values
    }
}

impl<'a> Handler<f64> for Postprocess<'a> {
    fn sample_vec<D: Distribution<f64>>(&mut self, name: &str, _dist: D) -> Vec<f64> {
        let info = &self.layout.sites[self.site_idx];
        debug_assert_eq!(info.name, name);
        self.site_idx += 1;
        let u = &self.z[info.offset..info.offset + info.unconstrained_len];
        let mut x = Vec::with_capacity(info.len);
        transform::to_constrained::<f64>(info.support, info.event_len, u, &mut x);
        self.values.push((name.to_string(), x.clone()));
        x
    }
    fn observe<D: Distribution<f64>>(&mut self, _name: &str, _dist: D, _value: &[f64]) {}
    fn factor(&mut self, _name: &str, _log_factor: f64) {}
    fn deterministic(&mut self, name: &str, value: &[f64]) {
        self.values.push((name.to_string(), value.to_vec()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad::{self, Var};
    use crate::dist::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    struct EightSchools {
        y: Vec<f64>,
        sigma: Vec<f64>,
    }

    impl Model for EightSchools {
        fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
            let mu = h.sample("mu", Normal::new(0.0, 5.0));
            let tau = h.sample("tau", HalfCauchy::new(5.0));
            let theta = h.sample_vec("theta", Normal::new(mu, tau).expand(8));
            let mean: R = theta.iter().copied().sum::<R>() / 8.0;
            h.deterministic("theta_mean", &[mean]);
            h.observe("y", Normal::new(&theta, &self.sigma), &self.y);
        }
    }

    fn schools() -> EightSchools {
        EightSchools {
            y: vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
            sigma: vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        }
    }

    #[test]
    fn layout_discovery() {
        let m = schools();
        let layout = SiteLayout::discover(&m);
        assert_eq!(layout.dim, 10);
        assert_eq!(layout.sites.len(), 3);
        assert_eq!(layout.sites[1].support, Support::Positive);
        assert_eq!(layout.sites[2].offset, 2);
        assert_eq!(layout.deterministic, vec!["theta_mean".to_string()]);
    }

    #[test]
    fn log_density_matches_manual() {
        let m = schools();
        let layout = SiteLayout::discover(&m);
        let z: Vec<f64> = (0..10).map(|i| 0.1 * i as f64 - 0.3).collect();
        let lp = LogDensity::eval(&m, &z, &layout);
        // manual computation
        let mu = z[0];
        let tau = z[1].exp();
        let theta = &z[2..];
        let mut expect = Normal::<f64>::new(0.0, 5.0).log_prob_data(&[mu]);
        expect += HalfCauchy::<f64>::new(5.0).log_prob_data(&[tau]) + z[1]; // + log|d tau / d u|
        expect += Normal::<f64>::new(mu, tau).expand(8).log_prob_data(theta);
        expect += Normal::<f64>::new(&theta.to_vec(), &m.sigma).log_prob_data(&m.y);
        assert!((lp - expect).abs() < 1e-10, "{lp} vs {expect}");
    }

    #[test]
    fn log_density_gradient() {
        let m = schools();
        let layout = SiteLayout::discover(&m);
        let z: Vec<f64> = (0..10).map(|i| 0.1 * i as f64 - 0.3).collect();
        ad::reset();
        let zv = Var::leaves(&z);
        let lp = LogDensity::eval(&m, &zv, &layout);
        let g = ad::gradient(lp, &zv);
        let fd = ad::finite_diff(|z| LogDensity::eval(&m, z, &layout), &z, 1e-6);
        for (a, b) in g.iter().zip(&fd) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // tape stays small: 3 latent sites + observe + transforms + deterministic
        assert!(ad::tape_len() < 60, "tape has {} nodes", ad::tape_len());
    }

    #[test]
    fn tracer_and_postprocess() {
        let m = schools();
        let layout = SiteLayout::discover(&m);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let trace = Tracer::new(&mut rng).trace(&m);
        assert_eq!(trace.sites.len(), 5);
        assert_eq!(trace.get("tau").unwrap().kind, SiteKind::Latent);
        assert!(trace.get("tau").unwrap().value[0] > 0.0);
        assert_eq!(trace.get("y").unwrap().kind, SiteKind::Observed);
        assert_eq!(trace.get("theta_mean").unwrap().kind, SiteKind::Deterministic);

        // unconstrain the traced values and check the log joint matches
        let values = trace.latent_values();
        let z = layout.unconstrain(&values, &vec![0.0; layout.dim]);
        let lp = LogDensity::eval(&m, &z, &layout);
        let logdet = z[1]; // tau = exp(u)
        assert!((lp - logdet - trace.log_joint()).abs() < 1e-9);

        // postprocess reproduces constrained values and deterministic sites
        let post = Postprocess::run(&m, &z, &layout);
        assert_eq!(post.len(), 4);
        let (n, tau) = &post[1];
        assert_eq!(n, "tau");
        assert!((tau[0] - values["tau"][0]).abs() < 1e-12);
        let theta_mean = &post[3].1[0];
        let expect: f64 = values["theta"].iter().sum::<f64>() / 8.0;
        assert!((theta_mean - expect).abs() < 1e-12);

        // replaying values reproduces them
        let trace2 = Tracer::with_values(&mut rng, &values).trace(&m);
        assert_eq!(trace2.get("theta").unwrap().value, values["theta"]);
    }

    struct SimplexModel;
    impl Model for SimplexModel {
        fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
            let p = h.sample_vec("p", Dirichlet::new(&[1.0, 1.0, 1.0][..]));
            h.observe("obs", Categorical::new(&p).expand(3), &[0.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn simplex_site_layout_and_gradient() {
        let layout = SiteLayout::discover(&SimplexModel);
        assert_eq!(layout.dim, 2);
        assert_eq!(layout.sites[0].len, 3);
        let z = [0.4, -0.2];
        ad::reset();
        let zv = Var::leaves(&z);
        let lp = LogDensity::eval(&SimplexModel, &zv, &layout);
        let g = ad::gradient(lp, &zv);
        let fd = ad::finite_diff(|z| LogDensity::eval(&SimplexModel, z, &layout), &z, 1e-6);
        for (a, b) in g.iter().zip(&fd) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
