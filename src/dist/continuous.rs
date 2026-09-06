//! Univariate continuous distributions.

use rand::{Rng, RngCore};
use rand_distr::{Distribution as _, StandardNormal};

use super::{broadcast_len, univariate, Distribution, IntoParam, Param, Support, Value};
use super::{LN_2, LN_2PI, LN_PI};
use crate::ad::Real;
use crate::special::{digamma, ln_beta, ln_gamma};

macro_rules! impl_common {
    ($name:ident, [$($p:ident),*]) => {
        impl<'a, R: Real> $name<'a, R> {
            /// Set the batch size (number of iid draws) when all parameters are scalars.
            pub fn expand(mut self, n: usize) -> Self {
                self.n = broadcast_len(&[$(self.$p.len()),*], Some(n));
                self
            }
        }
    };
}

// ------------------------------------------------------------------ Normal ---

/// Normal distribution `N(loc, scale)`.
#[derive(Clone, Copy, Debug)]
pub struct Normal<'a, R> {
    pub loc: Param<'a, R>,
    pub scale: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Normal<'a, R> {
    pub fn new(loc: impl IntoParam<'a, R>, scale: impl IntoParam<'a, R>) -> Self {
        let (loc, scale) = (loc.into_param(), scale.into_param());
        let n = broadcast_len(&[loc.len(), scale.len()], None);
        Normal { loc, scale, n }
    }
}
impl_common!(Normal, [loc, scale]);

impl<'a, R: Real> Distribution<R> for Normal<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Real
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        // hoist log(scale) and 1/scale when the scale is shared across the plate
        let shared = self.scale.is_scalar();
        let s0 = self.scale.get(0);
        let (ls0, inv0) = (s0.ln(), 1.0 / s0);
        univariate([self.loc, self.scale], x, self.n, |[mu, s], x| {
            let (ls, inv_s) = if shared { (ls0, inv0) } else { (s.ln(), 1.0 / s) };
            let z = (x - mu) * inv_s;
            let lp = -0.5 * z * z - ls - 0.5 * LN_2PI;
            if !R::DIFFERENTIABLE {
                return (lp, [0.0; 2], 0.0);
            }
            let dz = -z * inv_s; // d lp / dx
            (lp, [-dz, (z * z - 1.0) * inv_s], dz)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let e: f64 = rng.sample(StandardNormal);
            *o = self.loc.get(i) + self.scale.get(i) * e;
        }
    }
}

// --------------------------------------------------------------- LogNormal ---

/// Log-normal distribution: `log(x) ~ N(loc, scale)`.
#[derive(Clone, Copy, Debug)]
pub struct LogNormal<'a, R> {
    pub loc: Param<'a, R>,
    pub scale: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> LogNormal<'a, R> {
    pub fn new(loc: impl IntoParam<'a, R>, scale: impl IntoParam<'a, R>) -> Self {
        let (loc, scale) = (loc.into_param(), scale.into_param());
        let n = broadcast_len(&[loc.len(), scale.len()], None);
        LogNormal { loc, scale, n }
    }
}
impl_common!(LogNormal, [loc, scale]);

impl<'a, R: Real> Distribution<R> for LogNormal<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Positive
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.loc, self.scale], x, self.n, |[mu, s], x| {
            if x <= 0.0 {
                return (f64::NEG_INFINITY, [0.0; 2], 0.0);
            }
            let lx = x.ln();
            let inv_s = 1.0 / s;
            let z = (lx - mu) * inv_s;
            let lp = -0.5 * z * z - s.ln() - 0.5 * LN_2PI - lx;
            if !R::DIFFERENTIABLE {
                return (lp, [0.0; 2], 0.0);
            }
            (lp, [z * inv_s, (z * z - 1.0) * inv_s], (-z * inv_s - 1.0) / x)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let e: f64 = rng.sample(StandardNormal);
            *o = (self.loc.get(i) + self.scale.get(i) * e).exp();
        }
    }
}

// -------------------------------------------------------------- HalfNormal ---

/// Half-normal distribution on `[0, inf)` with the given scale.
#[derive(Clone, Copy, Debug)]
pub struct HalfNormal<'a, R> {
    pub scale: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> HalfNormal<'a, R> {
    pub fn new(scale: impl IntoParam<'a, R>) -> Self {
        let scale = scale.into_param();
        let n = broadcast_len(&[scale.len()], None);
        HalfNormal { scale, n }
    }
}
impl_common!(HalfNormal, [scale]);

impl<'a, R: Real> Distribution<R> for HalfNormal<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Positive
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.scale], x, self.n, |[s], x| {
            if x < 0.0 {
                return (f64::NEG_INFINITY, [0.0], 0.0);
            }
            let inv_s = 1.0 / s;
            let z = x * inv_s;
            let lp = LN_2 - 0.5 * z * z - s.ln() - 0.5 * LN_2PI;
            if !R::DIFFERENTIABLE {
                return (lp, [0.0], 0.0);
            }
            (lp, [(z * z - 1.0) * inv_s], -z * inv_s)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let e: f64 = rng.sample(StandardNormal);
            *o = self.scale.get(i) * e.abs();
        }
    }
}

// ------------------------------------------------------------------ Cauchy ---

/// Cauchy distribution with location `loc` and scale `scale`.
#[derive(Clone, Copy, Debug)]
pub struct Cauchy<'a, R> {
    pub loc: Param<'a, R>,
    pub scale: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Cauchy<'a, R> {
    pub fn new(loc: impl IntoParam<'a, R>, scale: impl IntoParam<'a, R>) -> Self {
        let (loc, scale) = (loc.into_param(), scale.into_param());
        let n = broadcast_len(&[loc.len(), scale.len()], None);
        Cauchy { loc, scale, n }
    }
}
impl_common!(Cauchy, [loc, scale]);

#[inline]
fn cauchy_lp(mu: f64, s: f64, x: f64, grad: bool) -> (f64, [f64; 2], f64) {
    let inv_s = 1.0 / s;
    let z = (x - mu) * inv_s;
    let w = 1.0 + z * z;
    let lp = -LN_PI - s.ln() - w.ln();
    if !grad {
        return (lp, [0.0; 2], 0.0);
    }
    let dz = -2.0 * z / w;
    let dx = dz * inv_s;
    (lp, [-dx, (-1.0 - dz * z) * inv_s], dx)
}

impl<'a, R: Real> Distribution<R> for Cauchy<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Real
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.loc, self.scale], x, self.n, |[mu, s], x| {
            cauchy_lp(mu, s, x, R::DIFFERENTIABLE)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let u: f64 = rng.random();
            *o = self.loc.get(i) + self.scale.get(i) * (std::f64::consts::PI * (u - 0.5)).tan();
        }
    }
}

// -------------------------------------------------------------- HalfCauchy ---

/// Half-Cauchy distribution on `[0, inf)`.
#[derive(Clone, Copy, Debug)]
pub struct HalfCauchy<'a, R> {
    pub scale: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> HalfCauchy<'a, R> {
    pub fn new(scale: impl IntoParam<'a, R>) -> Self {
        let scale = scale.into_param();
        let n = broadcast_len(&[scale.len()], None);
        HalfCauchy { scale, n }
    }
}
impl_common!(HalfCauchy, [scale]);

impl<'a, R: Real> Distribution<R> for HalfCauchy<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Positive
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.scale], x, self.n, |[s], x| {
            if x < 0.0 {
                return (f64::NEG_INFINITY, [0.0], 0.0);
            }
            let (lp, [_, ds], dx) = cauchy_lp(0.0, s, x, R::DIFFERENTIABLE);
            (lp + LN_2, [ds], dx)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let u: f64 = rng.random();
            *o = self.scale.get(i) * (std::f64::consts::FRAC_PI_2 * u).tan();
        }
    }
}

// ---------------------------------------------------------------- StudentT ---

/// Student's t distribution with `df` degrees of freedom, location and scale.
#[derive(Clone, Copy, Debug)]
pub struct StudentT<'a, R> {
    pub df: Param<'a, R>,
    pub loc: Param<'a, R>,
    pub scale: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> StudentT<'a, R> {
    pub fn new(df: impl IntoParam<'a, R>, loc: impl IntoParam<'a, R>, scale: impl IntoParam<'a, R>) -> Self {
        let (df, loc, scale) = (df.into_param(), loc.into_param(), scale.into_param());
        let n = broadcast_len(&[df.len(), loc.len(), scale.len()], None);
        StudentT { df, loc, scale, n }
    }
}
impl_common!(StudentT, [df, loc, scale]);

impl<'a, R: Real> Distribution<R> for StudentT<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Real
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.df, self.loc, self.scale], x, self.n, |[nu, mu, s], x| {
            let inv_s = 1.0 / s;
            let z = (x - mu) * inv_s;
            let w = 1.0 + z * z / nu;
            let lp = ln_gamma(0.5 * (nu + 1.0))
                - ln_gamma(0.5 * nu)
                - 0.5 * (nu * std::f64::consts::PI).ln()
                - s.ln()
                - 0.5 * (nu + 1.0) * w.ln();
            if !R::DIFFERENTIABLE {
                return (lp, [0.0; 3], 0.0);
            }
            let dz = -(nu + 1.0) * z / (nu * w);
            let dx = dz * inv_s;
            let dnu = 0.5
                * (digamma(0.5 * (nu + 1.0)) - digamma(0.5 * nu) - 1.0 / nu - w.ln()
                    + (nu + 1.0) * z * z / (nu * nu * w));
            (lp, [dnu, -dx, (-1.0 - dz * z) * inv_s], dx)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let t = rand_distr::StudentT::new(self.df.get(i)).expect("StudentT: df must be > 0");
            let e: f64 = t.sample(rng);
            *o = self.loc.get(i) + self.scale.get(i) * e;
        }
    }
}

// ----------------------------------------------------------------- Uniform ---

/// Continuous uniform distribution on `(low, high)`.
///
/// The bounds define the support used for the unconstraining transform, so for
/// latent sites they must be constants.
#[derive(Clone, Copy, Debug)]
pub struct Uniform<'a, R> {
    pub low: Param<'a, R>,
    pub high: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Uniform<'a, R> {
    pub fn new(low: impl IntoParam<'a, R>, high: impl IntoParam<'a, R>) -> Self {
        let (low, high) = (low.into_param(), high.into_param());
        let n = broadcast_len(&[low.len(), high.len()], None);
        Uniform { low, high, n }
    }
}
impl_common!(Uniform, [low, high]);

impl<'a, R: Real> Distribution<R> for Uniform<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        // Only scalar constant bounds are representable as a single support.
        // Vector bounds are validated at transform time.
        if self.low.is_scalar() && self.high.is_scalar() {
            Support::Interval(self.low.get(0), self.high.get(0))
        } else {
            let lo = (0..self.n).map(|i| self.low.get(i)).fold(f64::INFINITY, f64::min);
            let hi = (0..self.n)
                .map(|i| self.high.get(i))
                .fold(f64::NEG_INFINITY, f64::max);
            let all_same = (0..self.n).all(|i| self.low.get(i) == lo && self.high.get(i) == hi);
            assert!(all_same, "Uniform with heterogeneous bounds cannot be a latent site; use a scalar bound and rescale, or observe it");
            Support::Interval(lo, hi)
        }
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.low, self.high], x, self.n, |[lo, hi], x| {
            if x < lo || x > hi {
                return (f64::NEG_INFINITY, [0.0; 2], 0.0);
            }
            let w = hi - lo;
            let lp = -w.ln();
            (lp, [1.0 / w, -1.0 / w], 0.0)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let u: f64 = rng.random();
            let lo = self.low.get(i);
            *o = lo + (self.high.get(i) - lo) * u;
        }
    }
}

// ------------------------------------------------------------- Exponential ---

/// Exponential distribution with the given rate.
#[derive(Clone, Copy, Debug)]
pub struct Exponential<'a, R> {
    pub rate: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Exponential<'a, R> {
    pub fn new(rate: impl IntoParam<'a, R>) -> Self {
        let rate = rate.into_param();
        let n = broadcast_len(&[rate.len()], None);
        Exponential { rate, n }
    }
}
impl_common!(Exponential, [rate]);

impl<'a, R: Real> Distribution<R> for Exponential<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Positive
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.rate], x, self.n, |[r], x| {
            if x < 0.0 {
                return (f64::NEG_INFINITY, [0.0], 0.0);
            }
            (r.ln() - r * x, [1.0 / r - x], -r)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let e: f64 = rng.sample(rand_distr::Exp1);
            *o = e / self.rate.get(i);
        }
    }
}

// ------------------------------------------------------------------- Gamma ---

/// Gamma distribution with shape `concentration` and `rate`.
#[derive(Clone, Copy, Debug)]
pub struct Gamma<'a, R> {
    pub concentration: Param<'a, R>,
    pub rate: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Gamma<'a, R> {
    pub fn new(concentration: impl IntoParam<'a, R>, rate: impl IntoParam<'a, R>) -> Self {
        let (concentration, rate) = (concentration.into_param(), rate.into_param());
        let n = broadcast_len(&[concentration.len(), rate.len()], None);
        Gamma {
            concentration,
            rate,
            n,
        }
    }
}
impl_common!(Gamma, [concentration, rate]);

impl<'a, R: Real> Distribution<R> for Gamma<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Positive
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.concentration, self.rate], x, self.n, |[a, b], x| {
            if x <= 0.0 {
                return (f64::NEG_INFINITY, [0.0; 2], 0.0);
            }
            let lx = x.ln();
            let lb = b.ln();
            let lp = a * lb - ln_gamma(a) + (a - 1.0) * lx - b * x;
            if !R::DIFFERENTIABLE {
                return (lp, [0.0; 2], 0.0);
            }
            (lp, [lb - digamma(a) + lx, a / b - x], (a - 1.0) / x - b)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let g = rand_distr::Gamma::new(self.concentration.get(i), 1.0 / self.rate.get(i))
                .expect("Gamma: invalid parameters");
            *o = g.sample(rng);
        }
    }
}

// ------------------------------------------------------------ InverseGamma ---

/// Inverse-gamma distribution: `1/x ~ Gamma(concentration, rate)`.
#[derive(Clone, Copy, Debug)]
pub struct InverseGamma<'a, R> {
    pub concentration: Param<'a, R>,
    pub rate: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> InverseGamma<'a, R> {
    pub fn new(concentration: impl IntoParam<'a, R>, rate: impl IntoParam<'a, R>) -> Self {
        let (concentration, rate) = (concentration.into_param(), rate.into_param());
        let n = broadcast_len(&[concentration.len(), rate.len()], None);
        InverseGamma {
            concentration,
            rate,
            n,
        }
    }
}
impl_common!(InverseGamma, [concentration, rate]);

impl<'a, R: Real> Distribution<R> for InverseGamma<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Positive
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.concentration, self.rate], x, self.n, |[a, b], x| {
            if x <= 0.0 {
                return (f64::NEG_INFINITY, [0.0; 2], 0.0);
            }
            let lx = x.ln();
            let lb = b.ln();
            let lp = a * lb - ln_gamma(a) - (a + 1.0) * lx - b / x;
            if !R::DIFFERENTIABLE {
                return (lp, [0.0; 2], 0.0);
            }
            (
                lp,
                [lb - digamma(a) - lx, a / b - 1.0 / x],
                -(a + 1.0) / x + b / (x * x),
            )
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let g = rand_distr::Gamma::new(self.concentration.get(i), 1.0 / self.rate.get(i))
                .expect("InverseGamma: invalid parameters");
            let y: f64 = g.sample(rng);
            *o = 1.0 / y;
        }
    }
}

// -------------------------------------------------------------------- Beta ---

/// Beta distribution with concentrations `alpha`, `beta`.
#[derive(Clone, Copy, Debug)]
pub struct Beta<'a, R> {
    pub alpha: Param<'a, R>,
    pub beta: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Beta<'a, R> {
    pub fn new(alpha: impl IntoParam<'a, R>, beta: impl IntoParam<'a, R>) -> Self {
        let (alpha, beta) = (alpha.into_param(), beta.into_param());
        let n = broadcast_len(&[alpha.len(), beta.len()], None);
        Beta { alpha, beta, n }
    }
}
impl_common!(Beta, [alpha, beta]);

impl<'a, R: Real> Distribution<R> for Beta<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::UnitInterval
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.alpha, self.beta], x, self.n, |[a, b], x| {
            if x <= 0.0 || x >= 1.0 {
                return (f64::NEG_INFINITY, [0.0; 2], 0.0);
            }
            let lx = x.ln();
            let l1x = (-x).ln_1p();
            let lp = (a - 1.0) * lx + (b - 1.0) * l1x - ln_beta(a, b);
            if !R::DIFFERENTIABLE {
                return (lp, [0.0; 2], 0.0);
            }
            let dab = digamma(a + b);
            (
                lp,
                [lx - digamma(a) + dab, l1x - digamma(b) + dab],
                (a - 1.0) / x - (b - 1.0) / (1.0 - x),
            )
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let d =
                rand_distr::Beta::new(self.alpha.get(i), self.beta.get(i)).expect("Beta: invalid parameters");
            *o = d.sample(rng);
        }
    }
}

// ------------------------------------------------------------------ Pareto ---

/// Pareto distribution with `scale` (minimum) and shape `alpha`.
#[derive(Clone, Copy, Debug)]
pub struct Pareto<'a, R> {
    pub scale: Param<'a, R>,
    pub alpha: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Pareto<'a, R> {
    pub fn new(scale: impl IntoParam<'a, R>, alpha: impl IntoParam<'a, R>) -> Self {
        let (scale, alpha) = (scale.into_param(), alpha.into_param());
        let n = broadcast_len(&[scale.len(), alpha.len()], None);
        Pareto { scale, alpha, n }
    }
}
impl_common!(Pareto, [scale, alpha]);

impl<'a, R: Real> Distribution<R> for Pareto<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        assert!(self.scale.is_scalar(), "Pareto latent sites need a scalar scale");
        Support::GreaterThan(self.scale.get(0))
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.scale, self.alpha], x, self.n, |[m, a], x| {
            if x < m {
                return (f64::NEG_INFINITY, [0.0; 2], 0.0);
            }
            let lm = m.ln();
            let lx = x.ln();
            let lp = a.ln() + a * lm - (a + 1.0) * lx;
            (lp, [a / m, 1.0 / a + lm - lx], -(a + 1.0) / x)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let u: f64 = rng.random();
            *o = self.scale.get(i) * (1.0 - u).powf(-1.0 / self.alpha.get(i));
        }
    }
}

// ----------------------------------------------------------------- Laplace ---

/// Laplace (double exponential) distribution.
#[derive(Clone, Copy, Debug)]
pub struct Laplace<'a, R> {
    pub loc: Param<'a, R>,
    pub scale: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Laplace<'a, R> {
    pub fn new(loc: impl IntoParam<'a, R>, scale: impl IntoParam<'a, R>) -> Self {
        let (loc, scale) = (loc.into_param(), scale.into_param());
        let n = broadcast_len(&[loc.len(), scale.len()], None);
        Laplace { loc, scale, n }
    }
}
impl_common!(Laplace, [loc, scale]);

impl<'a, R: Real> Distribution<R> for Laplace<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Real
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.loc, self.scale], x, self.n, |[mu, b], x| {
            let d = x - mu;
            let ad = d.abs();
            let lp = -(2.0 * b).ln() - ad / b;
            let sgn = if d >= 0.0 { 1.0 } else { -1.0 };
            (lp, [sgn / b, -1.0 / b + ad / (b * b)], -sgn / b)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let u: f64 = rng.random::<f64>() - 0.5;
            let s = if u >= 0.0 { 1.0 } else { -1.0 };
            *o = self.loc.get(i) - self.scale.get(i) * s * (1.0 - 2.0 * u.abs()).ln();
        }
    }
}

// --------------------------------------------------------- ImproperUniform ---

/// An improper flat prior with the given support (log density zero everywhere
/// on the support). Useful for "uninformative" priors in models where the
/// posterior is still proper.
#[derive(Clone, Copy, Debug)]
pub struct ImproperUniform {
    pub support: Support,
    n: usize,
}

impl ImproperUniform {
    pub fn new(support: Support, len: usize) -> Self {
        ImproperUniform { support, n: len }
    }
}

impl<R: Real> Distribution<R> for ImproperUniform {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        self.support
    }
    fn log_prob_value(&self, _x: Value<'_, R>) -> R {
        R::zero()
    }
    fn sample(&self, _rng: &mut dyn RngCore, _out: &mut [f64]) {
        panic!("cannot sample from an ImproperUniform distribution");
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::ad::Var;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn rng() -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(0)
    }

    // Reference values computed with scipy.stats.*.logpdf.
    #[test]
    fn normal_values() {
        let d = Normal::new(0.5, 2.0);
        let lp: f64 = d.log_prob_data(&[1.3]);
        assert!((lp - (-1.6920857138)).abs() < 1e-9);
        let d = Normal::new(&[0.0, 1.0][..], &[1.0, 3.0][..]);
        let lp: f64 = d.log_prob_data(&[0.2, -1.0]);
        let expected = -0.9389385332 + (-2.2397730441);
        assert!((lp - expected).abs() < 1e-9, "{lp} vs {expected}");
    }

    #[test]
    fn normal_grads() {
        // params: [mu, sigma, x0, x1]
        check_grad(
            |p| Normal::new(p[0], p[1]).expand(2).log_prob(&p[2..4]),
            |p| Normal::new(p[0], p[1]).expand(2).log_prob(&p[2..4]),
            &[0.3, 1.7, -0.4, 2.2],
            1e-6,
        );
        // vector loc, scalar constant scale, observed data
        check_grad(
            |p| Normal::new(&p[0..2], 0.7).log_prob_data(&[1.0, -2.0]),
            |p| Normal::new(&p[0..2], 0.7).log_prob_data(&[1.0, -2.0]),
            &[0.3, 1.7],
            1e-6,
        );
    }

    #[test]
    fn lognormal_halfnormal_grads() {
        check_grad(
            |p| LogNormal::new(p[0], p[1]).log_prob(&p[2..3]),
            |p| LogNormal::new(p[0], p[1]).log_prob(&p[2..3]),
            &[0.3, 0.8, 1.9],
            1e-6,
        );
        let lp: f64 = LogNormal::new(0.3, 0.8).log_prob_data(&[1.9]);
        assert!((lp - (-1.4289489302)).abs() < 1e-8, "{lp}");
        check_grad(
            |p| HalfNormal::new(p[0]).log_prob(&p[1..2]),
            |p| HalfNormal::new(p[0]).log_prob(&p[1..2]),
            &[1.3, 0.6],
            1e-6,
        );
        let lp: f64 = HalfNormal::new(1.3).log_prob_data(&[0.6]);
        assert!((lp - (-0.5946644929)).abs() < 1e-8, "{lp}");
    }

    #[test]
    fn cauchy_grads() {
        check_grad(
            |p| Cauchy::new(p[0], p[1]).log_prob(&p[2..3]),
            |p| Cauchy::new(p[0], p[1]).log_prob(&p[2..3]),
            &[0.3, 0.8, 1.9],
            1e-6,
        );
        let lp: f64 = Cauchy::new(0.3, 0.8).log_prob_data(&[1.9]);
        assert!((lp - (-2.5310242470)).abs() < 1e-8, "{lp}");
        check_grad(
            |p| HalfCauchy::new(p[0]).log_prob(&p[1..2]),
            |p| HalfCauchy::new(p[0]).log_prob(&p[1..2]),
            &[0.8, 1.9],
            1e-6,
        );
        let lp: f64 = HalfCauchy::new(0.8).log_prob_data(&[1.9]);
        assert!((lp - (-2.1216452395)).abs() < 1e-8, "{lp}");
    }

    #[test]
    fn student_t_grads() {
        check_grad(
            |p| StudentT::new(p[0], p[1], p[2]).log_prob(&p[3..4]),
            |p| StudentT::new(p[0], p[1], p[2]).log_prob(&p[3..4]),
            &[3.5, 0.2, 1.4, -1.1],
            1e-6,
        );
        let lp: f64 = StudentT::new(3.5, 0.2, 1.4).log_prob_data(&[-1.1]);
        assert!((lp - (-1.8214494706)).abs() < 1e-8, "{lp}");
    }

    #[test]
    fn uniform_exponential_gamma_grads() {
        check_grad(
            |p| Uniform::new(p[0], p[1]).log_prob(&p[2..3]),
            |p| Uniform::new(p[0], p[1]).log_prob(&p[2..3]),
            &[-1.0, 3.0, 0.5],
            1e-6,
        );
        let lp: f64 = Uniform::new(-1.0, 3.0).log_prob_data(&[5.0]);
        assert_eq!(lp, f64::NEG_INFINITY);
        check_grad(
            |p| Exponential::new(p[0]).log_prob(&p[1..2]),
            |p| Exponential::new(p[0]).log_prob(&p[1..2]),
            &[2.5, 0.7],
            1e-6,
        );
        check_grad(
            |p| Gamma::new(p[0], p[1]).log_prob(&p[2..3]),
            |p| Gamma::new(p[0], p[1]).log_prob(&p[2..3]),
            &[2.5, 0.7, 1.9],
            1e-6,
        );
        let lp: f64 = Gamma::new(2.5, 0.7).log_prob_data(&[1.9]);
        assert!((lp - (-1.5435894011)).abs() < 1e-8, "{lp}");
        check_grad(
            |p| InverseGamma::new(p[0], p[1]).log_prob(&p[2..3]),
            |p| InverseGamma::new(p[0], p[1]).log_prob(&p[2..3]),
            &[2.5, 0.7, 1.9],
            1e-6,
        );
        let lp: f64 = InverseGamma::new(2.5, 0.7).log_prob_data(&[1.9]);
        assert!((lp - (-3.7912798846)).abs() < 1e-8, "{lp}");
    }

    #[test]
    fn beta_pareto_laplace_grads() {
        check_grad(
            |p| Beta::new(p[0], p[1]).log_prob(&p[2..3]),
            |p| Beta::new(p[0], p[1]).log_prob(&p[2..3]),
            &[2.5, 0.7, 0.3],
            1e-6,
        );
        let lp: f64 = Beta::new(2.5, 0.7).log_prob_data(&[0.3]);
        assert!((lp - (-1.3591020132)).abs() < 1e-8, "{lp}");
        check_grad(
            |p| Pareto::new(p[0], p[1]).log_prob(&p[2..3]),
            |p| Pareto::new(p[0], p[1]).log_prob(&p[2..3]),
            &[1.0, 1.5, 2.3],
            1e-6,
        );
        let lp: f64 = Pareto::new(1.0, 1.5).log_prob_data(&[2.3]);
        assert!((lp - (-1.6768076992)).abs() < 1e-8, "{lp}");
        check_grad(
            |p| Laplace::new(p[0], p[1]).log_prob(&p[2..3]),
            |p| Laplace::new(p[0], p[1]).log_prob(&p[2..3]),
            &[0.4, 1.5, 2.3],
            1e-6,
        );
        let lp: f64 = Laplace::new(0.4, 1.5).log_prob_data(&[2.3]);
        assert!((lp - (-2.3652789553)).abs() < 1e-8, "{lp}");
    }

    #[test]
    fn improper_uniform() {
        let d = ImproperUniform::new(Support::Positive, 2);
        let lp: f64 = d.log_prob_data(&[1.0, 2.0]);
        assert_eq!(lp, 0.0);
        assert_eq!(
            <ImproperUniform as Distribution<f64>>::support(&d),
            Support::Positive
        );
    }

    #[test]
    fn single_tape_node_per_site() {
        crate::ad::reset();
        let xs = Var::leaves(&[0.1; 1000]);
        let mu = Var::new(0.0);
        let n0 = crate::ad::tape_len();
        let _ = Normal::new(mu, 1.0).expand(1000).log_prob(&xs);
        assert_eq!(crate::ad::tape_len(), n0 + 1);
    }

    #[test]
    fn sampling_moments() {
        let n = 200_000;
        let check = |name: &str, d: &dyn Fn(&mut Xoshiro256PlusPlus) -> f64, mean: f64, var: f64| {
            let mut r2 = rng();
            let xs: Vec<f64> = (0..n).map(|_| d(&mut r2)).collect();
            let (m, v) = mean_var(&xs);
            assert!(
                (m - mean).abs() < 0.03 * (1.0 + mean.abs()),
                "{name} mean {m} vs {mean}"
            );
            assert!((v - var).abs() < 0.05 * (1.0 + var), "{name} var {v} vs {var}");
        };
        check(
            "normal",
            &|r| Normal::<f64>::new(1.0, 2.0).sample_vec(r)[0],
            1.0,
            4.0,
        );
        check(
            "lognormal",
            &|r| LogNormal::<f64>::new(0.0, 0.5).sample_vec(r)[0],
            (0.125f64).exp(),
            ((0.25f64).exp() - 1.0) * (0.25f64).exp(),
        );
        check(
            "halfnormal",
            &|r| HalfNormal::<f64>::new(2.0).sample_vec(r)[0],
            2.0 * (2.0 / std::f64::consts::PI).sqrt(),
            4.0 * (1.0 - 2.0 / std::f64::consts::PI),
        );
        check(
            "uniform",
            &|r| Uniform::<f64>::new(-1.0, 3.0).sample_vec(r)[0],
            1.0,
            16.0 / 12.0,
        );
        check(
            "exponential",
            &|r| Exponential::<f64>::new(2.0).sample_vec(r)[0],
            0.5,
            0.25,
        );
        check(
            "gamma",
            &|r| Gamma::<f64>::new(3.0, 2.0).sample_vec(r)[0],
            1.5,
            0.75,
        );
        check(
            "invgamma",
            &|r| InverseGamma::<f64>::new(4.0, 2.0).sample_vec(r)[0],
            2.0 / 3.0,
            4.0 / 18.0,
        );
        check(
            "beta",
            &|r| Beta::<f64>::new(2.0, 3.0).sample_vec(r)[0],
            0.4,
            0.04,
        );
        check(
            "pareto",
            &|r| Pareto::<f64>::new(1.0, 4.0).sample_vec(r)[0],
            4.0 / 3.0,
            4.0 / 18.0,
        );
        check(
            "laplace",
            &|r| Laplace::<f64>::new(0.5, 1.5).sample_vec(r)[0],
            0.5,
            4.5,
        );
        check(
            "student_t",
            &|r| StudentT::<f64>::new(7.0, 0.5, 1.5).sample_vec(r)[0],
            0.5,
            2.25 * 7.0 / 5.0,
        );
        // Cauchy has no mean; check the median instead.
        let mut r2 = rng();
        let mut xs: Vec<f64> = (0..n)
            .map(|_| Cauchy::<f64>::new(0.5, 1.5).sample_vec(&mut r2)[0])
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((xs[n / 2] - 0.5).abs() < 0.03);
    }
}
