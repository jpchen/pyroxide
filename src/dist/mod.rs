//! Probability distributions with analytic gradients.
//!
//! Every distribution implements [`Distribution<R>`] for both `f64` and
//! [`Var`](crate::ad::Var). Log densities are computed in closed form together
//! with their partial derivatives, and pushed onto the autodiff tape as a
//! *single* node (see [`crate::ad`]). This is what makes gradient evaluation
//! competitive with compiled code.
//!
//! # Parameters
//!
//! Parameters are passed as anything implementing [`IntoParam`]: a scalar
//! (`f64` constant or differentiable `R`), or a slice / `Vec` of either. Slices
//! broadcast against each other and against the batch size set with
//! [`expand`](Normal::expand):
//!
//! ```
//! use pyroxide::dist::{Normal, Distribution};
//! // a plate of 3 independent normals with a shared scale
//! let d = Normal::<f64>::new(&[0.0, 1.0, 2.0][..], 1.0);
//! assert_eq!(d.len(), 3);
//! // a plate of 10 iid standard normals
//! let d = Normal::<f64>::new(0.0, 1.0).expand(10);
//! assert_eq!(d.len(), 10);
//! let lp: f64 = d.log_prob_data(&[0.0; 10]);
//! assert!((lp - 10.0 * (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-12);
//! ```

use rand::RngCore;

use crate::ad::{NodeBuilder, Real, Var};

mod continuous;
mod discrete;
mod multivariate;

pub use continuous::*;
pub use discrete::*;
pub use multivariate::*;

/// The support of a distribution, used to choose the bijection to
/// unconstrained space for gradient-based inference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Support {
    /// The whole real line.
    Real,
    /// `(0, inf)`.
    Positive,
    /// `(0, 1)`.
    UnitInterval,
    /// `(lo, hi)`.
    Interval(f64, f64),
    /// `(lo, inf)`.
    GreaterThan(f64),
    /// `(-inf, hi)`.
    LessThan(f64),
    /// Probability vectors summing to one (event dimension `k`).
    Simplex,
    /// `{0, 1}` (discrete).
    Boolean,
    /// Non-negative integers (discrete).
    NonNegativeInteger,
    /// `{0, ..., k-1}` (discrete).
    IntegerInterval(i64, i64),
    /// Strictly increasing vectors (event dimension `k`).
    OrderedVector,
    /// Lower Cholesky factors of correlation matrices (event is a row-major
    /// `k x k` matrix with `event_len = k * k`).
    CorrCholesky,
}

impl Support {
    /// True for discrete supports, which cannot be latent in gradient-based
    /// kernels.
    pub fn is_discrete(&self) -> bool {
        matches!(
            self,
            Support::Boolean | Support::NonNegativeInteger | Support::IntegerInterval(_, _)
        )
    }
}

/// Restricts a distribution to strictly increasing event vectors.
///
/// The density is the base density on the ordered region (the `1/k!`
/// normalizer is dropped, as in Stan's `ordered` type); sampling sorts a base
/// draw. Use it to break label switching in mixtures:
///
/// ```
/// use pyroxide::dist::{Normal, Ordered, Distribution, Support};
/// let d = Ordered::new(Normal::<f64>::new(0.0, 5.0).expand(3), 3);
/// assert_eq!(d.support(), Support::OrderedVector);
/// assert_eq!(d.event_len(), 3);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Ordered<D> {
    pub base: D,
    k: usize,
}

impl<D> Ordered<D> {
    /// `k` is the event length (the base distribution's `len` must be a multiple).
    pub fn new(base: D, k: usize) -> Self {
        Ordered { base, k }
    }
}

impl<R: Real, D: Distribution<R>> Distribution<R> for Ordered<D> {
    fn len(&self) -> usize {
        self.base.len()
    }
    fn event_len(&self) -> usize {
        self.k
    }
    fn support(&self) -> Support {
        Support::OrderedVector
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        let n = x.len();
        for row in 0..n / self.k {
            for i in 1..self.k {
                if x.get(row * self.k + i) <= x.get(row * self.k + i - 1) {
                    return R::constant(f64::NEG_INFINITY);
                }
            }
        }
        self.base.log_prob_value(x)
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        self.base.sample(rng, out);
        for row in out.chunks_mut(self.k) {
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
}

/// A distribution parameter: either a scalar or a per-element slice, either
/// constant (`f64`) or differentiable (`R`).
#[derive(Clone, Copy, Debug)]
pub enum Param<'a, R> {
    /// Differentiable scalar.
    S(R),
    /// Differentiable vector.
    V(&'a [R]),
    /// Constant scalar.
    C(f64),
    /// Constant vector.
    CV(&'a [f64]),
}

impl<'a, R: Real> Param<'a, R> {
    /// Number of elements (1 for scalars).
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Param::S(_) | Param::C(_) => 1,
            Param::V(v) => v.len(),
            Param::CV(v) => v.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True if the parameter is a scalar (broadcast against the batch).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        matches!(self, Param::S(_) | Param::C(_))
    }

    /// Value of element `i` (scalars broadcast).
    #[inline]
    pub fn get(&self, i: usize) -> f64 {
        match self {
            Param::S(r) => r.value(),
            Param::C(c) => *c,
            Param::V(v) => v[i].value(),
            Param::CV(v) => v[i],
        }
    }

    /// Element `i` as an `R` (constants are lifted).
    #[inline]
    pub fn get_r(&self, i: usize) -> R {
        match self {
            Param::S(r) => *r,
            Param::C(c) => R::constant(*c),
            Param::V(v) => v[i],
            Param::CV(v) => R::constant(v[i]),
        }
    }

    /// Build a vector parameter from a slice of `R` (useful in generic code).
    #[inline]
    pub fn vec(v: &'a [R]) -> Self {
        Param::V(v)
    }

    /// Materialize into a `Vec<R>` of length `n` (broadcasting scalars).
    pub fn to_vec(&self, n: usize) -> Vec<R> {
        (0..n).map(|i| self.get_r(i)).collect()
    }
}

/// Accumulates the partial derivative of a node with respect to a parameter,
/// handling scalar broadcasting (scalars receive the sum of per-element
/// partials; vectors receive one partial per element).
pub struct ParamGrad<'a, R: Real> {
    param: Param<'a, R>,
    sum: f64,
}

impl<'a, R: Real> ParamGrad<'a, R> {
    #[inline]
    pub fn new(param: Param<'a, R>) -> Self {
        ParamGrad { param, sum: 0.0 }
    }

    /// Record partial `d` for element `i`.
    #[inline]
    pub fn add(&mut self, b: &mut R::Node, i: usize, d: f64) {
        match self.param {
            Param::S(_) => self.sum += d,
            Param::V(v) => b.add(v[i], d),
            _ => {}
        }
    }

    /// Flush the accumulated scalar partial (call once, after the loop).
    #[inline]
    pub fn finish(self, b: &mut R::Node) {
        if let Param::S(r) = self.param {
            b.add(r, self.sum);
        }
    }
}

/// Element type of a parameter slice. Implemented for `f64` (constant data,
/// valid for any `R`) and for [`Var`] (when `R = Var`).
pub trait ParamElem<R>: Copy {
    fn slice_param(xs: &[Self]) -> Param<'_, R>;
}

impl<R: Real> ParamElem<R> for f64 {
    #[inline]
    fn slice_param(xs: &[f64]) -> Param<'_, R> {
        Param::CV(xs)
    }
}

impl ParamElem<Var> for Var {
    #[inline]
    fn slice_param(xs: &[Var]) -> Param<'_, Var> {
        Param::V(xs)
    }
}

/// Conversion into a [`Param`]. Implemented for `f64`, `R`, `&[T]`, `&Vec<T>`,
/// `&[T; N]` (with `T` = `f64` or `R`), and `Param` itself.
pub trait IntoParam<'a, R> {
    fn into_param(self) -> Param<'a, R>;
}

impl<'a, R: Real> IntoParam<'a, R> for f64 {
    #[inline]
    fn into_param(self) -> Param<'a, R> {
        Param::C(self)
    }
}

impl<'a> IntoParam<'a, Var> for Var {
    #[inline]
    fn into_param(self) -> Param<'a, Var> {
        Param::S(self)
    }
}

impl<'a, R: Real> IntoParam<'a, R> for Param<'a, R> {
    #[inline]
    fn into_param(self) -> Param<'a, R> {
        self
    }
}

impl<'a, R: Real, T: ParamElem<R>> IntoParam<'a, R> for &'a [T] {
    #[inline]
    fn into_param(self) -> Param<'a, R> {
        T::slice_param(self)
    }
}

impl<'a, R: Real, T: ParamElem<R>> IntoParam<'a, R> for &'a Vec<T> {
    #[inline]
    fn into_param(self) -> Param<'a, R> {
        T::slice_param(self.as_slice())
    }
}

impl<'a, R: Real, T: ParamElem<R>, const N: usize> IntoParam<'a, R> for &'a [T; N] {
    #[inline]
    fn into_param(self) -> Param<'a, R> {
        T::slice_param(self.as_slice())
    }
}

/// The argument of a log density: either latent (differentiable) values or
/// observed data.
#[derive(Clone, Copy, Debug)]
pub enum Value<'a, R> {
    Latent(&'a [R]),
    Data(&'a [f64]),
}

impl<'a, R: Real> Value<'a, R> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Value::Latent(v) => v.len(),
            Value::Data(v) => v.len(),
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn get(&self, i: usize) -> f64 {
        match self {
            Value::Latent(v) => v[i].value(),
            Value::Data(v) => v[i],
        }
    }
    /// Record partial `d` with respect to element `i` (no-op for data).
    #[inline]
    pub fn add_grad(&self, b: &mut R::Node, i: usize, d: f64) {
        if let Value::Latent(v) = self {
            b.add(v[i], d);
        }
    }
}

/// A probability distribution over vectors of length [`len`](Distribution::len).
///
/// For univariate families, `len` is the batch size (independent draws whose
/// log densities are summed). For multivariate families ([`Dirichlet`],
/// [`MultivariateNormal`]) `len` is `batch * event_len`.
pub trait Distribution<R: Real> {
    /// Total number of scalars in one draw.
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size of one event (1 for univariate families).
    #[inline]
    fn event_len(&self) -> usize {
        1
    }

    /// Support of each event.
    fn support(&self) -> Support;

    /// Summed log density (or mass) of `x`, which has length [`len`](Self::len).
    fn log_prob_value(&self, x: Value<'_, R>) -> R;

    /// Log density of latent (differentiable) values.
    #[inline]
    fn log_prob(&self, x: &[R]) -> R {
        debug_assert_eq!(x.len(), self.len(), "log_prob: wrong length");
        self.log_prob_value(Value::Latent(x))
    }

    /// Log density of observed data.
    #[inline]
    fn log_prob_data(&self, x: &[f64]) -> R {
        debug_assert_eq!(x.len(), self.len(), "log_prob_data: wrong length");
        self.log_prob_value(Value::Data(x))
    }

    /// Draw one sample (length [`len`](Self::len)) into `out`.
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]);

    /// Draw one sample into a new vector.
    fn sample_vec(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        self.sample(rng, &mut out);
        out
    }
}

/// Resolve the batch size from a set of parameter lengths and an explicit
/// `expand` size. Panics on incompatible lengths.
pub(crate) fn broadcast_len(lens: &[usize], expand: Option<usize>) -> usize {
    let mut n = expand.unwrap_or(1);
    for &l in lens {
        if l == 1 {
            continue;
        }
        if n == 1 || n == l {
            n = l;
        } else {
            panic!("incompatible parameter lengths: {lens:?} with batch size {n}");
        }
    }
    n
}

/// Shared implementation of batched univariate log densities.
///
/// `f(params, x) -> (log_prob, d/dparams, d/dx)` is evaluated per element; the
/// helper accumulates the value and pushes one tape node with partials for
/// every differentiable input.
#[inline]
pub(crate) fn univariate<R: Real, const K: usize>(
    params: [Param<'_, R>; K],
    x: Value<'_, R>,
    n: usize,
    f: impl Fn([f64; K], f64) -> (f64, [f64; K], f64),
) -> R {
    debug_assert_eq!(x.len(), n);
    let mut total = 0.0;
    let mut b = R::begin_node(n + K);
    let mut grads: [ParamGrad<'_, R>; K] = std::array::from_fn(|k| ParamGrad::new(params[k]));
    for i in 0..n {
        let p: [f64; K] = std::array::from_fn(|k| params[k].get(i));
        let (lp, dp, dx) = f(p, x.get(i));
        total += lp;
        if R::DIFFERENTIABLE {
            for k in 0..K {
                grads[k].add(&mut b, i, dp[k]);
            }
            x.add_grad(&mut b, i, dx);
        }
    }
    if R::DIFFERENTIABLE {
        for g in grads {
            g.finish(&mut b);
        }
    }
    b.finish(total)
}

pub(crate) const LN_2PI: f64 = 1.8378770664093453;
pub(crate) const LN_PI: f64 = 1.1447298858494002;
pub(crate) const LN_2: f64 = std::f64::consts::LN_2;

#[cfg(test)]
pub(crate) mod testing {
    //! Shared test helpers for distribution gradient checks.
    use super::*;
    use crate::ad;

    /// Check the autodiff gradient of `f` against central finite differences.
    /// `f` takes the flat parameter vector as `Var`s / `f64`s.
    pub fn check_grad<FV, FF>(fv: FV, ff: FF, x: &[f64], tol: f64)
    where
        FV: Fn(&[Var]) -> Var,
        FF: Fn(&[f64]) -> f64,
    {
        ad::reset();
        let xs = Var::leaves(x);
        let y = fv(&xs);
        let y0 = ff(x);
        assert!(
            (y.value() - y0).abs() < 1e-9 * (1.0 + y0.abs()),
            "value mismatch: Var {} vs f64 {}",
            y.value(),
            y0
        );
        let g = ad::gradient(y, &xs);
        let fd = ad::finite_diff(&ff, x, 1e-6);
        for (i, (a, b)) in g.iter().zip(&fd).enumerate() {
            assert!(
                (a - b).abs() < tol * (1.0 + b.abs()),
                "gradient mismatch at {i}: ad {a} vs fd {b} (full ad {g:?}, fd {fd:?})"
            );
        }
    }

    pub fn mean_var(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let m = xs.iter().sum::<f64>() / n;
        let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
        (m, v)
    }
}
