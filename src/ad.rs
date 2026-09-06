//! Reverse-mode automatic differentiation on a thread-local tape.
//!
//! The design follows the Stan math library rather than JAX: a model is
//! written once, generically over a scalar type [`Real`], and is then run either
//! with `f64` (plain evaluation, zero overhead) or with [`Var`] (every arithmetic
//! operation records a node on a thread-local [`Tape`]; a single backward sweep
//! produces the gradient with respect to all leaves).
//!
//! Two things make this fast enough to compete with XLA-compiled code:
//!
//! 1. **Flat tape.** Nodes are stored in structure-of-arrays form
//!    (`starts`, `parents`, `weights`). The backward sweep is one tight loop with
//!    no pointer chasing and no allocation.
//! 2. **Vector nodes with analytic partials.** Distributions do not build their
//!    log density out of scalar operations. Instead they compute the value and
//!    the partial derivative with respect to each input in closed form and push
//!    a single node via [`Real::begin_node`]. A `Normal` plate with 3000
//!    observations costs one node with a handful of inputs, not ~15000 scalar
//!    nodes.
//!
//! The tape is reused between gradient evaluations ([`reset`]) so steady-state
//! MCMC never allocates for autodiff.
//!
//! # Example
//! ```
//! use pyroxide::ad::{self, Real, Var};
//!
//! fn f<R: Real>(x: R, y: R) -> R { (x * y).exp() + x.ln() }
//!
//! ad::reset();
//! let x = Var::new(2.0);
//! let y = Var::new(0.5);
//! let z = f(x, y);
//! let grads = ad::gradient(z, &[x, y]);
//! let e = (2.0f64 * 0.5).exp();
//! assert!((grads[0] - (0.5 * e + 0.5)).abs() < 1e-12);
//! assert!((grads[1] - 2.0 * e).abs() < 1e-12);
//! // The same function evaluated with plain floats:
//! assert_eq!(f(2.0f64, 0.5), z.value());
//! ```

use std::cell::RefCell;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::special::{digamma, ln_gamma};

/// Structure-of-arrays record of every operation performed on [`Var`]s.
///
/// Node `i` has parents `parents[starts[i]..starts[i+1]]` with local partial
/// derivatives `weights[starts[i]..starts[i+1]]`.
#[derive(Default, Debug)]
pub struct Tape {
    starts: Vec<u32>,
    parents: Vec<u32>,
    weights: Vec<f64>,
    adj: Vec<f64>,
    /// scratch used by [`NodeBuilder`] to avoid allocating per node
    scratch_parents: Vec<u32>,
    scratch_weights: Vec<f64>,
}

impl Tape {
    fn new() -> Self {
        let mut t = Tape::default();
        t.starts.push(0);
        t
    }

    #[inline]
    fn len(&self) -> usize {
        self.starts.len() - 1
    }

    #[inline]
    fn push_leaf(&mut self) -> u32 {
        let idx = self.len() as u32;
        self.starts.push(self.parents.len() as u32);
        idx
    }

    #[inline]
    fn push1(&mut self, p: u32, w: f64) -> u32 {
        let idx = self.len() as u32;
        self.parents.push(p);
        self.weights.push(w);
        self.starts.push(self.parents.len() as u32);
        idx
    }

    #[inline]
    fn push2(&mut self, p1: u32, w1: f64, p2: u32, w2: f64) -> u32 {
        let idx = self.len() as u32;
        self.parents.push(p1);
        self.parents.push(p2);
        self.weights.push(w1);
        self.weights.push(w2);
        self.starts.push(self.parents.len() as u32);
        idx
    }

    fn clear(&mut self) {
        self.starts.clear();
        self.starts.push(0);
        self.parents.clear();
        self.weights.clear();
    }

    /// Reverse sweep from `out`. Afterwards `adj[i]` holds d out / d node_i.
    fn backward(&mut self, out: u32) {
        let n = self.len();
        self.adj.clear();
        self.adj.resize(n, 0.0);
        let out = out as usize;
        self.adj[out] = 1.0;
        let adj = &mut self.adj;
        let starts = &self.starts;
        let parents = &self.parents;
        let weights = &self.weights;
        for i in (0..=out).rev() {
            let a = adj[i];
            if a == 0.0 {
                continue;
            }
            let (s, e) = (starts[i] as usize, starts[i + 1] as usize);
            for k in s..e {
                adj[parents[k] as usize] += a * weights[k];
            }
        }
    }
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

/// Clear the thread-local tape (keeping its memory), starting a fresh gradient
/// computation. Every `Var` created before `reset` becomes invalid.
pub fn reset() {
    TAPE.with(|t| t.borrow_mut().clear());
}

/// Number of nodes currently on the thread-local tape.
pub fn tape_len() -> usize {
    TAPE.with(|t| t.borrow().len())
}

/// Run the backward sweep from `out` and return `d out / d v` for each `v` in `wrt`.
pub fn gradient(out: Var, wrt: &[Var]) -> Vec<f64> {
    let mut g = vec![0.0; wrt.len()];
    gradient_into(out, wrt, &mut g);
    g
}

/// Like [`gradient`] but writes into a caller-supplied buffer (no allocation
/// besides the tape's own reused adjoint buffer).
pub fn gradient_into(out: Var, wrt: &[Var], grad: &mut [f64]) {
    assert_eq!(wrt.len(), grad.len());
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        t.backward(out.idx);
        for (g, v) in grad.iter_mut().zip(wrt) {
            *g = t.adj[v.idx as usize];
        }
    });
}

/// A differentiable scalar: a value plus an index into the thread-local tape.
///
/// `Var` is `Copy` and 16 bytes; all arithmetic operators are overloaded, and
/// mixing with `f64` on either side is supported.
#[derive(Clone, Copy)]
pub struct Var {
    val: f64,
    idx: u32,
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Var({}, #{})", self.val, self.idx)
    }
}

impl Var {
    /// Create a new independent variable (leaf) on the tape.
    #[inline]
    pub fn new(val: f64) -> Var {
        let idx = TAPE.with(|t| t.borrow_mut().push_leaf());
        Var { val, idx }
    }

    /// Create leaves for every element of `vals` in order.
    pub fn leaves(vals: &[f64]) -> Vec<Var> {
        vals.iter().map(|&v| Var::new(v)).collect()
    }

    /// Value of this variable.
    #[inline]
    pub fn value(self) -> f64 {
        self.val
    }

    #[inline]
    fn unary(self, val: f64, w: f64) -> Var {
        let idx = TAPE.with(|t| t.borrow_mut().push1(self.idx, w));
        Var { val, idx }
    }

    #[inline]
    fn binary(self, other: Var, val: f64, w1: f64, w2: f64) -> Var {
        let idx = TAPE.with(|t| t.borrow_mut().push2(self.idx, w1, other.idx, w2));
        Var { val, idx }
    }
}

/// Builder for an n-ary tape node with caller-computed partials.
///
/// Obtained from [`Real::begin_node`]. For `f64` this is a zero-cost no-op; for
/// [`Var`] it accumulates `(parent, partial)` pairs into a reusable scratch
/// buffer and pushes a single node on `finish`.
pub trait NodeBuilder<R> {
    /// Register `input` as a parent with local partial derivative `partial`.
    fn add(&mut self, input: R, partial: f64);
    /// Push the node with the given value and return the resulting scalar.
    fn finish(self, value: f64) -> R;
}

/// [`NodeBuilder`] for plain floats: does nothing.
pub struct F64Node;

impl NodeBuilder<f64> for F64Node {
    #[inline(always)]
    fn add(&mut self, _input: f64, _partial: f64) {}
    #[inline(always)]
    fn finish(self, value: f64) -> f64 {
        value
    }
}

/// [`NodeBuilder`] for [`Var`].
pub struct VarNode {
    parents: Vec<u32>,
    weights: Vec<f64>,
}

impl NodeBuilder<Var> for VarNode {
    #[inline]
    fn add(&mut self, input: Var, partial: f64) {
        self.parents.push(input.idx);
        self.weights.push(partial);
    }
    fn finish(mut self, value: f64) -> Var {
        let idx = TAPE.with(|t| {
            let mut t = t.borrow_mut();
            let idx = t.len() as u32;
            t.parents.extend_from_slice(&self.parents);
            t.weights.extend_from_slice(&self.weights);
            let end = t.parents.len() as u32;
            t.starts.push(end);
            // hand the scratch buffers back for reuse
            self.parents.clear();
            self.weights.clear();
            t.scratch_parents = std::mem::take(&mut self.parents);
            t.scratch_weights = std::mem::take(&mut self.weights);
            idx
        });
        Var { val: value, idx }
    }
}

/// The scalar abstraction models are written against.
///
/// Implemented for `f64` (evaluation) and [`Var`] (differentiation). Generic
/// code should use the methods here (`exp`, `ln`, ...) rather than the inherent
/// `f64` methods so it works for both.
pub trait Real:
    Copy
    + fmt::Debug
    + PartialOrd
    + Send
    + Sync
    + 'static
    + crate::dist::ParamElem<Self>
    + for<'a> crate::dist::IntoParam<'a, Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sum
{
    /// Builder type returned by [`Real::begin_node`].
    type Node: NodeBuilder<Self>;

    /// Lift a constant (not differentiated).
    fn constant(v: f64) -> Self;
    /// Read the numeric value.
    fn value(self) -> f64;
    /// Does this type record derivatives? (`false` for `f64`)
    const DIFFERENTIABLE: bool;

    /// Begin an n-ary node. Call [`NodeBuilder::add`] for each input, then
    /// [`NodeBuilder::finish`] with the node's value.
    ///
    /// Other `Var` arithmetic may be performed between `begin_node` and
    /// `finish` (the builder owns its own scratch buffers), but doing so is
    /// unnecessary: partials are plain `f64`s.
    fn begin_node(capacity_hint: usize) -> Self::Node;

    /// Convenience: node with explicit inputs and partials of equal length.
    fn node(value: f64, inputs: &[Self], partials: &[f64]) -> Self {
        debug_assert_eq!(inputs.len(), partials.len());
        let mut b = Self::begin_node(inputs.len());
        for (i, p) in inputs.iter().zip(partials) {
            b.add(*i, *p);
        }
        b.finish(value)
    }

    fn zero() -> Self {
        Self::constant(0.0)
    }
    fn one() -> Self {
        Self::constant(1.0)
    }

    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn powf(self, p: f64) -> Self;
    fn powi(self, p: i32) -> Self;
    fn square(self) -> Self {
        self * self
    }
    fn recip(self) -> Self;
    fn log1p(self) -> Self;
    fn expm1(self) -> Self;
    fn abs(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    /// logistic sigmoid `1 / (1 + exp(-x))`
    fn sigmoid(self) -> Self;
    /// `log(sigmoid(x)) = -softplus(-x)`, numerically stable
    fn log_sigmoid(self) -> Self;
    /// `log(1 + exp(x))`, numerically stable
    fn softplus(self) -> Self;
    /// `log Gamma(x)`
    fn ln_gamma(self) -> Self;
    /// Fused `self * m + add` with a constant multiplier: one tape node.
    fn mul_add(self, m: f64, add: Self) -> Self {
        self * m + add
    }
    /// Fused `self * m + add` with differentiable `m`: one tape node.
    fn fma(self, m: Self, add: Self) -> Self {
        self * m + add
    }
    /// Dot product with constant weights as one node (see [`dot_const`]).
    fn dot_const(xs: &[Self], w: &[f64]) -> Self {
        let total: f64 = xs.iter().zip(w).map(|(x, w)| x.value() * w).sum();
        let mut b = Self::begin_node(xs.len());
        for (&x, &w) in xs.iter().zip(w) {
            b.add(x, w);
        }
        b.finish(total)
    }
    /// Row-major `n x k` constant matrix times `v` (see [`matvec_const`]).
    fn matvec_const(mat: &[f64], n: usize, k: usize, v: &[Self]) -> Vec<Self> {
        (0..n).map(|i| Self::dot_const(v, &mat[i * k..(i + 1) * k])).collect()
    }
    /// Elementwise maximum by value (gradient flows to the larger argument).
    fn max(self, other: Self) -> Self {
        if self.value() >= other.value() {
            self
        } else {
            other
        }
    }
    fn min(self, other: Self) -> Self {
        if self.value() <= other.value() {
            self
        } else {
            other
        }
    }
    fn is_finite(self) -> bool {
        self.value().is_finite()
    }
}

// ---------------------------------------------------------------- f64 --------

impl Real for f64 {
    type Node = F64Node;
    const DIFFERENTIABLE: bool = false;
    #[inline(always)]
    fn constant(v: f64) -> f64 {
        v
    }
    #[inline(always)]
    fn value(self) -> f64 {
        self
    }
    #[inline(always)]
    fn begin_node(_capacity_hint: usize) -> F64Node {
        F64Node
    }
    #[inline(always)]
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    #[inline(always)]
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    #[inline(always)]
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn powf(self, p: f64) -> f64 {
        f64::powf(self, p)
    }
    #[inline(always)]
    fn powi(self, p: i32) -> f64 {
        f64::powi(self, p)
    }
    #[inline(always)]
    fn recip(self) -> f64 {
        1.0 / self
    }
    #[inline(always)]
    fn log1p(self) -> f64 {
        f64::ln_1p(self)
    }
    #[inline(always)]
    fn expm1(self) -> f64 {
        f64::exp_m1(self)
    }
    #[inline(always)]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline(always)]
    fn sin(self) -> f64 {
        f64::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> f64 {
        f64::cos(self)
    }
    #[inline(always)]
    fn tanh(self) -> f64 {
        f64::tanh(self)
    }
    #[inline(always)]
    fn sigmoid(self) -> f64 {
        sigmoid_f64(self)
    }
    #[inline(always)]
    fn log_sigmoid(self) -> f64 {
        -softplus_f64(-self)
    }
    #[inline(always)]
    fn softplus(self) -> f64 {
        softplus_f64(self)
    }
    #[inline(always)]
    fn ln_gamma(self) -> f64 {
        ln_gamma(self)
    }
}

/// Numerically stable logistic sigmoid for `f64`.
#[inline]
pub fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable `log(1 + exp(x))` for `f64`.
#[inline]
pub fn softplus_f64(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `(softplus(x), sigmoid(x))` sharing a single `exp`.
#[inline]
pub fn softplus_sigmoid_f64(x: f64) -> (f64, f64) {
    let e = (-x.abs()).exp(); // in (0, 1]
    let l1pe = e.ln_1p();
    if x >= 0.0 {
        (x + l1pe, 1.0 / (1.0 + e))
    } else {
        (l1pe, e / (1.0 + e))
    }
}

/// Numerically stable `log(exp(a) + exp(b))`.
#[inline]
pub fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

// ---------------------------------------------------------------- Var --------

impl Real for Var {
    type Node = VarNode;
    const DIFFERENTIABLE: bool = true;

    #[inline]
    fn constant(v: f64) -> Var {
        // constants are leaves too; their adjoint is simply never read
        Var::new(v)
    }
    #[inline(always)]
    fn value(self) -> f64 {
        self.val
    }
    fn begin_node(capacity_hint: usize) -> VarNode {
        let (mut parents, mut weights) = TAPE.with(|t| {
            let mut t = t.borrow_mut();
            (
                std::mem::take(&mut t.scratch_parents),
                std::mem::take(&mut t.scratch_weights),
            )
        });
        parents.reserve(capacity_hint);
        weights.reserve(capacity_hint);
        VarNode { parents, weights }
    }

    #[inline]
    fn exp(self) -> Var {
        let e = self.val.exp();
        self.unary(e, e)
    }
    #[inline]
    fn ln(self) -> Var {
        self.unary(self.val.ln(), 1.0 / self.val)
    }
    #[inline]
    fn sqrt(self) -> Var {
        let s = self.val.sqrt();
        self.unary(s, 0.5 / s)
    }
    #[inline]
    fn powf(self, p: f64) -> Var {
        let v = self.val.powf(p);
        self.unary(v, p * self.val.powf(p - 1.0))
    }
    #[inline]
    fn powi(self, p: i32) -> Var {
        let v = self.val.powi(p);
        self.unary(v, p as f64 * self.val.powi(p - 1))
    }
    #[inline]
    fn recip(self) -> Var {
        let r = 1.0 / self.val;
        self.unary(r, -r * r)
    }
    #[inline]
    fn log1p(self) -> Var {
        self.unary(self.val.ln_1p(), 1.0 / (1.0 + self.val))
    }
    #[inline]
    fn expm1(self) -> Var {
        let v = self.val.exp_m1();
        self.unary(v, v + 1.0)
    }
    #[inline]
    fn abs(self) -> Var {
        self.unary(self.val.abs(), if self.val >= 0.0 { 1.0 } else { -1.0 })
    }
    #[inline]
    fn sin(self) -> Var {
        self.unary(self.val.sin(), self.val.cos())
    }
    #[inline]
    fn cos(self) -> Var {
        self.unary(self.val.cos(), -self.val.sin())
    }
    #[inline]
    fn tanh(self) -> Var {
        let t = self.val.tanh();
        self.unary(t, 1.0 - t * t)
    }
    #[inline]
    fn sigmoid(self) -> Var {
        let s = sigmoid_f64(self.val);
        self.unary(s, s * (1.0 - s))
    }
    #[inline]
    fn log_sigmoid(self) -> Var {
        self.unary(-softplus_f64(-self.val), sigmoid_f64(-self.val))
    }
    #[inline]
    fn softplus(self) -> Var {
        self.unary(softplus_f64(self.val), sigmoid_f64(self.val))
    }
    #[inline]
    fn ln_gamma(self) -> Var {
        self.unary(ln_gamma(self.val), digamma(self.val))
    }
    #[inline]
    fn mul_add(self, m: f64, add: Var) -> Var {
        self.binary(add, self.val * m + add.val, m, 1.0)
    }
    #[inline]
    fn fma(self, m: Var, add: Var) -> Var {
        let val = self.val * m.val + add.val;
        let idx = TAPE.with(|t| {
            let mut t = t.borrow_mut();
            let idx = t.len() as u32;
            t.parents.extend_from_slice(&[self.idx, m.idx, add.idx]);
            t.weights.extend_from_slice(&[m.val, self.val, 1.0]);
            let end = t.parents.len() as u32;
            t.starts.push(end);
            idx
        });
        Var { val, idx }
    }
    /// Single tape borrow, parents and weights pushed straight onto the tape.
    fn dot_const(xs: &[Var], w: &[f64]) -> Var {
        debug_assert_eq!(xs.len(), w.len());
        let total: f64 = xs.iter().zip(w).map(|(x, w)| x.val * w).sum();
        let idx = TAPE.with(|t| {
            let mut t = t.borrow_mut();
            let idx = t.len() as u32;
            t.parents.extend(xs.iter().map(|x| x.idx));
            t.weights.extend_from_slice(w);
            let end = t.parents.len() as u32;
            t.starts.push(end);
            idx
        });
        Var { val: total, idx }
    }
    /// All `n` output nodes are pushed under one tape borrow.
    fn matvec_const(mat: &[f64], n: usize, k: usize, v: &[Var]) -> Vec<Var> {
        debug_assert_eq!(mat.len(), n * k);
        debug_assert_eq!(v.len(), k);
        let mut out = Vec::with_capacity(n);
        TAPE.with(|t| {
            let mut t = t.borrow_mut();
            t.parents.reserve(n * k);
            t.weights.reserve(n * k);
            t.starts.reserve(n);
            for i in 0..n {
                let row = &mat[i * k..(i + 1) * k];
                let mut total = 0.0;
                for (x, w) in v.iter().zip(row) {
                    total += x.val * w;
                }
                let idx = t.len() as u32;
                t.parents.extend(v.iter().map(|x| x.idx));
                t.weights.extend_from_slice(row);
                let end = t.parents.len() as u32;
                t.starts.push(end);
                out.push(Var { val: total, idx });
            }
        });
        out
    }
}

impl Add for Var {
    type Output = Var;
    #[inline]
    fn add(self, o: Var) -> Var {
        self.binary(o, self.val + o.val, 1.0, 1.0)
    }
}
impl Sub for Var {
    type Output = Var;
    #[inline]
    fn sub(self, o: Var) -> Var {
        self.binary(o, self.val - o.val, 1.0, -1.0)
    }
}
impl Mul for Var {
    type Output = Var;
    #[inline]
    fn mul(self, o: Var) -> Var {
        self.binary(o, self.val * o.val, o.val, self.val)
    }
}
impl Div for Var {
    type Output = Var;
    #[inline]
    fn div(self, o: Var) -> Var {
        let r = 1.0 / o.val;
        self.binary(o, self.val * r, r, -self.val * r * r)
    }
}
impl Neg for Var {
    type Output = Var;
    #[inline]
    fn neg(self) -> Var {
        self.unary(-self.val, -1.0)
    }
}
impl Add<f64> for Var {
    type Output = Var;
    #[inline]
    fn add(self, o: f64) -> Var {
        self.unary(self.val + o, 1.0)
    }
}
impl Sub<f64> for Var {
    type Output = Var;
    #[inline]
    fn sub(self, o: f64) -> Var {
        self.unary(self.val - o, 1.0)
    }
}
impl Mul<f64> for Var {
    type Output = Var;
    #[inline]
    fn mul(self, o: f64) -> Var {
        self.unary(self.val * o, o)
    }
}
impl Div<f64> for Var {
    type Output = Var;
    #[inline]
    fn div(self, o: f64) -> Var {
        self.unary(self.val / o, 1.0 / o)
    }
}
impl Add<Var> for f64 {
    type Output = Var;
    #[inline]
    fn add(self, o: Var) -> Var {
        o + self
    }
}
impl Sub<Var> for f64 {
    type Output = Var;
    #[inline]
    fn sub(self, o: Var) -> Var {
        o.unary(self - o.val, -1.0)
    }
}
impl Mul<Var> for f64 {
    type Output = Var;
    #[inline]
    fn mul(self, o: Var) -> Var {
        o * self
    }
}
impl Div<Var> for f64 {
    type Output = Var;
    #[inline]
    fn div(self, o: Var) -> Var {
        let r = 1.0 / o.val;
        o.unary(self * r, -self * r * r)
    }
}
impl AddAssign for Var {
    #[inline]
    fn add_assign(&mut self, o: Var) {
        *self = *self + o;
    }
}
impl SubAssign for Var {
    #[inline]
    fn sub_assign(&mut self, o: Var) {
        *self = *self - o;
    }
}
impl MulAssign for Var {
    #[inline]
    fn mul_assign(&mut self, o: Var) {
        *self = *self * o;
    }
}
impl DivAssign for Var {
    #[inline]
    fn div_assign(&mut self, o: Var) {
        *self = *self / o;
    }
}
impl PartialEq for Var {
    fn eq(&self, o: &Var) -> bool {
        self.val == o.val
    }
}
impl PartialOrd for Var {
    fn partial_cmp(&self, o: &Var) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&o.val)
    }
}
impl Sum for Var {
    /// Summing many `Var`s creates a single n-ary node.
    fn sum<I: Iterator<Item = Var>>(iter: I) -> Var {
        let mut b = Var::begin_node(8);
        let mut total = 0.0;
        for v in iter {
            total += v.val;
            b.add(v, 1.0);
        }
        b.finish(total)
    }
}

// ---------------------------------------------------------- vector helpers ----

/// Sum of a slice as a single node.
pub fn sum<R: Real>(xs: &[R]) -> R {
    let total: f64 = xs.iter().map(|x| x.value()).sum();
    let mut b = R::begin_node(xs.len());
    for &x in xs {
        b.add(x, 1.0);
    }
    b.finish(total)
}

/// Dot product of a differentiable vector with constant weights, as one node.
pub fn dot_const<R: Real>(xs: &[R], w: &[f64]) -> R {
    debug_assert_eq!(xs.len(), w.len());
    R::dot_const(xs, w)
}

/// Dot product of two differentiable vectors, as one node.
pub fn dot<R: Real>(xs: &[R], ys: &[R]) -> R {
    debug_assert_eq!(xs.len(), ys.len());
    let total: f64 = xs.iter().zip(ys).map(|(x, y)| x.value() * y.value()).sum();
    let mut b = R::begin_node(2 * xs.len());
    for (&x, &y) in xs.iter().zip(ys) {
        b.add(x, y.value());
        b.add(y, x.value());
    }
    b.finish(total)
}

/// Dense matrix (row-major, `n x k`) times differentiable vector `v` (length `k`).
/// Each output is one node with `k` parents.
pub fn matvec_const<R: Real>(mat: &[f64], n: usize, k: usize, v: &[R]) -> Vec<R> {
    debug_assert_eq!(mat.len(), n * k);
    debug_assert_eq!(v.len(), k);
    R::matvec_const(mat, n, k, v)
}

/// `log(sum(exp(xs)))` as a single node.
pub fn logsumexp<R: Real>(xs: &[R]) -> R {
    let m = xs.iter().map(|x| x.value()).fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return R::constant(f64::NEG_INFINITY);
    }
    let s: f64 = xs.iter().map(|x| (x.value() - m).exp()).sum();
    let val = m + s.ln();
    let mut b = R::begin_node(xs.len());
    for &x in xs {
        b.add(x, (x.value() - val).exp());
    }
    b.finish(val)
}

/// Finite-difference gradient of `f` at `x` (central differences); for tests.
pub fn finite_diff<F: Fn(&[f64]) -> f64>(f: F, x: &[f64], eps: f64) -> Vec<f64> {
    let mut g = vec![0.0; x.len()];
    let mut xp = x.to_vec();
    for i in 0..x.len() {
        xp[i] = x[i] + eps;
        let fp = f(&xp);
        xp[i] = x[i] - eps;
        let fm = f(&xp);
        xp[i] = x[i];
        g[i] = (fp - fm) / (2.0 * eps);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_grad<F: Fn(&[Var]) -> Var, G: Fn(&[f64]) -> f64>(f: F, g: G, x: &[f64]) {
        reset();
        let xs = Var::leaves(x);
        let y = f(&xs);
        assert!((y.value() - g(x)).abs() < 1e-12, "value mismatch");
        let grad = gradient(y, &xs);
        let fd = finite_diff(&g, x, 1e-6);
        for (a, b) in grad.iter().zip(&fd) {
            assert!((a - b).abs() < 1e-5 * (1.0 + b.abs()), "grad {a} vs fd {b}");
        }
    }

    #[test]
    fn arithmetic_grads() {
        check_grad(
            |x| x[0] * x[1] + x[0] / x[1] - x[1] + 3.0 * x[0] - 2.0 / x[1],
            |x| x[0] * x[1] + x[0] / x[1] - x[1] + 3.0 * x[0] - 2.0 / x[1],
            &[1.3, -0.7],
        );
    }

    #[test]
    fn unary_grads() {
        check_grad(
            |x| {
                x[0].exp()
                    + x[1].ln()
                    + x[1].sqrt()
                    + x[0].powf(3.0)
                    + x[0].sin() * x[1].cos()
                    + x[0].tanh()
                    + x[1].log1p()
                    + x[0].expm1()
                    + x[0].sigmoid()
                    + x[0].softplus()
                    + x[0].log_sigmoid()
                    + x[1].ln_gamma()
                    + x[1].recip()
                    + (-x[0]).abs()
                    + x[1].powi(2)
            },
            |x| {
                x[0].exp()
                    + x[1].ln()
                    + x[1].sqrt()
                    + x[0].powf(3.0)
                    + x[0].sin() * x[1].cos()
                    + x[0].tanh()
                    + x[1].ln_1p()
                    + x[0].exp_m1()
                    + sigmoid_f64(x[0])
                    + softplus_f64(x[0])
                    + (-softplus_f64(-x[0]))
                    + ln_gamma(x[1])
                    + 1.0 / x[1]
                    + x[0].abs()
                    + x[1].powi(2)
            },
            &[0.4, 2.3],
        );
    }

    #[test]
    fn vector_helpers() {
        let w = [0.5, -1.5, 2.0];
        check_grad(
            |x| dot_const(x, &w) + sum(x) * logsumexp(x) + dot(x, x),
            |x| {
                let d: f64 = x.iter().zip(&w).map(|(a, b)| a * b).sum();
                let s: f64 = x.iter().sum();
                let lse = x.iter().map(|v| v.exp()).sum::<f64>().ln();
                let dd: f64 = x.iter().map(|v| v * v).sum();
                d + s * lse + dd
            },
            &[0.1, 0.2, -0.3],
        );
    }

    #[test]
    fn fused_ops_match_unfused() {
        let w = [0.5, -1.5, 2.0, 0.25, 3.0, -0.75];
        check_grad(
            |x| {
                let mv = matvec_const(&w, 2, 3, &x[0..3]);
                x[0].mul_add(1.5, x[1]) + x[1].fma(x[2], x[0]) + mv[0] * mv[1] + dot_const(&x[0..3], &w[3..6])
            },
            |x| {
                let mv = matvec_const(&w, 2, 3, &x[0..3]);
                x[0].mul_add(1.5, x[1]) + x[1].fma(x[2], x[0]) + mv[0] * mv[1] + dot_const(&x[0..3], &w[3..6])
            },
            &[0.3, -0.7, 1.1],
        );
        for &x in &[-40.0, -3.0, -0.1, 0.0, 0.5, 7.0, 40.0] {
            let (sp, sg) = softplus_sigmoid_f64(x);
            assert!((sp - softplus_f64(x)).abs() < 1e-14 * (1.0 + sp.abs()));
            assert!((sg - sigmoid_f64(x)).abs() < 1e-15);
        }
    }

    #[test]
    fn reuse_variable_many_times() {
        // x used in many places: adjoints must accumulate
        check_grad(
            |x| {
                let mut acc = x[0];
                for _ in 0..10 {
                    acc = acc * x[0] + x[0];
                }
                acc
            },
            |x| {
                let mut acc = x[0];
                for _ in 0..10 {
                    acc = acc * x[0] + x[0];
                }
                acc
            },
            &[0.9],
        );
    }

    #[test]
    fn tape_reset_reuses_memory() {
        reset();
        let x = Var::new(1.0);
        let _ = (x * x).exp();
        let n1 = tape_len();
        reset();
        assert_eq!(tape_len(), 0);
        let x = Var::new(1.0);
        let _ = (x * x).exp();
        assert_eq!(tape_len(), n1);
    }

    #[test]
    fn sum_iterator_is_one_node() {
        reset();
        let xs = Var::leaves(&[1.0, 2.0, 3.0]);
        let n0 = tape_len();
        let s: Var = xs.iter().copied().sum();
        assert_eq!(tape_len(), n0 + 1);
        assert_eq!(s.value(), 6.0);
        let g = gradient(s, &xs);
        assert_eq!(g, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn generic_function_matches_f64() {
        fn f<R: Real>(x: R) -> R {
            (x * 2.0 + 1.0).ln() * x.exp() - x.sqrt() / 3.0
        }
        reset();
        let v = Var::new(1.7);
        assert!((f(v).value() - f(1.7f64)).abs() < 1e-14);
    }
}
