//! Discrete distributions. These can be observed (and sampled from), but not
//! used as latent sites in gradient-based kernels.

use rand::{Rng, RngCore};
use rand_distr::Distribution as _;

use super::{broadcast_len, univariate, Distribution, IntoParam, Param, Support, Value};
use crate::ad::{sigmoid_f64, softplus_sigmoid_f64, NodeBuilder, Real};
use crate::special::{ln_binomial, ln_gamma};

/// Either a probability or a logit parameterization.
#[derive(Clone, Copy, Debug)]
pub enum ProbParam<'a, R> {
    Probs(Param<'a, R>),
    Logits(Param<'a, R>),
}

impl<'a, R: Real> ProbParam<'a, R> {
    fn param(&self) -> Param<'a, R> {
        match self {
            ProbParam::Probs(p) | ProbParam::Logits(p) => *p,
        }
    }
    /// Probability of element `i`.
    fn prob(&self, i: usize) -> f64 {
        match self {
            ProbParam::Probs(p) => p.get(i),
            ProbParam::Logits(l) => sigmoid_f64(l.get(i)),
        }
    }
}

// --------------------------------------------------------------- Bernoulli ---

/// Bernoulli distribution, parameterized by probabilities or logits.
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli<'a, R> {
    pub p: ProbParam<'a, R>,
    n: usize,
}

impl<'a, R: Real> Bernoulli<'a, R> {
    pub fn new(probs: impl IntoParam<'a, R>) -> Self {
        let p = probs.into_param();
        let n = broadcast_len(&[p.len()], None);
        Bernoulli {
            p: ProbParam::Probs(p),
            n,
        }
    }
    pub fn logits(logits: impl IntoParam<'a, R>) -> Self {
        let p = logits.into_param();
        let n = broadcast_len(&[p.len()], None);
        Bernoulli {
            p: ProbParam::Logits(p),
            n,
        }
    }
    pub fn expand(mut self, n: usize) -> Self {
        self.n = broadcast_len(&[self.p.param().len()], Some(n));
        self
    }
}

impl<'a, R: Real> Distribution<R> for Bernoulli<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::Boolean
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        match self.p {
            ProbParam::Probs(p) => univariate([p], x, self.n, |[p], x| {
                if x != 0.0 && x != 1.0 {
                    return (f64::NEG_INFINITY, [0.0], 0.0);
                }
                if x == 1.0 {
                    (p.ln(), [1.0 / p], 0.0)
                } else {
                    ((-p).ln_1p(), [-1.0 / (1.0 - p)], 0.0)
                }
            }),
            ProbParam::Logits(l) => bernoulli_logits(l, x, self.n),
        }
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let u: f64 = rng.random();
            *o = if u < self.p.prob(i) { 1.0 } else { 0.0 };
        }
    }
}

/// Bernoulli(logits) over a plate.
fn bernoulli_logits<R: Real>(l: Param<'_, R>, x: Value<'_, R>, n: usize) -> R {
    let mut total = 0.0;
    let mut b = R::begin_node(n + 1);
    let mut g = super::ParamGrad::new(l);
    for i in 0..n {
        let (li, xi) = (l.get(i), x.get(i));
        if xi != 0.0 && xi != 1.0 {
            total = f64::NEG_INFINITY;
            continue;
        }
        // log p(x) = x*l - softplus(l); softplus and sigmoid share one exp
        let (sp, sg) = softplus_sigmoid_f64(li);
        total += xi * li - sp;
        if R::DIFFERENTIABLE {
            g.add(&mut b, i, xi - sg);
        }
    }
    g.finish(&mut b);
    b.finish(total)
}

// ---------------------------------------------------------------- Binomial ---

/// Binomial distribution with `total_count` trials, parameterized by
/// probabilities or logits. `total_count` is treated as data (not differentiated).
#[derive(Clone, Copy, Debug)]
pub struct Binomial<'a, R> {
    pub total_count: Param<'a, R>,
    pub p: ProbParam<'a, R>,
    n: usize,
}

impl<'a, R: Real> Binomial<'a, R> {
    pub fn new(total_count: impl IntoParam<'a, R>, probs: impl IntoParam<'a, R>) -> Self {
        let (total_count, p) = (total_count.into_param(), probs.into_param());
        let n = broadcast_len(&[total_count.len(), p.len()], None);
        Binomial {
            total_count,
            p: ProbParam::Probs(p),
            n,
        }
    }
    pub fn logits(total_count: impl IntoParam<'a, R>, logits: impl IntoParam<'a, R>) -> Self {
        let (total_count, p) = (total_count.into_param(), logits.into_param());
        let n = broadcast_len(&[total_count.len(), p.len()], None);
        Binomial {
            total_count,
            p: ProbParam::Logits(p),
            n,
        }
    }
    pub fn expand(mut self, n: usize) -> Self {
        self.n = broadcast_len(&[self.total_count.len(), self.p.param().len()], Some(n));
        self
    }
}

impl<'a, R: Real> Distribution<R> for Binomial<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        let max = (0..self.n).map(|i| self.total_count.get(i)).fold(0.0, f64::max);
        Support::IntegerInterval(0, max as i64)
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        // total_count enters as a constant: wrap it so `univariate` never
        // differentiates it.
        let tc = self.total_count;
        match self.p {
            ProbParam::Probs(p) => binomial_probs(tc, p, x, self.n),
            ProbParam::Logits(l) => binomial_logits(tc, l, x, self.n),
        }
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let n = self.total_count.get(i).round() as u64;
            let d = rand_distr::Binomial::new(n, self.p.prob(i)).expect("Binomial: invalid p");
            *o = d.sample(rng) as f64;
        }
    }
}

fn binomial_probs<R: Real>(tc: Param<'_, R>, p: Param<'_, R>, x: Value<'_, R>, n: usize) -> R {
    let mut total = 0.0;
    let mut b = R::begin_node(n + 1);
    let mut g = super::ParamGrad::new(p);
    for i in 0..n {
        let (nn, pp, xx) = (tc.get(i), p.get(i), x.get(i));
        if xx < 0.0 || xx > nn {
            total = f64::NEG_INFINITY;
            continue;
        }
        let lp = ln_binomial(nn, xx) + xx * pp.ln() + (nn - xx) * (-pp).ln_1p();
        total += lp;
        if R::DIFFERENTIABLE {
            g.add(&mut b, i, xx / pp - (nn - xx) / (1.0 - pp));
        }
    }
    g.finish(&mut b);
    b.finish(total)
}

fn binomial_logits<R: Real>(tc: Param<'_, R>, l: Param<'_, R>, x: Value<'_, R>, n: usize) -> R {
    let mut total = 0.0;
    let mut b = R::begin_node(n + 1);
    let mut g = super::ParamGrad::new(l);
    for i in 0..n {
        let (nn, ll, xx) = (tc.get(i), l.get(i), x.get(i));
        if xx < 0.0 || xx > nn {
            total = f64::NEG_INFINITY;
            continue;
        }
        // x*l - n*softplus(l) + log C(n, x)
        let (sp, sg) = softplus_sigmoid_f64(ll);
        let lp = ln_binomial(nn, xx) + xx * ll - nn * sp;
        total += lp;
        if R::DIFFERENTIABLE {
            g.add(&mut b, i, xx - nn * sg);
        }
    }
    g.finish(&mut b);
    b.finish(total)
}

// ----------------------------------------------------------------- Poisson ---

/// Poisson distribution with the given rate.
#[derive(Clone, Copy, Debug)]
pub struct Poisson<'a, R> {
    pub rate: Param<'a, R>,
    n: usize,
}

impl<'a, R: Real> Poisson<'a, R> {
    pub fn new(rate: impl IntoParam<'a, R>) -> Self {
        let rate = rate.into_param();
        let n = broadcast_len(&[rate.len()], None);
        Poisson { rate, n }
    }
    pub fn expand(mut self, n: usize) -> Self {
        self.n = broadcast_len(&[self.rate.len()], Some(n));
        self
    }
}

impl<'a, R: Real> Distribution<R> for Poisson<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::NonNegativeInteger
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        univariate([self.rate], x, self.n, |[r], x| {
            if x < 0.0 {
                return (f64::NEG_INFINITY, [0.0], 0.0);
            }
            (x * r.ln() - r - ln_gamma(x + 1.0), [x / r - 1.0], 0.0)
        })
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let d = rand_distr::Poisson::new(self.rate.get(i)).expect("Poisson: invalid rate");
            let v: f64 = d.sample(rng);
            *o = v;
        }
    }
}

// ------------------------------------------------------------- Categorical ---

/// Categorical distribution over `{0, ..., k-1}`.
///
/// The parameter is either a single vector of length `k` (shared across the
/// batch, set the batch with [`expand`](Self::expand)) or a row-major `n x k`
/// matrix (one row per observation, set `k` with [`with_k`](Self::with_k)).
#[derive(Clone, Copy, Debug)]
pub struct Categorical<'a, R> {
    pub p: ProbParam<'a, R>,
    k: usize,
    n: usize,
}

impl<'a, R: Real> Categorical<'a, R> {
    pub fn new(probs: impl IntoParam<'a, R>) -> Self {
        let p = probs.into_param();
        Categorical {
            p: ProbParam::Probs(p),
            k: p.len(),
            n: 1,
        }
    }
    pub fn logits(logits: impl IntoParam<'a, R>) -> Self {
        let p = logits.into_param();
        Categorical {
            p: ProbParam::Logits(p),
            k: p.len(),
            n: 1,
        }
    }
    /// Interpret the parameter as an `n x k` matrix with the given `k`.
    pub fn with_k(mut self, k: usize) -> Self {
        let total = self.p.param().len();
        assert!(
            total.is_multiple_of(k),
            "Categorical: parameter length {total} not divisible by k={k}"
        );
        self.k = k;
        self.n = total / k;
        self
    }
    /// Number of iid draws sharing one parameter vector.
    pub fn expand(mut self, n: usize) -> Self {
        assert_eq!(
            self.p.param().len(),
            self.k,
            "expand requires a shared parameter vector"
        );
        self.n = n;
        self
    }
    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.k
    }
    #[inline]
    fn row(&self, i: usize) -> usize {
        if self.p.param().len() == self.k {
            0
        } else {
            i * self.k
        }
    }
    fn probs_row(&self, i: usize, out: &mut [f64]) {
        let p = self.p.param();
        let r = self.row(i);
        match self.p {
            ProbParam::Probs(_) => {
                for j in 0..self.k {
                    out[j] = p.get(r + j);
                }
            }
            ProbParam::Logits(_) => {
                let m = (0..self.k)
                    .map(|j| p.get(r + j))
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut s = 0.0;
                for j in 0..self.k {
                    out[j] = (p.get(r + j) - m).exp();
                    s += out[j];
                }
                for o in out.iter_mut() {
                    *o /= s;
                }
            }
        }
    }
}

impl<'a, R: Real> Distribution<R> for Categorical<'a, R> {
    fn len(&self) -> usize {
        self.n
    }
    fn support(&self) -> Support {
        Support::IntegerInterval(0, self.k as i64 - 1)
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        let p = self.p.param();
        let shared = p.len() == self.k;
        let mut total = 0.0;
        let mut b = R::begin_node(if shared { self.k } else { self.n * self.k });
        // accumulate gradient for a shared parameter vector
        let mut shared_grad = vec![0.0; if shared { self.k } else { 0 }];
        for i in 0..self.n {
            let xi = x.get(i);
            if xi < 0.0 || xi >= self.k as f64 || xi.fract() != 0.0 {
                total = f64::NEG_INFINITY;
                continue;
            }
            let xi = xi as usize;
            let r = self.row(i);
            match self.p {
                ProbParam::Probs(_) => {
                    let pj = p.get(r + xi);
                    total += pj.ln();
                    if R::DIFFERENTIABLE {
                        if shared {
                            shared_grad[xi] += 1.0 / pj;
                        } else if let Param::V(v) = p {
                            b.add(v[r + xi], 1.0 / pj);
                        }
                    }
                }
                ProbParam::Logits(_) => {
                    let m = (0..self.k)
                        .map(|j| p.get(r + j))
                        .fold(f64::NEG_INFINITY, f64::max);
                    let s: f64 = (0..self.k).map(|j| (p.get(r + j) - m).exp()).sum();
                    let lse = m + s.ln();
                    total += p.get(r + xi) - lse;
                    if R::DIFFERENTIABLE {
                        for j in 0..self.k {
                            let d = (if j == xi { 1.0 } else { 0.0 }) - (p.get(r + j) - lse).exp();
                            if shared {
                                shared_grad[j] += d;
                            } else if let Param::V(v) = p {
                                b.add(v[r + j], d);
                            }
                        }
                    }
                }
            }
        }
        if R::DIFFERENTIABLE && shared {
            if let Param::V(v) = p {
                for j in 0..self.k {
                    b.add(v[j], shared_grad[j]);
                }
            }
        }
        b.finish(total)
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        let mut probs = vec![0.0; self.k];
        for (i, o) in out.iter_mut().enumerate() {
            self.probs_row(i, &mut probs);
            let u: f64 = rng.random();
            let mut acc = 0.0;
            let mut idx = self.k - 1;
            for (j, p) in probs.iter().enumerate() {
                acc += p;
                if u < acc {
                    idx = j;
                    break;
                }
            }
            *o = idx as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn bernoulli_values_and_grads() {
        let lp1: f64 = Bernoulli::new(0.3).log_prob_data(&[1.0]);
        let lp0: f64 = Bernoulli::new(0.3).log_prob_data(&[0.0]);
        assert!((lp1 - 0.3f64.ln()).abs() < 1e-12);
        assert!((lp0 - 0.7f64.ln()).abs() < 1e-12);
        let l = (0.3f64 / 0.7).ln();
        let lp1l: f64 = Bernoulli::logits(l).log_prob_data(&[1.0]);
        assert!((lp1l - lp1).abs() < 1e-12);
        let data = [1.0, 0.0, 0.0, 1.0, 1.0];
        check_grad(
            |p| Bernoulli::new(p[0]).expand(5).log_prob_data(&data),
            |p| Bernoulli::new(p[0]).expand(5).log_prob_data(&data),
            &[0.3],
            1e-6,
        );
        check_grad(
            |p| Bernoulli::logits(&p[0..5]).log_prob_data(&data),
            |p| Bernoulli::logits(&p[0..5]).log_prob_data(&data),
            &[0.3, -1.0, 2.0, 0.1, -0.4],
            1e-6,
        );
    }

    #[test]
    fn binomial_values_and_grads() {
        let lp: f64 = Binomial::new(10.0, 0.3).log_prob_data(&[4.0]);
        assert!((lp - (-1.6088333502)).abs() < 1e-8, "{lp}");
        let l = (0.3f64 / 0.7).ln();
        let lpl: f64 = Binomial::logits(10.0, l).log_prob_data(&[4.0]);
        assert!((lpl - lp).abs() < 1e-10);
        let counts = [10.0, 20.0, 5.0];
        let data = [4.0, 11.0, 0.0];
        check_grad(
            |p| Binomial::new(&counts[..], &p[0..3]).log_prob_data(&data),
            |p| Binomial::new(&counts[..], &p[0..3]).log_prob_data(&data),
            &[0.3, 0.6, 0.1],
            1e-6,
        );
        check_grad(
            |p| Binomial::logits(&counts[..], p[0]).log_prob_data(&data),
            |p| Binomial::logits(&counts[..], p[0]).log_prob_data(&data),
            &[0.4],
            1e-6,
        );
        // numerically stable for huge counts (pyro issue #1706)
        let lp: f64 = Binomial::logits(5_000_000.0, (3849.0f64 / 5e6).ln() - (1.0 - 3849.0 / 5e6).ln())
            .log_prob_data(&[3849.0]);
        assert!(lp.is_finite());
    }

    #[test]
    fn poisson_values_and_grads() {
        let lp: f64 = Poisson::new(2.5).log_prob_data(&[4.0]);
        assert!((lp - (-2.0128909029)).abs() < 1e-8, "{lp}");
        let data = [4.0, 0.0, 7.0];
        check_grad(
            |p| Poisson::new(&p[0..3]).log_prob_data(&data),
            |p| Poisson::new(&p[0..3]).log_prob_data(&data),
            &[2.5, 0.4, 6.0],
            1e-6,
        );
    }

    #[test]
    fn categorical_values_and_grads() {
        let lp: f64 = Categorical::new(&[0.2, 0.5, 0.3][..]).log_prob_data(&[1.0]);
        assert!((lp - 0.5f64.ln()).abs() < 1e-12);
        let data = [1.0, 2.0, 0.0, 1.0];
        check_grad(
            |p| Categorical::new(&p[0..3]).expand(4).log_prob_data(&data),
            |p| Categorical::new(&p[0..3]).expand(4).log_prob_data(&data),
            &[0.2, 0.5, 0.3],
            1e-6,
        );
        check_grad(
            |p| Categorical::logits(&p[0..3]).expand(4).log_prob_data(&data),
            |p| Categorical::logits(&p[0..3]).expand(4).log_prob_data(&data),
            &[0.2, -0.5, 1.3],
            1e-6,
        );
        // per-observation logits (2 x 3 matrix)
        let data2 = [1.0, 2.0];
        check_grad(
            |p| Categorical::logits(&p[0..6]).with_k(3).log_prob_data(&data2),
            |p| Categorical::logits(&p[0..6]).with_k(3).log_prob_data(&data2),
            &[0.2, -0.5, 1.3, 0.0, 0.7, -2.0],
            1e-6,
        );
    }

    #[test]
    fn discrete_sampling_moments() {
        let mut r = Xoshiro256PlusPlus::seed_from_u64(1);
        let n = 100_000;
        let xs: Vec<f64> = (0..n)
            .map(|_| Bernoulli::<f64>::new(0.3).sample_vec(&mut r)[0])
            .collect();
        assert!((mean_var(&xs).0 - 0.3).abs() < 0.01);
        let xs: Vec<f64> = (0..n)
            .map(|_| Binomial::<f64>::new(10.0, 0.3).sample_vec(&mut r)[0])
            .collect();
        let (m, v) = mean_var(&xs);
        assert!((m - 3.0).abs() < 0.03 && (v - 2.1).abs() < 0.1);
        let xs: Vec<f64> = (0..n)
            .map(|_| Poisson::<f64>::new(2.5).sample_vec(&mut r)[0])
            .collect();
        let (m, v) = mean_var(&xs);
        assert!((m - 2.5).abs() < 0.03 && (v - 2.5).abs() < 0.1);
        let xs: Vec<f64> = (0..n)
            .map(|_| Categorical::<f64>::new(&[0.2, 0.5, 0.3][..]).sample_vec(&mut r)[0])
            .collect();
        let frac1 = xs.iter().filter(|&&x| x == 1.0).count() as f64 / n as f64;
        assert!((frac1 - 0.5).abs() < 0.01);
    }
}
