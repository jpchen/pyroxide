//! Multivariate distributions.

use rand::{Rng, RngCore};
use rand_distr::{Distribution as _, StandardNormal};

use super::{Distribution, IntoParam, Param, Support, Value, LN_2PI};
use crate::ad::{NodeBuilder, Real};
use crate::linalg;
use crate::special::{digamma, ln_gamma};

// --------------------------------------------------------------- Dirichlet ---

/// Dirichlet distribution over the `k`-simplex.
///
/// `concentration` has length `k` (shared across the batch) or `n * k`
/// (row-major, one row per draw; set `k` with [`with_k`](Self::with_k)).
#[derive(Clone, Copy, Debug)]
pub struct Dirichlet<'a, R> {
    pub concentration: Param<'a, R>,
    k: usize,
    n: usize,
}

impl<'a, R: Real> Dirichlet<'a, R> {
    pub fn new(concentration: impl IntoParam<'a, R>) -> Self {
        let concentration = concentration.into_param();
        assert!(concentration.len() >= 2, "Dirichlet needs at least 2 categories");
        Dirichlet {
            k: concentration.len(),
            n: 1,
            concentration,
        }
    }
    /// Interpret the concentration as an `n x k` matrix.
    pub fn with_k(mut self, k: usize) -> Self {
        let total = self.concentration.len();
        assert!(total.is_multiple_of(k));
        self.k = k;
        self.n = total / k;
        self
    }
    /// Number of iid draws sharing one concentration vector.
    pub fn expand(mut self, n: usize) -> Self {
        assert_eq!(self.concentration.len(), self.k);
        self.n = n;
        self
    }
    #[inline]
    fn row(&self, i: usize) -> usize {
        if self.concentration.len() == self.k {
            0
        } else {
            i * self.k
        }
    }
}

impl<'a, R: Real> Distribution<R> for Dirichlet<'a, R> {
    fn len(&self) -> usize {
        self.n * self.k
    }
    fn event_len(&self) -> usize {
        self.k
    }
    fn support(&self) -> Support {
        Support::Simplex
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        let a = self.concentration;
        let shared = a.len() == self.k;
        let mut shared_grad = vec![0.0; if shared { self.k } else { 0 }];
        let mut total = 0.0;
        let mut b = R::begin_node(self.n * self.k + a.len());
        for i in 0..self.n {
            let r = self.row(i);
            let xo = i * self.k;
            let asum: f64 = (0..self.k).map(|j| a.get(r + j)).sum();
            let mut lp = ln_gamma(asum);
            let mut valid = true;
            for j in 0..self.k {
                let aj = a.get(r + j);
                let xj = x.get(xo + j);
                if xj <= 0.0 {
                    valid = false;
                    break;
                }
                lp += -ln_gamma(aj) + (aj - 1.0) * xj.ln();
            }
            if !valid {
                total = f64::NEG_INFINITY;
                continue;
            }
            total += lp;
            if R::DIFFERENTIABLE {
                let dsum = digamma(asum);
                for j in 0..self.k {
                    let aj = a.get(r + j);
                    let xj = x.get(xo + j);
                    x.add_grad(&mut b, xo + j, (aj - 1.0) / xj);
                    let da = dsum - digamma(aj) + xj.ln();
                    if shared {
                        shared_grad[j] += da;
                    } else if let Param::V(v) = a {
                        b.add(v[r + j], da);
                    }
                }
            }
        }
        if R::DIFFERENTIABLE && shared {
            if let Param::V(v) = a {
                for j in 0..self.k {
                    b.add(v[j], shared_grad[j]);
                }
            }
        }
        b.finish(total)
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        for i in 0..self.n {
            let r = self.row(i);
            let row = &mut out[i * self.k..(i + 1) * self.k];
            let mut s = 0.0;
            for j in 0..self.k {
                let g = rand_distr::Gamma::new(self.concentration.get(r + j), 1.0)
                    .expect("Dirichlet: invalid concentration");
                row[j] = g.sample(rng);
                s += row[j];
            }
            for v in row.iter_mut() {
                *v /= s;
            }
        }
    }
}

// ------------------------------------------------------ MultivariateNormal ---

/// Multivariate normal distribution parameterized by a location vector and the
/// lower-triangular Cholesky factor of the covariance (row-major `k x k`).
///
/// Use [`from_covariance`](Self::from_covariance) for a constant covariance
/// matrix, or [`new`](Self::new) with a (possibly differentiable) `scale_tril`.
#[derive(Clone, Debug)]
pub struct MultivariateNormal<'a, R> {
    pub loc: Param<'a, R>,
    pub scale_tril: Param<'a, R>,
    owned_tril: Option<Vec<f64>>,
    k: usize,
    n: usize,
}

impl<'a, R: Real> MultivariateNormal<'a, R> {
    /// `scale_tril` is the lower Cholesky factor `L` with `cov = L L^T`, row-major.
    pub fn new(loc: impl IntoParam<'a, R>, scale_tril: impl IntoParam<'a, R>) -> Self {
        let (loc, scale_tril) = (loc.into_param(), scale_tril.into_param());
        let k = loc.len();
        assert_eq!(scale_tril.len(), k * k, "scale_tril must be k x k");
        MultivariateNormal {
            loc,
            scale_tril,
            owned_tril: None,
            k,
            n: 1,
        }
    }

    /// Construct from a constant (row-major) covariance matrix.
    pub fn from_covariance(loc: impl IntoParam<'a, R>, cov: &[f64]) -> Self {
        let loc = loc.into_param();
        let k = loc.len();
        assert_eq!(cov.len(), k * k, "covariance must be k x k");
        let tril = linalg::cholesky(cov, k).expect("covariance is not positive definite");
        MultivariateNormal {
            loc,
            scale_tril: Param::CV(&[]), // replaced by owned_tril at use time
            owned_tril: Some(tril),
            k,
            n: 1,
        }
    }

    /// Number of iid draws.
    pub fn expand(mut self, n: usize) -> Self {
        self.n = n;
        self
    }

    fn tril_get(&self, i: usize) -> f64 {
        match &self.owned_tril {
            Some(t) => t[i],
            None => self.scale_tril.get(i),
        }
    }

    fn tril_vec(&self) -> Vec<f64> {
        (0..self.k * self.k).map(|i| self.tril_get(i)).collect()
    }
}

impl<'a, R: Real> Distribution<R> for MultivariateNormal<'a, R> {
    fn len(&self) -> usize {
        self.n * self.k
    }
    fn event_len(&self) -> usize {
        self.k
    }
    fn support(&self) -> Support {
        Support::Real
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        let k = self.k;
        let l = self.tril_vec();
        let mut log_det = 0.0;
        for j in 0..k {
            log_det += l[j * k + j].ln();
        }
        let mut total = 0.0;
        let mut b = R::begin_node(self.n * k + k + k * k);
        let mut d = vec![0.0; k];
        let mut y = vec![0.0; k];
        let mut dx = vec![0.0; k];
        let mut loc_grad = vec![0.0; k];
        // gradient wrt L accumulates over the batch: L^{-T} (y y^T - I) (lower part)
        let mut l_grad = vec![0.0; if R::DIFFERENTIABLE { k * k } else { 0 }];
        let mut w = vec![0.0; k * k];
        for i in 0..self.n {
            for j in 0..k {
                d[j] = x.get(i * k + j) - self.loc.get(j);
            }
            // y = L^{-1} d
            linalg::solve_lower(&l, k, &d, &mut y);
            let quad: f64 = y.iter().map(|v| v * v).sum();
            total += -0.5 * quad - log_det - 0.5 * k as f64 * LN_2PI;
            if R::DIFFERENTIABLE {
                // dx = -L^{-T} y
                linalg::solve_lower_transpose(&l, k, &y, &mut dx);
                for j in 0..k {
                    dx[j] = -dx[j];
                    x.add_grad(&mut b, i * k + j, dx[j]);
                    loc_grad[j] -= dx[j];
                }
                if !matches!(self.scale_tril, Param::C(_) | Param::CV(_)) || self.owned_tril.is_none() {
                    // M = y y^T - I ; solve L^T W = M column by column
                    for r in 0..k {
                        for c in 0..k {
                            w[r * k + c] = y[r] * y[c] - if r == c { 1.0 } else { 0.0 };
                        }
                    }
                    let mut col = vec![0.0; k];
                    let mut sol = vec![0.0; k];
                    for c in 0..k {
                        for r in 0..k {
                            col[r] = w[r * k + c];
                        }
                        linalg::solve_lower_transpose(&l, k, &col, &mut sol);
                        for r in c..k {
                            l_grad[r * k + c] += sol[r];
                        }
                    }
                }
            }
        }
        if R::DIFFERENTIABLE {
            match self.loc {
                Param::S(r) => b.add(r, loc_grad.iter().sum()),
                Param::V(v) => {
                    for j in 0..k {
                        b.add(v[j], loc_grad[j]);
                    }
                }
                _ => {}
            }
            if let Param::V(v) = self.scale_tril {
                for r in 0..k {
                    for c in 0..=r {
                        b.add(v[r * k + c], l_grad[r * k + c]);
                    }
                }
            }
        }
        b.finish(total)
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        let k = self.k;
        let l = self.tril_vec();
        let mut eps = vec![0.0; k];
        for i in 0..self.n {
            for e in eps.iter_mut() {
                *e = rng.sample(StandardNormal);
            }
            let row = &mut out[i * k..(i + 1) * k];
            for r in 0..k {
                let mut acc = self.loc.get(r);
                for c in 0..=r {
                    acc += l[r * k + c] * eps[c];
                }
                row[r] = acc;
            }
        }
    }
}

// ------------------------------------------------------------ LKJCholesky ---

/// LKJ distribution over lower Cholesky factors of `k x k` correlation
/// matrices (row-major), with density proportional to `det(L L^T)^(eta - 1)`.
/// `eta = 1` is uniform over correlation matrices; `eta > 1` favors identity.
///
/// The concentration is a constant (not differentiated). Sampling uses the
/// C-vine construction of Lewandowski, Kurowicka & Joe (2009).
#[derive(Clone, Copy, Debug)]
pub struct LKJCholesky {
    pub dim: usize,
    pub concentration: f64,
    n: usize,
}

impl LKJCholesky {
    pub fn new(dim: usize, concentration: f64) -> Self {
        assert!(dim >= 2, "LKJCholesky needs dim >= 2");
        assert!(concentration > 0.0, "LKJCholesky concentration must be positive");
        LKJCholesky {
            dim,
            concentration,
            n: 1,
        }
    }
    /// Number of iid draws.
    pub fn expand(mut self, n: usize) -> Self {
        self.n = n;
        self
    }
    /// Log normalizing constant (numpyro's `LKJCholesky.log_prob`).
    fn log_normalizer(&self) -> f64 {
        let dm1 = (self.dim - 1) as f64;
        let alpha = self.concentration + 0.5 * dm1;
        let denominator = ln_gamma(alpha) * dm1;
        // multigammaln(alpha - 0.5, dim - 1)
        let a = alpha - 0.5;
        let p = self.dim - 1;
        let mut numerator = (p * (p - 1)) as f64 / 4.0 * std::f64::consts::PI.ln();
        for j in 0..p {
            numerator += ln_gamma(a - 0.5 * j as f64);
        }
        let pi_constant = 0.5 * dm1 * std::f64::consts::PI.ln();
        pi_constant + numerator - denominator
    }
}

impl<R: Real> Distribution<R> for LKJCholesky {
    fn len(&self) -> usize {
        self.n * self.dim * self.dim
    }
    fn event_len(&self) -> usize {
        self.dim * self.dim
    }
    fn support(&self) -> Support {
        Support::CorrCholesky
    }
    fn log_prob_value(&self, x: Value<'_, R>) -> R {
        let k = self.dim;
        let norm = self.log_normalizer();
        let mut total = 0.0;
        let mut b = R::begin_node(self.n * (k - 1));
        for e in 0..self.n {
            for i in 1..k {
                // order_i = 2 eta - 3 + k - i  (0-indexed diagonal i)
                let order = 2.0 * self.concentration - 3.0 + (k - i) as f64;
                let lii = x.get(e * k * k + i * k + i);
                if lii <= 0.0 {
                    total = f64::NEG_INFINITY;
                    continue;
                }
                total += order * lii.ln();
                if R::DIFFERENTIABLE {
                    x.add_grad(&mut b, e * k * k + i * k + i, order / lii);
                }
            }
            total -= norm;
        }
        b.finish(total)
    }
    fn sample(&self, rng: &mut dyn RngCore, out: &mut [f64]) {
        let k = self.dim;
        let marginal = self.concentration + 0.5 * (k - 2) as f64;
        for e in 0..self.n {
            let l = &mut out[e * k * k..(e + 1) * k * k];
            l.iter_mut().for_each(|v| *v = 0.0);
            l[0] = 1.0;
            for i in 1..k {
                let mut sum_sq = 0.0;
                for j in 0..i {
                    // partial correlation at tree level j: 2 Beta(b, b) - 1, b = eta + (k - 2 - j)/2
                    let bconc = marginal - 0.5 * j as f64;
                    let beta = rand_distr::Beta::new(bconc, bconc).expect("LKJ: invalid concentration");
                    let z = 2.0 * beta.sample(rng) - 1.0;
                    let lij = if j == 0 { z } else { z * (1.0 - sum_sq).sqrt() };
                    l[i * k + j] = lij;
                    sum_sq += lij * lij;
                }
                l[i * k + i] = (1.0 - sum_sq).max(0.0).sqrt();
            }
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
    fn dirichlet_values_and_grads() {
        let lp: f64 = Dirichlet::new(&[1.5, 2.0, 0.7][..]).log_prob_data(&[0.2, 0.5, 0.3]);
        assert!((lp - 0.7717963326).abs() < 1e-8, "{lp}");
        // gradient wrt concentration and x (x kept on the simplex by fd? no:
        // finite differences perturb off the simplex, which is fine for the
        // density formula as written).
        check_grad(
            |p| Dirichlet::new(&p[0..3]).log_prob(&p[3..6]),
            |p| Dirichlet::new(&p[0..3]).log_prob(&p[3..6]),
            &[1.5, 2.0, 0.7, 0.2, 0.5, 0.3],
            1e-6,
        );
        // batched, shared concentration
        let xs = [0.2, 0.5, 0.3, 0.6, 0.3, 0.1];
        check_grad(
            |p| Dirichlet::new(&p[0..3]).expand(2).log_prob_data(&xs),
            |p| Dirichlet::new(&p[0..3]).expand(2).log_prob_data(&xs),
            &[1.5, 2.0, 0.7],
            1e-6,
        );
    }

    #[test]
    fn mvn_values_and_grads() {
        let cov = [2.0, 0.3, 0.3, 1.0];
        let d = MultivariateNormal::from_covariance(&[0.5, -1.0][..], &cov);
        let lp: f64 = d.log_prob_data(&[1.0, 0.0]);
        assert!((lp - (-2.6718998916)).abs() < 1e-8, "{lp}");
        let tril = linalg::cholesky(&cov, 2).unwrap();
        // grads wrt loc and x, constant tril
        check_grad(
            |p| MultivariateNormal::new(&p[0..2], &tril[..]).log_prob(&p[2..4]),
            |p| MultivariateNormal::new(&p[0..2], &tril[..]).log_prob(&p[2..4]),
            &[0.5, -1.0, 1.0, 0.0],
            1e-6,
        );
        // grads wrt tril (lower entries only matter)
        let x = [1.0, 0.0, -0.3, 0.7, 2.0, 1.0];
        check_grad(
            |p| {
                MultivariateNormal::new(&[0.5, -1.0][..], &p[0..4])
                    .expand(3)
                    .log_prob_data(&x)
            },
            |p| {
                MultivariateNormal::new(&[0.5, -1.0][..], &p[0..4])
                    .expand(3)
                    .log_prob_data(&x)
            },
            &[tril[0], 0.0, tril[2], tril[3]],
            1e-6,
        );
    }

    #[test]
    fn lkj_cholesky_values_grads_and_sampling() {
        // reference values from numpyro.distributions.LKJCholesky
        let l3 = [
            1.0,
            0.0,
            0.0,
            0.291312612452,
            0.9566279119,
            0.0,
            -0.604367777117,
            0.302707412747,
            0.736958487467,
        ];
        for (eta, expect) in [
            (2.0, -1.358953077397629),
            (0.7, -1.9188054592391546),
            (1.0, -1.6406533610647958),
        ] {
            let lp: f64 = LKJCholesky::new(3, eta).log_prob_data(&l3);
            assert!((lp - expect).abs() < 1e-8, "eta {eta}: {lp} vs {expect}");
        }
        let l4 = {
            let mut v = vec![0.0; 16];
            let rows = [
                [1.0, 0.0, 0.0, 0.0],
                [0.291312612452, 0.9566279119, 0.0, 0.0],
                [-0.604367777117, 0.302707412747, 0.736958487467, 0.0],
                [0.800499021761, -0.118293752127, 0.029352741434, 0.586810293824],
            ];
            for i in 0..4 {
                v[i * 4..(i + 1) * 4].copy_from_slice(&rows[i]);
            }
            v
        };
        let lp: f64 = LKJCholesky::new(4, 1.5).log_prob_data(&l4);
        assert!((lp - (-2.795172178538924)).abs() < 1e-8, "{lp}");
        // gradient wrt the matrix entries (only the diagonal matters)
        check_grad(
            |p| LKJCholesky::new(3, 2.0).log_prob(&p[0..9]),
            |p| LKJCholesky::new(3, 2.0).log_prob(&p[0..9]),
            &l3,
            1e-6,
        );
        // sampling: off-diagonal correlations have Var = 1 / (2 eta + k - 1)
        let mut r = Xoshiro256PlusPlus::seed_from_u64(5);
        for (k, eta) in [(2usize, 2.0), (3, 1.0), (4, 0.8)] {
            let d: &dyn Distribution<f64> = &LKJCholesky::new(k, eta);
            let n = 40_000;
            let mut m2 = 0.0;
            for _ in 0..n {
                let l = d.sample_vec(&mut r);
                // correlation between the last two variables: row k-1 dot row k-2
                let rho: f64 = (0..k).map(|j| l[(k - 1) * k + j] * l[(k - 2) * k + j]).sum();
                m2 += rho * rho / n as f64;
                for i in 0..k {
                    let nrm: f64 = (0..k).map(|j| l[i * k + j] * l[i * k + j]).sum();
                    assert!((nrm - 1.0).abs() < 1e-10);
                }
            }
            let a = eta + 0.5 * (k - 2) as f64;
            let expect = 1.0 / (2.0 * a + 1.0);
            assert!(
                (m2 - expect).abs() < 0.01,
                "k={k} eta={eta}: E[rho^2] {m2} vs {expect}"
            );
        }
    }

    #[test]
    fn multivariate_sampling() {
        let mut r = Xoshiro256PlusPlus::seed_from_u64(3);
        let n = 100_000;
        let d = Dirichlet::<f64>::new(&[1.5, 2.0, 0.7][..]);
        let mut m = [0.0; 3];
        for _ in 0..n {
            let s = d.sample_vec(&mut r);
            assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            for j in 0..3 {
                m[j] += s[j] / n as f64;
            }
        }
        for j in 0..3 {
            assert!((m[j] - [1.5, 2.0, 0.7][j] / 4.2).abs() < 0.01);
        }
        let cov = [2.0, 0.3, 0.3, 1.0];
        let d = MultivariateNormal::<f64>::from_covariance(&[0.5, -1.0][..], &cov);
        let xs: Vec<Vec<f64>> = (0..n).map(|_| d.sample_vec(&mut r)).collect();
        let x0: Vec<f64> = xs.iter().map(|v| v[0]).collect();
        let x1: Vec<f64> = xs.iter().map(|v| v[1]).collect();
        let (m0, v0) = mean_var(&x0);
        let (m1, v1) = mean_var(&x1);
        assert!((m0 - 0.5).abs() < 0.02 && (m1 + 1.0).abs() < 0.02);
        assert!((v0 - 2.0).abs() < 0.05 && (v1 - 1.0).abs() < 0.03);
        let c: f64 = x0.iter().zip(&x1).map(|(a, b)| (a - m0) * (b - m1)).sum::<f64>() / (n - 1) as f64;
        assert!((c - 0.3).abs() < 0.03);
    }
}
