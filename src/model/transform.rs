//! Bijections between constrained supports and unconstrained Euclidean space.
//!
//! Gradient-based kernels run in unconstrained space. Each latent site with a
//! constrained [`Support`] is mapped through the corresponding transform and the
//! log absolute determinant of the Jacobian is added to the log density.

use crate::ad::{NodeBuilder, Real};
use crate::dist::Support;

/// Number of unconstrained parameters needed to represent `constrained_len`
/// constrained values (with events of size `event_len`).
pub fn unconstrained_len(support: Support, constrained_len: usize, event_len: usize) -> usize {
    match support {
        Support::Simplex => {
            debug_assert!(constrained_len.is_multiple_of(event_len));
            (constrained_len / event_len) * (event_len - 1)
        }
        Support::CorrCholesky => {
            // event is a k x k matrix; k(k-1)/2 free parameters
            let k = corr_dim(event_len);
            (constrained_len / event_len) * (k * (k - 1) / 2)
        }
        _ => constrained_len,
    }
}

/// Side length `k` of a `k x k` event stored as `event_len = k*k` scalars.
pub fn corr_dim(event_len: usize) -> usize {
    let k = (event_len as f64).sqrt().round() as usize;
    assert_eq!(k * k, event_len, "CorrCholesky event must be a square matrix");
    k
}

/// Map unconstrained values `u` to constrained values (appended to `out`) and
/// return the summed log |det J| of the transform.
pub fn to_constrained<R: Real>(support: Support, event_len: usize, u: &[R], out: &mut Vec<R>) -> R {
    match support {
        Support::Real => {
            out.extend_from_slice(u);
            R::zero()
        }
        Support::Positive => positive(u, 0.0, out),
        Support::GreaterThan(lo) => positive(u, lo, out),
        Support::LessThan(hi) => {
            // x = hi - exp(u)
            let mut total = 0.0;
            let mut b = R::begin_node(u.len());
            for &ui in u {
                let e = ui.value().exp();
                out.push(R::node(hi - e, &[ui], &[-e]));
                total += ui.value();
                b.add(ui, 1.0);
            }
            b.finish(total)
        }
        Support::UnitInterval => interval(u, 0.0, 1.0, out),
        Support::Interval(lo, hi) => interval(u, lo, hi, out),
        Support::Simplex => simplex(u, event_len, out),
        Support::OrderedVector => ordered(u, event_len, out),
        Support::CorrCholesky => corr_cholesky(u, corr_dim(event_len), out),
        Support::Boolean | Support::NonNegativeInteger | Support::IntegerInterval(_, _) => {
            panic!("discrete latent sites are not supported by gradient-based inference; observe them or marginalize")
        }
    }
}

/// Inverse of [`to_constrained`]: map constrained values to unconstrained space.
pub fn to_unconstrained(support: Support, event_len: usize, x: &[f64]) -> Vec<f64> {
    match support {
        Support::Real => x.to_vec(),
        Support::Positive => x.iter().map(|v| v.ln()).collect(),
        Support::GreaterThan(lo) => x.iter().map(|v| (v - lo).ln()).collect(),
        Support::LessThan(hi) => x.iter().map(|v| (hi - v).ln()).collect(),
        Support::UnitInterval => x.iter().map(|v| logit(*v)).collect(),
        Support::Interval(lo, hi) => x.iter().map(|v| logit((v - lo) / (hi - lo))).collect(),
        Support::Simplex => {
            let k = event_len;
            let mut out = Vec::with_capacity(x.len() / k * (k - 1));
            for row in x.chunks(k) {
                let mut remaining = 1.0;
                for i in 0..k - 1 {
                    let z = row[i] / remaining;
                    out.push(logit(z) + ((k - 1 - i) as f64).ln());
                    remaining -= row[i];
                }
            }
            out
        }
        Support::OrderedVector => {
            let k = event_len;
            let mut out = Vec::with_capacity(x.len());
            for row in x.chunks(k) {
                out.push(row[0]);
                for i in 1..k {
                    out.push((row[i] - row[i - 1]).ln());
                }
            }
            out
        }
        Support::CorrCholesky => {
            let k = corr_dim(event_len);
            let mut out = Vec::with_capacity(x.len() / event_len * (k * (k - 1) / 2));
            for l in x.chunks(event_len) {
                for i in 1..k {
                    let mut sum_sq = 0.0;
                    for j in 0..i {
                        let z = l[i * k + j] / (1.0 - sum_sq).sqrt();
                        out.push(0.5 * ((1.0 + z) / (1.0 - z)).ln()); // atanh
                        sum_sq += l[i * k + j] * l[i * k + j];
                    }
                }
            }
            out
        }
        _ => panic!("discrete supports have no unconstraining transform"),
    }
}

/// Ordered vector: `x_0 = u_0`, `x_i = x_{i-1} + exp(u_i)`; log|J| = sum_{i>0} u_i.
fn ordered<R: Real>(u: &[R], k: usize, out: &mut Vec<R>) -> R {
    assert!(u.len().is_multiple_of(k), "ordered: bad unconstrained length");
    let mut total = 0.0;
    let mut b = R::begin_node(u.len());
    for row in u.chunks(k) {
        let mut prev = row[0];
        out.push(prev);
        for &ui in &row[1..] {
            prev += ui.exp();
            out.push(prev);
            total += ui.value();
            b.add(ui, 1.0);
        }
    }
    b.finish(total)
}

/// Lower Cholesky factor of a correlation matrix from `k(k-1)/2` unconstrained
/// values (Stan's `cholesky_corr_constrain`): `z = tanh(u)` are partial
/// correlations, filled row by row with signed stick breaking. Output is the
/// row-major `k x k` matrix.
/// log|J| = sum log(1 - z^2) + sum_{i, 1<=j<i} 0.5 log(1 - sum_{l<j} L_il^2).
fn corr_cholesky<R: Real>(u: &[R], k: usize, out: &mut Vec<R>) -> R {
    let m = k * (k - 1) / 2;
    assert!(m > 0, "corr_cholesky: dimension must be >= 2");
    assert!(
        u.len().is_multiple_of(m),
        "corr_cholesky: bad unconstrained length"
    );
    let mut logdet = R::zero();
    let n = u.len() / m;
    for e in 0..n {
        let uu = &u[e * m..(e + 1) * m];
        let start = out.len();
        out.resize(start + k * k, R::zero());
        let l = &mut out[start..];
        l[0] = R::one();
        let mut idx = 0;
        for i in 1..k {
            let mut sum_sq = R::zero();
            for j in 0..i {
                let z = uu[idx].tanh();
                idx += 1;
                logdet += (-z * z + 1.0).ln();
                let lij = if j == 0 {
                    z
                } else {
                    let rem = -sum_sq + 1.0;
                    logdet += rem.ln() * 0.5;
                    z * rem.sqrt()
                };
                l[i * k + j] = lij;
                sum_sq += lij * lij;
            }
            l[i * k + i] = (-sum_sq + 1.0).sqrt();
        }
    }
    logdet
}

#[inline]
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// `x = lo + exp(u)`, log|J| = sum(u).
fn positive<R: Real>(u: &[R], lo: f64, out: &mut Vec<R>) -> R {
    let mut total = 0.0;
    let mut b = R::begin_node(u.len());
    for &ui in u {
        let e = ui.value().exp();
        out.push(R::node(lo + e, &[ui], &[e]));
        total += ui.value();
        b.add(ui, 1.0);
    }
    b.finish(total)
}

/// `x = lo + (hi - lo) * sigmoid(u)`, log|J| = sum(log(hi-lo) + log_sigmoid(u) + log_sigmoid(-u)).
fn interval<R: Real>(u: &[R], lo: f64, hi: f64, out: &mut Vec<R>) -> R {
    let w = hi - lo;
    let lw = w.ln();
    let mut total = 0.0;
    let mut b = R::begin_node(u.len());
    for &ui in u {
        let uv = ui.value();
        let s = crate::ad::sigmoid_f64(uv);
        out.push(R::node(lo + w * s, &[ui], &[w * s * (1.0 - s)]));
        // log sigmoid(u) + log sigmoid(-u)
        total += lw - crate::ad::softplus_f64(-uv) - crate::ad::softplus_f64(uv);
        b.add(ui, 1.0 - 2.0 * s);
    }
    b.finish(total)
}

/// Stick-breaking transform from `R^{k-1}` to the `k`-simplex (Stan / numpyro
/// parameterization). log|J| = sum_i [log y_i - softplus(u_i - log(k-1-i))].
fn simplex<R: Real>(u: &[R], k: usize, out: &mut Vec<R>) -> R {
    assert!(u.len().is_multiple_of(k - 1), "simplex: bad unconstrained length");
    let mut logdet = R::zero();
    for row in u.chunks(k - 1) {
        let mut remaining = R::one();
        for (i, &ui) in row.iter().enumerate() {
            let offset = ((k - 1 - i) as f64).ln();
            let shifted = ui - offset;
            let z = shifted.sigmoid();
            let y = z * remaining;
            logdet = logdet + y.ln() - shifted.softplus();
            out.push(y);
            remaining -= y;
        }
        out.push(remaining);
    }
    logdet
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad::{self, Var};

    fn roundtrip(support: Support, event_len: usize, x: &[f64]) {
        let u = to_unconstrained(support, event_len, x);
        let mut back = Vec::new();
        to_constrained::<f64>(support, event_len, &u, &mut back);
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() < 1e-10, "{support:?}: {x:?} -> {u:?} -> {back:?}");
        }
    }

    #[test]
    fn roundtrips() {
        roundtrip(Support::Real, 1, &[-1.0, 2.5]);
        roundtrip(Support::Positive, 1, &[0.1, 7.0]);
        roundtrip(Support::GreaterThan(1.0), 1, &[1.1, 7.0]);
        roundtrip(Support::LessThan(1.0), 1, &[0.9, -7.0]);
        roundtrip(Support::UnitInterval, 1, &[0.2, 0.99]);
        roundtrip(Support::Interval(-2.0, 3.0), 1, &[-1.9, 2.5]);
        roundtrip(Support::Simplex, 3, &[0.2, 0.5, 0.3, 0.7, 0.1, 0.2]);
        roundtrip(Support::Simplex, 2, &[0.4, 0.6]);
    }

    /// Check log|det J| against a finite-difference Jacobian (square transforms
    /// only) and the AD gradient of logdet against finite differences.
    fn check_logdet(support: Support, event_len: usize, u: &[f64]) {
        let n = u.len();
        let mut out = Vec::new();
        let logdet = to_constrained::<f64>(support, event_len, u, &mut out);
        if out.len() == n {
            // numerical Jacobian
            let eps = 1e-6;
            let mut jac = vec![0.0; n * n];
            for j in 0..n {
                let mut up = u.to_vec();
                up[j] += eps;
                let mut xp = Vec::new();
                to_constrained::<f64>(support, event_len, &up, &mut xp);
                let mut um = u.to_vec();
                um[j] -= eps;
                let mut xm = Vec::new();
                to_constrained::<f64>(support, event_len, &um, &mut xm);
                for i in 0..n {
                    jac[i * n + j] = (xp[i] - xm[i]) / (2.0 * eps);
                }
            }
            let det = det(&jac, n).abs();
            assert!(
                (det.ln() - logdet).abs() < 1e-5,
                "{support:?}: logdet {logdet} vs numeric {}",
                det.ln()
            );
        }
        // gradient of logdet and of outputs via AD
        ad::reset();
        let uv = Var::leaves(u);
        let mut outv = Vec::new();
        let ld = to_constrained::<Var>(support, event_len, &uv, &mut outv);
        let g = ad::gradient(ld, &uv);
        let fd = ad::finite_diff(
            |u| {
                let mut o = Vec::new();
                to_constrained::<f64>(support, event_len, u, &mut o)
            },
            u,
            1e-6,
        );
        for (a, b) in g.iter().zip(&fd) {
            assert!((a - b).abs() < 1e-5, "{support:?} logdet grad {a} vs {b}");
        }
        for (i, o) in outv.iter().enumerate() {
            let g = ad::gradient(*o, &uv);
            let fd = ad::finite_diff(
                |u| {
                    let mut o = Vec::new();
                    to_constrained::<f64>(support, event_len, u, &mut o);
                    o[i]
                },
                u,
                1e-6,
            );
            for (a, b) in g.iter().zip(&fd) {
                assert!((a - b).abs() < 1e-5, "{support:?} output {i} grad {a} vs {b}");
            }
        }
    }

    fn det(m: &[f64], n: usize) -> f64 {
        // LU with partial pivoting
        let mut a = m.to_vec();
        let mut d = 1.0;
        for c in 0..n {
            let mut p = c;
            for r in c + 1..n {
                if a[r * n + c].abs() > a[p * n + c].abs() {
                    p = r;
                }
            }
            if a[p * n + c] == 0.0 {
                return 0.0;
            }
            if p != c {
                for j in 0..n {
                    a.swap(c * n + j, p * n + j);
                }
                d = -d;
            }
            d *= a[c * n + c];
            for r in c + 1..n {
                let f = a[r * n + c] / a[c * n + c];
                for j in c..n {
                    a[r * n + j] -= f * a[c * n + j];
                }
            }
        }
        d
    }

    #[test]
    fn logdets_and_grads() {
        check_logdet(Support::Positive, 1, &[0.3, -1.2]);
        check_logdet(Support::GreaterThan(2.0), 1, &[0.3, -1.2]);
        check_logdet(Support::LessThan(2.0), 1, &[0.3, -1.2]);
        check_logdet(Support::UnitInterval, 1, &[0.3, -1.2, 4.0]);
        check_logdet(Support::Interval(-1.0, 4.0), 1, &[0.3, -1.2]);
        // simplex is not square; only the gradient checks run, plus a
        // direct check of the closed form against the k-1 dimensional Jacobian
        check_logdet(Support::Simplex, 3, &[0.3, -1.2]);
        check_logdet(Support::Simplex, 4, &[0.3, -1.2, 0.8, 0.1, 0.2, -0.3]);
    }

    #[test]
    fn ordered_and_corr_cholesky_match_numpyro() {
        // OrderedTransform reference values
        let u = [0.5, -1.0, 0.3];
        let mut x = Vec::new();
        let ld = to_constrained::<f64>(Support::OrderedVector, 3, &u, &mut x);
        let expect = [0.5, 0.8678794411714423, 2.2177382487474455];
        for (a, b) in x.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((ld - (-0.7)).abs() < 1e-12);
        roundtrip(Support::OrderedVector, 3, &expect);
        check_logdet(Support::OrderedVector, 3, &u);

        // CorrCholeskyTransform reference values (k = 3 and k = 4)
        let y = [0.3, -0.7, 0.4];
        let mut l = Vec::new();
        let ld = to_constrained::<f64>(Support::CorrCholesky, 9, &y, &mut l);
        let expect = [
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
        for (a, b) in l.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-9, "{l:?}");
        }
        assert!((ld - (-0.9263991987030626)).abs() < 1e-9, "{ld}");
        roundtrip(Support::CorrCholesky, 9, &l);
        check_logdet(Support::CorrCholesky, 9, &y);
        let y4 = [0.3, -0.7, 0.4, 1.1, -0.2, 0.05];
        let mut l4 = Vec::new();
        let ld4 = to_constrained::<f64>(Support::CorrCholesky, 16, &y4, &mut l4);
        assert!((ld4 - (-3.0362469300853387)).abs() < 1e-9, "{ld4}");
        assert!((l4[15] - 0.586810293824).abs() < 1e-9);
        // rows have unit norm
        for i in 0..4 {
            let n: f64 = (0..4).map(|j| l4[i * 4 + j] * l4[i * 4 + j]).sum();
            assert!((n - 1.0).abs() < 1e-12);
        }
        roundtrip(Support::CorrCholesky, 16, &l4);
        check_logdet(Support::CorrCholesky, 16, &y4);
        assert_eq!(unconstrained_len(Support::CorrCholesky, 32, 16), 12);
    }

    #[test]
    fn simplex_logdet_matches_reduced_jacobian() {
        // Treat the map u -> (y_0, ..., y_{k-2}) as square in R^{k-1}
        let k = 4;
        let u = [0.3, -1.2, 0.8];
        let mut out = Vec::new();
        let logdet = to_constrained::<f64>(Support::Simplex, k, &u, &mut out);
        let n = k - 1;
        let eps = 1e-6;
        let mut jac = vec![0.0; n * n];
        for j in 0..n {
            let mut up = u.to_vec();
            up[j] += eps;
            let mut xp = Vec::new();
            to_constrained::<f64>(Support::Simplex, k, &up, &mut xp);
            let mut um = u.to_vec();
            um[j] -= eps;
            let mut xm = Vec::new();
            to_constrained::<f64>(Support::Simplex, k, &um, &mut xm);
            for i in 0..n {
                jac[i * n + j] = (xp[i] - xm[i]) / (2.0 * eps);
            }
        }
        let d = det(&jac, n).abs().ln();
        assert!((d - logdet).abs() < 1e-5, "{logdet} vs {d}");
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
