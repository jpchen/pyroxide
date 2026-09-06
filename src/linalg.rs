//! Small dense linear-algebra helpers (row-major `f64` matrices).
//!
//! Sizes in MCMC mass-matrix adaptation are modest (tens to a few hundred
//! dimensions), so simple triple loops are used; no BLAS dependency.

/// Cholesky factorization `A = L L^T` of a symmetric positive definite matrix
/// (row-major, `n x n`). Returns the lower-triangular `L` (upper part zero), or
/// `None` if the matrix is not positive definite.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n);
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        if !(s > 0.0) || !s.is_finite() {
            return None;
        }
        let d = s.sqrt();
        l[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    Some(l)
}

/// Solve `L y = b` for lower-triangular `L` (forward substitution).
#[inline]
pub fn solve_lower(l: &[f64], n: usize, b: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        let row = &l[i * n..i * n + i];
        for (k, lik) in row.iter().enumerate() {
            s -= lik * y[k];
        }
        y[i] = s / l[i * n + i];
    }
}

/// Solve `L^T x = b` for lower-triangular `L` (back substitution).
#[inline]
pub fn solve_lower_transpose(l: &[f64], n: usize, b: &[f64], x: &mut [f64]) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
}

/// Dense symmetric matrix-vector product `y = A x` (row-major).
#[inline]
pub fn matvec(a: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let row = &a[i * n..(i + 1) * n];
        let mut s = 0.0;
        for (aij, xj) in row.iter().zip(x) {
            s += aij * xj;
        }
        y[i] = s;
    }
}

/// `y = L x` for lower-triangular `L`.
#[inline]
pub fn tril_matvec(l: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let row = &l[i * n..i * n + i + 1];
        let mut s = 0.0;
        for (lij, xj) in row.iter().zip(x) {
            s += lij * xj;
        }
        y[i] = s;
    }
}

/// Inverse of a symmetric positive definite matrix via Cholesky.
pub fn spd_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = cholesky(a, n)?;
    let mut inv = vec![0.0; n * n];
    let mut e = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut x = vec![0.0; n];
    for j in 0..n {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[j] = 1.0;
        solve_lower(&l, n, &e, &mut y);
        solve_lower_transpose(&l, n, &y, &mut x);
        for i in 0..n {
            inv[i * n + j] = x[i];
        }
    }
    Some(inv)
}

/// Log determinant of an SPD matrix via Cholesky.
pub fn spd_logdet(a: &[f64], n: usize) -> Option<f64> {
    let l = cholesky(a, n)?;
    Some(2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_and_solves() {
        let a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let l = cholesky(&a, 3).unwrap();
        // reconstruct
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += l[i * 3 + k] * l[j * 3 + k];
                }
                assert!((s - a[i * 3 + j]).abs() < 1e-12);
            }
        }
        let b = [1.0, -2.0, 0.5];
        let mut y = [0.0; 3];
        let mut x = [0.0; 3];
        solve_lower(&l, 3, &b, &mut y);
        solve_lower_transpose(&l, 3, &y, &mut x);
        let mut ax = [0.0; 3];
        matvec(&a, 3, &x, &mut ax);
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-12);
        }
        let inv = spd_inverse(&a, 3).unwrap();
        let mut prod = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    prod[i * 3 + j] += a[i * 3 + k] * inv[k * 3 + j];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i * 3 + j] - e).abs() < 1e-12);
            }
        }
        assert!((spd_logdet(&a, 3).unwrap() - 44.6f64.ln()).abs() < 1e-10);
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
    }
}
