//! Special functions needed by the distributions and their gradients.

/// `log |Gamma(x)|`.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Digamma function `d/dx log Gamma(x)`.
///
/// Uses the reflection formula for negative arguments, the recurrence to shift
/// the argument above 10, and the asymptotic expansion.
pub fn digamma(mut x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let mut result = 0.0;
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN; // poles
        }
        // reflection: psi(1 - x) - psi(x) = pi cot(pi x)
        result -= std::f64::consts::PI / (std::f64::consts::PI * x).tan();
        x = 1.0 - x;
    }
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += x.ln() - 0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2
                    * (1.0 / 120.0
                        - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 * (1.0 / 132.0)))));
    result
}

/// Trigamma function `d^2/dx^2 log Gamma(x)`.
pub fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        let s = (std::f64::consts::PI * x).sin();
        result += std::f64::consts::PI * std::f64::consts::PI / (s * s);
        x = 1.0 - x;
        // trigamma(1-x) + trigamma(x) = pi^2 / sin^2(pi x)
        return result - trigamma(x);
    }
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += inv + 0.5 * inv2
        + inv * inv2
            * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 * (1.0 / 30.0))));
    result
}

/// `log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)`.
#[inline]
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// `log (n choose k)` for real-valued `n`, `k`.
#[inline]
pub fn ln_binomial(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

/// Error function.
#[inline]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Complementary error function.
#[inline]
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// Standard normal CDF.
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal log CDF, stable in the lower tail.
pub fn normal_log_cdf(x: f64) -> f64 {
    if x < -5.0 {
        // asymptotic expansion: log(phi(x)/(-x)) + log(1 - 1/x^2 + 3/x^4 ...)
        let x2 = x * x;
        -0.5 * x2 - (-x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            + (1.0 - 1.0 / x2 + 3.0 / (x2 * x2) - 15.0 / (x2 * x2 * x2)).ln()
    } else {
        normal_cdf(x).ln()
    }
}

/// Inverse of the standard normal CDF (Acklam's algorithm with one Newton refinement).
pub fn normal_quantile(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    let x = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    // one step of Newton refinement
    let e = normal_cdf(x) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digamma_known_values() {
        // psi(1) = -gamma
        assert!((digamma(1.0) + 0.5772156649015329).abs() < 1e-12);
        // psi(0.5) = -gamma - 2 ln 2
        assert!((digamma(0.5) + 0.5772156649015329 + 2.0 * 2f64.ln()).abs() < 1e-12);
        // psi(10) = H_9 - gamma
        let h9: f64 = (1..=9).map(|k| 1.0 / k as f64).sum();
        assert!((digamma(10.0) - (h9 - 0.5772156649015329)).abs() < 1e-12);
        // negative argument via reflection
        assert!((digamma(-0.5) - 0.03648997397857652).abs() < 1e-10);
    }

    #[test]
    fn digamma_is_derivative_of_lgamma() {
        for &x in &[0.3, 1.0, 2.5, 7.0, 40.0] {
            let h = 1e-5;
            let fd = (ln_gamma(x + h) - ln_gamma(x - h)) / (2.0 * h);
            assert!((digamma(x) - fd).abs() < 1e-7, "x={x}");
        }
    }

    #[test]
    fn trigamma_is_derivative_of_digamma() {
        for &x in &[0.3, 1.0, 2.5, 7.0, 40.0] {
            let h = 1e-5;
            let fd = (digamma(x + h) - digamma(x - h)) / (2.0 * h);
            assert!((trigamma(x) - fd).abs() < 1e-6, "x={x}");
        }
        assert!((trigamma(1.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-10);
    }

    #[test]
    fn normal_cdf_quantile_roundtrip() {
        for &p in &[1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-6] {
            let x = normal_quantile(p);
            assert!((normal_cdf(x) - p).abs() < 1e-12, "p={p}");
        }
        assert!((normal_log_cdf(-10.0) - (-53.23128515051247)).abs() < 1e-6);
        assert!((normal_log_cdf(1.0) - normal_cdf(1.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn ln_binomial_matches_exact() {
        // 10 choose 3 = 120
        assert!((ln_binomial(10.0, 3.0) - 120f64.ln()).abs() < 1e-10);
    }
}
