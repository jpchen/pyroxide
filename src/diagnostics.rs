//! Convergence diagnostics: autocorrelation, effective sample size, Gelman–Rubin
//! R-hat (plain and split), highest posterior density intervals, and the
//! summary table. A port of `numpyro.diagnostics`.
//!
//! All functions take chains as `&[Vec<f64>]` (one vector of draws per chain).

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::infer::Array;

/// Smallest `n >= target` whose only prime factors are 2, 3 and 5
/// (like `scipy.fftpack.next_fast_len`).
pub fn next_fast_len(target: usize) -> usize {
    if target <= 2 {
        return target;
    }
    let mut n = target;
    loop {
        let mut m = n;
        while m % 2 == 0 {
            m /= 2;
        }
        while m % 3 == 0 {
            m /= 3;
        }
        while m % 5 == 0 {
            m /= 5;
        }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

/// Autocorrelation of `x` at lags `0..n` via FFT. With `bias = true` the
/// biased estimator (divides by `n`) is used, as in Stan; the unbiased one
/// divides by `n - k`.
pub fn autocorrelation(x: &[f64], bias: bool) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let m = next_fast_len(n);
    let m2 = 2 * m;
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut buf: Vec<Complex<f64>> = (0..m2)
        .map(|i| Complex::new(if i < n { x[i] - mean } else { 0.0 }, 0.0))
        .collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(m2);
    let ifft = planner.plan_fft_inverse(m2);
    fft.process(&mut buf);
    for v in buf.iter_mut() {
        *v = Complex::new(v.norm_sqr(), 0.0);
    }
    ifft.process(&mut buf);
    let mut ac: Vec<f64> = buf[..n].iter().map(|c| c.re / m2 as f64).collect();
    if !bias {
        for (k, v) in ac.iter_mut().enumerate() {
            *v /= (n - k) as f64;
        }
    }
    let a0 = ac[0];
    for v in ac.iter_mut() {
        *v /= a0;
    }
    ac
}

/// Autocovariance of `x` (autocorrelation times the population variance).
pub fn autocovariance(x: &[f64], bias: bool) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    autocorrelation(x, bias).into_iter().map(|a| a * var).collect()
}

fn chain_variance_stats(chains: &[Vec<f64>]) -> (f64, f64) {
    let c = chains.len();
    let n = chains[0].len() as f64;
    let mut var_within = 0.0;
    let mut means = Vec::with_capacity(c);
    for ch in chains {
        let m = ch.iter().sum::<f64>() / n;
        let v = ch.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
        var_within += v / c as f64;
        means.push(m);
    }
    let mut var_estimator = var_within * (n - 1.0) / n;
    if c > 1 {
        let gm = means.iter().sum::<f64>() / c as f64;
        let var_between = means.iter().map(|m| (m - gm) * (m - gm)).sum::<f64>() / (c as f64 - 1.0);
        var_estimator += var_between;
    } else {
        var_within = var_estimator;
    }
    (var_within, var_estimator)
}

/// Gelman–Rubin potential scale reduction factor. Requires >= 2 chains of
/// >= 2 draws.
pub fn gelman_rubin(chains: &[Vec<f64>]) -> f64 {
    assert!(chains.len() >= 2 && chains[0].len() >= 2);
    let (w, v) = chain_variance_stats(chains);
    (v / w).sqrt()
}

/// Split R-hat: each chain is split in half and the halves treated as chains.
pub fn split_gelman_rubin(chains: &[Vec<f64>]) -> f64 {
    let n = chains[0].len();
    assert!(n >= 4, "split_gelman_rubin needs at least 4 draws");
    let half = n / 2;
    let mut split: Vec<Vec<f64>> = Vec::with_capacity(2 * chains.len());
    for ch in chains {
        split.push(ch[..half].to_vec());
    }
    for ch in chains {
        split.push(ch[n - half..].to_vec());
    }
    gelman_rubin(&split)
}

/// Effective sample size across chains (Geyer's initial monotone sequence
/// estimator, following Stan / numpyro).
pub fn effective_sample_size(chains: &[Vec<f64>], bias: bool) -> f64 {
    let c = chains.len();
    let n = chains[0].len();
    assert!(n >= 2);
    // mean autocovariance across chains at each lag
    let mut gamma = vec![0.0; n];
    for ch in chains {
        for (g, a) in gamma.iter_mut().zip(autocovariance(ch, bias)) {
            *g += a / c as f64;
        }
    }
    let (var_within, var_estimator) = chain_variance_stats(chains);
    let mut rho: Vec<f64> = gamma
        .iter()
        .map(|g| 1.0 - (var_within - g) / var_estimator)
        .collect();
    rho[0] = 1.0;
    // initial positive sequence: pair up lags
    let mut big_rho: Vec<f64> = (0..n / 2).map(|k| rho[2 * k] + rho[2 * k + 1]).collect();
    if big_rho.is_empty() {
        return (c * n) as f64;
    }
    // initial monotone sequence
    let mut running_min = f64::INFINITY;
    for (k, r) in big_rho.iter_mut().enumerate() {
        if k == 0 {
            continue;
        }
        let clipped = r.max(0.0);
        running_min = running_min.min(clipped);
        *r = running_min;
    }
    let tau = -1.0 + 2.0 * big_rho.iter().sum::<f64>();
    (c * n) as f64 / tau
}

/// Highest posterior density interval containing probability mass `prob`.
pub fn hpdi(x: &[f64], prob: f64) -> (f64, f64) {
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mass = sorted.len();
    let idx_len = (prob * mass as f64) as usize;
    if idx_len == 0 || idx_len >= mass {
        return (sorted[0], sorted[mass - 1]);
    }
    let mut best = 0;
    let mut best_width = f64::INFINITY;
    for i in 0..(mass - idx_len) {
        let w = sorted[i + idx_len] - sorted[i];
        if w < best_width {
            best_width = w;
            best = i;
        }
    }
    (sorted[best], sorted[best + idx_len])
}

fn median(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Summary statistics for one element of a site.
#[derive(Clone, Debug)]
pub struct ElementSummary {
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub hpdi_low: f64,
    pub hpdi_high: f64,
    pub n_eff: f64,
    pub r_hat: f64,
}

/// Summary of all elements of a site.
#[derive(Clone, Debug)]
pub struct SiteSummary {
    pub name: String,
    pub elements: Vec<ElementSummary>,
}

/// Summarize a site array (chains x draws x len).
pub fn summarize(name: &str, arr: &Array, prob: f64) -> SiteSummary {
    let elements = (0..arr.len)
        .map(|j| {
            let per_chain = arr.column_per_chain(j);
            let flat: Vec<f64> = per_chain.iter().flatten().copied().collect();
            let n = flat.len() as f64;
            let mean = flat.iter().sum::<f64>() / n;
            let std = if n > 1.0 {
                (flat.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
            } else {
                0.0
            };
            let (lo, hi) = hpdi(&flat, prob);
            let n_eff = if arr.draws >= 2 { effective_sample_size(&per_chain, true) } else { f64::NAN };
            let r_hat = if arr.draws >= 4 { split_gelman_rubin(&per_chain) } else { f64::NAN };
            ElementSummary {
                mean,
                std,
                median: median(&flat),
                hpdi_low: lo,
                hpdi_high: hi,
                n_eff,
                r_hat,
            }
        })
        .collect();
    SiteSummary {
        name: name.to_string(),
        elements,
    }
}

/// Render summaries as a numpyro-style table.
pub fn format_summary(summaries: &[SiteSummary], prob: f64) -> String {
    let mut rows: Vec<(String, &ElementSummary)> = Vec::new();
    for s in summaries {
        if s.elements.len() == 1 {
            rows.push((s.name.clone(), &s.elements[0]));
        } else {
            for (i, e) in s.elements.iter().enumerate() {
                rows.push((format!("{}[{}]", s.name, i), e));
            }
        }
    }
    let width = rows.iter().map(|(n, _)| n.len()).max().unwrap_or(4).max(10);
    let lo = format!("{:.1}%", 50.0 * (1.0 - prob));
    let hi = format!("{:.1}%", 50.0 * (1.0 + prob));
    let mut out = String::new();
    out.push('\n');
    out.push_str(&format!(
        "{:>width$} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}\n",
        "", "mean", "std", "median", lo, hi, "n_eff", "r_hat",
        width = width
    ));
    for (name, e) in rows {
        out.push_str(&format!(
            "{:>width$} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}\n",
            name, e.mean, e.std, e.median, e.hpdi_low, e.hpdi_high, e.n_eff, e.r_hat,
            width = width
        ));
    }
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_fast_len_matches_scipy() {
        let cases = [(433, 450), (124, 125), (25, 25), (300, 300), (1, 1), (3, 3), (7, 8)];
        for (t, e) in cases {
            assert_eq!(next_fast_len(t), e, "target {t}");
        }
    }

    #[test]
    fn autocorrelation_matches_numpyro() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let actual = autocorrelation(&x, false);
        let expected = [1.0, 0.78, 0.52, 0.21, -0.13, -0.52, -0.94, -1.4, -1.91, -2.45];
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() < 0.01, "{actual:?}");
        }
        let biased = autocorrelation(&x, true);
        for (k, (b, e)) in biased.iter().zip(&expected).enumerate() {
            let e = e * (10 - k) as f64 / 10.0;
            assert!((b - e).abs() < 0.01);
        }
        let cov = autocovariance(&x, false);
        let expected_cov = [8.25, 6.42, 4.25, 1.75, -1.08, -4.25, -7.75, -11.58, -15.75, -20.25];
        for (a, e) in cov.iter().zip(&expected_cov) {
            assert!((a - e).abs() < 0.01, "{cov:?}");
        }
        // biased estimator of white noise has tiny tails; unbiased does not
        let mut rng = 12345u64;
        let noise: Vec<f64> = (0..20000)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        let ac = autocorrelation(&noise, true);
        assert!(ac[ac.len() - 100..].iter().all(|v| v.abs() < 0.01));
        let ac = autocorrelation(&noise, false);
        assert!(ac[ac.len() - 100..].iter().any(|v| v.abs() > 0.1));
    }

    #[test]
    fn gelman_rubin_matches_numpyro() {
        let x = vec![
            (0..10).map(|i| i as f64).collect::<Vec<_>>(),
            (0..10).map(|i| i as f64 + 1.0).collect::<Vec<_>>(),
        ];
        assert!((gelman_rubin(&x) - 0.98).abs() < 0.01);
        // split r-hat agrees with r-hat on the split chains
        let y = vec![
            vec![0.3, -1.2, 0.8, 2.0, -0.4, 0.9, 1.1, -0.7, 0.2, 0.5],
            vec![1.3, 0.2, -0.8, 1.0, 0.4, -0.9, 2.1, 0.7, -0.2, 0.0],
        ];
        let split = vec![
            y[0][..5].to_vec(),
            y[1][..5].to_vec(),
            y[0][5..].to_vec(),
            y[1][5..].to_vec(),
        ];
        assert!((split_gelman_rubin(&y) - gelman_rubin(&split)).abs() < 1e-12);
    }

    #[test]
    fn effective_sample_size_matches_numpyro() {
        // x = arange(1000).reshape(100, 10): 100 chains of 10 draws
        let chains: Vec<Vec<f64>> = (0..100)
            .map(|c| (0..10).map(|d| (c * 10 + d) as f64).collect())
            .collect();
        let ess = effective_sample_size(&chains, false);
        assert!((ess - 52.64).abs() < 0.01, "{ess}");
    }

    #[test]
    fn hpdi_of_symmetric_and_skewed() {
        // uniform grid: 80% interval spans 80% of the range
        let x: Vec<f64> = (0..10001).map(|i| i as f64 / 10000.0).collect();
        let (lo, hi) = hpdi(&x, 0.8);
        assert!((hi - lo - 0.8).abs() < 1e-3);
        // exponential quantiles: the HPDI starts at 0
        let e: Vec<f64> = (1..20000).map(|i| -(1.0 - i as f64 / 20000.0).ln()).collect();
        let (lo, hi) = hpdi(&e, 0.2);
        assert!(lo.abs() < 0.01 && (hi - 0.223).abs() < 0.01, "{lo} {hi}");
    }

    #[test]
    fn iid_normal_ess_is_about_n() {
        let mut rng = 987654321u64;
        let mut normal = || {
            // Box-Muller on a xorshift
            let mut u = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (u1, u2) = (u(), u());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let chains: Vec<Vec<f64>> = (0..4).map(|_| (0..2000).map(|_| normal()).collect()).collect();
        let ess = effective_sample_size(&chains, true);
        assert!(ess > 6000.0 && ess < 10000.0, "{ess}");
        let rhat = split_gelman_rubin(&chains);
        assert!((rhat - 1.0).abs() < 0.02, "{rhat}");
    }
}
