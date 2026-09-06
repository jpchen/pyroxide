//! Warmup adaptation: dual averaging for the step size, Welford estimation of
//! the (inverse) mass matrix, and the Stan-style windowed schedule.
//!
//! Ports the corresponding pieces of `numpyro.infer.hmc_util`.

use rand::RngCore;
use rand_distr::StandardNormal;

use crate::linalg;

/// Dual averaging (Nesterov 2009) as adapted for MCMC by Hoffman & Gelman
/// (2014): drives a statistic (here `target_accept - accept_prob`) to zero by
/// adapting a parameter (here `log step_size`).
#[derive(Clone, Debug)]
pub struct DualAveraging {
    pub t0: f64,
    pub kappa: f64,
    pub gamma: f64,
    x_t: f64,
    x_avg: f64,
    g_avg: f64,
    t: f64,
    prox_center: f64,
}

impl DualAveraging {
    /// Default hyperparameters from the NUTS paper: `t0 = 10`, `kappa = 0.75`,
    /// `gamma = 0.05`.
    pub fn new(prox_center: f64) -> Self {
        Self::with_params(prox_center, 10.0, 0.75, 0.05)
    }

    pub fn with_params(prox_center: f64, t0: f64, kappa: f64, gamma: f64) -> Self {
        DualAveraging {
            t0,
            kappa,
            gamma,
            x_t: 0.0,
            x_avg: 0.0,
            g_avg: 0.0,
            t: 0.0,
            prox_center,
        }
    }

    /// Incorporate the statistic `g` observed at the current iterate.
    pub fn update(&mut self, g: f64) {
        self.t += 1.0;
        let t = self.t;
        self.g_avg = (1.0 - 1.0 / (t + self.t0)) * self.g_avg + g / (t + self.t0);
        self.x_t = self.prox_center - t.sqrt() / self.gamma * self.g_avg;
        let w = t.powf(-self.kappa);
        self.x_avg = (1.0 - w) * self.x_avg + w * self.x_t;
    }

    /// Current iterate.
    pub fn x(&self) -> f64 {
        self.x_t
    }

    /// Weighted average of the iterates (used after warmup).
    pub fn x_avg(&self) -> f64 {
        self.x_avg
    }
}

/// Welford's online (co)variance estimator.
#[derive(Clone, Debug)]
pub struct Welford {
    diagonal: bool,
    dim: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
    count: usize,
}

impl Welford {
    pub fn new(dim: usize, diagonal: bool) -> Self {
        Welford {
            diagonal,
            dim,
            mean: vec![0.0; dim],
            m2: vec![0.0; if diagonal { dim } else { dim * dim }],
            count: 0,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn update(&mut self, sample: &[f64]) {
        debug_assert_eq!(sample.len(), self.dim);
        self.count += 1;
        let n = self.count as f64;
        if self.diagonal {
            for i in 0..self.dim {
                let delta_pre = sample[i] - self.mean[i];
                self.mean[i] += delta_pre / n;
                let delta_post = sample[i] - self.mean[i];
                self.m2[i] += delta_pre * delta_post;
            }
        } else {
            let d = self.dim;
            let mut delta_pre = vec![0.0; d];
            for i in 0..d {
                delta_pre[i] = sample[i] - self.mean[i];
                self.mean[i] += delta_pre[i] / n;
            }
            for i in 0..d {
                let delta_post_i = sample[i] - self.mean[i];
                for j in 0..d {
                    self.m2[i * d + j] += delta_post_i * delta_pre[j];
                }
            }
        }
    }

    /// The estimated variance (length `dim`) or covariance (`dim * dim`,
    /// row-major). With `regularize`, applies Stan's shrinkage toward a small
    /// multiple of the identity for numerical stability.
    pub fn covariance(&self, regularize: bool) -> Vec<f64> {
        let n = self.count as f64;
        let mut cov: Vec<f64> = self.m2.iter().map(|m| m / (n - 1.0)).collect();
        if regularize {
            let scale = n / (n + 5.0);
            let shrink = 1e-3 * (5.0 / (n + 5.0));
            if self.diagonal {
                for c in cov.iter_mut() {
                    *c = scale * *c + shrink;
                }
            } else {
                let d = self.dim;
                for i in 0..d {
                    for j in 0..d {
                        cov[i * d + j] *= scale;
                    }
                    cov[i * d + i] += shrink;
                }
            }
        }
        cov
    }
}

/// A contiguous window `[start, end]` (inclusive) of warmup iterations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptWindow {
    pub start: usize,
    pub end: usize,
}

/// Stan's windowed adaptation schedule: an initial fast-adaptation buffer,
/// doubling slow-adaptation windows (mass matrix updated at each window end),
/// and a final fast buffer.
pub fn build_adaptation_schedule(num_steps: usize) -> Vec<AdaptWindow> {
    let mut schedule = Vec::new();
    if num_steps < 20 {
        schedule.push(AdaptWindow {
            start: 0,
            end: num_steps.saturating_sub(1),
        });
        return schedule;
    }
    let (mut start_buffer, mut end_buffer, mut init_window) = (75usize, 50usize, 25usize);
    if start_buffer + end_buffer + init_window > num_steps {
        start_buffer = (0.15 * num_steps as f64) as usize;
        end_buffer = (0.1 * num_steps as f64) as usize;
        init_window = num_steps - start_buffer - end_buffer;
    }
    schedule.push(AdaptWindow {
        start: 0,
        end: start_buffer - 1,
    });
    let end_window_start = num_steps - end_buffer;
    let mut next_size = init_window;
    let mut next_start = start_buffer;
    while next_start < end_window_start {
        let cur_start = next_start;
        let mut cur_size = next_size;
        if 3 * cur_size <= end_window_start - cur_start {
            next_size = 2 * cur_size;
        } else {
            cur_size = end_window_start - cur_start;
        }
        next_start = cur_start + cur_size;
        schedule.push(AdaptWindow {
            start: cur_start,
            end: next_start - 1,
        });
    }
    schedule.push(AdaptWindow {
        start: end_window_start,
        end: num_steps - 1,
    });
    schedule
}

/// The (inverse) mass matrix of a Euclidean-Gaussian kinetic energy, either
/// diagonal or dense.
#[derive(Clone, Debug)]
pub enum MassMatrix {
    Diag {
        /// Inverse mass matrix diagonal `M^{-1}`.
        inv: Vec<f64>,
        /// `sqrt(M) = 1 / sqrt(M^{-1})`, for drawing momenta.
        sqrt_mass: Vec<f64>,
    },
    Dense {
        dim: usize,
        /// Row-major inverse mass matrix `M^{-1}`.
        inv: Vec<f64>,
        /// Lower Cholesky factor `L` with `M^{-1} = L L^T`; momenta are drawn by
        /// solving `L^T r = eps` so that `r ~ N(0, M)`.
        chol_inv: Vec<f64>,
    },
}

impl MassMatrix {
    pub fn identity(dim: usize, dense: bool) -> Self {
        if dense {
            let mut inv = vec![0.0; dim * dim];
            for i in 0..dim {
                inv[i * dim + i] = 1.0;
            }
            MassMatrix::Dense {
                dim,
                chol_inv: inv.clone(),
                inv,
            }
        } else {
            MassMatrix::Diag {
                inv: vec![1.0; dim],
                sqrt_mass: vec![1.0; dim],
            }
        }
    }

    /// Build from an inverse mass matrix (diagonal of length `dim` or dense of
    /// length `dim * dim`).
    pub fn from_inverse(inv: Vec<f64>, dim: usize) -> Self {
        if inv.len() == dim {
            let sqrt_mass = inv.iter().map(|v| 1.0 / v.sqrt()).collect();
            MassMatrix::Diag { inv, sqrt_mass }
        } else {
            assert_eq!(inv.len(), dim * dim, "inverse mass matrix has wrong size");
            let chol_inv = linalg::cholesky(&inv, dim)
                .expect("inverse mass matrix is not positive definite");
            MassMatrix::Dense { dim, inv, chol_inv }
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            MassMatrix::Diag { inv, .. } => inv.len(),
            MassMatrix::Dense { dim, .. } => *dim,
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, MassMatrix::Dense { .. })
    }

    /// The inverse mass matrix (diagonal or row-major dense).
    pub fn inverse(&self) -> &[f64] {
        match self {
            MassMatrix::Diag { inv, .. } => inv,
            MassMatrix::Dense { inv, .. } => inv,
        }
    }

    /// Kinetic energy `0.5 r^T M^{-1} r`.
    #[inline]
    pub fn kinetic(&self, r: &[f64]) -> f64 {
        match self {
            MassMatrix::Diag { inv, .. } => {
                0.5 * r.iter().zip(inv).map(|(r, m)| r * r * m).sum::<f64>()
            }
            MassMatrix::Dense { dim, inv, .. } => {
                let mut s = 0.0;
                for i in 0..*dim {
                    let row = &inv[i * dim..(i + 1) * dim];
                    let mut vi = 0.0;
                    for (a, b) in row.iter().zip(r) {
                        vi += a * b;
                    }
                    s += vi * r[i];
                }
                0.5 * s
            }
        }
    }

    /// Velocity `v = M^{-1} r`.
    #[inline]
    pub fn velocity(&self, r: &[f64], v: &mut [f64]) {
        match self {
            MassMatrix::Diag { inv, .. } => {
                for ((v, r), m) in v.iter_mut().zip(r).zip(inv) {
                    *v = r * m;
                }
            }
            MassMatrix::Dense { dim, inv, .. } => linalg::matvec(inv, *dim, r, v),
        }
    }

    /// Draw `r ~ N(0, M)`.
    pub fn sample_momentum(&self, rng: &mut dyn RngCore, r: &mut [f64]) {
        match self {
            MassMatrix::Diag { sqrt_mass, .. } => {
                for (r, s) in r.iter_mut().zip(sqrt_mass) {
                    let e: f64 = rand::Rng::sample(rng, StandardNormal);
                    *r = e * s;
                }
            }
            MassMatrix::Dense { dim, chol_inv, .. } => {
                let eps: Vec<f64> = (0..*dim)
                    .map(|_| rand::Rng::sample(rng, StandardNormal))
                    .collect();
                linalg::solve_lower_transpose(chol_inv, *dim, &eps, r);
            }
        }
    }
}

/// Configuration for [`WarmupAdapter`].
#[derive(Clone, Debug)]
pub struct AdaptConfig {
    pub adapt_step_size: bool,
    pub adapt_mass_matrix: bool,
    pub dense_mass: bool,
    pub target_accept_prob: f64,
    pub regularize_mass_matrix: bool,
    /// Dual averaging shrinks the log step size toward `log(multiplier *
    /// step_size)` at the start of each window. Hoffman & Gelman recommend 10
    /// for HMC (favoring larger steps early); random-walk kernels use 1.
    pub prox_center_multiplier: f64,
    /// Restart dual averaging (re-centered at the current step size) whenever
    /// the mass matrix is updated at a window end. Right for HMC, whose
    /// acceptance rate responds sharply to the step size; random-walk kernels
    /// keep one continuous trajectory instead because the final 50-iteration
    /// window is too short for them to re-converge.
    pub restart_step_size_per_window: bool,
}

impl Default for AdaptConfig {
    fn default() -> Self {
        AdaptConfig {
            adapt_step_size: true,
            adapt_mass_matrix: true,
            dense_mass: false,
            target_accept_prob: 0.8,
            regularize_mass_matrix: true,
            prox_center_multiplier: 10.0,
            restart_step_size_per_window: true,
        }
    }
}

/// Adapts the step size (dual averaging, every iteration) and the mass matrix
/// (Welford, at the end of each slow window) during warmup.
#[derive(Clone, Debug)]
pub struct WarmupAdapter {
    cfg: AdaptConfig,
    schedule: Vec<AdaptWindow>,
    num_steps: usize,
    window_idx: usize,
    da: DualAveraging,
    welford: Welford,
    pub step_size: f64,
    pub mass: MassMatrix,
}

impl WarmupAdapter {
    /// `find_reasonable` optionally tunes the initial step size (see
    /// [`crate::infer::hmc`]).
    pub fn new(
        num_steps: usize,
        dim: usize,
        step_size: f64,
        inverse_mass_matrix: Option<Vec<f64>>,
        cfg: AdaptConfig,
        find_reasonable: Option<&mut dyn FnMut(f64, &MassMatrix) -> f64>,
    ) -> Self {
        let mass = match inverse_mass_matrix {
            Some(inv) => {
                let m = MassMatrix::from_inverse(inv, dim);
                assert_eq!(
                    m.is_dense(),
                    cfg.dense_mass,
                    "inverse_mass_matrix shape does not match dense_mass"
                );
                m
            }
            None => MassMatrix::identity(dim, cfg.dense_mass),
        };
        let mut step_size = step_size;
        if cfg.adapt_step_size {
            if let Some(f) = find_reasonable {
                step_size = f(step_size, &mass);
            }
        }
        let da = DualAveraging::new((cfg.prox_center_multiplier * step_size).ln());
        let welford = Welford::new(dim, !cfg.dense_mass);
        WarmupAdapter {
            schedule: build_adaptation_schedule(num_steps),
            num_steps,
            window_idx: 0,
            da,
            welford,
            step_size,
            mass,
            cfg,
        }
    }

    pub fn schedule(&self) -> &[AdaptWindow] {
        &self.schedule
    }

    pub fn window_idx(&self) -> usize {
        self.window_idx
    }

    /// Update after warmup iteration `t` with the observed acceptance
    /// probability and the new position `z` (unconstrained).
    pub fn update(
        &mut self,
        t: usize,
        accept_prob: f64,
        z: &[f64],
        mut find_reasonable: Option<&mut dyn FnMut(f64, &MassMatrix) -> f64>,
    ) {
        if self.cfg.adapt_step_size {
            self.da.update(self.cfg.target_accept_prob - accept_prob);
            let log_ss = if t == self.num_steps - 1 {
                self.da.x_avg()
            } else {
                self.da.x()
            };
            self.step_size = log_ss.exp().clamp(f64::MIN_POSITIVE, f64::MAX);
        }
        let num_windows = self.schedule.len();
        let is_middle = self.window_idx > 0 && self.window_idx < num_windows - 1;
        if self.cfg.adapt_mass_matrix && is_middle {
            self.welford.update(z);
        }
        let at_window_end = t == self.schedule[self.window_idx].end;
        if at_window_end {
            self.window_idx += 1;
        }
        if at_window_end && is_middle {
            if self.cfg.adapt_mass_matrix {
                let cov = self.welford.covariance(self.cfg.regularize_mass_matrix);
                self.mass = MassMatrix::from_inverse(cov, z.len());
                self.welford = Welford::new(z.len(), !self.cfg.dense_mass);
            }
            if self.cfg.adapt_step_size && self.cfg.restart_step_size_per_window {
                if let Some(f) = find_reasonable.as_mut() {
                    self.step_size = f(self.step_size, &self.mass);
                }
                self.da = DualAveraging::new(self.cfg.prox_center_multiplier.ln() + self.step_size.ln());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn dual_averaging_optimizes_quadratic() {
        // minimize (x + 1)^2 by feeding its gradient (port of numpyro test)
        let mut da = DualAveraging::with_params(0.0, 10.0, 0.75, 0.5);
        for _ in 0..10 {
            let x = da.x();
            let g = 2.0 * (x + 1.0);
            da.update(g);
        }
        assert!((da.x_avg() + 1.0).abs() < 1e-3, "{}", da.x_avg());
    }

    #[test]
    fn welford_covariance_matches_sample_covariance() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let a = [1.0, 0.5, 0.0, -0.3, 1.2, 0.0, 0.7, 0.2, 0.9];
        let target: Vec<f64> = (0..9)
            .map(|idx| {
                let (i, j) = (idx / 3, idx % 3);
                (0..3).map(|k| a[i * 3 + k] * a[j * 3 + k]).sum()
            })
            .collect();
        let mut diag = Welford::new(3, true);
        let mut dense = Welford::new(3, false);
        let mut samples = Vec::new();
        for _ in 0..4000 {
            let eps: Vec<f64> = (0..3).map(|_| rand::Rng::sample(&mut rng, StandardNormal)).collect();
            let x: Vec<f64> = (0..3)
                .map(|i| (0..3).map(|k| a[i * 3 + k] * eps[k]).sum::<f64>() + i as f64)
                .collect();
            diag.update(&x);
            dense.update(&x);
            samples.push(x);
        }
        let cov = dense.covariance(false);
        let var = diag.covariance(false);
        for i in 0..3 {
            for j in 0..3 {
                assert!((cov[i * 3 + j] - target[i * 3 + j]).abs() < 0.08, "cov {i}{j}");
            }
            assert!((var[i] - target[i * 3 + i]).abs() < 0.08);
            assert!((var[i] - cov[i * 3 + i]).abs() < 1e-10);
        }
        // regularization shrinks toward identity slightly
        let reg = diag.covariance(true);
        let n = 4000.0;
        assert!((reg[0] - (n / (n + 5.0) * var[0] + 1e-3 * 5.0 / (n + 5.0))).abs() < 1e-12);
    }

    #[test]
    fn adaptation_schedule_matches_numpyro() {
        let cases: Vec<(usize, Vec<(usize, usize)>)> = vec![
            (18, vec![(0, 17)]),
            (50, vec![(0, 6), (7, 44), (45, 49)]),
            (100, vec![(0, 14), (15, 89), (90, 99)]),
            (150, vec![(0, 74), (75, 99), (100, 149)]),
            (200, vec![(0, 74), (75, 99), (100, 149), (150, 199)]),
            (280, vec![(0, 74), (75, 99), (100, 229), (230, 279)]),
            (1000, vec![(0, 74), (75, 99), (100, 149), (150, 249), (250, 449), (450, 949), (950, 999)]),
        ];
        for (n, expected) in cases {
            let s = build_adaptation_schedule(n);
            let got: Vec<(usize, usize)> = s.iter().map(|w| (w.start, w.end)).collect();
            assert_eq!(got, expected, "num_steps={n}");
        }
    }

    #[test]
    fn mass_matrix_momentum_and_kinetic() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let inv = vec![2.0, 0.3, 0.3, 0.5];
        let m = MassMatrix::from_inverse(inv.clone(), 2);
        // r ~ N(0, M) where M = inv^{-1}
        let mut r = [0.0; 2];
        let n = 100_000;
        let mut c = [0.0; 4];
        for _ in 0..n {
            m.sample_momentum(&mut rng, &mut r);
            for i in 0..2 {
                for j in 0..2 {
                    c[i * 2 + j] += r[i] * r[j] / n as f64;
                }
            }
        }
        let mass = linalg::spd_inverse(&inv, 2).unwrap();
        for k in 0..4 {
            assert!((c[k] - mass[k]).abs() < 0.03, "{c:?} vs {mass:?}");
        }
        let r = [1.0, -2.0];
        let mut v = [0.0; 2];
        m.velocity(&r, &mut v);
        assert!((v[0] - (2.0 - 0.6)).abs() < 1e-12 && (v[1] - (0.3 - 1.0)).abs() < 1e-12);
        assert!((m.kinetic(&r) - 0.5 * (r[0] * v[0] + r[1] * v[1])).abs() < 1e-12);
        let d = MassMatrix::from_inverse(vec![4.0, 0.25], 2);
        assert!((d.kinetic(&r) - 0.5 * (4.0 + 0.25 * 4.0)).abs() < 1e-12);
    }

    /// Port of numpyro's `test_warmup_adapter`.
    #[test]
    fn warmup_adapter_windows() {
        let num_steps = 150;
        let schedule = build_adaptation_schedule(num_steps);
        let mut fr = |ss: f64, _m: &MassMatrix| if ss < 1.0 { ss * 4.0 } else { ss / 4.0 };
        let z = vec![1.0; 3];
        let mut wa = WarmupAdapter::new(num_steps, 3, 1.0, None, AdaptConfig::default(), Some(&mut fr));
        assert_eq!(wa.step_size, 0.25);
        assert_eq!(wa.mass.inverse(), &[1.0, 1.0, 1.0]);
        assert_eq!(wa.window_idx(), 0);

        let w = schedule[0];
        for t in w.start..=w.end {
            let ap = 0.7 + 0.1 * t as f64 / (w.end - w.start) as f64;
            wa.update(t, ap, &z, Some(&mut fr));
        }
        let last = 0.25;
        assert_eq!(wa.window_idx(), 1);
        assert!(wa.step_size < last, "step size decreases when accept_prob < target");
        assert_eq!(wa.mass.inverse(), &[1.0, 1.0, 1.0]);

        let w = schedule[1];
        let z2 = vec![2.0; 3];
        let len = (w.end - w.start) as f64;
        for t in w.start..=w.end {
            let ap = 0.8 + 0.1 * (t - w.start) as f64 / len;
            wa.update(t, ap, &z2, Some(&mut fr));
        }
        assert_eq!(wa.window_idx(), 2);
        let last = wa.step_size;
        // constant z in the window: covariance is pure regularization
        let reg = 1e-3 * (5.0 / ((w.end + 1 - w.start) as f64 + 5.0));
        for v in wa.mass.inverse() {
            assert!((v - reg).abs() < 1e-7, "{v} vs {reg}");
        }

        let w = schedule[2];
        for t in w.start..=w.end {
            let zt: Vec<f64> = z.iter().map(|v| v * t as f64).collect();
            wa.update(t, 0.8, &zt, Some(&mut fr));
        }
        assert_eq!(wa.window_idx(), 3);
        // accept_prob == target: log step size sits at the prox center log(10 * last)
        assert!((wa.step_size - last * 10.0).abs() < 1e-6, "{} vs {}", wa.step_size, last * 10.0);
        for v in wa.mass.inverse() {
            assert!((v - reg).abs() < 1e-7);
        }
    }
}
