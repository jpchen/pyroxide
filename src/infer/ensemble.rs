//! Gradient-free ensemble samplers: affine-invariant ensemble sampling (AIES,
//! Goodman & Weare 2010 / `emcee`) and ensemble slice sampling (ESS,
//! Karamanis & Beutler 2020 / `zeus`). Ports of `numpyro.infer.AIES` / `ESS`.
//!
//! An ensemble of `n` walkers (even, ideally `>= 2 * dim`) is split in two
//! halves; each half is updated using the other half as the "complementary
//! ensemble", which keeps the update a valid Markov chain on the joint
//! ensemble. The `MCMC` driver reports every walker as a chain.
//!
//! Only `Potential::value` is used, so these work for densities without
//! gradients.

use rand::Rng;
use rand_distr::StandardNormal;

use super::mcmc::{ChainRng, Kernel};
use super::potential::Potential;

/// Shared ensemble state: walker positions (row-major `n x d`) and cached log
/// densities.
#[derive(Clone, Debug)]
pub struct EnsembleState {
    pub i: usize,
    pub num_warmup: usize,
    pub n: usize,
    pub dim: usize,
    pub z: Vec<f64>,
    pub log_density: Vec<f64>,
    pub accept_prob: f64,
    pub mean_accept_prob: f64,
    /// ESS: adaptive scale factor `mu`.
    pub mu: f64,
    n_expansions: usize,
    n_contractions: usize,
    perm: Vec<usize>,
    proposal: Vec<f64>,
}

impl EnsembleState {
    fn walker(&self, w: usize) -> &[f64] {
        &self.z[w * self.dim..(w + 1) * self.dim]
    }
}

fn init_ensemble_state<P: Potential>(p: &P, zs: Vec<Vec<f64>>, num_warmup: usize, mu: f64) -> EnsembleState {
    let n = zs.len();
    assert!(
        n >= 4 && n.is_multiple_of(2),
        "ensemble samplers need an even number of walkers >= 4 (got {n})"
    );
    let dim = zs[0].len();
    let mut z = Vec::with_capacity(n * dim);
    let mut log_density = Vec::with_capacity(n);
    for w in &zs {
        z.extend_from_slice(w);
        log_density.push(-p.value(w));
    }
    EnsembleState {
        i: 0,
        num_warmup,
        n,
        dim,
        z,
        log_density,
        accept_prob: 0.0,
        mean_accept_prob: 0.0,
        mu,
        n_expansions: 0,
        n_contractions: 0,
        perm: (0..n).collect(),
        proposal: vec![0.0; dim],
    }
}

/// Pick two distinct indices from `0..n`.
fn two_distinct(rng: &mut ChainRng, n: usize) -> (usize, usize) {
    let a = rng.random_range(0..n);
    let mut b = rng.random_range(0..n - 1);
    if b >= a {
        b += 1;
    }
    (a, b)
}

// -------------------------------------------------------------------- AIES ---

/// Proposal move for [`AIES`].
#[derive(Clone, Copy, Debug)]
pub enum AiesMove {
    /// Differential evolution (Nelson et al. 2013): `x + γ (x_j - x_k)` with
    /// `γ ~ N(g0, g0 σ)`, `g0 = 2.38 / sqrt(2 d)` by default. numpyro's default.
    DE { sigma: f64, g0: Option<f64> },
    /// Goodman & Weare stretch move with scale `a`.
    Stretch { a: f64 },
}

/// Affine-invariant ensemble sampler.
pub struct AIES<P: Potential> {
    potential: P,
    pub move_: AiesMove,
    /// Randomly permute walkers before splitting into halves each iteration.
    pub randomize_split: bool,
}

impl<P: Potential> AIES<P> {
    pub fn new(potential: P) -> Self {
        AIES {
            potential,
            move_: AiesMove::DE {
                sigma: 1e-5,
                g0: None,
            },
            randomize_split: false,
        }
    }
    pub fn stretch_move(mut self, a: f64) -> Self {
        self.move_ = AiesMove::Stretch { a };
        self
    }
    pub fn de_move(mut self, sigma: f64, g0: Option<f64>) -> Self {
        self.move_ = AiesMove::DE { sigma, g0 };
        self
    }
    pub fn randomize_split(mut self, b: bool) -> Self {
        self.randomize_split = b;
        self
    }

    /// Update the `active` half using the `inactive` half. Returns the number of
    /// accepted proposals.
    fn update_half(
        &self,
        st: &mut EnsembleState,
        active: &[usize],
        inactive: &[usize],
        rng: &mut ChainRng,
    ) -> usize {
        let d = st.dim;
        let mut accepted = 0;
        for &w in active {
            let (log_factor, ok) = match self.move_ {
                AiesMove::DE { sigma, g0 } => {
                    let g = g0.unwrap_or(2.38 / (2.0 * d as f64).sqrt());
                    let gamma: f64 = g + g * sigma * rng.sample::<f64, _>(StandardNormal);
                    let (j, k) = two_distinct(rng, inactive.len());
                    let (xj, xk) = (inactive[j], inactive[k]);
                    for i in 0..d {
                        st.proposal[i] = st.z[w * d + i] + gamma * (st.z[xj * d + i] - st.z[xk * d + i]);
                    }
                    (0.0, true)
                }
                AiesMove::Stretch { a } => {
                    let u: f64 = rng.random();
                    let zz = ((a - 1.0) * u + 1.0).powi(2) / a;
                    let r = inactive[rng.random_range(0..inactive.len())];
                    for i in 0..d {
                        let xr = st.z[r * d + i];
                        st.proposal[i] = xr - (xr - st.z[w * d + i]) * zz;
                    }
                    ((d as f64 - 1.0) * zz.ln(), true)
                }
            };
            if !ok {
                continue;
            }
            let lp_new = -self.potential.value(&st.proposal);
            let log_accept = log_factor + lp_new - st.log_density[w];
            if log_accept.is_finite() && rng.random::<f64>().ln() < log_accept {
                st.z[w * d..(w + 1) * d].copy_from_slice(&st.proposal);
                st.log_density[w] = lp_new;
                accepted += 1;
            }
        }
        accepted
    }
}

fn split_halves(st: &mut EnsembleState, randomize: bool, rng: &mut ChainRng) {
    if randomize {
        // Fisher–Yates
        for i in (1..st.n).rev() {
            let j = rng.random_range(0..=i);
            st.perm.swap(i, j);
        }
    }
}

impl<P: Potential> Kernel for AIES<P> {
    type State = EnsembleState;

    fn potential(&self) -> &dyn Potential {
        &self.potential
    }

    fn is_ensemble(&self) -> bool {
        true
    }

    fn init(&self, _z: Vec<f64>, _num_warmup: usize, _rng: &mut ChainRng) -> EnsembleState {
        panic!("AIES is an ensemble sampler; use MCMC (which calls init_ensemble)")
    }

    fn init_ensemble(&self, zs: Vec<Vec<f64>>, num_warmup: usize, _rng: &mut ChainRng) -> EnsembleState {
        init_ensemble_state(&self.potential, zs, num_warmup, 1.0)
    }

    fn step(&self, st: &mut EnsembleState, rng: &mut ChainRng) {
        split_halves(st, self.randomize_split, rng);
        let half = st.n / 2;
        let (first, second): (Vec<usize>, Vec<usize>) = (st.perm[..half].to_vec(), st.perm[half..].to_vec());
        let mut accepted = self.update_half(st, &first, &second, rng);
        accepted += self.update_half(st, &second, &first, rng);
        let accept_prob = accepted as f64 / st.n as f64;
        let warmup = st.i < st.num_warmup;
        st.i += 1;
        let n = if warmup { st.i } else { st.i - st.num_warmup };
        st.mean_accept_prob += (accept_prob - st.mean_accept_prob) / n as f64;
        st.accept_prob = accept_prob;
    }

    fn position(st: &EnsembleState) -> &[f64] {
        st.walker(0)
    }

    fn position_of(st: &EnsembleState, w: usize) -> &[f64] {
        st.walker(w)
    }

    fn stat_names(&self) -> &'static [&'static str] {
        &["accept_prob", "mean_accept_prob"]
    }

    fn stats(&self, st: &EnsembleState, out: &mut [f64]) {
        out[0] = st.accept_prob;
        out[1] = st.mean_accept_prob;
    }
}

// --------------------------------------------------------------------- ESS ---

/// Ensemble slice sampler with the differential move.
pub struct ESS<P: Potential> {
    potential: P,
    pub randomize_split: bool,
    /// Maximum stepping-out / shrinking iterations per walker.
    pub max_iter: usize,
    pub init_mu: f64,
    /// Adapt `mu` during warmup from the expansion/contraction counts.
    pub tune_mu: bool,
}

impl<P: Potential> ESS<P> {
    pub fn new(potential: P) -> Self {
        ESS {
            potential,
            randomize_split: true,
            max_iter: 10_000,
            init_mu: 1.0,
            tune_mu: true,
        }
    }
    pub fn randomize_split(mut self, b: bool) -> Self {
        self.randomize_split = b;
        self
    }
    pub fn init_mu(mut self, mu: f64) -> Self {
        self.init_mu = mu;
        self
    }
    pub fn tune_mu(mut self, b: bool) -> Self {
        self.tune_mu = b;
        self
    }

    fn update_half(&self, st: &mut EnsembleState, active: &[usize], inactive: &[usize], rng: &mut ChainRng) {
        let d = st.dim;
        let mut direction = vec![0.0; d];
        let mut x = vec![0.0; d];
        for &w in active {
            // differential move: direction = 2 mu (x_j - x_k), j != k in the other half
            let (j, k) = two_distinct(rng, inactive.len());
            let (xj, xk) = (inactive[j], inactive[k]);
            for i in 0..d {
                direction[i] = 2.0 * st.mu * (st.z[xj * d + i] - st.z[xk * d + i]);
            }
            let e: f64 = rng.sample(rand_distr::Exp1);
            let log_height = st.log_density[w] - e;
            let logp_at = |t: f64, x: &mut [f64], st: &EnsembleState| {
                for i in 0..d {
                    x[i] = st.z[w * d + i] + t * direction[i];
                }
                -self.potential.value(x)
            };
            // stepping out
            let mut left = -rng.random::<f64>();
            let mut right = left + 1.0;
            let mut iter = 0;
            while iter < self.max_iter && logp_at(left, &mut x, st) > log_height {
                left -= 1.0;
                st.n_expansions += 1;
                iter += 1;
            }
            iter = 0;
            while iter < self.max_iter && logp_at(right, &mut x, st) > log_height {
                right += 1.0;
                st.n_expansions += 1;
                iter += 1;
            }
            // shrinking
            iter = 0;
            loop {
                let t = left + (right - left) * rng.random::<f64>();
                let lp = logp_at(t, &mut x, st);
                if lp >= log_height || iter >= self.max_iter {
                    st.z[w * d..(w + 1) * d].copy_from_slice(&x);
                    st.log_density[w] = lp;
                    break;
                }
                if t < 0.0 {
                    left = t;
                } else {
                    right = t;
                }
                st.n_contractions += 1;
                iter += 1;
            }
        }
    }
}

impl<P: Potential> Kernel for ESS<P> {
    type State = EnsembleState;

    fn potential(&self) -> &dyn Potential {
        &self.potential
    }

    fn is_ensemble(&self) -> bool {
        true
    }

    fn init(&self, _z: Vec<f64>, _num_warmup: usize, _rng: &mut ChainRng) -> EnsembleState {
        panic!("ESS is an ensemble sampler; use MCMC (which calls init_ensemble)")
    }

    fn init_ensemble(&self, zs: Vec<Vec<f64>>, num_warmup: usize, _rng: &mut ChainRng) -> EnsembleState {
        init_ensemble_state(&self.potential, zs, num_warmup, self.init_mu)
    }

    fn step(&self, st: &mut EnsembleState, rng: &mut ChainRng) {
        split_halves(st, self.randomize_split, rng);
        let half = st.n / 2;
        let (first, second): (Vec<usize>, Vec<usize>) = (st.perm[..half].to_vec(), st.perm[half..].to_vec());
        st.n_expansions = 0;
        st.n_contractions = 0;
        self.update_half(st, &first, &second, rng);
        self.update_half(st, &second, &first, rng);
        let warmup = st.i < st.num_warmup;
        if self.tune_mu && warmup {
            // zeus tuning rule: mu <- mu * 2 n_exp / (n_exp + n_con)
            let ne = st.n_expansions.max(1) as f64;
            let nc = st.n_contractions as f64;
            st.mu *= 2.0 * ne / (ne + nc);
        }
        st.i += 1;
        // slice sampling always accepts
        st.accept_prob = 1.0;
        st.mean_accept_prob = 1.0;
    }

    fn position(st: &EnsembleState) -> &[f64] {
        st.walker(0)
    }

    fn position_of(st: &EnsembleState, w: usize) -> &[f64] {
        st.walker(w)
    }

    fn stat_names(&self) -> &'static [&'static str] {
        &["mu", "n_expansions", "n_contractions"]
    }

    fn stats(&self, st: &EnsembleState, out: &mut [f64]) {
        out[0] = st.mu;
        out[1] = st.n_expansions as f64;
        out[2] = st.n_contractions as f64;
    }
}
