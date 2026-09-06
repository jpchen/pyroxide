//! Barker proposal Metropolis–Hastings (Livingstone & Zanella 2022).
//!
//! A gradient-based MH kernel whose proposal is skew-symmetric in the
//! direction of the gradient: each coordinate increment `z_i ~ N(0, ε²)` keeps
//! its sign with probability `sigmoid(z_i ∂_i log π(x))` and is flipped
//! otherwise. Compared with MALA it is robust to poorly tuned step sizes and to
//! heavy tails, and it is competitive with HMC in low to moderate dimension.
//! Port of `numpyro.infer.BarkerMH`, including its use of the HMC warmup
//! adapter (step size by dual averaging toward 0.4 acceptance, diagonal or
//! dense mass matrix by Welford).

use rand::Rng;
use rand_distr::StandardNormal;

use super::adapt::{AdaptConfig, MassMatrix, WarmupAdapter};
use super::mcmc::{ChainRng, Kernel};
use super::potential::Potential;
use crate::ad::{sigmoid_f64, softplus_f64};
use crate::linalg;

/// Barker proposal MH kernel.
pub struct BarkerMH<P: Potential> {
    potential: P,
    pub step_size: f64,
    pub adapt: AdaptConfig,
}

impl<P: Potential> BarkerMH<P> {
    pub fn new(potential: P) -> Self {
        BarkerMH {
            potential,
            step_size: 1.0,
            adapt: AdaptConfig {
                target_accept_prob: 0.4,
                // like random-walk MH: one continuous dual-averaging trajectory
                // centred on the current step; the acceptance signal is too noisy
                // for HMC's per-window restarts (see infer::mh)
                prox_center_multiplier: 1.0,
                restart_step_size_per_window: false,
                ..AdaptConfig::default()
            },
        }
    }
    pub fn step_size(mut self, s: f64) -> Self {
        self.step_size = s;
        self
    }
    pub fn adapt_step_size(mut self, b: bool) -> Self {
        self.adapt.adapt_step_size = b;
        self
    }
    pub fn adapt_mass_matrix(mut self, b: bool) -> Self {
        self.adapt.adapt_mass_matrix = b;
        self
    }
    pub fn dense_mass(mut self, b: bool) -> Self {
        self.adapt.dense_mass = b;
        self
    }
    pub fn target_accept_prob(mut self, p: f64) -> Self {
        self.adapt.target_accept_prob = p;
        self
    }
    pub fn potential_ref(&self) -> &P {
        &self.potential
    }
}

/// Per-chain state of [`BarkerMH`].
#[derive(Clone, Debug)]
pub struct BarkerState {
    pub i: usize,
    pub num_warmup: usize,
    pub z: Vec<f64>,
    pub potential_energy: f64,
    pub z_grad: Vec<f64>,
    pub accept_prob: f64,
    pub mean_accept_prob: f64,
    pub adapt: WarmupAdapter,
    y: Vec<f64>,
    y_grad: Vec<f64>,
    dx: Vec<f64>,
    scratch: Vec<f64>,
    scratch2: Vec<f64>,
}

/// `out = S g` where `S^T S = M^{-1}` (`S = sqrt(M^{-1})` diagonal, or `L^T`
/// for the dense Cholesky factor `L L^T = M^{-1}`).
fn scale_grad(mass: &MassMatrix, g: &[f64], out: &mut [f64]) {
    match mass {
        MassMatrix::Diag { inv, .. } => {
            for i in 0..g.len() {
                out[i] = inv[i].sqrt() * g[i];
            }
        }
        MassMatrix::Dense { dim, chol_inv, .. } => {
            // L^T g
            for i in 0..*dim {
                let mut s = 0.0;
                for k in i..*dim {
                    s += chol_inv[k * dim + i] * g[k];
                }
                out[i] = s;
            }
        }
    }
}

/// `out = S^T dx` (the increment mapped back to position space).
fn scale_increment(mass: &MassMatrix, dx: &[f64], out: &mut [f64]) {
    match mass {
        MassMatrix::Diag { inv, .. } => {
            for i in 0..dx.len() {
                out[i] = inv[i].sqrt() * dx[i];
            }
        }
        MassMatrix::Dense { dim, chol_inv, .. } => linalg::tril_matvec(chol_inv, *dim, dx, out),
    }
}

impl<P: Potential> Kernel for BarkerMH<P> {
    type State = BarkerState;

    fn potential(&self) -> &dyn Potential {
        &self.potential
    }

    fn init(&self, z: Vec<f64>, num_warmup: usize, _rng: &mut ChainRng) -> BarkerState {
        let d = z.len();
        let mut z_grad = vec![0.0; d];
        let pe = self.potential.value_and_grad(&z, &mut z_grad);
        let adapt = WarmupAdapter::new(num_warmup, d, self.step_size, None, self.adapt.clone(), None);
        BarkerState {
            i: 0,
            num_warmup,
            z,
            potential_energy: pe,
            z_grad,
            accept_prob: 0.0,
            mean_accept_prob: 0.0,
            adapt,
            y: vec![0.0; d],
            y_grad: vec![0.0; d],
            dx: vec![0.0; d],
            scratch: vec![0.0; d],
            scratch2: vec![0.0; d],
        }
    }

    fn step(&self, st: &mut BarkerState, rng: &mut ChainRng) {
        let warmup = st.i < st.num_warmup;
        let d = st.z.len();
        let eps = st.adapt.step_size;
        let mass = &st.adapt.mass;
        // scaled gradient of log density at x: -S ∇U(x)
        scale_grad(mass, &st.z_grad, &mut st.scratch);
        for i in 0..d {
            let z: f64 = eps * rng.sample::<f64, _>(StandardNormal);
            let grad_logp_scaled = -st.scratch[i];
            let p_keep = sigmoid_f64(z * grad_logp_scaled);
            let b = if rng.random::<f64>() < p_keep { 1.0 } else { -1.0 };
            st.dx[i] = b * z;
        }
        scale_increment(mass, &st.dx, &mut st.scratch2);
        for i in 0..d {
            st.y[i] = st.z[i] + st.scratch2[i];
        }
        let y_pe = self.potential.value_and_grad(&st.y, &mut st.y_grad);
        // log accept ratio = U(x) - U(y) + Σ_i [softplus(dx_i ∂_iU(x)) - softplus(-dx_i ∂_iU(y))]
        // (gradients of the potential, scaled by S; the Barker proposal density
        // is 2 φ(dx) σ(dx ∇log π) so the reverse move contributes σ(-dx ∇log π(y)))
        scale_grad(mass, &st.y_grad, &mut st.scratch2);
        let mut log_ratio = st.potential_energy - y_pe;
        for i in 0..d {
            let gux = st.scratch[i];
            let guy = st.scratch2[i];
            log_ratio += softplus_f64(st.dx[i] * gux) - softplus_f64(-st.dx[i] * guy);
        }
        if log_ratio.is_nan() {
            log_ratio = f64::NEG_INFINITY;
        }
        let accept_prob = log_ratio.exp().min(1.0);
        if rng.random::<f64>() < accept_prob {
            std::mem::swap(&mut st.z, &mut st.y);
            std::mem::swap(&mut st.z_grad, &mut st.y_grad);
            st.potential_energy = y_pe;
        }
        if warmup {
            let t = st.i;
            let z = &st.z;
            st.adapt.update(t, accept_prob, z, None);
        }
        st.i += 1;
        let n = if warmup { st.i } else { st.i - st.num_warmup };
        st.mean_accept_prob += (accept_prob - st.mean_accept_prob) / n as f64;
        st.accept_prob = accept_prob;
    }

    fn position(st: &BarkerState) -> &[f64] {
        &st.z
    }

    fn stat_names(&self) -> &'static [&'static str] {
        &["accept_prob", "step_size", "potential_energy", "mean_accept_prob"]
    }

    fn stats(&self, st: &BarkerState, out: &mut [f64]) {
        out[0] = st.accept_prob;
        out[1] = st.adapt.step_size;
        out[2] = st.potential_energy;
        out[3] = st.mean_accept_prob;
    }
}
