//! Metropolis-Adjusted Microcanonical Sampler (MAMS; Robnik, Cohn-Gordon &
//! Seljak 2025), a.k.a. adjusted MCLMC.
//!
//! Microcanonical Langevin Monte Carlo replaces HMC's Gaussian momentum with a
//! unit-norm momentum on the sphere and integrates the isokinetic
//! ("energy-sampling Hamiltonian") dynamics of Steeg & Galstyan / Robnik et
//! al. with a two-stage McLachlan splitting. MAMS draws a fresh unit momentum
//! each transition, integrates for a (Halton-jittered) number of steps chosen
//! from a trajectory length `L`, and accepts or rejects with the exact energy
//! change, so it targets the posterior exactly. In benchmarks it typically
//! matches or beats NUTS in effective samples per gradient on smooth targets.
//!
//! Port of the `numpyro.contrib.microcanonical.MAMS` kernel (itself a port of
//! BlackJAX's `adjusted_mclmc`): McLachlan integrator, closed-form ESH momentum
//! update, `L` from posterior variances (`1.3 * sqrt(Σ var)`) after the first
//! half of warmup, dual averaging of the step size toward 0.9 acceptance in
//! two phases. Requires `dim >= 2`.

use rand::Rng;
use rand_distr::StandardNormal;

use super::adapt::DualAveraging;
use super::mcmc::{ChainRng, Kernel};
use super::potential::Potential;

const MCLACHLAN_B1: f64 = 0.193_183_327_503_783_6;
const MCLACHLAN: [f64; 5] = [MCLACHLAN_B1, 0.5, 1.0 - 2.0 * MCLACHLAN_B1, 0.5, MCLACHLAN_B1];
const VELOCITY_VERLET: [f64; 3] = [0.5, 1.0, 0.5];
const L_RATIO_MAX: f64 = 2.0;

/// Integrator splitting for the isokinetic dynamics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Integrator {
    /// Two-stage minimal-norm scheme (two gradients per step); default.
    McLachlan,
    /// Standard leapfrog (one gradient per step).
    VelocityVerlet,
}

/// MAMS kernel.
pub struct MAMS<P: Potential> {
    potential: P,
    pub target_accept_prob: f64,
    /// Initial step size (default `0.2 sqrt(d)`).
    pub step_size: Option<f64>,
    /// Initial trajectory length (default `sqrt(d)`).
    pub trajectory_length: Option<f64>,
    /// Jitter the number of steps with a Halton sequence (default true).
    pub jitter: bool,
    /// Learn a diagonal preconditioner from warmup variances.
    pub preconditioning: bool,
    pub divergence_threshold: f64,
    pub max_num_steps: usize,
    pub integrator: Integrator,
    /// Fractions of warmup spent in the two dual-averaging phases.
    pub frac_da1: f64,
    pub frac_da2: f64,
    /// Variance-based `L` scaling (`L = tuning_factor * sqrt(Σ var)`).
    pub tuning_factor: f64,
}

impl<P: Potential> MAMS<P> {
    pub fn new(potential: P) -> Self {
        MAMS {
            potential,
            target_accept_prob: 0.9,
            step_size: None,
            trajectory_length: None,
            jitter: true,
            preconditioning: false,
            divergence_threshold: 1000.0,
            max_num_steps: 1024,
            integrator: Integrator::McLachlan,
            frac_da1: 0.5,
            frac_da2: 0.5,
            tuning_factor: 1.3,
        }
    }
    pub fn target_accept_prob(mut self, p: f64) -> Self {
        self.target_accept_prob = p;
        self
    }
    pub fn step_size(mut self, s: f64) -> Self {
        self.step_size = Some(s);
        self
    }
    pub fn trajectory_length(mut self, l: f64) -> Self {
        self.trajectory_length = Some(l);
        self
    }
    pub fn jitter(mut self, b: bool) -> Self {
        self.jitter = b;
        self
    }
    pub fn preconditioning(mut self, b: bool) -> Self {
        self.preconditioning = b;
        self
    }
    pub fn integrator(mut self, i: Integrator) -> Self {
        self.integrator = i;
        self
    }
    pub fn max_num_steps(mut self, n: usize) -> Self {
        self.max_num_steps = n;
        self
    }
    pub fn potential_ref(&self) -> &P {
        &self.potential
    }

    fn coefficients(&self) -> &'static [f64] {
        match self.integrator {
            Integrator::McLachlan => &MCLACHLAN,
            Integrator::VelocityVerlet => &VELOCITY_VERLET,
        }
    }

    /// One ESH momentum update (Steeg & Galstyan 2021, eq. 16). `u` is the
    /// unit momentum, `g` the gradient of the *log density*. Returns the
    /// kinetic-energy change; writes the kinetic gradient (velocity) to `kg`.
    fn momentum_update(
        &self,
        u: &mut [f64],
        g: &[f64],
        sqrt_inv_mass: &[f64],
        delta_scale: f64,
        ws: &mut Ws,
    ) -> f64 {
        let d = u.len();
        let dm1 = (d - 1) as f64;
        let mut gnorm2 = 0.0;
        for i in 0..d {
            ws.e[i] = g[i] * sqrt_inv_mass[i];
            gnorm2 += ws.e[i] * ws.e[i];
        }
        let gnorm = gnorm2.sqrt();
        if gnorm > 1e-13 {
            for v in ws.e.iter_mut() {
                *v /= gnorm;
            }
        }
        let mut proj = 0.0;
        for i in 0..d {
            proj += u[i] * ws.e[i];
        }
        let delta = delta_scale * gnorm / dm1;
        let zeta = (-delta).exp();
        let mut norm2 = 0.0;
        for i in 0..d {
            let v = ws.e[i] * (1.0 - zeta) * (1.0 + zeta + proj * (1.0 - zeta)) + 2.0 * zeta * u[i];
            ws.tmp[i] = v;
            norm2 += v * v;
        }
        let norm = norm2.sqrt();
        for i in 0..d {
            u[i] = if norm > 1e-13 { ws.tmp[i] / norm } else { ws.tmp[i] };
            ws.kg[i] = u[i] * sqrt_inv_mass[i];
        }
        (delta - std::f64::consts::LN_2 + (1.0 + proj + (1.0 - proj) * zeta * zeta).ln()) * dm1
    }

    /// One deterministic isokinetic integrator step. Returns the kinetic-energy
    /// change; updates `z`, `u`, `logp`, `grad` in place.
    #[allow(clippy::too_many_arguments)]
    fn isokinetic_step(
        &self,
        step: f64,
        z: &mut [f64],
        u: &mut [f64],
        logp: &mut f64,
        grad: &mut [f64],
        sqrt_inv_mass: &[f64],
        ws: &mut Ws,
    ) -> f64 {
        let coefs = self.coefficients();
        let mut ke = 0.0;
        for (i, &c) in coefs[..coefs.len() - 1].iter().enumerate() {
            if i % 2 == 0 {
                ke += self.momentum_update(u, grad, sqrt_inv_mass, step * c, ws);
            } else {
                for k in 0..z.len() {
                    z[k] += step * c * ws.kg[k];
                }
                *logp = -self.potential.value_and_grad(z, grad);
                for g in grad.iter_mut() {
                    *g = -*g;
                }
            }
        }
        ke += self.momentum_update(u, grad, sqrt_inv_mass, step * coefs[coefs.len() - 1], ws);
        ke
    }

    /// One Metropolis-adjusted trajectory. Returns (accept_prob, energy_change,
    /// num_steps, diverging).
    fn transition(&self, st: &mut MamsState, rng: &mut ChainRng) -> (f64, f64, usize, bool) {
        let d = st.z.len();
        // fresh unit momentum
        let mut norm2 = 0.0;
        for i in 0..d {
            let v: f64 = rng.sample(StandardNormal);
            st.u[i] = v;
            norm2 += v * v;
        }
        let norm = norm2.sqrt();
        st.u.iter_mut().for_each(|v| *v /= norm);
        let avg_steps = st.l / st.step_size;
        let num_steps = if self.jitter {
            (0.5 + halton(st.step_index) * rescale(avg_steps)).round()
        } else {
            avg_steps.ceil()
        };
        let num_steps = (num_steps.max(1.0) as usize).min(self.max_num_steps);
        st.step_index += 1;
        st.y.copy_from_slice(&st.z);
        st.y_grad.copy_from_slice(&st.grad);
        let mut logp = st.log_density;
        let mut ke = 0.0;
        for _ in 0..num_steps {
            let (mut y, mut u, mut yg) = (
                std::mem::take(&mut st.y),
                std::mem::take(&mut st.u),
                std::mem::take(&mut st.y_grad),
            );
            ke += self.isokinetic_step(
                st.step_size,
                &mut y,
                &mut u,
                &mut logp,
                &mut yg,
                &st.sqrt_inv_mass,
                &mut st.ws,
            );
            st.y = y;
            st.u = u;
            st.y_grad = yg;
        }
        let mut delta = logp - st.log_density - ke;
        if delta.is_nan() || !st.y.iter().all(|v| v.is_finite()) {
            delta = f64::NEG_INFINITY;
        }
        let accept_prob = delta.min(0.0).exp();
        let diverging = -delta > self.divergence_threshold;
        if rng.random::<f64>() < accept_prob {
            std::mem::swap(&mut st.z, &mut st.y);
            std::mem::swap(&mut st.grad, &mut st.y_grad);
            st.log_density = logp;
        }
        (accept_prob, -delta, num_steps, diverging)
    }
}

/// `(i+1)`-th element of the base-2 van der Corput (Halton) sequence.
fn halton(i: usize) -> f64 {
    let mut n = i + 1;
    let mut x = 0.0;
    let mut f = 0.5;
    while n > 0 {
        if n % 2 == 1 {
            x += f;
        }
        n /= 2;
        f *= 0.5;
    }
    x
}

/// `s` such that `round(U(0,1) * s + 0.5)` has mean `mu`.
fn rescale(mu: f64) -> f64 {
    let k = (2.0 * mu - 1.0).floor();
    let x = k * (mu - 0.5 * (k + 1.0)) / (k + 1.0 - mu);
    k + x
}

#[derive(Clone, Debug, Default)]
struct Ws {
    e: Vec<f64>,
    tmp: Vec<f64>,
    kg: Vec<f64>,
}

/// Per-chain state of [`MAMS`].
#[derive(Clone, Debug)]
pub struct MamsState {
    pub i: usize,
    pub num_warmup: usize,
    pub z: Vec<f64>,
    /// Log density (negative potential) at `z`.
    pub log_density: f64,
    /// Gradient of the log density at `z`.
    pub grad: Vec<f64>,
    pub l: f64,
    pub step_size: f64,
    pub sqrt_inv_mass: Vec<f64>,
    pub accept_prob: f64,
    pub mean_accept_prob: f64,
    pub energy_change: f64,
    pub num_steps: usize,
    pub diverging: bool,
    step_index: usize,
    da: DualAveraging,
    phase1_end: usize,
    // streaming mean / second moment collected in the second half of phase 1
    collect_from: usize,
    sum_w: f64,
    mean: Vec<f64>,
    m2: Vec<f64>,
    u: Vec<f64>,
    y: Vec<f64>,
    y_grad: Vec<f64>,
    ws: Ws,
}

impl<P: Potential> Kernel for MAMS<P> {
    type State = MamsState;

    fn potential(&self) -> &dyn Potential {
        &self.potential
    }

    fn init(&self, z: Vec<f64>, num_warmup: usize, _rng: &mut ChainRng) -> MamsState {
        let d = z.len();
        assert!(
            d >= 2,
            "MAMS needs at least 2 dimensions (the isokinetic dynamics divide by d - 1)"
        );
        let mut grad = vec![0.0; d];
        let pe = self.potential.value_and_grad(&z, &mut grad);
        grad.iter_mut().for_each(|g| *g = -*g);
        let sd = (d as f64).sqrt();
        let l = self.trajectory_length.unwrap_or(sd);
        let step_size = self.step_size.unwrap_or(0.2 * sd);
        let phase1 = (num_warmup as f64 * self.frac_da1).round() as usize;
        MamsState {
            i: 0,
            num_warmup,
            z,
            log_density: -pe,
            grad,
            l,
            step_size,
            sqrt_inv_mass: vec![1.0; d],
            accept_prob: 0.0,
            mean_accept_prob: 0.0,
            energy_change: 0.0,
            num_steps: 0,
            diverging: false,
            step_index: 0,
            da: DualAveraging::new((10.0 * step_size).ln()),
            phase1_end: phase1,
            collect_from: phase1 - phase1 / 2,
            sum_w: 0.0,
            mean: vec![0.0; d],
            m2: vec![0.0; d],
            u: vec![0.0; d],
            y: vec![0.0; d],
            y_grad: vec![0.0; d],
            ws: Ws {
                e: vec![0.0; d],
                tmp: vec![0.0; d],
                kg: vec![0.0; d],
            },
        }
    }

    fn step(&self, st: &mut MamsState, rng: &mut ChainRng) {
        let warmup = st.i < st.num_warmup;
        let (accept_prob, energy_change, num_steps, diverging) = self.transition(st, rng);
        if warmup {
            // dual averaging on log step size, clipped to [1e-10, L]
            st.da.update(self.target_accept_prob - accept_prob);
            let phase1_last = st.i + 1 == st.phase1_end;
            let last = st.i + 1 == st.num_warmup;
            let log_ss = if phase1_last || last {
                st.da.x_avg()
            } else {
                st.da.x()
            };
            st.step_size = log_ss.exp().clamp(1e-10, st.l);
            // collect posterior variances during the second half of phase 1
            if st.i >= st.collect_from && st.i < st.phase1_end {
                st.sum_w += 1.0;
                for k in 0..st.z.len() {
                    st.mean[k] += st.z[k];
                    st.m2[k] += st.z[k] * st.z[k];
                }
            }
            if phase1_last && st.sum_w > 1.0 {
                let d = st.z.len();
                let mut var_sum = 0.0;
                let mut vars = vec![0.0; d];
                for k in 0..d {
                    let m = st.mean[k] / st.sum_w;
                    vars[k] = (st.m2[k] / st.sum_w - m * m).max(1e-10);
                    var_sum += vars[k];
                }
                let new_l = self.tuning_factor * var_sum.sqrt();
                let change = (new_l / st.l).clamp(1.0 / L_RATIO_MAX, L_RATIO_MAX);
                st.l *= change;
                st.step_size *= change;
                if self.preconditioning {
                    for k in 0..d {
                        st.sqrt_inv_mass[k] = vars[k].sqrt();
                    }
                    st.l = self.tuning_factor * (d as f64).sqrt();
                }
                // phase 2: restart dual averaging around the new step size
                st.da = DualAveraging::new((10.0 * st.step_size).ln());
            }
        }
        st.i += 1;
        let n = if warmup { st.i } else { st.i - st.num_warmup };
        st.mean_accept_prob += (accept_prob - st.mean_accept_prob) / n as f64;
        st.accept_prob = accept_prob;
        st.energy_change = energy_change;
        st.num_steps = num_steps;
        st.diverging = diverging;
    }

    fn position(st: &MamsState) -> &[f64] {
        &st.z
    }

    fn stat_names(&self) -> &'static [&'static str] {
        &[
            "accept_prob",
            "step_size",
            "num_steps",
            "diverging",
            "energy_change",
            "potential_energy",
            "mean_accept_prob",
            "trajectory_length",
        ]
    }

    fn stats(&self, st: &MamsState, out: &mut [f64]) {
        out[0] = st.accept_prob;
        out[1] = st.step_size;
        out[2] = st.num_steps as f64;
        out[3] = if st.diverging { 1.0 } else { 0.0 };
        out[4] = st.energy_change;
        out[5] = -st.log_density;
        out[6] = st.mean_accept_prob;
        out[7] = st.l;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_and_rescale() {
        // van der Corput base 2: 1/2, 1/4, 3/4, 1/8, 5/8, ...
        let h: Vec<f64> = (0..5).map(halton).collect();
        assert_eq!(h, vec![0.5, 0.25, 0.75, 0.125, 0.625]);
        // round(U * s + 0.5) has mean mu: check by quadrature over a grid
        for &mu in &[1.5, 3.7, 10.2] {
            let s = rescale(mu);
            let n = 100_000;
            let m: f64 = (0..n)
                .map(|i| ((i as f64 + 0.5) / n as f64 * s + 0.5).round())
                .sum::<f64>()
                / n as f64;
            assert!((m - mu).abs() < 1e-3, "mu={mu}: {m}");
        }
    }

    #[test]
    fn isokinetic_step_conserves_energy_on_gaussian() {
        use crate::infer::FnPotential;
        use crate::Var;
        let d = 4;
        let pot = FnPotential::new(d, |z: &[Var]| z.iter().map(|x| *x * *x * 0.5).sum::<Var>());
        let k = MAMS::new(&pot);
        let mut rng = rand::SeedableRng::seed_from_u64(0);
        let mut st = k.init(vec![0.3, -1.0, 0.5, 2.0], 0, &mut rng);
        let mut u = vec![0.5, 0.5, 0.5, 0.5];
        let mut z = st.z.clone();
        let mut grad = st.grad.clone();
        let mut logp = st.log_density;
        let mut ke = 0.0;
        let sqrt_inv = vec![1.0; d];
        for _ in 0..200 {
            ke += k.isokinetic_step(0.05, &mut z, &mut u, &mut logp, &mut grad, &sqrt_inv, &mut st.ws);
            let n: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-12, "momentum stays on the sphere");
        }
        // energy error: logp change should balance the kinetic change
        let delta = logp - st.log_density - ke;
        assert!(delta.abs() < 0.05, "energy drift {delta}");
        assert!((z[0] - st.z[0]).abs() > 0.1, "position moved");
    }
}
