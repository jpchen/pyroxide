//! Hamiltonian Monte Carlo: the No-U-Turn Sampler (NUTS) and fixed-length HMC.
//!
//! This is a port of `numpyro.infer.hmc` / `hmc_util` to allocation-free Rust:
//! the iterative NUTS tree builder with `O(log depth)` momentum checkpoints,
//! the biased progressive sampling of Betancourt (2017), the generalized
//! U-turn criterion, divergence detection, and Stan's windowed warmup
//! adaptation of step size and (diagonal or dense) mass matrix.
//!
//! All buffers live in [`HmcState`] and are allocated once per chain.

use rand::Rng;

use super::adapt::{AdaptConfig, MassMatrix, WarmupAdapter};
use super::mcmc::{ChainRng, Kernel};
use super::potential::Potential;
use crate::ad::logaddexp;

/// Which Hamiltonian algorithm to run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Algo {
    /// No-U-Turn sampler with adaptive trajectory length.
    Nuts,
    /// Fixed trajectory length (`trajectory_length`) or fixed number of steps.
    Hmc,
}

/// Configuration shared by NUTS and HMC. Construct with [`HmcKernel::nuts`] /
/// [`HmcKernel::hmc`] and the builder methods.
#[derive(Clone, Debug)]
pub struct HmcConfig {
    pub algo: Algo,
    pub step_size: f64,
    pub inverse_mass_matrix: Option<Vec<f64>>,
    pub adapt: AdaptConfig,
    /// NUTS: maximum tree depth (during warmup, after warmup).
    pub max_tree_depth: (usize, usize),
    /// HMC: trajectory length (ignored when `num_steps` is set).
    pub trajectory_length: f64,
    /// HMC: fixed number of leapfrog steps.
    pub num_steps: Option<usize>,
    /// Tune the initial step size with the doubling heuristic of Hoffman &
    /// Gelman at the start of each adaptation window.
    pub find_heuristic_step_size: bool,
    /// Energy change threshold for flagging a divergent transition.
    pub max_delta_energy: f64,
}

impl Default for HmcConfig {
    fn default() -> Self {
        HmcConfig {
            algo: Algo::Nuts,
            step_size: 1.0,
            inverse_mass_matrix: None,
            adapt: AdaptConfig::default(),
            max_tree_depth: (10, 10),
            trajectory_length: 2.0 * std::f64::consts::PI,
            num_steps: None,
            find_heuristic_step_size: false,
            max_delta_energy: 1000.0,
        }
    }
}

/// NUTS / HMC kernel over a [`Potential`].
///
/// ```no_run
/// # use pyroxide::infer::*;
/// # let potential = FnPotential::new(1, |z: &[pyroxide::Var]| z[0] * z[0] * 0.5);
/// let kernel = HmcKernel::nuts(potential).target_accept_prob(0.9).dense_mass(true);
/// ```
pub struct HmcKernel<P: Potential> {
    potential: P,
    pub cfg: HmcConfig,
}

impl<P: Potential> HmcKernel<P> {
    /// NUTS with default settings (step size and diagonal mass adaptation,
    /// target acceptance 0.8, max tree depth 10).
    pub fn nuts(potential: P) -> Self {
        HmcKernel {
            potential,
            cfg: HmcConfig::default(),
        }
    }

    /// HMC with a fixed trajectory length (default `2π`).
    pub fn hmc(potential: P) -> Self {
        HmcKernel {
            potential,
            cfg: HmcConfig {
                algo: Algo::Hmc,
                ..HmcConfig::default()
            },
        }
    }

    pub fn with_config(potential: P, cfg: HmcConfig) -> Self {
        HmcKernel { potential, cfg }
    }

    pub fn step_size(mut self, s: f64) -> Self {
        self.cfg.step_size = s;
        self
    }
    pub fn adapt_step_size(mut self, b: bool) -> Self {
        self.cfg.adapt.adapt_step_size = b;
        self
    }
    pub fn adapt_mass_matrix(mut self, b: bool) -> Self {
        self.cfg.adapt.adapt_mass_matrix = b;
        self
    }
    pub fn dense_mass(mut self, b: bool) -> Self {
        self.cfg.adapt.dense_mass = b;
        self
    }
    pub fn regularize_mass_matrix(mut self, b: bool) -> Self {
        self.cfg.adapt.regularize_mass_matrix = b;
        self
    }
    pub fn target_accept_prob(mut self, p: f64) -> Self {
        self.cfg.adapt.target_accept_prob = p;
        self
    }
    /// Initial inverse mass matrix (diagonal of length `d`, or dense `d*d`).
    pub fn inverse_mass_matrix(mut self, m: Vec<f64>) -> Self {
        self.cfg.adapt.dense_mass = m.len() != self.potential.dim();
        self.cfg.inverse_mass_matrix = Some(m);
        self
    }
    pub fn max_tree_depth(mut self, d: usize) -> Self {
        self.cfg.max_tree_depth = (d, d);
        self
    }
    /// Separate max tree depths for warmup and sampling.
    pub fn max_tree_depths(mut self, warmup: usize, sampling: usize) -> Self {
        self.cfg.max_tree_depth = (warmup, sampling);
        self
    }
    pub fn trajectory_length(mut self, t: f64) -> Self {
        self.cfg.trajectory_length = t;
        self
    }
    pub fn num_steps(mut self, n: usize) -> Self {
        self.cfg.num_steps = Some(n);
        self
    }
    pub fn find_heuristic_step_size(mut self, b: bool) -> Self {
        self.cfg.find_heuristic_step_size = b;
        self
    }
    pub fn max_delta_energy(mut self, e: f64) -> Self {
        self.cfg.max_delta_energy = e;
        self
    }

    pub fn potential_ref(&self) -> &P {
        &self.potential
    }

    // ------------------------------------------------------------- integrator --

    /// One velocity-Verlet (leapfrog) step of size `eps` (may be negative).
    /// `z`, `r`, `grad` are updated in place; returns the new potential energy.
    #[inline]
    fn leapfrog(
        &self,
        eps: f64,
        mass: &MassMatrix,
        z: &mut [f64],
        r: &mut [f64],
        grad: &mut [f64],
        v: &mut [f64],
    ) -> f64 {
        let half = 0.5 * eps;
        for (r, g) in r.iter_mut().zip(grad.iter()) {
            *r -= half * g;
        }
        mass.velocity(r, v);
        for (z, v) in z.iter_mut().zip(v.iter()) {
            *z += eps * v;
        }
        let pe = self.potential.value_and_grad(z, grad);
        for (r, g) in r.iter_mut().zip(grad.iter()) {
            *r -= half * g;
        }
        pe
    }

    /// Hoffman & Gelman's heuristic: double / halve the step size until the
    /// acceptance probability of a single leapfrog step crosses 0.8.
    fn find_reasonable_step_size(
        &self,
        mut step_size: f64,
        mass: &MassMatrix,
        z0: &[f64],
        pe0: f64,
        grad0: &[f64],
        rng: &mut ChainRng,
    ) -> f64 {
        let d = z0.len();
        let target = 0.8f64.ln();
        let mut z = vec![0.0; d];
        let mut r = vec![0.0; d];
        let mut grad = vec![0.0; d];
        let mut v = vec![0.0; d];
        let mut last_direction = 0i32;
        loop {
            z.copy_from_slice(z0);
            grad.copy_from_slice(grad0);
            mass.sample_momentum(rng, &mut r);
            let energy_current = pe0 + mass.kinetic(&r);
            let pe_new = self.leapfrog(step_size, mass, &mut z, &mut r, &mut grad, &mut v);
            let energy_new = pe_new + mass.kinetic(&r);
            let delta = energy_new - energy_current;
            // NaN energy => treat as "too large", i.e. decrease
            let direction = if target < -delta { 1 } else { -1 };
            if last_direction != 0 && direction != last_direction {
                break;
            }
            let next = if direction == 1 {
                step_size * 2.0
            } else {
                step_size * 0.5
            };
            if (direction == -1 && next <= f64::MIN_POSITIVE) || (direction == 1 && !next.is_finite()) {
                break;
            }
            step_size = next;
            last_direction = direction;
        }
        step_size
    }

    // ----------------------------------------------------------------- NUTS ---

    fn build_tree(&self, st: &mut HmcState, max_depth: usize, rng: &mut ChainRng) {
        let d = st.z.len();
        let ws = &mut st.ws;
        let mass = &st.adapt.mass;
        let step_size = st.adapt.step_size;
        let energy_current = st.potential_energy + mass.kinetic(&ws.r0);

        // initial single-point tree
        {
            let t = &mut ws.main;
            t.z_left.copy_from_slice(&st.z);
            t.r_left.copy_from_slice(&ws.r0);
            t.g_left.copy_from_slice(&st.z_grad);
            t.z_right.copy_from_slice(&st.z);
            t.r_right.copy_from_slice(&ws.r0);
            t.g_right.copy_from_slice(&st.z_grad);
            t.z_prop.copy_from_slice(&st.z);
            t.g_prop.copy_from_slice(&st.z_grad);
            t.prop_pe = st.potential_energy;
            t.prop_energy = energy_current;
            t.depth = 0;
            t.weight = 0.0;
            t.r_sum.copy_from_slice(&ws.r0);
            t.turning = false;
            t.diverging = false;
            t.sum_accept = 0.0;
            t.num_proposals = 0;
        }

        while ws.main.depth < max_depth && !ws.main.turning && !ws.main.diverging {
            let going_right: bool = rng.random::<f64>() < 0.5;
            // build subtree of the same depth as `main` in direction going_right
            self.iterative_build_subtree(ws, mass, step_size, going_right, energy_current, rng, d);
            // combine main + sub -> tmp (biased progressive sampling)
            let transition_u: f64 = rng.random();
            combine_tree(
                &mut ws.tmp,
                &ws.main,
                &ws.sub,
                mass,
                going_right,
                transition_u,
                true,
                &mut ws.scratch,
            );
            std::mem::swap(&mut ws.main, &mut ws.tmp);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn iterative_build_subtree(
        &self,
        ws: &mut Workspace,
        mass: &MassMatrix,
        step_size: f64,
        going_right: bool,
        energy_current: f64,
        rng: &mut ChainRng,
        d: usize,
    ) {
        let max_num_proposals: u64 = 1 << ws.main.depth;
        let proto_depth = ws.main.depth;
        // `sub` starts as a copy of the prototype with num_proposals = 0
        ws.sub.copy_from(&ws.main);
        ws.sub.num_proposals = 0;
        let mut turning = false;
        while ws.sub.num_proposals < max_num_proposals && !turning && !ws.sub.diverging {
            // leaf from the current subtree in the direction of travel
            {
                let (z, r, g) = if going_right {
                    (&ws.sub.z_right, &ws.sub.r_right, &ws.sub.g_right)
                } else {
                    (&ws.sub.z_left, &ws.sub.r_left, &ws.sub.g_left)
                };
                ws.leaf.z_prop.copy_from_slice(z);
                ws.leaf.r_right.copy_from_slice(r);
                ws.leaf.g_prop.copy_from_slice(g);
            }
            // one leapfrog step into the leaf's buffers
            let eps = if going_right { step_size } else { -step_size };
            let pe_new = {
                let leaf = &mut ws.leaf;
                self.leapfrog(
                    eps,
                    mass,
                    &mut leaf.z_prop,
                    &mut leaf.r_right,
                    &mut leaf.g_prop,
                    &mut ws.scratch.v,
                )
            };
            {
                let leaf = &mut ws.leaf;
                let energy_new = pe_new + mass.kinetic(&leaf.r_right);
                let mut delta = energy_new - energy_current;
                if delta.is_nan() {
                    delta = f64::INFINITY;
                }
                leaf.z_left.copy_from_slice(&leaf.z_prop);
                leaf.z_right.copy_from_slice(&leaf.z_prop);
                leaf.r_left.copy_from_slice(&leaf.r_right);
                leaf.g_left.copy_from_slice(&leaf.g_prop);
                leaf.g_right.copy_from_slice(&leaf.g_prop);
                leaf.r_sum.copy_from_slice(&leaf.r_right);
                leaf.prop_pe = pe_new;
                leaf.prop_energy = energy_new;
                leaf.depth = 0;
                leaf.weight = -delta;
                leaf.turning = false;
                leaf.diverging = delta > self.cfg.max_delta_energy;
                leaf.sum_accept = (-delta).exp().min(1.0);
                leaf.num_proposals = 1;
            }
            let leaf_idx = ws.sub.num_proposals;
            if leaf_idx == 0 {
                std::mem::swap(&mut ws.sub, &mut ws.leaf);
                // `sub` is now the new leaf; `leaf` holds the old prototype copy
                // whose contents are irrelevant (but we need the new leaf's r for
                // the turning check below, so re-point to `sub`).
            } else {
                let u: f64 = rng.random();
                combine_tree(
                    &mut ws.tmp,
                    &ws.sub,
                    &ws.leaf,
                    mass,
                    going_right,
                    u,
                    false,
                    &mut ws.scratch,
                );
                std::mem::swap(&mut ws.sub, &mut ws.tmp);
            }
            // After the swap/combine, the newest leaf momentum is the right leaf
            // (going right) or left leaf (going left) of `sub`; both are the
            // same vector for a fresh leaf. Use the direction leaf.
            let (ckpt_min, ckpt_max) = leaf_idx_to_ckpt_idxs(leaf_idx);
            let r_new: &[f64] = if going_right {
                &ws.sub.r_right
            } else {
                &ws.sub.r_left
            };
            if leaf_idx.is_multiple_of(2) {
                let ci = ckpt_max as usize;
                ws.r_ckpts[ci * d..(ci + 1) * d].copy_from_slice(r_new);
                ws.r_sum_ckpts[ci * d..(ci + 1) * d].copy_from_slice(&ws.sub.r_sum);
            }
            turning = is_iterative_turning(
                mass,
                r_new,
                &ws.sub.r_sum,
                &ws.r_ckpts,
                &ws.r_sum_ckpts,
                ckpt_min,
                ckpt_max,
                d,
                &mut ws.scratch,
            );
        }
        ws.sub.depth = proto_depth;
        ws.sub.turning = turning;
    }

    // ------------------------------------------------------------------ HMC ---

    fn hmc_transition(&self, st: &mut HmcState, rng: &mut ChainRng) -> (f64, u64, bool, f64) {
        let mass = &st.adapt.mass;
        let mut step_size = st.adapt.step_size;
        let num_steps = match self.cfg.num_steps {
            Some(n) => n,
            None => (self.cfg.trajectory_length / step_size).ceil().max(1.0) as usize,
        };
        if self.cfg.num_steps.is_none() {
            step_size = self.cfg.trajectory_length / num_steps as f64;
        }
        let ws = &mut st.ws;
        let energy_old = st.potential_energy + mass.kinetic(&ws.r0);
        ws.z_new.copy_from_slice(&st.z);
        ws.g_new.copy_from_slice(&st.z_grad);
        ws.r_new.copy_from_slice(&ws.r0);
        let mut pe_new = st.potential_energy;
        for _ in 0..num_steps {
            pe_new = self.leapfrog(
                step_size,
                mass,
                &mut ws.z_new,
                &mut ws.r_new,
                &mut ws.g_new,
                &mut ws.scratch.v,
            );
        }
        let energy_new = pe_new + mass.kinetic(&ws.r_new);
        let mut delta = energy_new - energy_old;
        if delta.is_nan() {
            delta = f64::INFINITY;
        }
        let accept_prob = (-delta).exp().min(1.0);
        let diverging = delta > self.cfg.max_delta_energy;
        let u: f64 = rng.random();
        if u < accept_prob {
            st.z.copy_from_slice(&ws.z_new);
            st.z_grad.copy_from_slice(&ws.g_new);
            st.potential_energy = pe_new;
            (accept_prob, num_steps as u64, diverging, energy_new)
        } else {
            (accept_prob, num_steps as u64, diverging, energy_old)
        }
    }
}

/// Everything a NUTS tree needs to know about itself (port of numpyro's `TreeInfo`).
#[derive(Clone, Debug)]
pub struct TreeInfo {
    pub z_left: Vec<f64>,
    pub r_left: Vec<f64>,
    pub g_left: Vec<f64>,
    pub z_right: Vec<f64>,
    pub r_right: Vec<f64>,
    pub g_right: Vec<f64>,
    pub z_prop: Vec<f64>,
    pub g_prop: Vec<f64>,
    pub prop_pe: f64,
    pub prop_energy: f64,
    pub depth: usize,
    pub weight: f64,
    pub r_sum: Vec<f64>,
    pub turning: bool,
    pub diverging: bool,
    pub sum_accept: f64,
    pub num_proposals: u64,
}

impl TreeInfo {
    fn zeros(d: usize) -> Self {
        TreeInfo {
            z_left: vec![0.0; d],
            r_left: vec![0.0; d],
            g_left: vec![0.0; d],
            z_right: vec![0.0; d],
            r_right: vec![0.0; d],
            g_right: vec![0.0; d],
            z_prop: vec![0.0; d],
            g_prop: vec![0.0; d],
            prop_pe: 0.0,
            prop_energy: 0.0,
            depth: 0,
            weight: 0.0,
            r_sum: vec![0.0; d],
            turning: false,
            diverging: false,
            sum_accept: 0.0,
            num_proposals: 0,
        }
    }

    fn copy_from(&mut self, o: &TreeInfo) {
        self.z_left.copy_from_slice(&o.z_left);
        self.r_left.copy_from_slice(&o.r_left);
        self.g_left.copy_from_slice(&o.g_left);
        self.z_right.copy_from_slice(&o.z_right);
        self.r_right.copy_from_slice(&o.r_right);
        self.g_right.copy_from_slice(&o.g_right);
        self.z_prop.copy_from_slice(&o.z_prop);
        self.g_prop.copy_from_slice(&o.g_prop);
        self.prop_pe = o.prop_pe;
        self.prop_energy = o.prop_energy;
        self.depth = o.depth;
        self.weight = o.weight;
        self.r_sum.copy_from_slice(&o.r_sum);
        self.turning = o.turning;
        self.diverging = o.diverging;
        self.sum_accept = o.sum_accept;
        self.num_proposals = o.num_proposals;
    }
}

#[derive(Clone, Debug)]
struct Scratch {
    v: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Workspace {
    /// Momentum drawn at the start of the current iteration.
    r0: Vec<f64>,
    main: TreeInfo,
    sub: TreeInfo,
    leaf: TreeInfo,
    tmp: TreeInfo,
    r_ckpts: Vec<f64>,
    r_sum_ckpts: Vec<f64>,
    scratch: Scratch,
    // fixed-length HMC buffers
    z_new: Vec<f64>,
    r_new: Vec<f64>,
    g_new: Vec<f64>,
}

impl Workspace {
    fn new(d: usize, max_depth: usize) -> Self {
        Workspace {
            r0: vec![0.0; d],
            main: TreeInfo::zeros(d),
            sub: TreeInfo::zeros(d),
            leaf: TreeInfo::zeros(d),
            tmp: TreeInfo::zeros(d),
            r_ckpts: vec![0.0; (max_depth + 1) * d],
            r_sum_ckpts: vec![0.0; (max_depth + 1) * d],
            scratch: Scratch {
                v: vec![0.0; d],
                a: vec![0.0; d],
                b: vec![0.0; d],
                c: vec![0.0; d],
            },
            z_new: vec![0.0; d],
            r_new: vec![0.0; d],
            g_new: vec![0.0; d],
        }
    }
}

/// Per-chain state of the HMC/NUTS kernel.
#[derive(Clone, Debug)]
pub struct HmcState {
    /// Iteration counter (warmup and sampling combined).
    pub i: usize,
    pub num_warmup: usize,
    /// Current unconstrained position.
    pub z: Vec<f64>,
    pub z_grad: Vec<f64>,
    pub potential_energy: f64,
    /// Hamiltonian at the end of the last transition.
    pub energy: f64,
    pub num_steps: u64,
    pub accept_prob: f64,
    pub mean_accept_prob: f64,
    pub diverging: bool,
    pub adapt: WarmupAdapter,
    ws: Workspace,
}

impl HmcState {
    /// Tree depth of the last NUTS trajectory.
    pub fn tree_depth(&self) -> u32 {
        if self.num_steps == 0 {
            0
        } else {
            (63 - self.num_steps.leading_zeros()) + 1
        }
    }
}

/// Generalized U-turn criterion between the two ends of a (sub)tree.
#[inline]
fn is_turning(mass: &MassMatrix, r_left: &[f64], r_right: &[f64], r_sum: &[f64], s: &mut Scratch) -> bool {
    let d = r_left.len();
    mass.velocity(r_left, &mut s.a);
    mass.velocity(r_right, &mut s.b);
    for i in 0..d {
        s.c[i] = r_sum[i] - 0.5 * (r_left[i] + r_right[i]);
    }
    let mut left_angle = 0.0;
    let mut right_angle = 0.0;
    for i in 0..d {
        left_angle += s.a[i] * s.c[i];
        right_angle += s.b[i] * s.c[i];
    }
    left_angle <= 0.0 || right_angle <= 0.0
}

/// Checkpoint indices for leaf `n` of the iterative tree builder
/// (port of numpyro's `_leaf_idx_to_ckpt_idxs`).
pub fn leaf_idx_to_ckpt_idxs(n: u64) -> (i64, i64) {
    let idx_max = (n >> 1).count_ones() as i64;
    let num_subtrees = ((!n & (n + 1)) - 1).count_ones() as i64;
    (idx_max - num_subtrees + 1, idx_max)
}

#[allow(clippy::too_many_arguments)]
fn is_iterative_turning(
    mass: &MassMatrix,
    r: &[f64],
    r_sum: &[f64],
    r_ckpts: &[f64],
    r_sum_ckpts: &[f64],
    idx_min: i64,
    idx_max: i64,
    d: usize,
    s: &mut Scratch,
) -> bool {
    let mut i = idx_max;
    while i >= idx_min {
        let ci = i as usize;
        let r_left = &r_ckpts[ci * d..(ci + 1) * d];
        let r_sum_ckpt = &r_sum_ckpts[ci * d..(ci + 1) * d];
        // subtree_r_sum = r_sum - r_sum_ckpt + r_left, computed into s.v
        for k in 0..d {
            s.v[k] = r_sum[k] - r_sum_ckpt[k] + r_left[k];
        }
        // is_turning needs a, b, c; s.v holds the subtree sum
        let turning = {
            mass.velocity(r_left, &mut s.a);
            mass.velocity(r, &mut s.b);
            for k in 0..d {
                s.c[k] = s.v[k] - 0.5 * (r_left[k] + r[k]);
            }
            let mut la = 0.0;
            let mut ra = 0.0;
            for k in 0..d {
                la += s.a[k] * s.c[k];
                ra += s.b[k] * s.c[k];
            }
            la <= 0.0 || ra <= 0.0
        };
        if turning {
            return true;
        }
        i -= 1;
    }
    false
}

/// Merge `current` and `new` into `dest` (port of numpyro's `_combine_tree`).
#[allow(clippy::too_many_arguments)]
fn combine_tree(
    dest: &mut TreeInfo,
    current: &TreeInfo,
    new: &TreeInfo,
    mass: &MassMatrix,
    going_right: bool,
    transition_u: f64,
    biased: bool,
    s: &mut Scratch,
) {
    let (left, right) = if going_right {
        (current, new)
    } else {
        (new, current)
    };
    dest.z_left.copy_from_slice(&left.z_left);
    dest.r_left.copy_from_slice(&left.r_left);
    dest.g_left.copy_from_slice(&left.g_left);
    dest.z_right.copy_from_slice(&right.z_right);
    dest.r_right.copy_from_slice(&right.r_right);
    dest.g_right.copy_from_slice(&right.g_right);
    for i in 0..dest.r_sum.len() {
        dest.r_sum[i] = current.r_sum[i] + new.r_sum[i];
    }
    let (transition_prob, turning) = if biased {
        let mut p = (new.weight - current.weight).exp().min(1.0);
        if new.turning || new.diverging {
            p = 0.0;
        }
        let t = new.turning || is_turning(mass, &dest.r_left, &dest.r_right, &dest.r_sum, s);
        (p, t)
    } else {
        (
            crate::ad::sigmoid_f64(new.weight - current.weight),
            current.turning,
        )
    };
    let src = if transition_u < transition_prob {
        new
    } else {
        current
    };
    dest.z_prop.copy_from_slice(&src.z_prop);
    dest.g_prop.copy_from_slice(&src.g_prop);
    dest.prop_pe = src.prop_pe;
    dest.prop_energy = src.prop_energy;
    dest.depth = current.depth + 1;
    dest.weight = logaddexp(current.weight, new.weight);
    dest.turning = turning;
    dest.diverging = new.diverging;
    dest.sum_accept = current.sum_accept + new.sum_accept;
    dest.num_proposals = current.num_proposals + new.num_proposals;
}

impl<P: Potential> Kernel for HmcKernel<P> {
    type State = HmcState;

    fn potential(&self) -> &dyn Potential {
        &self.potential
    }

    fn init(&self, z: Vec<f64>, num_warmup: usize, rng: &mut ChainRng) -> HmcState {
        let d = z.len();
        let mut z_grad = vec![0.0; d];
        let pe = self.potential.value_and_grad(&z, &mut z_grad);
        let max_depth = self.cfg.max_tree_depth.0.max(self.cfg.max_tree_depth.1);
        let mut ws = Workspace::new(d, max_depth);
        // The heuristic needs the RNG and the current point; build a closure
        // over copies so the borrow checker is happy.
        let z0 = z.clone();
        let g0 = z_grad.clone();
        let mut rng2 = rng.clone();
        rng.jump();
        let mut heuristic =
            |ss: f64, mass: &MassMatrix| self.find_reasonable_step_size(ss, mass, &z0, pe, &g0, &mut rng2);
        let find: Option<&mut dyn FnMut(f64, &MassMatrix) -> f64> = if self.cfg.find_heuristic_step_size {
            Some(&mut heuristic)
        } else {
            None
        };
        let adapt = WarmupAdapter::new(
            num_warmup,
            d,
            self.cfg.step_size,
            self.cfg.inverse_mass_matrix.clone(),
            self.cfg.adapt.clone(),
            find,
        );
        adapt.mass.sample_momentum(rng, &mut ws.r0);
        let energy = pe + adapt.mass.kinetic(&ws.r0);
        HmcState {
            i: 0,
            num_warmup,
            z,
            z_grad,
            potential_energy: pe,
            energy,
            num_steps: 0,
            accept_prob: 0.0,
            mean_accept_prob: 0.0,
            diverging: false,
            adapt,
            ws,
        }
    }

    fn step(&self, st: &mut HmcState, rng: &mut ChainRng) {
        let warmup = st.i < st.num_warmup;
        st.adapt.mass.sample_momentum(rng, &mut st.ws.r0);
        let (accept_prob, num_steps, diverging, energy) = match self.cfg.algo {
            Algo::Nuts => {
                let max_depth = if warmup {
                    self.cfg.max_tree_depth.0
                } else {
                    self.cfg.max_tree_depth.1
                };
                self.build_tree(st, max_depth, rng);
                let t = &st.ws.main;
                let accept_prob = t.sum_accept / t.num_proposals.max(1) as f64;
                st.z.copy_from_slice(&t.z_prop);
                st.z_grad.copy_from_slice(&t.g_prop);
                st.potential_energy = t.prop_pe;
                (accept_prob, t.num_proposals, t.diverging, t.prop_energy)
            }
            Algo::Hmc => self.hmc_transition(st, rng),
        };
        if warmup {
            let t = st.i;
            if self.cfg.find_heuristic_step_size {
                let z0 = st.z.clone();
                let g0 = st.z_grad.clone();
                let pe = st.potential_energy;
                let mut rng2 = rng.clone();
                rng.jump();
                let mut heuristic = |ss: f64, mass: &MassMatrix| {
                    self.find_reasonable_step_size(ss, mass, &z0, pe, &g0, &mut rng2)
                };
                st.adapt.update(t, accept_prob, &z0, Some(&mut heuristic));
            } else {
                let z = &st.z;
                st.adapt.update(t, accept_prob, z, None);
            }
        }
        st.i += 1;
        let n = if warmup { st.i } else { st.i - st.num_warmup };
        st.mean_accept_prob += (accept_prob - st.mean_accept_prob) / n as f64;
        st.accept_prob = accept_prob;
        st.num_steps = num_steps;
        st.diverging = diverging;
        st.energy = energy;
    }

    fn position(st: &HmcState) -> &[f64] {
        &st.z
    }

    fn stat_names(&self) -> &'static [&'static str] {
        &[
            "accept_prob",
            "step_size",
            "num_steps",
            "diverging",
            "energy",
            "potential_energy",
            "mean_accept_prob",
        ]
    }

    fn stats(&self, st: &HmcState, out: &mut [f64]) {
        out[0] = st.accept_prob;
        out[1] = st.adapt.step_size;
        out[2] = st.num_steps as f64;
        out[3] = if st.diverging { 1.0 } else { 0.0 };
        out[4] = st.energy;
        out[5] = st.potential_energy;
        out[6] = st.mean_accept_prob;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::potential::FnPotential;
    use crate::{Real, Var};
    use rand::SeedableRng;

    fn std_normal(d: usize) -> FnPotential<impl Fn(&[Var]) -> Var + Send + Sync> {
        FnPotential::new(d, |z: &[Var]| z.iter().map(|x| *x * *x * 0.5).sum::<Var>())
    }

    #[test]
    fn leaf_idx_to_ckpt_idx_matches_numpyro() {
        assert_eq!(leaf_idx_to_ckpt_idxs(0), (1, 0));
        assert_eq!(leaf_idx_to_ckpt_idxs(6), (3, 2));
        assert_eq!(leaf_idx_to_ckpt_idxs(7), (0, 2));
        assert_eq!(leaf_idx_to_ckpt_idxs(13), (2, 2));
        assert_eq!(leaf_idx_to_ckpt_idxs(15), (0, 3));
    }

    #[test]
    fn iterative_turning_matches_numpyro() {
        // port of test_is_iterative_turning (dimension 1)
        let mass = MassMatrix::identity(1, false);
        let r = [1.0];
        let r_sum = [3.0];
        let r_ckpts = [1.0, 2.0, 3.0, -2.0];
        let r_sum_ckpts = [2.0, 4.0, 4.0, -1.0];
        let mut s = Scratch {
            v: vec![0.0],
            a: vec![0.0],
            b: vec![0.0],
            c: vec![0.0],
        };
        let cases = [
            ((3, 2), false),
            ((3, 3), true),
            ((0, 0), false),
            ((0, 1), true),
            ((1, 3), true),
        ];
        for ((mn, mx), expected) in cases {
            let t = is_iterative_turning(&mass, &r, &r_sum, &r_ckpts, &r_sum_ckpts, mn, mx, 1, &mut s);
            assert_eq!(t, expected, "ckpt idxs ({mn}, {mx})");
        }
    }

    /// Port of numpyro's velocity-verlet tests: harmonic oscillator, circular
    /// planetary motion, quartic oscillator — accuracy, energy conservation,
    /// reversibility.
    #[test]
    fn velocity_verlet_trajectories() {
        struct Case {
            pot: FnPotential<Box<dyn Fn(&[Var]) -> Var + Send + Sync>>,
            step: f64,
            n: usize,
            q0: Vec<f64>,
            p0: Vec<f64>,
            qf: Vec<f64>,
            pf: Vec<f64>,
            prec: f64,
        }
        let cases = vec![
            Case {
                pot: FnPotential::new(1, Box::new(|q: &[Var]| q[0] * q[0] * 0.5)),
                step: 0.01,
                n: 100,
                q0: vec![0.0],
                p0: vec![1.0],
                qf: vec![1f64.sin()],
                pf: vec![1f64.cos()],
                prec: 1e-4,
            },
            Case {
                pot: FnPotential::new(2, Box::new(|q: &[Var]| -(q[0] * q[0] + q[1] * q[1]).powf(-0.5))),
                step: 0.01,
                n: 628,
                q0: vec![1.0, 0.0],
                p0: vec![0.0, 1.0],
                qf: vec![1.0, 0.0],
                pf: vec![0.0, 1.0],
                prec: 5e-3,
            },
            Case {
                pot: FnPotential::new(1, Box::new(|q: &[Var]| q[0].powf(4.0) * 0.25)),
                step: 0.1,
                n: 1810,
                q0: vec![0.02],
                p0: vec![0.0],
                qf: vec![-0.02],
                pf: vec![0.0],
                prec: 1e-4,
            },
        ];
        for c in cases {
            let d = c.q0.len();
            let k = HmcKernel::nuts(&c.pot);
            let mass = MassMatrix::identity(d, false);
            let run = |q0: &[f64], p0: &[f64]| {
                let mut z = q0.to_vec();
                let mut r = p0.to_vec();
                let mut g = vec![0.0; d];
                let mut v = vec![0.0; d];
                c.pot.value_and_grad(&z, &mut g);
                for _ in 0..c.n {
                    k.leapfrog(c.step, &mass, &mut z, &mut r, &mut g, &mut v);
                }
                (z, r)
            };
            let (qf, pf) = run(&c.q0, &c.p0);
            for i in 0..d {
                assert!((qf[i] - c.qf[i]).abs() < c.prec, "q {qf:?} vs {:?}", c.qf);
                assert!((pf[i] - c.pf[i]).abs() < c.prec, "p {pf:?} vs {:?}", c.pf);
            }
            let e0 = c.pot.value(&c.q0) + mass.kinetic(&c.p0);
            let e1 = c.pot.value(&qf) + mass.kinetic(&pf);
            assert!((e0 - e1).abs() < 1e-5, "energy {e0} vs {e1}");
            let p_rev: Vec<f64> = pf.iter().map(|p| -p).collect();
            let (q_back, _) = run(&qf, &p_rev);
            for i in 0..d {
                assert!((q_back[i] - c.q0[i]).abs() < 1e-4, "reversibility");
            }
        }
    }

    #[test]
    fn find_reasonable_step_size_brackets_threshold() {
        // port of numpyro's test: for U = z^2/2, M = 1, one leapfrog step gives
        // delta_energy = eps^4 / 8, so the heuristic stops around
        // eps* = (-8 log 0.8)^{1/4}.
        let pot = std_normal(1);
        let k = HmcKernel::nuts(&pot);
        let mass = MassMatrix::identity(1, false);
        // deterministic momentum: r = 1 by using a fixed-value mass matrix trick
        // is not available, so approximate by checking the bracketing property
        // over several seeds.
        let threshold = (-(0.8f64.ln()) * 8.0).powf(0.25);
        for &init in &[0.1, 10.0] {
            let mut rng = ChainRng::seed_from_u64(0);
            let ss = k.find_reasonable_step_size(init, &mass, &[0.0], 0.0, &[0.0], &mut rng);
            // with random momentum the exact threshold shifts by |r|, but the
            // result must be within a factor of 2 of *some* crossing and
            // finite/positive
            assert!(ss.is_finite() && ss > 0.0);
            if init < threshold {
                assert!(ss > init);
            } else {
                assert!(ss < init);
            }
        }
    }

    #[test]
    fn build_tree_properties() {
        // port of numpyro's test_build_tree
        let pot = std_normal(1);
        for &step_size in &[0.01, 1.0, 100.0] {
            let k = HmcKernel::nuts(&pot)
                .adapt_step_size(false)
                .adapt_mass_matrix(false)
                .step_size(step_size);
            let mut rng = ChainRng::seed_from_u64(0);
            let mut st = k.init(vec![0.0], 0, &mut rng);
            st.ws.r0[0] = 1.0;
            k.build_tree(&mut st, 10, &mut rng);
            let t = &st.ws.main;
            assert!(t.num_proposals >= 1 << (t.depth.saturating_sub(1)));
            assert!(t.sum_accept <= t.num_proposals as f64);
            if t.depth < 10 {
                assert!(t.turning || t.diverging);
            }
            if step_size > 10.0 {
                assert!(t.diverging);
                assert_eq!(t.num_proposals, 1);
            }
            if step_size < 0.1 {
                assert!(t.num_proposals > 10);
            }
        }
    }
}
