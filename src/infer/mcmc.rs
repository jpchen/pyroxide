//! The [`Kernel`] trait, the [`MCMC`] driver (warmup, sampling, parallel
//! chains) and the [`Samples`] container.

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use super::potential::Potential;
use crate::diagnostics;

/// The random number generator used by every chain.
pub type ChainRng = Xoshiro256PlusPlus;

/// An MCMC transition kernel over a [`Potential`].
///
/// Kernels are shared (immutably) across chains; all per-chain mutable state,
/// including adaptation state and scratch buffers, lives in `State`.
pub trait Kernel: Sync {
    type State: Send + Clone;

    /// The target potential (used for initialization and postprocessing).
    fn potential(&self) -> &dyn Potential;

    /// Create the state for a chain starting at unconstrained position `z`.
    fn init(&self, z: Vec<f64>, num_warmup: usize, rng: &mut ChainRng) -> Self::State;

    /// Advance the chain by one transition. Warmup adaptation happens inside
    /// `step` while the state's iteration counter is below `num_warmup`.
    fn step(&self, state: &mut Self::State, rng: &mut ChainRng);

    /// Current unconstrained position.
    fn position(state: &Self::State) -> &[f64];

    /// Names of the per-iteration diagnostics written by [`Kernel::stats`].
    fn stat_names(&self) -> &'static [&'static str];

    /// Write per-iteration diagnostics (same order as `stat_names`).
    fn stats(&self, state: &Self::State, out: &mut [f64]);
}

/// How to choose each chain's initial point (in unconstrained space).
#[derive(Clone, Debug)]
pub enum InitStrategy {
    /// Uniform in `(-radius, radius)` on every unconstrained coordinate
    /// (numpyro's default, `radius = 2`).
    Uniform { radius: f64 },
    /// Start every chain at the given constrained values; unspecified sites
    /// are drawn uniformly as in `Uniform { radius: 2 }`.
    Value(HashMap<String, Vec<f64>>),
    /// Start at the given unconstrained vector.
    Unconstrained(Vec<f64>),
}

impl Default for InitStrategy {
    fn default() -> Self {
        InitStrategy::Uniform { radius: 2.0 }
    }
}

/// A `chains x draws x len` array of one site's samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    data: Vec<f64>,
    pub chains: usize,
    pub draws: usize,
    pub len: usize,
}

impl Array {
    pub fn new(chains: usize, draws: usize, len: usize) -> Self {
        Array {
            data: vec![0.0; chains * draws * len],
            chains,
            draws,
            len,
        }
    }

    /// The flat data, laid out as `[(chain, draw, element)]`.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// One draw (length `len`).
    #[inline]
    pub fn draw(&self, chain: usize, draw: usize) -> &[f64] {
        let s = (chain * self.draws + draw) * self.len;
        &self.data[s..s + self.len]
    }

    #[inline]
    fn draw_mut(&mut self, chain: usize, draw: usize) -> &mut [f64] {
        let s = (chain * self.draws + draw) * self.len;
        &mut self.data[s..s + self.len]
    }

    /// All draws of element `j`, chain by chain.
    pub fn column_per_chain(&self, j: usize) -> Vec<Vec<f64>> {
        (0..self.chains)
            .map(|c| (0..self.draws).map(|d| self.draw(c, d)[j]).collect())
            .collect()
    }

    /// All draws of element `j` with chains concatenated.
    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.chains)
            .flat_map(|c| (0..self.draws).map(move |d| (c, d)))
            .map(|(c, d)| self.draw(c, d)[j])
            .collect()
    }

    /// Per-element posterior mean over all chains.
    pub fn mean(&self) -> Vec<f64> {
        let n = (self.chains * self.draws) as f64;
        let mut m = vec![0.0; self.len];
        for c in 0..self.chains {
            for d in 0..self.draws {
                for (m, v) in m.iter_mut().zip(self.draw(c, d)) {
                    *m += v / n;
                }
            }
        }
        m
    }

    /// Per-element posterior standard deviation (ddof = 1).
    pub fn std(&self) -> Vec<f64> {
        let m = self.mean();
        let n = (self.chains * self.draws) as f64;
        let mut s = vec![0.0; self.len];
        for c in 0..self.chains {
            for d in 0..self.draws {
                for ((s, v), m) in s.iter_mut().zip(self.draw(c, d)).zip(&m) {
                    *s += (v - m) * (v - m) / (n - 1.0);
                }
            }
        }
        s.iter().map(|v| v.sqrt()).collect()
    }

    /// Scalar convenience: the mean of a length-1 site.
    pub fn scalar_mean(&self) -> f64 {
        assert_eq!(self.len, 1);
        self.mean()[0]
    }

    /// Scalar convenience: the std of a length-1 site.
    pub fn scalar_std(&self) -> f64 {
        assert_eq!(self.len, 1);
        self.std()[0]
    }

    /// Sample covariance (row-major `len x len`) over all chains and draws.
    pub fn covariance(&self) -> Vec<f64> {
        let m = self.mean();
        let n = (self.chains * self.draws) as f64;
        let k = self.len;
        let mut cov = vec![0.0; k * k];
        for c in 0..self.chains {
            for d in 0..self.draws {
                let x = self.draw(c, d);
                for i in 0..k {
                    for j in 0..k {
                        cov[i * k + j] += (x[i] - m[i]) * (x[j] - m[j]) / (n - 1.0);
                    }
                }
            }
        }
        cov
    }
}

/// Posterior samples of every site plus per-iteration diagnostics.
#[derive(Clone, Debug)]
pub struct Samples {
    /// `(site name, array)` in program order.
    pub sites: Vec<(String, Array)>,
    /// Kernel diagnostics per iteration (e.g. `accept_prob`, `num_steps`).
    pub extras: Vec<(String, Array)>,
    pub num_chains: usize,
    pub num_samples: usize,
}

impl Samples {
    /// Samples of a site. Panics if the site does not exist.
    pub fn get(&self, name: &str) -> &Array {
        self.try_get(name)
            .unwrap_or_else(|| panic!("no site named '{name}'; sites: {:?}", self.site_names()))
    }

    pub fn try_get(&self, name: &str) -> Option<&Array> {
        self.sites.iter().find(|(n, _)| n == name).map(|(_, a)| a)
    }

    /// A per-iteration diagnostic. Panics if unknown.
    pub fn extra(&self, name: &str) -> &Array {
        self.extras
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a)
            .unwrap_or_else(|| panic!("no diagnostic named '{name}'"))
    }

    pub fn site_names(&self) -> Vec<&str> {
        self.sites.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Number of divergent transitions (HMC/NUTS kernels).
    pub fn num_divergences(&self) -> usize {
        match self.extras.iter().find(|(n, _)| n == "diverging") {
            Some((_, a)) => a.data().iter().filter(|&&v| v > 0.5).count(),
            None => 0,
        }
    }

    /// Summary statistics for every site (mean, std, median, HPDI, n_eff, r_hat).
    pub fn summary(&self, prob: f64) -> Vec<diagnostics::SiteSummary> {
        self.sites
            .iter()
            .map(|(name, arr)| diagnostics::summarize(name, arr, prob))
            .collect()
    }

    /// Print a numpyro-style summary table.
    pub fn print_summary(&self) {
        print!("{}", diagnostics::format_summary(&self.summary(0.9), 0.9));
        let nd = self.num_divergences();
        println!("Number of divergences: {nd}");
    }
}

/// Runs warmup and sampling for one or more chains of a [`Kernel`].
///
/// ```no_run
/// # use pyroxide::infer::*;
/// # use pyroxide::Var;
/// let potential = FnPotential::new(2, |z: &[Var]| (z[0] * z[0] + z[1] * z[1]) * 0.5);
/// let mcmc = MCMC::new(HmcKernel::nuts(potential), 500, 1000).num_chains(4);
/// let samples = mcmc.run(0);
/// samples.print_summary();
/// ```
pub struct MCMC<K: Kernel> {
    pub kernel: K,
    pub num_warmup: usize,
    pub num_samples: usize,
    pub num_chains: usize,
    pub init: InitStrategy,
    /// Keep warmup draws in the output (they precede the post-warmup draws).
    pub collect_warmup: bool,
    /// Print a one-line progress report per chain.
    pub progress: bool,
    /// Run chains on separate threads (rayon). Sequential if false.
    pub parallel: bool,
}

impl<K: Kernel> MCMC<K> {
    pub fn new(kernel: K, num_warmup: usize, num_samples: usize) -> Self {
        MCMC {
            kernel,
            num_warmup,
            num_samples,
            num_chains: 1,
            init: InitStrategy::default(),
            collect_warmup: false,
            progress: false,
            parallel: true,
        }
    }
    pub fn num_chains(mut self, n: usize) -> Self {
        self.num_chains = n;
        self
    }
    pub fn init_strategy(mut self, s: InitStrategy) -> Self {
        self.init = s;
        self
    }
    /// Convenience for [`InitStrategy::Value`].
    pub fn init_to_value(mut self, values: HashMap<String, Vec<f64>>) -> Self {
        self.init = InitStrategy::Value(values);
        self
    }
    pub fn collect_warmup(mut self, b: bool) -> Self {
        self.collect_warmup = b;
        self
    }
    pub fn progress(mut self, b: bool) -> Self {
        self.progress = b;
        self
    }
    pub fn parallel(mut self, b: bool) -> Self {
        self.parallel = b;
        self
    }

    /// Find an initial point with finite potential energy (retrying random
    /// initializations as numpyro does).
    fn initial_position(&self, rng: &mut ChainRng) -> Vec<f64> {
        let p = self.kernel.potential();
        let d = p.dim();
        let radius = match &self.init {
            InitStrategy::Uniform { radius } => *radius,
            _ => 2.0,
        };
        let mut grad = vec![0.0; d];
        for attempt in 0..100 {
            let uniform: Vec<f64> = (0..d).map(|_| rng.random_range(-radius..radius)).collect();
            let z = match &self.init {
                InitStrategy::Uniform { .. } => uniform,
                InitStrategy::Value(values) => p.unconstrain(values, &uniform),
                InitStrategy::Unconstrained(z) => z.clone(),
            };
            let u = if d == 0 {
                0.0
            } else {
                p.value_and_grad(&z, &mut grad)
            };
            if u.is_finite() && grad.iter().all(|g| g.is_finite()) {
                return z;
            }
            if matches!(self.init, InitStrategy::Unconstrained(_)) {
                panic!("initial point has non-finite potential energy or gradient");
            }
            if attempt == 99 {
                panic!("could not find a valid initial point after 100 attempts; check the model for numerical problems or supply init values");
            }
        }
        unreachable!()
    }

    fn run_chain(&self, chain: usize, mut rng: ChainRng) -> ChainOutput {
        let z0 = self.initial_position(&mut rng);
        let d = z0.len();
        let mut state = self.kernel.init(z0, self.num_warmup, &mut rng);
        let names = self.kernel.stat_names();
        let total = self.num_warmup + self.num_samples;
        let collected = if self.collect_warmup {
            total
        } else {
            self.num_samples
        };
        let mut positions = Vec::with_capacity(collected * d);
        let mut stats = vec![Vec::with_capacity(collected); names.len()];
        let mut buf = vec![0.0; names.len()];
        let start = std::time::Instant::now();
        for it in 0..total {
            self.kernel.step(&mut state, &mut rng);
            if self.collect_warmup || it >= self.num_warmup {
                positions.extend_from_slice(K::position(&state));
                self.kernel.stats(&state, &mut buf);
                for (s, v) in stats.iter_mut().zip(&buf) {
                    s.push(*v);
                }
            }
            if self.progress && (it + 1 == self.num_warmup || it + 1 == total) {
                let phase = if it + 1 == self.num_warmup {
                    "warmup"
                } else {
                    "sample"
                };
                let mut b = vec![0.0; names.len()];
                self.kernel.stats(&state, &mut b);
                let ap = names
                    .iter()
                    .position(|n| *n == "mean_accept_prob")
                    .map(|i| b[i])
                    .unwrap_or(f64::NAN);
                let ss = names
                    .iter()
                    .position(|n| *n == "step_size")
                    .map(|i| b[i])
                    .unwrap_or(f64::NAN);
                eprintln!(
                    "chain {chain}: {phase} done ({} it, {:.2}s) mean accept prob {ap:.3}, step size {ss:.3e}",
                    it + 1,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        ChainOutput {
            positions,
            stats,
            dim: d,
        }
    }

    /// Run all chains and return postprocessed samples.
    pub fn run(&self, seed: u64) -> Samples {
        let master = ChainRng::seed_from_u64(seed);
        let rngs: Vec<ChainRng> = (0..self.num_chains)
            .scan(master, |r, _| {
                let mine = r.clone();
                r.long_jump();
                Some(mine)
            })
            .collect();
        let outputs: Vec<ChainOutput> = if self.parallel && self.num_chains > 1 {
            rngs.into_par_iter()
                .enumerate()
                .map(|(c, rng)| self.run_chain(c, rng))
                .collect()
        } else {
            rngs.into_iter()
                .enumerate()
                .map(|(c, rng)| self.run_chain(c, rng))
                .collect()
        };
        self.assemble(outputs)
    }

    fn assemble(&self, outputs: Vec<ChainOutput>) -> Samples {
        let p = self.kernel.potential();
        let names = self.kernel.stat_names();
        let draws = if self.collect_warmup {
            self.num_warmup + self.num_samples
        } else {
            self.num_samples
        };
        let chains = outputs.len();
        let d = outputs[0].dim;
        // discover site names/lengths from the first draw
        let first = if draws > 0 && d > 0 {
            p.postprocess(&outputs[0].positions[..d])
        } else if draws > 0 {
            p.postprocess(&[])
        } else {
            vec![]
        };
        let mut sites: Vec<(String, Array)> = first
            .iter()
            .map(|(n, v)| (n.clone(), Array::new(chains, draws, v.len())))
            .collect();
        for (c, out) in outputs.iter().enumerate() {
            for dr in 0..draws {
                let z = &out.positions[dr * d..(dr + 1) * d];
                let values = p.postprocess(z);
                for ((_, arr), (_, v)) in sites.iter_mut().zip(values.iter()) {
                    arr.draw_mut(c, dr).copy_from_slice(v);
                }
            }
        }
        let mut extras: Vec<(String, Array)> = names
            .iter()
            .map(|n| (n.to_string(), Array::new(chains, draws, 1)))
            .collect();
        for (c, out) in outputs.iter().enumerate() {
            for (k, (_, arr)) in extras.iter_mut().enumerate() {
                for dr in 0..draws {
                    arr.draw_mut(c, dr)[0] = out.stats[k][dr];
                }
            }
        }
        Samples {
            sites,
            extras,
            num_chains: chains,
            num_samples: draws,
        }
    }
}

struct ChainOutput {
    positions: Vec<f64>,
    stats: Vec<Vec<f64>>,
    dim: usize,
}
