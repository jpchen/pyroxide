//! Prior and posterior predictive sampling.
//!
//! [`Predictive`] runs a model forward under [`Tracer`], replaying latent values
//! from posterior [`Samples`] (or drawing them from the prior) and recording
//! every site — including observed sites, which are *sampled* rather than
//! conditioned on, and deterministic sites.
//!
//! ```no_run
//! # use pyroxide::prelude::*;
//! # struct M; impl Model for M { fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {} }
//! # let model = M;
//! # let samples = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 10, 10).run(0);
//! let pred = Predictive::new(&model).posterior(&samples).run(0);
//! let y_rep = pred.get("y");            // chains x draws x len
//! let prior = Predictive::new(&model).num_samples(500).run(1);
//! ```

use std::collections::HashMap;

use rand::SeedableRng;

use super::{Model, SiteKind, Tracer};
use crate::infer::{Array, ChainRng, Samples};

/// Builder for predictive simulations.
pub struct Predictive<'m, M: Model> {
    model: &'m M,
    posterior: Option<&'m Samples>,
    num_samples: usize,
    return_sites: Option<Vec<String>>,
}

impl<'m, M: Model> Predictive<'m, M> {
    pub fn new(model: &'m M) -> Self {
        Predictive {
            model,
            posterior: None,
            num_samples: 1000,
            return_sites: None,
        }
    }

    /// Replay latent sites from these posterior samples (one predictive draw per
    /// posterior draw, chain structure preserved).
    pub fn posterior(mut self, samples: &'m Samples) -> Self {
        self.posterior = Some(samples);
        self
    }

    /// Number of prior-predictive draws (ignored when `posterior` is set).
    pub fn num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Only keep these sites in the output.
    pub fn return_sites(mut self, names: &[&str]) -> Self {
        self.return_sites = Some(names.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Run the simulation. Observed sites appear under their own names with
    /// freshly sampled values; latent sites are the replayed (or prior) values.
    pub fn run(&self, seed: u64) -> Samples {
        let mut rng = ChainRng::seed_from_u64(seed);
        let (chains, draws) = match self.posterior {
            Some(s) => (s.num_chains, s.num_samples),
            None => (1, self.num_samples),
        };
        let mut sites: Vec<(String, Array)> = Vec::new();
        let mut values: HashMap<String, Vec<f64>> = HashMap::new();
        for c in 0..chains {
            for d in 0..draws {
                values.clear();
                if let Some(s) = self.posterior {
                    for (name, arr) in &s.sites {
                        values.insert(name.clone(), arr.draw(c, d).to_vec());
                    }
                }
                let trace = Tracer::predictive(&mut rng, &values).trace(self.model);
                for site in &trace.sites {
                    if site.kind == SiteKind::Factor {
                        continue;
                    }
                    if let Some(keep) = &self.return_sites {
                        if !keep.iter().any(|k| k == &site.name) {
                            continue;
                        }
                    }
                    let idx = match sites.iter().position(|(n, _)| n == &site.name) {
                        Some(i) => i,
                        None => {
                            sites.push((site.name.clone(), Array::new(chains, draws, site.value.len())));
                            sites.len() - 1
                        }
                    };
                    sites[idx].1.set_draw(c, d, &site.value);
                }
            }
        }
        Samples {
            sites,
            extras: Vec::new(),
            num_chains: chains,
            num_samples: draws,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::*;
    use crate::infer::{HmcKernel, ModelPotential, MCMC};
    use crate::model::Handler;
    use crate::Real;

    struct Coin {
        flips: Vec<f64>,
    }
    impl Model for Coin {
        fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
            let p = h.sample("p", Beta::new(1.0, 1.0));
            h.deterministic("odds", &[p / (-p + 1.0)]);
            h.observe("flips", Bernoulli::new(p).expand(self.flips.len()), &self.flips);
        }
    }

    #[test]
    fn prior_predictive_shapes_and_moments() {
        let model = Coin { flips: vec![1.0; 20] };
        let pred = Predictive::new(&model).num_samples(4000).run(0);
        assert_eq!(pred.get("p").draws, 4000);
        assert_eq!(pred.get("flips").len, 20);
        assert_eq!(pred.get("odds").len, 1);
        // prior p ~ U(0,1): E[p] = 0.5; prior predictive flips have mean 0.5
        assert!((pred.get("p").scalar_mean() - 0.5).abs() < 0.03);
        let m = pred.get("flips").mean();
        assert!(m.iter().all(|v| (v - 0.5).abs() < 0.05), "{m:?}");
    }

    #[test]
    fn posterior_predictive_replays_latents() {
        let model = Coin {
            flips: vec![1.0, 1.0, 1.0, 1.0, 0.0],
        };
        let samples = MCMC::new(HmcKernel::nuts(ModelPotential::new(&model)), 300, 2000)
            .num_chains(2)
            .run(0);
        let pred = Predictive::new(&model).posterior(&samples).run(1);
        assert_eq!(pred.num_chains, 2);
        assert_eq!(pred.get("flips").draws, 2000);
        // latent p is replayed exactly
        assert_eq!(pred.get("p").data(), samples.get("p").data());
        // posterior mean of p under Beta(1,1) prior with 4/5 heads is 5/7
        let p_mean = samples.get("p").scalar_mean();
        assert!((p_mean - 5.0 / 7.0).abs() < 0.03, "{p_mean}");
        // posterior predictive flips average the posterior mean of p
        let flips = pred.get("flips").mean();
        assert!(flips.iter().all(|v| (v - p_mean).abs() < 0.04), "{flips:?}");
        // return_sites filters
        let only = Predictive::new(&model)
            .posterior(&samples)
            .return_sites(&["odds"])
            .run(2);
        assert_eq!(only.site_names(), vec!["odds"]);
    }
}
