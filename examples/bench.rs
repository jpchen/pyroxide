//! Benchmark driver: runs a named model with a named kernel and prints one JSON
//! line with wall-clock time and effective sample sizes.
//!
//! ```text
//! cargo run --release --example bench -- --model eight_schools --algo nuts \
//!     --warmup 1000 --samples 1000 --chains 1 --seed 0
//! ```
//!
//! The same models are implemented in `benchmarks/numpyro_bench.py`.

use std::time::Instant;

use pyroxide::ad::{matvec_const, Real};
use pyroxide::diagnostics::effective_sample_size;
use pyroxide::dist::*;
use pyroxide::infer::*;
use pyroxide::model::{Handler, Model};

// ------------------------------------------------------------------ models ---

/// 100-dimensional standard normal.
struct Gauss {
    dim: usize,
}
impl Model for Gauss {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        h.sample_vec("z", Normal::new(0.0, 1.0).expand(self.dim));
    }
}

/// Bayesian logistic regression with N(0, 1) priors.
struct LogReg {
    x: Vec<f64>,
    n: usize,
    d: usize,
    y: Vec<f64>,
}
impl Model for LogReg {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let coefs = h.sample_vec("coefs", Normal::new(0.0, 1.0).expand(self.d));
        let logits = matvec_const(&self.x, self.n, self.d, &coefs);
        h.observe("obs", Bernoulli::logits(&logits), &self.y);
    }
}

/// Eight schools, non-centered.
struct EightSchools {
    y: Vec<f64>,
    sigma: Vec<f64>,
}
impl Model for EightSchools {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let mu = h.sample("mu", Normal::new(0.0, 5.0));
        let tau = h.sample("tau", HalfCauchy::new(5.0));
        let eta = h.sample_vec("eta", Normal::new(0.0, 1.0).expand(8));
        let theta: Vec<R> = eta.iter().map(|e| mu + tau * *e).collect();
        h.deterministic("theta", &theta);
        h.observe("y", Normal::new(&theta, &self.sigma), &self.y);
    }
}

/// Baseball batting averages: partially pooled model with a logit link
/// (numpyro `examples/baseball.py::partially_pooled_with_logit`).
struct Baseball {
    at_bats: Vec<f64>,
    hits: Vec<f64>,
}
impl Model for Baseball {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let loc = h.sample("loc", Normal::new(-1.0, 1.0));
        let scale = h.sample("scale", HalfCauchy::new(1.0));
        let alpha = h.sample_vec("alpha", Normal::new(loc, scale).expand(self.at_bats.len()));
        h.observe("obs", Binomial::logits(&self.at_bats, &alpha), &self.hits);
    }
}

/// Neal's funnel (10-d), non-centered (numpyro `examples/funnel.py`, reparam).
struct Funnel {
    dim: usize,
}
impl Model for Funnel {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let y = h.sample("y", Normal::new(0.0, 3.0));
        let x_base = h.sample_vec("x_base", Normal::new(0.0, 1.0).expand(self.dim - 1));
        let s = (y / 2.0).exp();
        let x: Vec<R> = x_base.iter().map(|b| *b * s).collect();
        h.deterministic("x", &x);
    }
}

/// Hierarchical linear regression over G groups with a shared prior
/// (a stress test with a larger latent dimension and data set).
struct HierRegression {
    groups: usize,
    n_per: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}
impl Model for HierRegression {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let mu_a = h.sample("mu_a", Normal::new(0.0, 5.0));
        let sigma_a = h.sample("sigma_a", HalfNormal::new(2.0));
        let mu_b = h.sample("mu_b", Normal::new(0.0, 5.0));
        let sigma_b = h.sample("sigma_b", HalfNormal::new(2.0));
        let a = h.sample_vec("a", Normal::new(mu_a, sigma_a).expand(self.groups));
        let b = h.sample_vec("b", Normal::new(mu_b, sigma_b).expand(self.groups));
        let sigma = h.sample("sigma", HalfNormal::new(1.0));
        let mut mean = Vec::with_capacity(self.x.len());
        for g in 0..self.groups {
            for i in 0..self.n_per {
                let idx = g * self.n_per + i;
                mean.push(a[g] + b[g] * self.x[idx]);
            }
        }
        h.observe("y", Normal::new(&mean, sigma), &self.y);
    }
}

// -------------------------------------------------------------------- data ---

fn read_csv(path: &str) -> Vec<Vec<f64>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|v| v.trim().parse::<f64>().unwrap()).collect())
        .collect()
}

fn logreg() -> LogReg {
    let dir = format!("{}/benchmarks/data", env!("CARGO_MANIFEST_DIR"));
    let x = read_csv(&format!("{dir}/logreg_X.csv"));
    let y = read_csv(&format!("{dir}/logreg_y.csv"));
    let (n, d) = (x.len(), x[0].len());
    LogReg {
        x: x.into_iter().flatten().collect(),
        n,
        d,
        y: y.into_iter().map(|r| r[0]).collect(),
    }
}

fn eight_schools() -> EightSchools {
    EightSchools {
        y: vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        sigma: vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    }
}

fn baseball() -> Baseball {
    // Efron & Morris (1975): hits in the first 45 at-bats of the 1970 season
    let hits = [
        18.0, 17.0, 16.0, 15.0, 14.0, 14.0, 13.0, 12.0, 11.0, 11.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 8.0,
        7.0,
    ];
    Baseball {
        at_bats: vec![45.0; 18],
        hits: hits.to_vec(),
    }
}

fn hier_regression() -> HierRegression {
    // deterministic synthetic data (same formula in the Python harness)
    let (groups, n_per) = (50usize, 40usize);
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut seed = 12345u64;
    let mut unif = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for g in 0..groups {
        let a = 1.0 + 0.5 * ((g as f64) * 0.7).sin();
        let b = -0.5 + 0.3 * ((g as f64) * 1.3).cos();
        for _ in 0..n_per {
            let xi = unif() * 4.0 - 2.0;
            // Box-Muller noise
            let (u1, u2) = (unif().max(1e-12), unif());
            let e = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            x.push(xi);
            y.push(a + b * xi + 0.5 * e);
        }
    }
    HierRegression { groups, n_per, x, y }
}

// ------------------------------------------------------------------ driver ---

struct Args {
    model: String,
    algo: String,
    warmup: usize,
    samples: usize,
    chains: usize,
    seed: u64,
    repeat: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: "eight_schools".into(),
        algo: "nuts".into(),
        warmup: 1000,
        samples: 1000,
        chains: 1,
        seed: 0,
        repeat: 1,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < argv.len() {
        match argv[i].as_str() {
            "--model" => a.model = argv[i + 1].clone(),
            "--algo" => a.algo = argv[i + 1].clone(),
            "--warmup" => a.warmup = argv[i + 1].parse().unwrap(),
            "--samples" => a.samples = argv[i + 1].parse().unwrap(),
            "--chains" => a.chains = argv[i + 1].parse().unwrap(),
            "--seed" => a.seed = argv[i + 1].parse().unwrap(),
            "--repeat" => a.repeat = argv[i + 1].parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    a
}

fn run_kernel<P: Potential>(pot: P, args: &Args) -> (Samples, f64) {
    let start = Instant::now();
    let samples = match args.algo.as_str() {
        "nuts" => MCMC::new(HmcKernel::nuts(pot), args.warmup, args.samples)
            .num_chains(args.chains)
            .run(args.seed),
        "hmc" => MCMC::new(HmcKernel::hmc(pot), args.warmup, args.samples)
            .num_chains(args.chains)
            .run(args.seed),
        "mh" => MCMC::new(
            MetropolisHastings::new(pot).dense_mass(true),
            args.warmup,
            args.samples,
        )
        .num_chains(args.chains)
        .run(args.seed),
        other => panic!("unknown algo {other}"),
    };
    (samples, start.elapsed().as_secs_f64())
}

fn run_model<M: Model>(model: &M, args: &Args) -> (Samples, f64) {
    let pot = ModelPotential::new(model);
    run_kernel(pot, args)
}

fn ess_stats(samples: &Samples) -> (f64, f64, f64) {
    let mut all = Vec::new();
    for (_, arr) in &samples.sites {
        for j in 0..arr.len {
            all.push(effective_sample_size(&arr.column_per_chain(j), true));
        }
    }
    let min = all.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let max_rhat = samples
        .summary(0.9)
        .iter()
        .flat_map(|s| s.elements.iter().map(|e| e.r_hat))
        .filter(|r| r.is_finite())
        .fold(0.0, f64::max);
    (min, mean, max_rhat)
}

fn main() {
    let args = parse_args();
    for rep in 0..args.repeat {
        let a = Args {
            seed: args.seed + rep as u64,
            ..Args {
                model: args.model.clone(),
                algo: args.algo.clone(),
                warmup: args.warmup,
                samples: args.samples,
                chains: args.chains,
                seed: 0,
                repeat: 1,
            }
        };
        let (samples, secs) = match a.model.as_str() {
            "gauss" => run_model(&Gauss { dim: 100 }, &a),
            "logreg" => run_model(&logreg(), &a),
            "eight_schools" => run_model(&eight_schools(), &a),
            "baseball" => run_model(&baseball(), &a),
            "funnel" => run_model(&Funnel { dim: 10 }, &a),
            "hier" => run_model(&hier_regression(), &a),
            other => panic!("unknown model {other}"),
        };
        let (ess_min, ess_mean, max_rhat) = ess_stats(&samples);
        let steps: f64 = samples
            .extras
            .iter()
            .find(|(n, _)| n == "num_steps")
            .map(|(_, arr)| arr.data().iter().sum())
            .unwrap_or(0.0);
        let mean_accept = samples
            .extras
            .iter()
            .find(|(n, _)| n == "mean_accept_prob")
            .map(|(_, arr)| {
                (0..arr.chains)
                    .map(|c| arr.draw(c, arr.draws - 1)[0])
                    .sum::<f64>()
                    / arr.chains as f64
            })
            .unwrap_or(f64::NAN);
        println!(
            "{{\"lib\": \"pyroxide\", \"model\": \"{}\", \"algo\": \"{}\", \"warmup\": {}, \"samples\": {}, \"chains\": {}, \"seed\": {}, \"time_s\": {:.4}, \"ess_min\": {:.1}, \"ess_mean\": {:.1}, \"ess_min_per_s\": {:.1}, \"max_rhat\": {:.4}, \"num_divergences\": {}, \"leapfrog_steps\": {}, \"mean_accept_prob\": {:.3}}}",
            a.model,
            a.algo,
            a.warmup,
            a.samples,
            a.chains,
            a.seed,
            secs,
            ess_min,
            ess_mean,
            ess_min / secs,
            max_rhat,
            samples.num_divergences(),
            steps as u64,
            mean_accept,
        );
    }
}
