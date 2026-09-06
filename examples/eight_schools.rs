//! Eight schools (Rubin 1981) under NUTS and Metropolis–Hastings, plus a
//! closed-form density with no model at all.
//!
//! ```text
//! cargo run --release --example eight_schools
//! ```

use pyroxide::prelude::*;

/// Non-centered parameterization.
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

fn main() {
    let model = EightSchools {
        y: vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        sigma: vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    };

    println!("== NUTS, 4 chains");
    let nuts = HmcKernel::nuts(ModelPotential::new(&model)).target_accept_prob(0.9);
    let samples = MCMC::new(nuts, 1000, 1000).num_chains(4).run(0);
    samples.print_summary();

    println!("== Metropolis–Hastings, dense proposal covariance");
    let mh = MetropolisHastings::new(ModelPotential::new(&model)).dense_mass(true);
    let samples = MCMC::new(mh, 10_000, 50_000).run(0);
    samples.print_summary();

    println!("== A density without a model: Rosenbrock banana");
    let banana = FnPotential::new(2, |z: &[Var]| {
        z[0] * z[0] * 0.5 + (z[1] - z[0] * z[0]).square() * 50.0
    });
    let samples = MCMC::new(HmcKernel::nuts(banana), 1000, 2000).run(1);
    samples.print_summary();
}
