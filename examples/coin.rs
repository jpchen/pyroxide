//! The "getting started" example from the book: estimate a coin's bias.
//!
//! ```text
//! cargo run --release --example coin
//! ```

use pyroxide::prelude::*;

/// A model is a struct holding its data ...
struct Coin {
    flips: Vec<f64>, // 1.0 = heads, 0.0 = tails
}

/// ... plus a generic `run` method describing how the data were generated.
impl Model for Coin {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        // a latent variable with a prior
        let p = h.sample("p", Beta::new(1.0, 1.0));
        // a derived quantity we want in the output
        h.deterministic("odds", &[p / (-p + 1.0)]);
        // the likelihood: condition the plate of flips on the data
        h.observe("flips", Bernoulli::new(p).expand(self.flips.len()), &self.flips);
    }
}

fn main() {
    let model = Coin {
        flips: vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    };

    // Turn the model into a potential energy and hand it to a kernel.
    let kernel = HmcKernel::nuts(ModelPotential::new(&model));

    // 500 warmup iterations, 1000 posterior draws, 4 chains in parallel.
    let samples = MCMC::new(kernel, 500, 1000).num_chains(4).run(0);

    samples.print_summary();
    println!("P(heads) ≈ {:.3}", samples.get("p").scalar_mean());

    // Posterior predictive: replicate the data set for each posterior draw.
    let pred = Predictive::new(&model).posterior(&samples).run(1);
    let heads_rep = pred.get("flips");
    let mean_heads: f64 = (0..heads_rep.draws)
        .map(|d| heads_rep.draw(0, d).iter().sum::<f64>())
        .sum::<f64>()
        / heads_rep.draws as f64;
    println!("observed heads: 6, posterior predictive mean heads: {mean_heads:.2}");
}
