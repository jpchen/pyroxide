//! Automatic guides generated from a model's latent-site layout.
//!
//! Each autoguide is an ordinary [`Model`] whose `run` declares parameters
//! with [`Handler::param`] and samples every latent site of the target model
//! from a tractable family in unconstrained space, pushed through the site's
//! constraining transform ([`Transformed`]).

use crate::ad::Real;
use crate::dist::{Delta, Distribution, MultivariateNormal, Normal, Support, Transformed};
use crate::model::{transform, Handler, Model, SiteLayout};

fn interior_init(support: Support, ulen: usize, event_len: usize) -> Vec<f64> {
    let mut out = Vec::new();
    transform::to_constrained::<f64>(support, event_len, &vec![0.0; ulen], &mut out);
    out
}

/// MAP estimation: a point mass per latent site.
///
/// Parameters are named `"{site}_auto_loc"` (constrained space).
pub struct AutoDelta {
    layout: SiteLayout,
}

impl AutoDelta {
    pub fn new<M: Model>(model: &M) -> Self {
        AutoDelta {
            layout: SiteLayout::discover(model),
        }
    }
}

impl Model for AutoDelta {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        for s in &self.layout.sites {
            let init = interior_init(s.support, s.unconstrained_len, s.event_len);
            let v = h.param(&format!("{}_auto_loc", s.name), &init, s.support);
            h.sample_vec(&s.name, Delta::new(&v).with_support(s.support));
        }
    }
}

/// Mean-field Gaussian guide: an independent `Normal(loc, scale)` on each
/// unconstrained coordinate, transformed to the site's support.
///
/// Parameters: `"{site}_auto_loc"` (unconstrained), `"{site}_auto_scale"` (positive).
pub struct AutoDiagonalNormal {
    layout: SiteLayout,
    pub init_scale: f64,
}

impl AutoDiagonalNormal {
    pub fn new<M: Model>(model: &M) -> Self {
        AutoDiagonalNormal {
            layout: SiteLayout::discover(model),
            init_scale: 0.1,
        }
    }
    pub fn init_scale(mut self, s: f64) -> Self {
        self.init_scale = s;
        self
    }
}

impl Model for AutoDiagonalNormal {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        for s in &self.layout.sites {
            let n = s.unconstrained_len;
            let loc = h.param(&format!("{}_auto_loc", s.name), &vec![0.0; n], Support::Real);
            let scale = h.param(
                &format!("{}_auto_scale", s.name),
                &vec![self.init_scale; n],
                Support::Positive,
            );
            h.sample_vec(
                &s.name,
                Transformed::new(Normal::new(&loc, &scale), s.support, s.event_len),
            );
        }
    }
}

/// Full-rank Gaussian guide over the concatenated unconstrained latents.
///
/// Parameters: `"auto_loc"` (length `d`) and `"auto_scale_tril"` (row-major
/// `d x d` lower Cholesky factor).
pub struct AutoMultivariateNormal {
    layout: SiteLayout,
    pub init_scale: f64,
}

impl AutoMultivariateNormal {
    pub fn new<M: Model>(model: &M) -> Self {
        AutoMultivariateNormal {
            layout: SiteLayout::discover(model),
            init_scale: 0.1,
        }
    }
    pub fn init_scale(mut self, s: f64) -> Self {
        self.init_scale = s;
        self
    }
}

impl Model for AutoMultivariateNormal {
    fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
        let d = self.layout.dim;
        let loc = h.param("auto_loc", &vec![0.0; d], Support::Real);
        let mut tril_init = vec![0.0; d * d];
        for i in 0..d {
            tril_init[i * d + i] = self.init_scale;
        }
        let tril = h.param("auto_scale_tril", &tril_init, Support::LowerCholesky);
        let mvn = MultivariateNormal::new(&loc, &tril);
        let u = h.draw(mvn);
        // joint log q of the unconstrained draw, corrected by each site's Jacobian
        let mvn = MultivariateNormal::new(&loc, &tril);
        let mut log_q = mvn.log_prob(&u);
        let mut xs = Vec::with_capacity(self.layout.sites.len());
        for s in &self.layout.sites {
            let us = &u[s.offset..s.offset + s.unconstrained_len];
            let mut x = Vec::with_capacity(s.len);
            let logdet = transform::to_constrained(s.support, s.event_len, us, &mut x);
            log_q -= logdet;
            xs.push(x);
        }
        // attribute the whole log q to the first site (only the sum matters)
        for (i, (s, x)) in self.layout.sites.iter().zip(xs).enumerate() {
            let lq = if i == 0 { log_q } else { R::zero() };
            h.sample_given(&s.name, x, lq);
        }
    }
}
