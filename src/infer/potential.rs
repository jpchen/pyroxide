//! The interface between models and inference: a potential energy function.
//!
//! Inference algorithms only ever see a [`Potential`]: a differentiable
//! function `U(z) = -log p(z)` on unconstrained `R^d`. [`ModelPotential`] turns
//! any [`Model`] into one; [`FnPotential`] wraps a closure for algorithm tests
//! and for users who already have a log density.

use std::collections::HashMap;

use crate::ad::{self, Var};
use crate::model::{LogDensity, Model, Postprocess, SiteLayout};

/// A differentiable potential energy `U(z) = -log p(z)` on unconstrained space.
pub trait Potential: Send + Sync {
    /// Dimension of the unconstrained parameter vector.
    fn dim(&self) -> usize;

    /// Potential energy at `z` (may be `+inf` or `NaN` for invalid states).
    fn value(&self, z: &[f64]) -> f64;

    /// Potential energy and its gradient at `z`.
    fn value_and_grad(&self, z: &[f64], grad: &mut [f64]) -> f64;

    /// Map an unconstrained state to named, constrained values. The default
    /// exposes the raw vector under the single name `"z"`.
    fn postprocess(&self, z: &[f64]) -> Vec<(String, Vec<f64>)> {
        vec![("z".to_string(), z.to_vec())]
    }

    /// Map named constrained values to an unconstrained vector, using
    /// `default` for anything not specified. The default implementation only
    /// understands the name `"z"`.
    fn unconstrain(&self, values: &HashMap<String, Vec<f64>>, default: &[f64]) -> Vec<f64> {
        match values.get("z") {
            Some(z) => z.clone(),
            None => default.to_vec(),
        }
    }
}

impl<P: Potential + ?Sized> Potential for &P {
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn value(&self, z: &[f64]) -> f64 {
        (**self).value(z)
    }
    fn value_and_grad(&self, z: &[f64], grad: &mut [f64]) -> f64 {
        (**self).value_and_grad(z, grad)
    }
    fn postprocess(&self, z: &[f64]) -> Vec<(String, Vec<f64>)> {
        (**self).postprocess(z)
    }
    fn unconstrain(&self, values: &HashMap<String, Vec<f64>>, default: &[f64]) -> Vec<f64> {
        (**self).unconstrain(values, default)
    }
}

impl<P: Potential + ?Sized> Potential for std::sync::Arc<P> {
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn value(&self, z: &[f64]) -> f64 {
        (**self).value(z)
    }
    fn value_and_grad(&self, z: &[f64], grad: &mut [f64]) -> f64 {
        (**self).value_and_grad(z, grad)
    }
    fn postprocess(&self, z: &[f64]) -> Vec<(String, Vec<f64>)> {
        (**self).postprocess(z)
    }
    fn unconstrain(&self, values: &HashMap<String, Vec<f64>>, default: &[f64]) -> Vec<f64> {
        (**self).unconstrain(values, default)
    }
}

/// The potential energy of a [`Model`]: minus its log joint density in
/// unconstrained space (including the log-Jacobian of the constraining
/// transforms).
pub struct ModelPotential<'m, M: Model> {
    pub model: &'m M,
    pub layout: SiteLayout,
}

impl<'m, M: Model> ModelPotential<'m, M> {
    /// Build a potential for `model`, discovering its site layout.
    pub fn new(model: &'m M) -> Self {
        let layout = SiteLayout::discover(model);
        ModelPotential { model, layout }
    }

    /// Log joint density (with Jacobian) at unconstrained `z`.
    pub fn log_density(&self, z: &[f64]) -> f64 {
        LogDensity::eval(self.model, z, &self.layout)
    }
}

impl<'m, M: Model> Potential for ModelPotential<'m, M> {
    fn dim(&self) -> usize {
        self.layout.dim
    }

    fn value(&self, z: &[f64]) -> f64 {
        -self.log_density(z)
    }

    fn value_and_grad(&self, z: &[f64], grad: &mut [f64]) -> f64 {
        ad::reset();
        let zv = Var::leaves(z);
        let lp = LogDensity::eval(self.model, &zv, &self.layout);
        ad::gradient_into(lp, &zv, grad);
        for g in grad.iter_mut() {
            *g = -*g;
        }
        -lp.value()
    }

    fn postprocess(&self, z: &[f64]) -> Vec<(String, Vec<f64>)> {
        Postprocess::run(self.model, z, &self.layout)
    }

    fn unconstrain(&self, values: &HashMap<String, Vec<f64>>, default: &[f64]) -> Vec<f64> {
        self.layout.unconstrain(values, default)
    }
}

/// A potential defined directly by a closure over [`Var`]s.
///
/// ```
/// use pyroxide::infer::{FnPotential, Potential};
/// use pyroxide::ad::{Real, Var};
/// // standard normal in 3 dimensions
/// let p = FnPotential::new(3, |z: &[Var]| z.iter().map(|x| *x * *x * 0.5).sum::<Var>());
/// let mut g = vec![0.0; 3];
/// let u = p.value_and_grad(&[1.0, 2.0, 3.0], &mut g);
/// assert!((u - 7.0).abs() < 1e-12);
/// assert_eq!(g, vec![1.0, 2.0, 3.0]);
/// ```
pub struct FnPotential<F> {
    dim: usize,
    f: F,
}

impl<F: Fn(&[Var]) -> Var + Send + Sync> FnPotential<F> {
    pub fn new(dim: usize, f: F) -> Self {
        FnPotential { dim, f }
    }
}

impl<F: Fn(&[Var]) -> Var + Send + Sync> Potential for FnPotential<F> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn value(&self, z: &[f64]) -> f64 {
        ad::reset();
        let zv = Var::leaves(z);
        (self.f)(&zv).value()
    }
    fn value_and_grad(&self, z: &[f64], grad: &mut [f64]) -> f64 {
        ad::reset();
        let zv = Var::leaves(z);
        let u = (self.f)(&zv);
        ad::gradient_into(u, &zv, grad);
        u.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::*;
    use crate::model::Handler;
    use crate::Real;

    struct Simple;
    impl Model for Simple {
        fn run<R: Real, H: Handler<R>>(&self, h: &mut H) {
            let s = h.sample("s", HalfNormal::new(1.0));
            h.observe("x", Normal::new(0.0, s).expand(3), &[0.5, -1.0, 2.0]);
        }
    }

    #[test]
    fn model_potential_grad_matches_fd() {
        let p = ModelPotential::new(&Simple);
        assert_eq!(p.dim(), 1);
        let z = [0.3];
        let mut g = [0.0];
        let u = p.value_and_grad(&z, &mut g);
        assert!((u - p.value(&z)).abs() < 1e-12);
        let fd = ad::finite_diff(|z| p.value(z), &z, 1e-6);
        assert!((g[0] - fd[0]).abs() < 1e-6);
        let post = p.postprocess(&z);
        assert_eq!(post[0].0, "s");
        assert!((post[0].1[0] - 0.3f64.exp()).abs() < 1e-12);
        let mut vals = HashMap::new();
        vals.insert("s".to_string(), vec![2.0]);
        let z = p.unconstrain(&vals, &[0.0]);
        assert!((z[0] - 2.0f64.ln()).abs() < 1e-12);
    }
}
