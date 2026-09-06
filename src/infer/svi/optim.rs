//! First-order optimizers over flat parameter vectors.

/// A gradient-descent optimizer for minimizing a loss.
pub trait Optimizer {
    /// Called once with the parameter dimension.
    fn init(&mut self, dim: usize);
    /// Update `params` in place given the gradient of the loss.
    fn step(&mut self, params: &mut [f64], grad: &[f64]);
}

/// Adam (Kingma & Ba 2015) with bias correction.
#[derive(Clone, Debug)]
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }
}

impl Optimizer for Adam {
    fn init(&mut self, dim: usize) {
        self.m = vec![0.0; dim];
        self.v = vec![0.0; dim];
        self.t = 0;
    }
    fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        self.t += 1;
        let t = self.t as f64;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            params[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }
    }
}

/// Adam with the gradient norm clipped to `clip_norm` before the update.
#[derive(Clone, Debug)]
pub struct ClippedAdam {
    pub adam: Adam,
    pub clip_norm: f64,
}

impl ClippedAdam {
    pub fn new(lr: f64, clip_norm: f64) -> Self {
        ClippedAdam {
            adam: Adam::new(lr),
            clip_norm,
        }
    }
}

impl Optimizer for ClippedAdam {
    fn init(&mut self, dim: usize) {
        self.adam.init(dim);
    }
    fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm > self.clip_norm {
            let s = self.clip_norm / norm;
            let clipped: Vec<f64> = grad.iter().map(|g| g * s).collect();
            self.adam.step(params, &clipped);
        } else {
            self.adam.step(params, grad);
        }
    }
}

/// Plain stochastic gradient descent with optional momentum.
#[derive(Clone, Debug)]
pub struct Sgd {
    pub lr: f64,
    pub momentum: f64,
    buf: Vec<f64>,
}

impl Sgd {
    pub fn new(lr: f64) -> Self {
        Sgd {
            lr,
            momentum: 0.0,
            buf: Vec::new(),
        }
    }
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }
}

impl Optimizer for Sgd {
    fn init(&mut self, dim: usize) {
        self.buf = vec![0.0; dim];
    }
    fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        for i in 0..params.len() {
            self.buf[i] = self.momentum * self.buf[i] + grad[i];
            params[i] -= self.lr * self.buf[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_grad(p: &[f64]) -> Vec<f64> {
        // loss = sum (p_i - i)^2
        p.iter().enumerate().map(|(i, v)| 2.0 * (v - i as f64)).collect()
    }

    #[test]
    fn optimizers_converge_on_quadratic() {
        for (name, mut opt) in [
            ("adam", Box::new(Adam::new(0.1)) as Box<dyn Optimizer>),
            ("clipped", Box::new(ClippedAdam::new(0.1, 1.0))),
            ("sgd", Box::new(Sgd::new(0.05).momentum(0.5))),
        ] {
            let mut p = vec![5.0, -3.0, 0.0];
            opt.init(3);
            for _ in 0..2000 {
                let g = quadratic_grad(&p);
                opt.step(&mut p, &g);
            }
            for (i, v) in p.iter().enumerate() {
                assert!((v - i as f64).abs() < 1e-2, "{name}: {p:?}");
            }
        }
    }
}
