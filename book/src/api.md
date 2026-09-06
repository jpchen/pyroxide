# API documentation

The full `rustdoc` reference is published alongside this book:

<a href="../api/pyroxide/index.html">pyroxide API documentation</a>

Locally:

```
cargo doc --no-deps --open
```

Key entry points:

| item | purpose |
|---|---|
| `pyroxide::prelude::*` | everything needed to write and fit a model |
| `pyroxide::model::{Model, Handler}` | the model interface |
| `pyroxide::dist` | distributions |
| `pyroxide::infer::{HmcKernel, MetropolisHastings, MCMC, Samples}` | inference |
| `pyroxide::infer::{Potential, FnPotential, ModelPotential}` | the model/inference boundary |
| `pyroxide::model::Predictive` | prior and posterior predictive simulation |
| `pyroxide::diagnostics` | ESS, R-hat, HPDI |
| `pyroxide::ad` | the autodiff tape and fused vector operations |
