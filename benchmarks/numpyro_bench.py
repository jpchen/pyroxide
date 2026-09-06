"""numpyro counterpart of `examples/bench.rs`.

Runs the same models with the same kernels and prints one JSON line per run.
Two timings are reported: `time_s` (a fresh `MCMC.run`, which includes XLA
compilation) and `time_s_cached` (a second run with the same shapes, where the
compiled program is reused). Both are on CPU with float64 enabled so the
arithmetic matches the Rust implementation.

    python benchmarks/numpyro_bench.py --model eight_schools --algo nuts \
        --warmup 1000 --samples 1000 --chains 1 --seed 0
"""
import argparse
import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import AIES, ESS, HMC, MCMC, NUTS, BarkerMH
from numpyro.infer.mcmc import MCMCKernel

try:  # only on the numpyro checkout carrying the microcanonical contrib module
    from numpyro.contrib.microcanonical import MAMS
except Exception:  # pragma: no cover
    MAMS = None

here = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ models ---

def gauss():
    numpyro.sample("z", dist.Normal(0.0, 1.0).expand([100]))


def make_logreg():
    X = jnp.asarray(np.loadtxt(os.path.join(here, "data", "logreg_X.csv"), delimiter=","))
    y = jnp.asarray(np.loadtxt(os.path.join(here, "data", "logreg_y.csv"), delimiter=","))

    def logreg():
        coefs = numpyro.sample("coefs", dist.Normal(0.0, 1.0).expand([X.shape[1]]))
        logits = X @ coefs
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    return logreg


def make_eight_schools():
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    def eight_schools():
        mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
        tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
        eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([8]))
        theta = numpyro.deterministic("theta", mu + tau * eta)
        numpyro.sample("y", dist.Normal(theta, sigma), obs=y)

    return eight_schools


def make_baseball():
    hits = jnp.array([18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7], dtype=float)
    at_bats = jnp.full((18,), 45.0)

    def baseball():
        loc = numpyro.sample("loc", dist.Normal(-1, 1))
        scale = numpyro.sample("scale", dist.HalfCauchy(1))
        with numpyro.plate("num_players", 18):
            alpha = numpyro.sample("alpha", dist.Normal(loc, scale))
            numpyro.sample("obs", dist.Binomial(at_bats, logits=alpha), obs=hits)

    return baseball


def funnel(dim=10):
    y = numpyro.sample("y", dist.Normal(0, 3))
    x_base = numpyro.sample("x_base", dist.Normal(0.0, 1.0).expand([dim - 1]))
    numpyro.deterministic("x", x_base * jnp.exp(y / 2))


def make_hier():
    groups, n_per = 50, 40
    # same xorshift + Box-Muller generator as examples/bench.rs
    seed = 12345

    def unif():
        nonlocal seed
        seed ^= (seed << 13) & 0xFFFFFFFFFFFFFFFF
        seed ^= seed >> 7
        seed ^= (seed << 17) & 0xFFFFFFFFFFFFFFFF
        return (seed >> 11) / float(1 << 53)

    xs, ys = [], []
    for g in range(groups):
        a = 1.0 + 0.5 * np.sin(g * 0.7)
        b = -0.5 + 0.3 * np.cos(g * 1.3)
        for _ in range(n_per):
            xi = unif() * 4.0 - 2.0
            u1, u2 = max(unif(), 1e-12), unif()
            e = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            xs.append(xi)
            ys.append(a + b * xi + 0.5 * e)
    x = jnp.asarray(np.array(xs).reshape(groups, n_per))
    y = jnp.asarray(np.array(ys).reshape(groups, n_per))

    def hier():
        mu_a = numpyro.sample("mu_a", dist.Normal(0, 5))
        sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(2))
        mu_b = numpyro.sample("mu_b", dist.Normal(0, 5))
        sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(2))
        a = numpyro.sample("a", dist.Normal(mu_a, sigma_a).expand([groups]))
        b = numpyro.sample("b", dist.Normal(mu_b, sigma_b).expand([groups]))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1))
        mean = a[:, None] + b[:, None] * x
        numpyro.sample("y", dist.Normal(mean, sigma), obs=y)

    return hier


# ----------------------------------------------------- Metropolis-Hastings ---

class MetropolisHastings(MCMCKernel):
    """Adaptive random-walk Metropolis-Hastings (numpyro has no built-in one;
    this follows the sketch in the `MCMCKernel` docstring and adds the same
    adaptation pyroxide uses: dual averaging on the log proposal scale toward
    0.234 acceptance and a Welford covariance estimate during warmup)."""

    sample_field = "u"

    def __init__(self, model, target_accept=0.234):
        self._model = model
        self._target = target_accept
        self._potential_fn = None
        self._postprocess_fn = None
        self._num_warmup = 0

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        from numpyro.infer.util import initialize_model

        rng_key, init_key = random.split(rng_key)
        info = initialize_model(init_key, self._model, model_args=model_args, model_kwargs=model_kwargs)
        self._potential_fn = info.potential_fn
        self._postprocess_fn = info.postprocess_fn
        self._num_warmup = num_warmup
        u_flat, self._unravel = jax.flatten_util.ravel_pytree(info.param_info.z)
        d = u_flat.shape[0]
        pe = self._potential_fn(info.param_info.z)
        state = dict(
            i=jnp.array(0),
            u=u_flat,
            pe=pe,
            accept_prob=jnp.array(0.0),
            mean_accept_prob=jnp.array(0.0),
            # dual averaging state on log step size
            log_step=jnp.log(2.38 / jnp.sqrt(d)),
            x_avg=jnp.array(0.0),
            g_avg=jnp.array(0.0),
            t=jnp.array(0.0),
            prox=jnp.log(2.38 / jnp.sqrt(d)),
            step=2.38 / jnp.sqrt(d),
            # welford
            mean=jnp.zeros(d),
            m2=jnp.zeros((d, d)),
            n=jnp.array(0.0),
            chol=jnp.eye(d),
            rng_key=rng_key,
        )
        return state

    def postprocess_fn(self, args, kwargs):
        # numpyro applies this to one collected `u` at a time
        unravel = self._unravel
        post = self._postprocess_fn  # initialize_model (static args) returns the constrain fn directly
        return lambda u: post(unravel(u))

    def sample(self, state, model_args, model_kwargs):
        key, key_eps, key_u = random.split(state["rng_key"], 3)
        d = state["u"].shape[0]
        eps = random.normal(key_eps, (d,))
        prop = state["u"] + state["step"] * (state["chol"] @ eps)
        pe_new = self._potential_fn(self._unravel(prop))
        delta = pe_new - state["pe"]
        delta = jnp.where(jnp.isnan(delta), jnp.inf, delta)
        accept_prob = jnp.minimum(1.0, jnp.exp(-delta))
        accept = random.uniform(key_u) < accept_prob
        u = jnp.where(accept, prop, state["u"])
        pe = jnp.where(accept, pe_new, state["pe"])
        i = state["i"]
        warmup = i < self._num_warmup

        # dual averaging (t0=10, kappa=0.75, gamma=0.05)
        t = state["t"] + 1.0
        g_avg = (1 - 1 / (t + 10.0)) * state["g_avg"] + (self._target - accept_prob) / (t + 10.0)
        log_step = state["prox"] - jnp.sqrt(t) / 0.05 * g_avg
        w = t ** (-0.75)
        x_avg = (1 - w) * state["x_avg"] + w * log_step
        final = i == self._num_warmup - 1
        step = jnp.where(final, jnp.exp(x_avg), jnp.exp(log_step))

        # welford covariance, refreshed into the proposal every 100 warmup iters after 100
        n = state["n"] + 1.0
        delta_pre = u - state["mean"]
        mean = state["mean"] + delta_pre / n
        delta_post = u - mean
        m2 = state["m2"] + jnp.outer(delta_post, delta_pre)
        refresh = warmup & (i >= 100) & ((i + 1) % 100 == 0)
        cov = m2 / jnp.maximum(n - 1.0, 1.0)
        cov = (n / (n + 5.0)) * cov + 1e-3 * (5.0 / (n + 5.0)) * jnp.eye(d)
        chol_new = jnp.linalg.cholesky(cov)
        chol = jnp.where(refresh, chol_new, state["chol"])

        new = dict(state)
        new.update(
            i=i + 1,
            u=u,
            pe=pe,
            accept_prob=accept_prob,
            t=jnp.where(warmup, t, state["t"]),
            g_avg=jnp.where(warmup, g_avg, state["g_avg"]),
            log_step=jnp.where(warmup, log_step, state["log_step"]),
            x_avg=jnp.where(warmup, x_avg, state["x_avg"]),
            step=jnp.where(warmup, step, state["step"]),
            mean=mean,
            m2=m2,
            n=n,
            chol=chol,
            rng_key=key,
        )
        nn = jnp.where(warmup, i + 1, i + 1 - self._num_warmup)
        new["mean_accept_prob"] = state["mean_accept_prob"] + (accept_prob - state["mean_accept_prob"]) / nn
        return new


# ------------------------------------------------------------------ driver ---

MODELS = {
    "gauss": lambda: gauss,
    "logreg": make_logreg,
    "eight_schools": make_eight_schools,
    "baseball": make_baseball,
    "funnel": lambda: funnel,
    "hier": make_hier,
}


def make_mcmc(model, args):
    if args.algo == "nuts":
        kernel = NUTS(model)
    elif args.algo == "hmc":
        kernel = HMC(model)
    elif args.algo == "mh":
        kernel = MetropolisHastings(model)
    elif args.algo == "barker":
        kernel = BarkerMH(model)
    elif args.algo == "aies":
        kernel = AIES(model)
    elif args.algo == "ess":
        # numpyro's ESS permutes the walker array in place each iteration when
        # randomize_split=True (its default), so per-chain series mix walkers
        # and the reported n_eff is inflated. Keep walker identity for a fair
        # comparison with pyroxide (which randomizes only the active/inactive split).
        kernel = ESS(model, randomize_split=False)
    elif args.algo == "mams":
        if MAMS is None:
            raise SystemExit("MAMS is not available in this numpyro installation")
        kernel = MAMS(model)
    else:
        raise ValueError(args.algo)
    ensemble = args.algo in ("aies", "ess")
    return MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="vectorized" if ensemble else ("parallel" if args.chains > 1 else "sequential"),
        progress_bar=False,
    )


def timed_run(mcmc, key):
    k = mcmc.sampler
    if isinstance(k, (HMC, NUTS)) or (MAMS is not None and isinstance(k, MAMS)):
        fields = ("diverging", "num_steps", "mean_accept_prob")
    elif isinstance(k, (AIES, ESS)):
        fields = ()
    else:
        fields = ("accept_prob", "mean_accept_prob")
    t0 = time.perf_counter()
    mcmc.run(key, extra_fields=fields)
    # force materialization of samples and extra fields
    samples = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    jax.block_until_ready(samples)
    jax.block_until_ready(extra)
    return time.perf_counter() - t0, samples, extra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="eight_schools")
    p.add_argument("--algo", default="nuts")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--chains", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--x32", action="store_true", help="use float32 (numpyro default) instead of float64")
    args = p.parse_args()

    if not args.x32:
        numpyro.enable_x64()
    numpyro.set_platform("cpu")
    if args.chains > 1:
        numpyro.set_host_device_count(args.chains)

    model = MODELS[args.model]()
    for rep in range(args.repeat):
        seed = args.seed + rep
        mcmc = make_mcmc(model, args)
        t_first, samples, extra = timed_run(mcmc, random.PRNGKey(seed))
        # second run with the same shapes reuses the compiled program
        mcmc2 = make_mcmc(model, args)
        t_cached, samples, extra = timed_run(mcmc2, random.PRNGKey(seed))

        ess = []
        rhat = []
        for name, v in samples.items():
            v = np.asarray(v)
            flat = v.reshape(v.shape[0], v.shape[1], -1)
            for j in range(flat.shape[2]):
                ess.append(float(effective_sample_size(flat[:, :, j])))
                if flat.shape[1] >= 4:
                    rhat.append(float(split_gelman_rubin(flat[:, :, j])))
        ess_min, ess_mean = min(ess), float(np.mean(ess))
        ndiv = int(np.sum(np.asarray(extra["diverging"]))) if "diverging" in extra else 0
        steps = int(np.sum(np.asarray(extra["num_steps"]))) if "num_steps" in extra else 0
        if "mean_accept_prob" in extra:
            map_ = np.asarray(extra["mean_accept_prob"])
            mean_accept = float(np.mean(map_[:, -1]))
        else:
            mean_accept = float("nan")
        print(json.dumps(dict(
            lib="numpyro" + ("_x32" if args.x32 else ""),
            model=args.model, algo=args.algo, warmup=args.warmup, samples=args.samples,
            chains=args.chains, seed=seed,
            time_s=round(t_first, 4), time_s_cached=round(t_cached, 4),
            ess_min=round(ess_min, 1), ess_mean=round(ess_mean, 1),
            ess_min_per_s=round(ess_min / t_first, 1), ess_min_per_s_cached=round(ess_min / t_cached, 1),
            max_rhat=round(max(rhat), 4) if rhat else float("nan"),
            num_divergences=ndiv, leapfrog_steps=steps, mean_accept_prob=round(mean_accept, 3),
        )), flush=True)


if __name__ == "__main__":
    main()
