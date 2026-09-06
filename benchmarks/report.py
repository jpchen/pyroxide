"""Merge benchmark JSONL results into benchmarks/RESULTS.md."""
import json
import os
import platform
import statistics
import subprocess
from collections import defaultdict

here = os.path.dirname(os.path.abspath(__file__))


def load(name):
    rows = []
    path = os.path.join(here, "results", name)
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def med(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return statistics.median(vals) if vals else float("nan")


def main():
    rows = load("pyroxide.jsonl") + load("numpyro.jsonl")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["algo"], r["lib"])].append(r)
    keys = sorted({(m, a) for (m, a, _) in groups}, key=lambda k: (["nuts", "hmc", "mh"].index(k[1]), k[0]))

    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        cpu = platform.processor()
    out = []
    out.append("# Benchmarks: pyroxide vs numpyro\n")
    out.append(f"Machine: {cpu}, {platform.system()} {platform.release()}. Single chain, CPU, float64 in both libraries.\n")
    out.append("`time` is wall-clock for warmup + sampling, median over repeats. numpyro `time` includes XLA compilation of the")
    out.append("sampler for that model; `cached` is a second run reusing the compiled program. `ESS/s` uses the minimum effective")
    out.append("sample size over all parameters divided by the (uncached) wall time.\n")
    out.append("| model | algo | warmup/samples | pyroxide time (s) | numpyro time (s) | numpyro cached (s) | speedup vs numpyro | speedup vs cached | pyroxide min ESS | numpyro min ESS | pyroxide ESS/s | numpyro ESS/s |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (model, algo) in keys:
        px = groups.get((model, algo, "pyroxide"), [])
        npy = groups.get((model, algo, "numpyro"), [])
        if not px and not npy:
            continue
        ws = f"{(px or npy)[0]['warmup']}/{(px or npy)[0]['samples']}"
        t_px = med(px, "time_s")
        t_np = med(npy, "time_s")
        t_npc = med(npy, "time_s_cached")
        out.append(
            f"| {model} | {algo} | {ws} | {t_px:.3f} | {t_np:.3f} | {t_npc:.3f} | {t_np / t_px:.1f}x | {t_npc / t_px:.1f}x | "
            f"{med(px, 'ess_min'):.0f} | {med(npy, 'ess_min'):.0f} | {med(px, 'ess_min_per_s'):.0f} | {med(npy, 'ess_min_per_s'):.0f} |"
        )
    out.append("")
    out.append("## Sampler diagnostics (median over repeats)\n")
    out.append("| model | algo | lib | leapfrog steps | mean accept | divergences | max R-hat |")
    out.append("|---|---|---|---:|---:|---:|---:|")
    for (model, algo) in keys:
        for lib in ["pyroxide", "numpyro"]:
            g = groups.get((model, algo, lib), [])
            if not g:
                continue
            out.append(
                f"| {model} | {algo} | {lib} | {med(g, 'leapfrog_steps'):.0f} | {med(g, 'mean_accept_prob'):.3f} | "
                f"{med(g, 'num_divergences'):.0f} | {med(g, 'max_rhat'):.3f} |"
            )
    text = "\n".join(out) + "\n"
    with open(os.path.join(here, "RESULTS.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
