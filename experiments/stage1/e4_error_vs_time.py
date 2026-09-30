"""E4: Accuracy vs running time for every method (paper Figs. 4-5).

For single-source queries (SSQ) and PageRank centrality (PRC) we sweep each
method's budget and plot the mean L1 error against wall-clock time:

* Power method: one point per iteration count, which traces its whole convergence curve.
* Walk-based methods (MCW, PW, PPW): T = c * n ln n walks, for several c.
* Forest-based methods (MCF(V), PF(V), PPF(V)): enough forests to cost about the same
  time as the matching T walks, T' = T * tau_walk / tau_forest.
  MCF is only run for PRC, because for a one-hot sigma it can do no better
  than MCW (Section 4.1). The V variants are run on undirected graphs only.

K follows the paper's defaults (eps = 0.5, B = 3).
"""

import math

import numpy as np
import pandas as pd

from common import (
    SOURCE_KINDS, budget, evaluate, graph, ground_truth, log, method_legend, method_line, mpl, nlogn,
    paper_alpha_label, parse_args, query_nodes, save_csv, save_fig, source_label, sources, too_expensive,
)
import pprlib as pl
from pprlib import theory

FACTORS = (1 / 64, 1 / 16, 1 / 4, 1)

# Number of points sampled (log-spaced) from each power-method convergence curve.
POWER_CURVE_POINTS = 25


def methods_for(g, kind):
    """Methods to run for this graph and source kind (see the module docstring)."""
    ms = ["mcw", "pw", "ppw"] + (["mcf"] if kind == "prc" else []) + ["pf", "ppf"]
    if g.is_undirected():
        ms += (["mcfv"] if kind == "prc" else []) + ["pfv", "ppfv"]
    return ms


def is_forest_method(m: str) -> bool:
    """True for the spanning-forest methods (mcf, pf, ppf and their V variants)."""
    return "f" in m[1:]


def power_curve(name, g, alpha, kind, qs):
    """Error and time of the power method after k iterations, averaged over sources."""
    rows = []
    for _tag, sigma in sources(g, kind, qs):
        pi = ground_truth(name, g, sigma, alpha)
        res = pl.power_iteration(g, sigma, alpha, tol=1e-14, max_iter=5000, exact=pi)
        per_iter = res.time / max(res.n_power_iterations, 1)
        err = res.history["error"]
        ks = np.unique(np.geomspace(1, err.size - 1, POWER_CURVE_POINTS).astype(int))
        rows.append(pd.DataFrame({"k": ks, "time": ks * per_iter, "l1": err[ks]}))
    return pd.concat(rows).groupby("k", as_index=False).mean()


def unit_times(g, alpha, seed):
    """Measured time of one alpha-random walk and of one spanning forest."""
    tw = theory.walk_time(g, alpha, n_walks=min(200_000, 20 * g.n), rng=seed)["tau_walk"]
    tf = theory.forest_statistics(g, alpha, n_forests=3, rng=seed)["tau_forest"]
    return tw, tf


def run(args):
    rows = []
    for name in args.datasets:
        g = graph(name)
        b = budget(name, args.quick)
        qs = query_nodes(g, b["queries"], args.seed)
        trials = max(2, b["trials"] // 4)
        for alpha in args.alphas:
            tw, tf = unit_times(g, alpha, args.seed)
            log(f"{name} alpha={alpha}: tau_walk={tw:.2e}s tau_forest={tf:.2e}s "
                f"(n tau_walk/tau_forest={g.n * tw / tf:.2f})")
            for kind in SOURCE_KINDS:
                pc = power_curve(name, g, alpha, kind, qs)
                for _, r in pc.iterrows():
                    rows.append(dict(dataset=name, alpha=alpha, source=kind, method="power", budget=r.k,
                                     time=r.time, l1=r.l1, l1_std=0.0))
                for f in FACTORS:
                    T = max(10, int(f * nlogn(g.n)))
                    if too_expensive(T, alpha):
                        log(f"skip {name} alpha={alpha} T={T:,} (budget guard)")
                        continue
                    # Forests matched in time to T walks.
                    F = max(1, math.ceil(T * tw / tf))
                    for m in methods_for(g, kind):
                        kw = dict(n_forests=F) if is_forest_method(m) else dict(n_walks=T)
                        s = evaluate(name, g, alpha, kind, qs, m, trials, args.seed, **kw)
                        rows.append(dict(dataset=name, alpha=alpha, source=kind, method=m,
                                         budget=F if "n_forests" in kw else T, **s))
                    log(f"{name} alpha={alpha} {kind} c={f:g}: done")
    df = pd.DataFrame(rows)
    save_csv(df, "e4_error_vs_time")
    plot(df)
    return df


def plot(df):
    """For each alpha, a grid of L1 error vs time: sources down the rows, datasets across."""
    plt = mpl()
    for alpha in sorted(df.alpha.unique()):
        d_a = df[df.alpha == alpha]
        names = list(dict.fromkeys(d_a.dataset))
        fig, axes = plt.subplots(2, len(names), figsize=(4.1 * len(names), 7.2), squeeze=False)
        seen = {}
        for j, name in enumerate(names):
            for i, kind in enumerate(SOURCE_KINDS):
                ax = axes[i, j]
                d = d_a[(d_a.dataset == name) & (d_a.source == kind)]
                for m in dict.fromkeys(d.method):
                    dm = d[d.method == m].sort_values("time")
                    # The power method has one point per iteration, so no markers.
                    kw = dict(marker=None) if m == "power" else {}
                    (line,) = method_line(ax, dm.time, dm.l1.clip(lower=1e-16), m, **kw)
                    seen.setdefault(m, line)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{name} · {source_label(kind)}", fontsize=10)
                ax.set_xlabel("time per query (s)")
                if j == 0:
                    ax.set_ylabel("mean L1 error")
        method_legend(fig, seen)
        fig.suptitle(f"E4 · Accuracy vs time, lower-left is better · {paper_alpha_label(alpha)}", x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        save_fig(fig, f"e4_error_vs_time_a{alpha}")


if __name__ == "__main__":
    run(parse_args(__doc__))
