"""E5: Effect of the power-iteration count K and the batch count B (paper Figs. 10-11).

* Varying K, with T fixed: a larger K means lower error (variance times alpha^(2K)) and a longer run (K m).
* Varying B, with T and the total number of power iterations fixed (K_batch = ceil(K_total / B)).
  The paper reports that the error first falls and then rises. The rise only
  appears once T/B falls below the level at which each batch contracts the
  residual (see E2c), so whether it shows up depends on T.

Single-source queries.
"""

import math

import pandas as pd

from common import (
    budget, evaluate, graph, log, method_legend, method_line, mpl, nlogn, paper_alpha_label,
    parse_args, query_nodes, save_csv, save_fig, too_expensive,
)
from pprlib import theory

K_GRID = (1, 2, 5, 10, 20, 50, 100, 200)
K_GRID_LARGE = (1, 5, 20, 100)  # a coarser grid on the 1.1M-node graph
B_GRID = (1, 2, 3, 4, 5, 6, 7)

WALK_METHODS = ["pw", "ppw"]

# x-axis label of each sweep.
SWEEP_LABELS = (("K", "power iterations K (per batch)"), ("B", "batches B (total power iterations fixed)"))
SWEEP_TITLES = {
    "K": "E5 · Varying K with T fixed: L1 error (top) and time (bottom)",
    "B": "E5 · Varying B with T and total power iterations fixed: L1 error (top) and time (bottom)",
}


def walk_budget(g, alpha, large: bool) -> int:
    """n ln n walks (1/16 of that on the large graph), divided by 4 until within the budget guard."""
    T = nlogn(g.n) // (16 if large else 1)
    while too_expensive(T, alpha):
        T //= 4
    return T


def run(args):
    rows = []
    for name in args.datasets:
        g = graph(name)
        b = budget(name, args.quick)
        qs = query_nodes(g, b["queries"], args.seed)
        trials = max(2, b["trials"] // 2)
        und = g.is_undirected()
        k_grid = K_GRID if not b["large"] else K_GRID_LARGE
        for alpha in args.alphas:
            T = walk_budget(g, alpha, b["large"])
            F = max(3, math.ceil(math.log(g.n)))
            forest_m = ["pf", "ppf"] + (["pfv", "ppfv"] if und else [])
            for K in k_grid:
                for m in WALK_METHODS + forest_m:
                    kw = dict(n_walks=T) if m in WALK_METHODS else dict(n_forests=F)
                    s = evaluate(name, g, alpha, "ssq", qs, m, trials, args.seed, K=K, **kw)
                    rows.append(dict(dataset=name, alpha=alpha, sweep="K", value=K, method=m, T=T, F=F, **s))
            log(f"{name} alpha={alpha}: K sweep done (T={T:,}, F={F})")
            K_total = 3 * theory.theoretical_K(alpha, 0.5 / math.sqrt(3))  # the default B=3 configuration
            for B in B_GRID:
                for m in ["ppw", "ppf"] + (["ppfv"] if und else []):
                    kw = dict(n_walks=T) if m == "ppw" else dict(n_forests=max(F, 2 * B))
                    s = evaluate(name, g, alpha, "ssq", qs, m, trials, args.seed,
                                 K=math.ceil(K_total / B), n_batches=B, **kw)
                    rows.append(dict(dataset=name, alpha=alpha, sweep="B", value=B, method=m, T=T, F=F,
                                     K_total=K_total, **s))
            log(f"{name} alpha={alpha}: B sweep done (K_total={K_total})")
    df = pd.DataFrame(rows)
    save_csv(df, "e5_parameters")
    plot(df)
    return df


def plot(df):
    """For each sweep: L1 error (top row) and time (bottom row), one column per (dataset, alpha)."""
    plt = mpl()
    for sweep, xlabel in SWEEP_LABELS:
        d_s = df[df.sweep == sweep]
        combos = list(dict.fromkeys(zip(d_s.dataset, d_s.alpha)))
        fig, axes = plt.subplots(2, len(combos), figsize=(3.9 * len(combos), 6.6), squeeze=False)
        seen = {}
        for j, (name, alpha) in enumerate(combos):
            d = d_s[(d_s.dataset == name) & (d_s.alpha == alpha)]
            top, bottom = axes[0, j], axes[1, j]
            for m in dict.fromkeys(d.method):
                dm = d[d.method == m].sort_values("value")
                (ln,) = method_line(top, dm.value, dm.l1.clip(lower=1e-16), m)
                method_line(bottom, dm.value, dm.time, m)
                seen.setdefault(m, ln)
            for ax in (top, bottom):
                ax.set_yscale("log")
            if sweep == "K":
                for ax in (top, bottom):
                    ax.set_xscale("log")
            top.set_title(f"{name} · {paper_alpha_label(alpha)}", fontsize=9)
            bottom.set_xlabel(xlabel)
        axes[0, 0].set_ylabel("mean L1 error")
        axes[1, 0].set_ylabel("time per query (s)")
        method_legend(fig, seen)
        fig.suptitle(SWEEP_TITLES[sweep], x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save_fig(fig, f"e5_vary_{sweep}")


if __name__ == "__main__":
    run(parse_args(__doc__))
