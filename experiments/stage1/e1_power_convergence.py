"""E1: Convergence of the deterministic power method.

We check three things:
* the error decays geometrically at rate alpha (a priori: ||r^(k) - pi||_1 <= 2 alpha^k);
* the a-posteriori bound alpha/(1-alpha) * ||r^(k) - r^(k-1)||_1 holds and is tight;
* the number of iterations needed for a tolerance grows like log(tol)/log(alpha),
  which is why small paper-alpha (alpha -> 1 here) is the hard case.
"""

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla

from common import (
    PALETTE, REFERENCE, SOURCE_KINDS, SURFACE, alpha_colors, graph, ground_truth, log, mpl,
    paper_alpha_label, parse_args, query_nodes, save_csv, save_fig, source_label, sources,
)
import pprlib as pl

ALPHAS = (0.5, 0.8, 0.9, 0.95, 0.99)

# Power-iteration settings: run well past the tolerances we report on.
POWER_TOL = 1e-13
POWER_MAX_ITER = 20000

# Errors below this are round-off and are left out of the rate fit.
ROUNDOFF_LEVEL = 1e-12

# Tolerances for which we count the iterations needed.
TOL_4, TOL_8 = 1e-4, 1e-8


def second_eigenvalue_modulus(g) -> float:
    """|lambda_2(P)|. The power method's asymptotic rate is alpha * |lambda_2|, which is at most alpha."""
    vals = spla.eigs(g.P.astype(float), k=3, which="LM", return_eigenvectors=False, maxiter=5000, tol=1e-8)
    mods = np.sort(np.abs(vals))[::-1]
    return float(mods[1])


def observed_slope(err: np.ndarray) -> float:
    """Slope of log(error) vs k, fitted where the error is above round-off level.

    exp(slope) is the geometric-mean decay factor per iteration.
    """
    k = np.arange(err.size)
    ok = (err > ROUNDOFF_LEVEL) & (k >= 1)
    return np.polyfit(k[ok], np.log(err[ok]), 1)[0] if ok.sum() > 2 else np.nan


def iterations_to(err: np.ndarray, tol: float):
    """First iteration whose error is below ``tol`` (NaN if it never gets there)."""
    return int(np.argmax(err < tol)) if (err < tol).any() else np.nan


def curve_rows(name, kind, alpha, err, apost):
    """One row per iteration: measured error and both bounds."""
    return [dict(dataset=name, source=kind, alpha=alpha, k=kk, error=err[kk],
                 apriori_bound=2 * alpha ** kk, aposteriori_bound=apost[kk])
            for kk in range(err.size)]


def run(args):
    rows, summary = [], []
    for name in args.datasets:
        g = graph(name)
        lam2 = second_eigenvalue_modulus(g)
        log(f"{name}: |lambda_2(P)| = {lam2:.4f}")
        q = query_nodes(g, 1, args.seed)
        for kind in SOURCE_KINDS:
            _tag, sigma = sources(g, kind, q)[0]
            for alpha in args.alphas:
                pi = ground_truth(name, g, sigma, alpha)
                res = pl.power_iteration(g, sigma, alpha, tol=POWER_TOL, max_iter=POWER_MAX_ITER, exact=pi)
                err = res.history["error"]
                apost = np.r_[np.nan, res.history["a_posteriori_bound"]]
                rows.extend(curve_rows(name, kind, alpha, err, apost))
                slope = observed_slope(err)
                summary.append(dict(
                    dataset=name, source=kind, alpha=alpha, paper_alpha=round(1 - alpha, 4),
                    observed_rate=float(np.exp(slope)), theory_rate=alpha, alpha_lambda2=alpha * lam2,
                    iters_to_1e4=iterations_to(err, TOL_4),
                    theory_iters_1e4=pl.theory.power_iterations_for_tol(alpha, TOL_4),
                    iters_to_1e8=iterations_to(err, TOL_8),
                    theory_iters_1e8=pl.theory.power_iterations_for_tol(alpha, TOL_8),
                    time_per_iter_ms=1e3 * res.time / max(res.n_power_iterations, 1),
                    apost_bound_holds=bool(np.all(
                        err[1:] <= res.history["a_posteriori_bound"] * (1 + 1e-9) + 1e-15)),
                ))
                log(f"{name} {kind} alpha={alpha}: rate {np.exp(slope):.4f} (theory {alpha}), "
                    f"{res.n_power_iterations} iters, {res.time:.2f}s")
    df, sm = pd.DataFrame(rows), pd.DataFrame(summary)
    save_csv(df, "e1_power_convergence_curves")
    save_csv(sm, "e1_power_convergence_summary")
    plot(df, sm, args)
    return sm


def plot(df, sm, args):
    """Error curves per (dataset, source), then observed vs theoretical rate."""
    plt = mpl()
    plot_curves(plt, df, args)
    plot_rates(plt, sm)


def plot_curves(plt, df, args):
    """Grid of L1 error vs iteration: sources down the rows, datasets across the columns."""
    names = list(dict.fromkeys(df.dataset))
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7), squeeze=False, sharey=True)
    cols = alpha_colors(args.alphas)
    for j, name in enumerate(names):
        for i, kind in enumerate(SOURCE_KINDS):
            ax = axes[i, j]
            d_k = df[(df.dataset == name) & (df.source == kind)]
            for a, c in zip(args.alphas, cols):
                d = d_k[d_k.alpha == a]
                ax.semilogy(d.k, d.error, color=c, label=paper_alpha_label(a))
                ax.semilogy(d.k, d.apriori_bound, color=c, linewidth=1, linestyle=":")
            ax.set_ylim(1e-14, 3)
            ax.set_xlim(0, d_k.k.max())
            ax.set_title(f"{name} · {source_label(kind)}")
            ax.set_xlabel("iteration k")
            if j == 0:
                ax.set_ylabel("L1 error ‖r⁽ᵏ⁾ − π‖₁")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("E1 · Power-method L1 error vs iteration: straight lines = geometric decay at rate α\n"
                 "solid = measured error, dotted = a-priori bound 2αᵏ", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, "e1_power_convergence")


def plot_rates(plt, sm):
    """Observed decay factor against alpha; points on the diagonal match theory."""
    blue = PALETTE["blue"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0.45, 1], [0.45, 1], color=REFERENCE, linewidth=1, label="rate = α (theory)")
    for kind, mk in zip(SOURCE_KINDS, ("o", "s")):
        d = sm[sm.source == kind]
        ax.plot(d.theory_rate, d.observed_rate, linestyle="none", marker=mk, markersize=8,
                color=blue, mfc=SURFACE if kind == "prc" else blue, label=source_label(kind))
    ax.set_xlabel("damping factor α")
    ax.set_ylabel("observed decay factor per iteration")
    ax.set_title("E1 · Observed convergence rate vs theory (all datasets)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "e1_power_rate")


if __name__ == "__main__":
    run(parse_args(__doc__, default_alphas=ALPHAS))
