"""E2: Empirical variance vs the paper's variance formulas.

For an unbiased estimator averaged over T i.i.d. samples, E||pi_hat - pi||_2^2
equals the single-sample variance divided by T. We therefore estimate the
single-sample variance as ``T * MSE`` over many independent trials and
compare it with:

(a) MCW:  Var = 1 - ||pi||_2^2                                         (Lemma 3.2)
(b) PW:   exact Var[x^(K)] and the bound alpha^(2K) (1 - ||pi||_2^2)    (Lemma 3.7)
(c) PPW:  per-walk variance of batch b, ||r_b||_1^2 - ||Pi r_b||_2^2    (Lemma 3.14),
          which should fall batch after batch as the residual shrinks.
"""

import numpy as np
import pandas as pd

from common import (
    FLOOR_SHADE, INK, INK_2, METHOD_STYLE, REFERENCE, SURFACE, graph, ground_truth, log, mpl,
    nlogn, paper_alpha_label, parse_args, query_nodes, save_csv, save_fig, sources,
)
import pprlib as pl
from pprlib import evaluation, theory

DATASETS = ["toy", "email-Eu-core", "ca-GrQc"]

# Exact variances below this are at the float64 round-off floor and cannot be compared.
ROUNDOFF_VARIANCE = 1e-20

# Number of PPW batches in part (c).
PPW_BATCHES = 6

# Smallest per-batch walk budget used in part (c).
SMALL_BATCH_WALKS = 500


def mse_single(results, pi, T):
    """T * MSE and its standard error (from the spread of per-trial squared errors)."""
    sq = np.array([np.sum((r.estimate - pi) ** 2) for r in results])
    return T * sq.mean(), T * sq.std(ddof=1) / np.sqrt(sq.size)


def walk_settings(alpha):
    """Walk count T and the K grid for part (b); alpha close to 1 needs a much wider K range."""
    if alpha < 0.95:
        return 2000, [0, 1, 2, 3, 5, 8, 12]
    return 500, [0, 1, 5, 10, 25, 50, 100, 200]


def run(args):
    trials = 100 if args.quick else 300
    rows_mcw, rows_pw, rows_ppw = [], [], []
    for name in args.datasets:
        g = graph(name)
        qs = query_nodes(g, 1 if name == "toy" else 3, args.seed)
        srcs = sources(g, "ssq", qs) + ([] if name == "toy" else sources(g, "prc", qs))
        for alpha in args.alphas:
            T, Ks = walk_settings(alpha)
            for tag, sigma in srcs:
                pi = ground_truth(name, g, sigma, alpha)
                # (a) MCW
                res = evaluation.run_trials(lambda r: pl.mcw(g, sigma, alpha, T, r), trials, args.seed)
                emp, se = mse_single(res, pi, T)
                rows_mcw.append(dict(dataset=name, alpha=alpha, source=tag, T=T, trials=trials,
                                     empirical=emp, se=se, theory=theory.walk_variance(pi)))
                # (b) PW for several K
                exact = theory.pw_variance_curve(g, pi, alpha, Ks)
                for K, ex in zip(Ks, exact):
                    res = evaluation.run_trials(
                        lambda r: pl.pw(g, sigma, alpha, T, K, r), trials, args.seed + 1)
                    emp, se = mse_single(res, pi, T)
                    rows_pw.append(dict(dataset=name, alpha=alpha, source=tag, K=K, T=T,
                                        empirical=emp, se=se, exact=ex,
                                        bound=theory.pw_variance_bound(pi, alpha, K)))
                log(f"{name} {tag} alpha={alpha}: MCW & PW done")
            # (c) PPW residual-driven variance decay, on the first single-source query
            rows_ppw.extend(ppw_rows(name, g, alpha, srcs[0], trials, args.seed))
            log(f"{name} alpha={alpha}: PPW done")
    a, b, c = pd.DataFrame(rows_mcw), pd.DataFrame(rows_pw), pd.DataFrame(rows_ppw)
    save_csv(a, "e2a_mcw_variance")
    save_csv(b, "e2b_pw_variance")
    save_csv(c, "e2c_ppw_variance")
    plot(a, b, c, args)
    report(a, b)
    return a, b, c


def ppw_rows(name, g, alpha, source, trials, seed):
    """Per-batch residual and per-walk variance of PPW for one source.

    Two per-batch budgets are used: a small one, and the paper's regime (about
    n ln n walks in total). PPW only contracts the residual when the per-batch
    walk count is large enough.
    """
    tag, sigma = source
    pi = ground_truth(name, g, sigma, alpha)
    Pi = pl.ppr_matrix(g, alpha)
    B, K = PPW_BATCHES, 1 if alpha < 0.95 else 10
    rows = []
    for T_b in sorted({SMALL_BATCH_WALKS, max(SMALL_BATCH_WALKS, nlogn(g.n) // 2)}):
        res = evaluation.run_trials(
            lambda r: pl.ppw(g, sigma, alpha, B * T_b, K, B, r, record_residuals=True),
            max(20, trials // 5), seed + 2)
        for b in range(B):
            v = [theory.residual_walk_variance(Pi, x.history["residuals"][b]) for x in res]
            rl1 = [x.history["residual_l1"][b] for x in res]
            rows.append(dict(dataset=name, alpha=alpha, source=tag, batch=b + 1, K=K, T_batch=T_b,
                             residual_l1=np.mean(rl1), per_walk_variance=np.mean(v),
                             mcw_variance=theory.walk_variance(pi),
                             pw_bound_same_K=theory.pw_variance_bound(pi, alpha, K)))
    return rows


def report(a, b):
    """Log how closely the empirical variances follow Lemmas 3.2 and 3.7."""
    a["rel_err"] = (a.empirical - a.theory) / a.theory
    b["rel_err_vs_exact"] = (b.empirical - b.exact) / b.exact.where(b.exact > ROUNDOFF_VARIANCE)
    violations = int((b.exact > b.bound * (1 + 1e-9)).sum())
    log(f"MCW: max |empirical/theory - 1| = {a.rel_err.abs().max():.3f}")
    log(f"PW (exact var > 1e-20): max |empirical/exact - 1| = {b.rel_err_vs_exact.abs().max():.3f}; "
        f"bound violated: {violations} times")


def plot(a, b, c, args):
    """One figure per part: (a) MCW identity plot, (b) PW variance vs K, (c) PPW variance by batch."""
    plt = mpl()
    plot_mcw(plt, a, args)
    plot_pw(plt, b, args)
    plot_ppw(plt, c, args)


def plot_mcw(plt, a, args):
    """(a) Empirical vs theoretical MCW variance; points should lie on the diagonal."""
    fig, ax = plt.subplots(figsize=(5, 4.4))
    lo, hi = a.theory.min() * 0.8, 1.05
    ax.plot([lo, hi], [lo, hi], color=REFERENCE, linewidth=1, label="empirical = theory")
    for alpha, mk in zip(args.alphas, ("o", "s")):
        d = a[a.alpha == alpha]
        ax.errorbar(d.theory, d.empirical, yerr=2 * d.se, linestyle="none", marker=mk, markersize=7,
                    color=METHOD_STYLE["mcw"]["color"], mfc=SURFACE if mk == "s" else None,
                    label=paper_alpha_label(alpha), elinewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("theory: 1 − ‖π‖₂²  (Lemma 3.2)")
    ax.set_ylabel("empirical single-walk variance  T·MSE  (±2 s.e.)")
    ax.set_title("E2a · MCW variance matches Lemma 3.2")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "e2a_mcw_variance")


def plot_pw(plt, b, args):
    """(b) PW variance vs K, one panel per (dataset, alpha), first single-source query."""
    n_names = b.dataset.nunique()
    fig, axes = plt.subplots(len(args.alphas), n_names,
                             figsize=(4.2 * n_names, 3.6 * len(args.alphas)), squeeze=False)
    names = list(dict.fromkeys(b.dataset))
    st = METHOD_STYLE["pw"]
    for (name, alpha), d in b.groupby(["dataset", "alpha"], sort=False):
        ax = axes[list(args.alphas).index(alpha), names.index(name)]
        d = d[d.source == d.source.iloc[0]]
        ax.plot(d.K, d.bound, color=st["color"], linestyle="--", linewidth=1.2, label="bound α²ᴷ(1−‖π‖²)")
        ax.plot(d.K, d.exact, color=st["color"], label="exact Var[x̂⁽ᴷ⁾]")
        ax.errorbar(d.K, d.empirical, yerr=2 * d.se, linestyle="none", marker="o", color=INK,
                    markersize=5, label="empirical T·MSE", elinewidth=1)
        ax.set_yscale("log")
        if d.exact.min() < 1e-22:
            shade_roundoff_floor(ax, d)
        ax.set_title(f"{name} · {d.source.iloc[0]} · {paper_alpha_label(alpha)}", fontsize=9)
        ax.set_xlabel("power iterations K")
        ax.set_ylabel("single-walk variance")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("E2b · PW: each power iteration shrinks the variance by at least α² (Lemma 3.7)",
                 x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, "e2b_pw_variance")


def shade_roundoff_floor(ax, d):
    """Grey out the region below which float64 cannot resolve the variance."""
    ax.axhspan(1e-40, 1e-24, color=FLOOR_SHADE, zorder=0)
    ax.text(d.K.max(), 3e-25, "float64 round-off floor", ha="right", va="top", fontsize=8, color=INK_2)
    ax.set_ylim(bottom=max(d.exact.min() / 10, 1e-34))


def plot_ppw(plt, c, args):
    """(c) PPW per-walk variance by batch, one line per per-batch walk budget."""
    names = list(dict.fromkeys(c.dataset))
    fig, axes = plt.subplots(len(args.alphas), len(names), figsize=(4.2 * len(names), 3.6 * len(args.alphas)),
                             squeeze=False, sharex=True)
    col = METHOD_STYLE["ppw"]["color"]
    for i, alpha in enumerate(args.alphas):
        for j, name in enumerate(names):
            ax = axes[i, j]
            d0 = c[(c.dataset == name) & (c.alpha == alpha)]
            budgets = sorted(d0.T_batch.unique())
            for T_b in budgets:
                d = d0[d0.T_batch == T_b]
                small = T_b == budgets[0] and len(budgets) > 1
                ax.semilogy(d.batch, d.per_walk_variance, color=col, marker="o",
                            linestyle="--" if small else "-", mfc=SURFACE if small else col,
                            markeredgecolor=col, label=f"{T_b:,} walks / batch")
            ax.axhline(d0.mcw_variance.iloc[0], color=METHOD_STYLE["mcw"]["color"], linewidth=1,
                       label="MCW per-walk variance")
            ax.set_title(f"{name} · {paper_alpha_label(alpha)} · K={int(d0.K.iloc[0])}", fontsize=9)
            ax.set_xlabel("batch b")
            if j == 0:
                ax.set_ylabel("per-walk variance ‖r_b‖₁² − ‖Π r_b‖₂²")
            ax.legend(fontsize=7)
    fig.suptitle("E2c · PPW (Lemma 3.14): with enough walks per batch the residual and variance shrink geometrically;\n"
                 "with too few (dashed), Monte Carlo noise amplified by 1/(1−α) makes the residual grow",
                 x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, "e2c_ppw_variance")


def select_datasets(requested):
    """E2 always includes the toy graph and only runs on DATASETS.

    The default (quick) selection, or any request containing an unknown
    dataset, is mapped onto DATASETS, keeping the toy graph.
    """
    if requested in (["email-Eu-core", "ca-GrQc"], None) or set(requested) - set(DATASETS):
        return [d for d in DATASETS if d in requested or d == "toy"] or DATASETS
    return requested


if __name__ == "__main__":
    args = parse_args(__doc__)
    args.datasets = select_datasets(args.datasets)
    run(args)
