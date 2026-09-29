"""Warm-start savings, predicted vs observed savings, and in-match influence timelines."""

import numpy as np
import pandas as pd

from common import FIXED_WINDOWS, GRID, INK, INK_2, PALETTE, mpl, parse_args, save_figure


def savings_figure(windows: pd.DataFrame):
    plt = mpl()
    warm = windows.dropna(subset=["warm_iterations"])
    fixed = warm[warm["mode"].eq("fixed")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    means = fixed.groupby("alpha")[["cold_iterations", "warm_iterations"]].mean()
    x = np.arange(len(means))
    left.bar(x - 0.18, means.cold_iterations, 0.36, color=INK_2, label="cold start (from σ)")
    left.bar(x + 0.18, means.warm_iterations, 0.36, color=PALETTE["blue"], label="warm start (previous window)")
    for i, (c, w) in enumerate(zip(means.cold_iterations, means.warm_iterations)):
        left.text(i - 0.18, c, f"{c:.1f}", ha="center", va="bottom")
        left.text(i + 0.18, w, f"{w:.1f}", ha="center", va="bottom")
    left.set(xticks=x, xticklabels=[f"α = {a:g}" for a in means.index], ylabel="mean power iterations",
             title="15-minute windows: iterations to tolerance")
    left.legend(loc="upper left")
    ok = warm.dropna(subset=["predicted_saving"])
    for alpha, color in zip(sorted(ok.alpha.unique()), [PALETTE["blue"], PALETTE["orange"]]):
        rows = ok[ok.alpha.eq(alpha)]
        right.scatter(rows.predicted_saving, rows.iterations_saved, s=14, alpha=0.5, color=color, label=f"α = {alpha:g}")
    if len(ok):
        lo = min(ok.predicted_saving.min(), ok.iterations_saved.min())
        hi = max(ok.predicted_saving.max(), ok.iterations_saved.max())
        right.plot([lo, hi], [lo, hi], color=INK, linewidth=1, linestyle="--", label="observed = predicted")
    right.set(xlabel="predicted saving  log(e_cold / e_warm) / log(1/ρ)", ylabel="observed iterations saved",
              title="Warm-start saving follows the convergence rate")
    right.legend(loc="upper left")
    fig.tight_layout()
    return fig


def timeline_figure(vectors: pd.DataFrame, network_id: str, team: str):
    plt = mpl()
    rows = vectors[vectors.network_id.eq(network_id) & vectors["mode"].eq("fixed") & vectors.alpha.eq(0.85)]
    order = [f"{a:g}+" if b is None else f"{a:g}-{b:g}" for a, b in FIXED_WINDOWS]
    table = rows.pivot_table(index="player", columns="window", values="score").reindex(columns=order)
    top = table.mean(axis=1).sort_values(ascending=False).index[:6]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = list(PALETTE.values())
    for color, player in zip(colors, top):
        ax.plot(range(len(order)), table.loc[player], marker="o", color=color, label=player)
    ax.set(xticks=range(len(order)), xticklabels=[f"{w}'" for w in order], xlabel="match window (minutes)",
           ylabel="PageRank (α = 0.85)", title=f"{team}: in-match passing influence, top 6 players")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, color=GRID)
    fig.tight_layout()
    return fig


def main():
    args = parse_args(__doc__)
    windows = pd.read_csv(args.output / "dynamic_windows.csv", encoding="utf-8")
    vectors = pd.read_csv(args.output / "dynamic_vectors.csv", encoding="utf-8")
    save_figure(savings_figure(windows), args.output / "figures" / "warm_vs_cold.png")
    first = windows.match_id.iloc[0]
    for (network_id, team), _ in windows[windows.match_id.eq(first)].groupby(["network_id", "team"]):
        save_figure(timeline_figure(vectors, network_id, team), args.output / "figures" / f"{network_id}_timeline.png")
    print("Stage 4 figures written.", flush=True)


if __name__ == "__main__":
    main()
