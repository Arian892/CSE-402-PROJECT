"""In-match PPR: one passing network per time window, solved by warm-started power iteration.

Each window's power iteration can start from the previous window's answer
instead of from sigma. Players are matched by label; players new to the
window get a uniform share and the vector is renormalized. The saving follows
from the linear convergence rate rho: iterations ~ log(tol / e0) / log(rho), so
a start error reduced from e_cold to e_warm saves about
log(e_cold / e_warm) / log(1 / rho) iterations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import pprlib as pl
from .networks import _team_events, build_network

FIXED_WINDOWS = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, None))


def substitution_windows(events: pd.DataFrame, team) -> list[tuple]:
    """Windows between the team's substitutions, cut at each substitution's match clock."""
    rows = _team_events(events, team)
    subs = rows.loc[rows.type.eq("Substitution") & rows.period.between(1, 4)]
    cuts = sorted({round(float(m) + float(s) / 60.0, 6) for m, s in zip(subs.minute, subs.second)} - {0.0})
    bounds = [0.0, *cuts]
    return [(a, b) for a, b in zip(bounds, bounds[1:])] + [(bounds[-1], None)]


def transfer(prev_labels: list, prev_vector: np.ndarray, graph: pl.Graph) -> np.ndarray:
    """Map a previous window's PPR onto this window's players (new players get 1/n), sum 1."""
    index = {label: i for i, label in enumerate(prev_labels)}
    n = graph.n
    x = np.array([prev_vector[index[label]] if label in index else 1.0 / n for label in graph.labels])
    total = x.sum()
    return x / total if total > 0 else np.full(n, 1.0 / n)


def _window_label(window) -> str:
    start, end = window
    return f"{start:g}+" if end is None else f"{start:g}-{end:g}"


def dynamic_ppr(events: pd.DataFrame, team, lineups: pd.DataFrame | None = None, *, windows=FIXED_WINDOWS,
                alpha: float = 0.85, tol: float = 1e-10, dangling: str = "self_loop") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uniform-source PPR for each window, cold-started from sigma and warm-started from the last window.

    Returns ``(windows_table, vectors_table)``. Windows without team events are skipped.
    ``warm_*`` columns are empty for the first solved window, which has no predecessor.
    """
    rows, vectors = [], []
    prev = None
    for window in windows:
        try:
            network = build_network(events, team, lineups, time_window=window, dangling=dangling)
        except ValueError as exc:
            if "no team events" in str(exc) or "no participating players" in str(exc):
                continue
            raise
        g = network.graph
        sigma = pl.uniform(g)
        exact = pl.exact_ppr(g, sigma, alpha)
        cold = pl.power_iteration(g, sigma, alpha, tol=tol, exact=exact)
        row = dict(window=_window_label(window), start=window[0], end=window[1], alpha=alpha,
                   n_players=g.n, n_edges=g.m, n_passes=int(network.W.sum()),
                   dangling_players=int(g.dangling_nodes.size),
                   cold_iterations=cold.n_power_iterations, cold_start_error=float(cold.history["error"][0]),
                   cold_final_error=pl.metrics.l1_error(cold.estimate, exact),
                   top_player=g.labels[int(np.argmax(exact))], top_score=float(exact.max()))
        errors = cold.history["error"]
        k = len(errors) - 1
        rate = (errors[-1] / errors[0]) ** (1.0 / k) if k > 0 and errors[0] > 0 and errors[-1] > 0 else np.nan
        row["observed_rate"] = float(rate)
        if prev is not None:
            prev_labels, prev_exact = prev
            x0 = transfer(prev_labels, prev_exact, g)
            warm = pl.power_iteration(g, sigma, alpha, tol=tol, x0=x0, exact=exact)
            common = [label for label in g.labels if label in prev_labels]
            tau = np.nan
            if len(common) >= 2:
                now = exact[[g.index(label) for label in common]]
                before = prev_exact[[prev_labels.index(label) for label in common]]
                tau = pl.metrics.kendall_tau(now, before)
            saving = cold.n_power_iterations - warm.n_power_iterations
            predicted = (math.log(row["cold_start_error"] / float(warm.history["error"][0])) / -math.log(rate)
                         if np.isfinite(rate) and 0 < rate < 1 and warm.history["error"][0] > 0 else np.nan)
            row.update(warm_iterations=warm.n_power_iterations, warm_start_error=float(warm.history["error"][0]),
                       warm_final_error=pl.metrics.l1_error(warm.estimate, exact),
                       iterations_saved=saving, predicted_saving=predicted, tau_vs_previous=tau,
                       players_entered=len(set(g.labels) - set(prev_labels)),
                       players_left=len(set(prev_labels) - set(g.labels)))
        rows.append(row)
        vectors += [dict(window=row["window"], start=window[0], alpha=alpha, player=label, score=float(s))
                    for label, s in zip(g.labels, exact)]
        prev = (list(g.labels), exact)
    return pd.DataFrame(rows), pd.DataFrame(vectors)
