"""Windowed PPR for every selected team-match: 15-minute and substitution windows, cold vs warm starts."""

import numpy as np
import pandas as pd

from common import (ALPHAS, FIXED_WINDOWS, MODES, dynamic_ppr, parse_args, save_csv, selected_matches,
                    statsbomb, substitution_windows)


def main():
    args = parse_args(__doc__)
    tables, vectors = [], []
    for match in selected_matches(args):
        mid = int(match["match_id"])
        events = statsbomb.load_events(args.data, mid)
        lineups = statsbomb.load_lineups(args.data, mid)
        for team in sorted(match["team_ids"]):
            context = dict(network_id=f"m{mid}_t{team}", match_id=mid, team_id=team,
                           team=str(events.loc[events.team_id.eq(team), "team"].iloc[0]),
                           competition=match["competition"])
            for mode in MODES:
                windows = FIXED_WINDOWS if mode == "fixed" else substitution_windows(events, team)
                for alpha in ALPHAS:
                    table, vecs = dynamic_ppr(events, team, lineups, windows=windows, alpha=alpha, tol=args.tol)
                    cols = [c for c in ("cold_final_error", "warm_final_error") if c in table]
                    bad = np.nanmax(table[cols].to_numpy()) if len(table) else 0.0
                    if bad > 1e-7:
                        raise ArithmeticError(f"power iteration missed the exact answer ({bad:.3g}) for {context['network_id']}")
                    tables.append(table.assign(mode=mode, **context))
                    vectors.append(vecs.assign(mode=mode, **context))
        print(f"Windows solved: match {mid}", flush=True)

    windows = pd.concat(tables, ignore_index=True)
    save_csv(windows, args.output / "dynamic_windows.csv")
    save_csv(pd.concat(vectors, ignore_index=True), args.output / "dynamic_vectors.csv")
    warm = windows.dropna(subset=["warm_iterations"])
    summary = warm.groupby(["mode", "alpha"]).agg(
        windows=("warm_iterations", "size"), mean_cold_iterations=("cold_iterations", "mean"),
        mean_warm_iterations=("warm_iterations", "mean"), mean_iterations_saved=("iterations_saved", "mean"),
        mean_cold_start_error=("cold_start_error", "mean"), mean_warm_start_error=("warm_start_error", "mean"),
        mean_tau_vs_previous=("tau_vs_previous", "mean")).reset_index()
    summary["percent_saved"] = 100 * summary.mean_iterations_saved / summary.mean_cold_iterations
    save_csv(summary, args.output / "warm_start_summary.csv")
    print(summary.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
