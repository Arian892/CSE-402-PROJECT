"""In-match windows and warm starts, on synthetic StatsBomb records."""

import numpy as np
import pytest

import pprlib as pl
from football import statsbomb
from football.dynamics import dynamic_ppr, substitution_windows, transfer

TEAM = {"id": 10, "name": "Test FC"}
PEOPLE = {1: "A", 2: "B", 3: "C", 4: "D"}


def event(index, kind, player=None, minute=0, second=0, **extra):
    row = dict(id=str(index), index=index, type={"name": kind}, team=TEAM, minute=minute, second=second,
               period=1, timestamp="00:00:00.000")
    if player is not None:
        row.update(player={"id": player, "name": PEOPLE[player]}, position={"name": "Center Forward"},
                   location=[50, 40])
    row.update(extra)
    return row


def passing(index, player, recipient, minute):
    return event(index, "Pass", player, minute,
                 **{"pass": {"recipient": {"id": recipient, "name": PEOPLE[recipient]}, "end_location": [60, 40]}})


def starting_xi():
    return event(1, "Starting XI", tactics={"lineup": [
        {"player": {"id": i, "name": PEOPLE[i]}, "position": {"name": "Center Forward"}} for i in (1, 2, 3)]})


def cycle(start_index, minute):
    return [passing(start_index, 1, 2, minute), passing(start_index + 1, 2, 3, minute),
            passing(start_index + 2, 3, 1, minute), passing(start_index + 3, 1, 3, minute)]


def test_transfer_matches_labels_and_shares_new_players():
    g = pl.Graph.from_edges([("B", "A", 1), ("A", "D", 1)], nodes=["A", "B", "D"])
    x = transfer(["A", "B", "C"], np.array([0.5, 0.3, 0.2]), g)
    assert x.sum() == pytest.approx(1.0)
    raw = np.array([0.5, 0.3, 1 / 3])
    np.testing.assert_allclose(x, raw / raw.sum())


def test_substitution_windows_cut_at_the_match_clock():
    rows = [starting_xi(), passing(2, 1, 2, 5),
            event(3, "Substitution", 3, 30, 30, substitution={"replacement": {"id": 4, "name": "D"}}),
            passing(4, 4, 1, 40)]
    events = statsbomb.events_frame(rows, 100)
    assert substitution_windows(events, 10) == [(0.0, 30.5), (30.5, None)]


def test_identical_windows_make_the_warm_start_almost_free():
    rows = [starting_xi(), *cycle(2, 3), *cycle(6, 18)]
    events = statsbomb.events_frame(rows, 100)
    table, vectors = dynamic_ppr(events, 10, windows=[(0, 15), (15, 30)], alpha=0.85)
    assert len(table) == 2
    second = table.iloc[1]
    assert second.warm_start_error < 1e-12
    assert second.warm_iterations <= 2 < second.cold_iterations
    assert second.iterations_saved == second.cold_iterations - second.warm_iterations
    assert table.cold_final_error.max() < 1e-8 and second.warm_final_error < 1e-8
    assert second.tau_vs_previous == pytest.approx(1.0)
    assert vectors.groupby("window").score.sum().to_numpy() == pytest.approx([1.0, 1.0])


def test_empty_windows_are_skipped_and_substitutes_tracked():
    rows = [starting_xi(), *cycle(2, 3),
            event(6, "Substitution", 3, 20, substitution={"replacement": {"id": 4, "name": "D"}}),
            passing(7, 4, 1, 50), passing(8, 1, 4, 51), passing(9, 1, 2, 52)]
    events = statsbomb.events_frame(rows, 100)
    table, _ = dynamic_ppr(events, 10, windows=[(0, 15), (60, 75), (45, 60)], alpha=0.85)
    assert table.window.tolist() == ["0-15", "45-60"]
    last = table.iloc[-1]
    assert last.players_entered == 1 and last.players_left == 1
    assert last.warm_final_error < 1e-8
