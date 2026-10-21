"""Tests du calcul Pocket Clock sur des actions synthétiques dont on connaît la réponse."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pocket_clock import compute_play  # noqa: E402
from model import survival, fit_model, predict, logit, expit, outcomes  # noqa: E402


def make_play(rusher_path, n_frames=40, snap=5, end=35, end_event="pass_forward", qb=(30.0, 26.0)):
    """Construit une action : un QB immobile, un rusher qui suit rusher_path(frame) -> (x, y)."""
    rows = []
    for f in range(1, n_frames + 1):
        ev = "ball_snap" if f == snap else (end_event if f == end else None)
        rows.append({"gameId": 1, "playId": 1, "nflId": 10, "frameId": f, "x": qb[0], "y": qb[1], "event": ev, "pff_role": "Pass"})
        x, y = rusher_path(f)
        rows.append({"gameId": 1, "playId": 1, "nflId": 20, "frameId": f, "x": x, "y": y, "event": ev, "pff_role": "Pass Rush"})
    return pd.DataFrame(rows)


def test_rusher_arrive_au_bon_moment():
    # Le rusher part à 10 yards du QB et avance de 0,5 yard par image après le snap (frame 5).
    # Il passe sous 2 yards quand 10 - 0,5*k < 2, soit k = 17 images après le snap -> 1,7 s.
    p = make_play(lambda f: (40.0 - 0.5 * max(0, f - 5), 26.0))
    play, rushers = compute_play(p)
    assert play["breached"] is True
    assert play["pocket_time"] == pytest.approx(1.7)
    assert play["first_rusher"] == 20
    assert rushers[0]["arrive"] == pytest.approx(1.7)


def test_poche_intacte_jusqu_au_lancer():
    # Le rusher reste à 5 yards : la poche n'est jamais percée, le chrono s'arrête au lancer (frame 35 -> 3,0 s).
    p = make_play(lambda f: (35.0, 26.0))
    play, rushers = compute_play(p)
    assert play["breached"] is False
    assert play["pocket_time"] == pytest.approx(3.0)
    assert play["release_time"] == pytest.approx(3.0)
    assert rushers[0]["arrive"] is None


def test_ignore_ce_qui_se_passe_apres_le_lancer():
    # Le rusher n'arrive qu'après le lancer (frame 35) : la poche doit être considérée comme intacte.
    p = make_play(lambda f: (31.0 if f > 36 else 35.0, 26.0))
    play, _ = compute_play(p)
    assert play["breached"] is False


def test_ignore_ce_qui_se_passe_avant_le_snap():
    # Un défenseur collé au QB avant le snap (alignement) ne doit pas déclencher le chrono.
    p = make_play(lambda f: (31.0 if f < 5 else 36.0, 26.0))
    play, _ = compute_play(p)
    assert play["breached"] is False


def test_sack_termine_la_fenetre():
    p = make_play(lambda f: (40.0 - 0.5 * max(0, f - 5), 26.0), end=25, end_event="qb_sack")
    play, _ = compute_play(p)
    assert play["release_time"] == pytest.approx(2.0)
    assert play["pocket_time"] == pytest.approx(1.7)


def test_rayon_parametrable():
    p = make_play(lambda f: (40.0 - 0.5 * max(0, f - 5), 26.0))
    # Rayon 3 yards : 10 - 0,5k < 3 -> k = 15 -> 1,5 s
    play, _ = compute_play(p, radius=3.0)
    assert play["pocket_time"] == pytest.approx(1.5)


def test_action_sans_snap_ignoree():
    p = make_play(lambda f: (35.0, 26.0))
    p["event"] = None
    assert compute_play(p) is None


def test_action_sans_rusher_ignoree():
    p = make_play(lambda f: (35.0, 26.0))
    p = p[p.pff_role == "Pass"]
    assert compute_play(p) is None


def test_kaplan_meier_simple():
    # 4 poches : percées à 1,0 s et 2,0 s ; deux lancers sans pression à 1,5 s et 3,0 s.
    df = pd.DataFrame({"pocket_time": [1.0, 2.0, 1.5, 3.0], "breached": [True, True, False, False]})
    curve, med = survival(df)
    # t=1,0 : 1 événement sur 4 à risque -> 75 %. t=2,0 : 1 sur 2 à risque -> 37,5 %.
    assert curve[10] == pytest.approx(75.0)
    assert curve[15] == pytest.approx(75.0)  # la censure à 1,5 s ne fait pas baisser la courbe
    assert curve[20] == pytest.approx(37.5)
    assert med == pytest.approx(2.0)


def test_kaplan_meier_sans_percee():
    df = pd.DataFrame({"pocket_time": [2.0, 3.0], "breached": [False, False]})
    curve, med = survival(df)
    assert all(v == 100 for v in curve)
    assert med is None


def test_logit_expit_inverses():
    for p in [0.01, 0.2, 0.5, 0.9]:
        assert expit(logit(p)) == pytest.approx(p)


def test_modele_retrouve_un_effet_connu():
    # Données simulées : l'équipe A protège bien (10 % de percées), B mal (50 %).
    rng = np.random.default_rng(0)
    rows = []
    for team, p in [("A", 0.10), ("B", 0.50)]:
        for i in range(2000):
            rows.append({"possessionTeam": team, "defensiveTeam": "Z", "n_rush": 4, "pff_playAction": 0,
                         "down": 1, "yardsToGo": 10, "breached": rng.random() < p,
                         "passResult": "C", "playResult": 5})
    m = fit_model(pd.DataFrame(rows), k=50)
    pa = predict(m, "A", "Z", 4, 0)
    pb = predict(m, "B", "Z", 4, 0)
    assert pa == pytest.approx(0.10, abs=0.02)
    assert pb == pytest.approx(0.50, abs=0.03)


def test_modele_shrinkage_petite_equipe():
    # Une équipe avec seulement 5 actions, toutes percées, ne doit pas être prédite à 100 %.
    rows = [{"possessionTeam": "BIG", "defensiveTeam": "Z", "n_rush": 4, "pff_playAction": 0, "down": 1, "yardsToGo": 10,
             "breached": i % 4 == 0, "passResult": "C", "playResult": 5} for i in range(1000)]
    rows += [{"possessionTeam": "TINY", "defensiveTeam": "Z", "n_rush": 4, "pff_playAction": 0, "down": 1, "yardsToGo": 10,
              "breached": True, "passResult": "S", "playResult": -7} for _ in range(5)]
    m = fit_model(pd.DataFrame(rows), k=100)
    assert predict(m, "TINY", "Z", 4, 0) < 0.45


def test_issues_case_vide_egale_case_parente():
    # Aucune action en 3e & long : la case situationnelle doit reprendre exactement la moyenne tactique.
    rows = [{"n_rush": 4, "pff_playAction": 0, "down": 1, "yardsToGo": 10 if i % 2 else 5,
             "breached": False, "passResult": "C", "playResult": 6} for i in range(200)]
    o = outcomes(pd.DataFrame(rows))
    assert o["0_4_0_11"]["n"] == 0
    assert o["0_4_0_11"]["yards"] == pytest.approx(o["0_4_0"]["yards"])


def test_issues_case_pleine_suit_les_donnees():
    rows = [{"n_rush": 5, "pff_playAction": 0, "down": 3, "yardsToGo": 10, "breached": True,
             "passResult": "S", "playResult": -7} for _ in range(5000)]
    o = outcomes(pd.DataFrame(rows), k=40)
    assert o["1_5_0_11"]["sack"] == pytest.approx(1.0, abs=0.01)
    assert o["1_5_0_11"]["yards"] == pytest.approx(-7, abs=0.1)
