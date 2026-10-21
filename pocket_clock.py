"""
Pocket Clock — NFL Big Data Bowl regional event (AWS)
Mesure, image par image, combien de temps la poche tient avant qu'un pass rusher
n'arrive à moins de 2 yards du quarterback.

Usage : python pocket_clock.py data
Sorties : plays_pocket_clock.csv (une ligne par action), rushers_pocket_clock.csv (une ligne par rusher et par action)
"""
import glob
import sys

import numpy as np
import pandas as pd

PRESSURE_RADIUS = 2.0   # yards
FPS = 10                # images par seconde du tracking
SNAP = {"ball_snap", "autoevent_ballsnap"}
END = {"pass_forward", "autoevent_passforward", "qb_sack", "qb_strip_sack", "run", "fumble"}


def compute_play(p, radius=PRESSURE_RADIUS):
    """Calcule le Pocket Clock d'une action.

    p : tracking d'une seule action, avec la colonne pff_role.
    Renvoie (ligne_action, lignes_rushers) ou None si l'action n'est pas exploitable
    (pas de snap, pas de QB ou pas de rusher identifiés).
    """
    ev = p[p.event.notna()][["frameId", "event"]].drop_duplicates()
    snap = ev[ev.event.isin(SNAP)].frameId.min()
    if pd.isna(snap):
        return None
    end_ev = ev[ev.event.isin(END) & (ev.frameId > snap)]
    end = end_ev.frameId.min() if len(end_ev) else p.frameId.max()
    win = p[(p.frameId >= snap) & (p.frameId <= end)]
    qb = win[win.pff_role == "Pass"].drop_duplicates("frameId").set_index("frameId")[["x", "y"]]
    rush = win[win.pff_role == "Pass Rush"]
    if qb.empty or rush.empty:
        return None
    rw = rush.join(qb, on="frameId", rsuffix="_qb").dropna(subset=["x_qb"])
    if rw.empty:
        return None
    rw = rw.assign(d=np.hypot(rw.x - rw.x_qb, rw.y - rw.y_qb))
    mind = rw.groupby("frameId").d.min()
    hit = mind[mind < radius]
    breached = len(hit) > 0
    first = None
    if breached:
        fr = hit.index.min()
        first = int(rw[rw.frameId == fr].sort_values("d").iloc[0].nflId)
    gid, pid = p.gameId.iloc[0], p.playId.iloc[0]
    rushers = []
    for nid, g in rw.groupby("nflId"):
        close = g[g.d < radius]
        rushers.append({"gameId": gid, "playId": pid, "nflId": int(nid),
                        "arrive": (close.frameId.min() - snap) / FPS if len(close) else None,
                        "min_d": round(float(g.d.min()), 3)})
    play = {"gameId": gid, "playId": pid,
            "pocket_time": ((hit.index.min() if breached else end) - snap) / FPS,
            "breached": breached, "release_time": (end - snap) / FPS, "first_rusher": first,
            "n_rush": int(rush.nflId.nunique()),
            "curve": [round(float(v), 2) for v in mind.values]}
    return play, rushers


def run(data):
    games = pd.read_csv(f"{data}/games.csv")
    plays = pd.read_csv(f"{data}/plays.csv")
    roles = pd.read_csv(f"{data}/pffScoutingData.csv")[["gameId", "playId", "nflId", "pff_role"]]
    prow, rrow = [], []
    for f in sorted(glob.glob(f"{data}/tracking/tracking_*.csv")):
        t = pd.read_csv(f).merge(roles, on=["gameId", "playId", "nflId"], how="left")
        for _, p in t.groupby(["gameId", "playId"]):
            r = compute_play(p)
            if r:
                prow.append(r[0]); rrow.extend(r[1])
    P = pd.DataFrame(prow).merge(
        plays[["gameId", "playId", "possessionTeam", "defensiveTeam", "passResult", "playResult",
               "down", "yardsToGo", "quarter", "playDescription", "pff_passCoverage",
               "pff_playAction", "dropBackType", "offenseFormation"]], on=["gameId", "playId"])
    P = P.merge(games[["gameId", "week"]], on="gameId")
    P.to_csv("plays_pocket_clock.csv", index=False)
    pd.DataFrame(rrow).to_csv("rushers_pocket_clock.csv", index=False)
    print(f"{len(P)} actions chronométrées, {P.breached.mean():.1%} de poches percées")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data")
