"""Construit dashboard_data.json à partir des sorties de pocket_clock.py."""
import json, sys
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
P = pd.read_csv("plays_pocket_clock.csv")
R = pd.read_csv("rushers_pocket_clock.csv")
players = pd.read_csv(f"{DATA}/players.csv")
pff = pd.read_csv(f"{DATA}/pffScoutingData.csv")
plays = pd.read_csv(f"{DATA}/plays.csv")
GRID = np.round(np.arange(0, 6.01, 0.1), 1)


def survival(df):
    """Kaplan-Meier : % de poches encore intactes à t secondes (le lancer = censure)."""
    d, e = df.pocket_time.values, df.breached.values.astype(bool)
    s, out = 1.0, []
    for t in GRID:
        at_risk = (d >= t - 1e-9).sum()
        ev = ((np.abs(d - t) < 0.05) & e).sum()
        if at_risk:
            s *= 1 - ev / at_risk
        out.append(round(s * 100, 1))
    med = next((float(t) for t, v in zip(GRID, out) if v <= 50), None)
    return out, med


def team_table(col):
    rows = []
    for team, g in P.groupby(col):
        curve, med = survival(g)
        rows.append({"team": team, "plays": len(g), "curve": curve, "median": med,
                     "breach": round(g.breached.mean() * 100, 1),
                     "sack": round((g.passResult == "S").mean() * 100, 1),
                     "intact25": curve[25], "intact35": curve[35]})
    return sorted(rows, key=lambda r: -r["intact25"])


overall, overall_med = survival(P)

# Ce que vaut chaque demi-seconde : issue de la passe selon le moment où la poche cède
P["bucket"] = np.where(P.breached, (np.floor(P.pocket_time * 2) / 2).clip(0, 4), -1)
buckets = []
for b, g in P.groupby("bucket"):
    buckets.append({"bucket": "Poche intacte" if b == -1 else (f"{b:.1f}–{b+0.5:.1f} s" if b < 4 else "4 s +"),
                    "n": len(g), "comp": round((g.passResult == "C").mean() * 100, 1),
                    "sack": round((g.passResult == "S").mean() * 100, 1),
                    "yards": round(g.playResult.mean(), 2)})

# Classement des pass rushers
R = R.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
R = R.merge(plays[["gameId", "playId", "defensiveTeam"]], on=["gameId", "playId"])
pr = pff[pff.pff_role == "Pass Rush"].groupby("nflId")[["pff_sack", "pff_hit", "pff_hurry"]].sum()
first = P.first_rusher.dropna().astype(int).value_counts()
rush = []
for nid, g in R.groupby("nflId"):
    if len(g) < 120:
        continue
    arr = g.arrive.dropna()
    rush.append({"id": int(nid), "name": g.displayName.iloc[0], "pos": g.officialPosition.iloc[0],
                 "team": g.defensiveTeam.mode().iloc[0], "snaps": len(g),
                 "getThere": round(len(arr) / len(g) * 100, 1),
                 "arrival": round(float(arr.median()), 2) if len(arr) else None,
                 "firstIn": int(first.get(nid, 0)),
                 "firstInRate": round(first.get(nid, 0) / len(g) * 100, 1),
                 "sacks": float(pr.loc[nid, "pff_sack"]) if nid in pr.index else 0,
                 "pressures": float(pr.loc[nid, ["pff_sack", "pff_hit", "pff_hurry"]].sum()) if nid in pr.index else 0})
rush = sorted(rush, key=lambda r: -r["firstInRate"])

# Actions à rejouer
pick = pd.concat([
    P[(P.passResult == "S") & P.breached].nsmallest(6, "pocket_time"),
    P[~P.breached & (P.passResult == "C")].nlargest(4, "release_time"),
    P[P.breached & (P.passResult == "C")].sort_values("playResult", ascending=False).head(3),
    P[P.breached & (P.passResult == "IN")].nsmallest(2, "pocket_time"),
]).drop_duplicates(["gameId", "playId"])
role = pff[["gameId", "playId", "nflId", "pff_role"]]
names = players.set_index("nflId").displayName.to_dict()
replays = []
for gid, sub in pick.groupby("gameId"):
    t = pd.read_csv(f"{DATA}/tracking/tracking_{gid}.csv").merge(role, on=["gameId", "playId", "nflId"], how="left")
    for _, pl in sub.iterrows():
        p = t[t.playId == pl.playId]
        ev = p[p.event.notna()][["frameId", "event"]].drop_duplicates()
        snap = int(ev[ev.event.isin({"ball_snap", "autoevent_ballsnap"})].frameId.min())
        p = p[(p.frameId >= snap - 5) & (p.frameId <= snap + 10 * pl.release_time + 12)]
        flip = p.playDirection.iloc[0] == "left"
        frames = []
        for fid, fr in p.groupby("frameId"):
            pts = []
            for _, r in fr.iterrows():
                x, y = (120 - r.x, 53.3 - r.y) if flip else (r.x, r.y)
                pts.append([round(x, 2), round(y, 2),
                            "B" if pd.isna(r.nflId) else ("Q" if r.pff_role == "Pass" else
                            "R" if r.pff_role == "Pass Rush" else
                            "K" if r.pff_role == "Pass Block" else
                            "O" if r.team == pl.possessionTeam else "D"),
                            "" if pd.isna(r.jerseyNumber) else int(r.jerseyNumber)])
            frames.append(pts)
        replays.append({"id": f"{gid}-{pl.playId}", "off": pl.possessionTeam, "def": pl.defensiveTeam,
                        "week": int(pl.week), "q": int(pl.quarter), "down": int(pl.down), "togo": int(pl.yardsToGo),
                        "result": pl.passResult, "yards": int(pl.playResult), "desc": pl.playDescription,
                        "pocket": pl.pocket_time, "breached": bool(pl.breached), "release": pl.release_time,
                        "snapIdx": 5, "first": names.get(pl.first_rusher) if pl.breached else None,
                        "curve": json.loads(pl.curve), "coverage": pl.pff_passCoverage if isinstance(pl.pff_passCoverage, str) else "",
                        "frames": frames})

# ---------- Pression PFF par action ----------
dpress = pff[pff.pff_role == "Pass Rush"].groupby(["gameId", "playId"])[["pff_hit", "pff_hurry", "pff_sack"]].max()
P = P.merge(dpress.reset_index(), on=["gameId", "playId"], how="left").fillna({"pff_hit": 0, "pff_hurry": 0, "pff_sack": 0})
P["pressure"] = (P[["pff_hit", "pff_hurry", "pff_sack"]].max(axis=1) > 0)
thrown = P.passResult.isin(["C", "I", "IN"])


def intact_at(df, idx=25):
    c, med = survival(df)
    return c[idx], med


# ---------- Fiche équipe (attaque et défense), avec rang sur 32 ----------
def card(col):
    rows = []
    for team, g in P.groupby(col):
        i25, med = intact_at(g)
        q4 = g[g.quarter >= 4]; q13 = g[g.quarter < 4]
        ttt = float(g[thrown.loc[g.index]].release_time.median())
        rows.append({"team": team, "plays": len(g),
                     "ttt": round(ttt, 2), "intact25": i25,
                     "median": med if med is not None else 6.0,
                     "pressure": round(g.pressure.mean() * 100, 1),
                     "sack": round((g.passResult == "S").mean() * 100, 1),
                     "margin": round((med if med is not None else 6.0) - ttt, 2),
                     "q4": round(intact_at(q4)[0] - intact_at(q13)[0], 1) if len(q4) > 30 else None})
    return rows


cards = {"offense": card("possessionTeam"), "defense": card("defensiveTeam")}

# ---------- Contexte tactique ----------
def ctx_group(label, mask):
    g = P[mask]
    i25, med = intact_at(g)
    return {"label": label, "n": int(len(g)), "intact25": i25, "median": med,
            "sack": round((g.passResult == "S").mean() * 100, 1)}


context = {
    "Play action": [ctx_group("Sans play action", P.pff_playAction == 0), ctx_group("Avec play action", P.pff_playAction == 1)],
    "Rushers envoyés": [ctx_group(f"{n} rushers", P.n_rush == n) for n in (3, 4, 5)] + [ctx_group("6 ou plus (blitz)", P.n_rush >= 6)],
    "Couverture": [ctx_group(c, P.pff_passCoverage == c) for c in
                   P.pff_passCoverage.value_counts().index[:6]],
    "Type de dropback": [ctx_group(d.replace("_", " ").capitalize(), P.dropBackType == d) for d in
                         P.dropBackType.value_counts().index[:4]],
}

# ---------- Fatigue des pass rushers ----------
Rf = R.merge(P[["gameId", "playId", "quarter"]], on=["gameId", "playId"])
Rf = Rf.sort_values(["gameId", "nflId", "playId"])
Rf["nth"] = Rf.groupby(["gameId", "nflId"]).cumcount() + 1
bins = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 99)]
fatigue_snap = []
for a, b in bins:
    g = Rf[(Rf.nth >= a) & (Rf.nth <= b)]
    fatigue_snap.append({"label": f"{a}–{b}" if b < 99 else f"{a}+", "n": int(len(g)),
                         "rate": round(g.arrive.notna().mean() * 100, 2),
                         "arrival": round(float(g.arrive.median()), 2)})
fatigue_q = []
for q in (1, 2, 3, 4):
    g = Rf[Rf.quarter == q] if q < 4 else Rf[Rf.quarter >= 4]
    fatigue_q.append({"label": f"Q{q}", "n": int(len(g)), "rate": round(g.arrive.notna().mean() * 100, 2),
                      "arrival": round(float(g.arrive.median()), 2)})

# ---------- Classement des joueurs de ligne offensive ----------
blk = pff[pff.pff_role == "Pass Block"].merge(plays[["gameId", "playId", "possessionTeam"]], on=["gameId", "playId"])
arr = R[["gameId", "playId", "nflId", "arrive"]].rename(columns={"nflId": "pff_nflIdBlockedPlayer", "arrive": "opp_arrive"})
blk = blk.merge(arr, on=["gameId", "playId", "pff_nflIdBlockedPlayer"], how="left")
blk = blk.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
oline = []
for nid, g in blk.groupby("nflId"):
    if len(g) < 150:
        continue
    assigned = g[g.pff_nflIdBlockedPlayer.notna() & g.pff_nflIdBlockedPlayer.isin(R.nflId)]
    assigned = assigned[assigned.set_index(["gameId", "playId"]).index.isin(P.set_index(["gameId", "playId"]).index)]
    if len(assigned) < 100:
        continue
    oline.append({"name": g.displayName.iloc[0], "pos": g.officialPosition.iloc[0],
                  "team": g.possessionTeam.mode().iloc[0], "snaps": int(len(g)),
                  "hold": round(assigned.opp_arrive.isna().mean() * 100, 1),
                  "sacks": int(g.pff_sackAllowed.sum()), "hits": int(g.pff_hitAllowed.sum()),
                  "hurries": int(g.pff_hurryAllowed.sum()),
                  "beaten": round(g.pff_beatenByDefender.mean() * 100, 1)})
oline = sorted(oline, key=lambda r: -r["hold"])

out = {"kpi": {"plays": len(P), "games": int(P.gameId.nunique()), "median": overall_med,
               "breach": round(P.breached.mean() * 100, 1),
               "compClean": buckets[0]["comp"], "sackClean": buckets[0]["sack"],
               "compBreach": round((P[P.breached].passResult == "C").mean() * 100, 1),
               "sackBreach": round((P[P.breached].passResult == "S").mean() * 100, 1)},
       "grid": GRID.tolist(), "overall": overall, "buckets": buckets,
       "offense": team_table("possessionTeam"), "defense": team_table("defensiveTeam"),
       "rushers": rush, "replays": replays, "cards": cards, "context": context,
       "fatigueSnap": fatigue_snap, "fatigueQ": fatigue_q, "oline": oline}
# ---------- Modèle de recommandation (entraîné sur toute la saison pour le dashboard) ----------
from model import fit_model, outcomes, RUSH
import os
mdl = fit_model(P.assign(n_rush=P.n_rush), k=100)
out["model"] = {"coef": {k_: round(float(v), 5) for k_, v in mdl["coef"].items()},
                "outcomes": outcomes(P),
                "usage": {t: {str(r): int(((P.defensiveTeam == t) & (P.n_rush.clip(3, 6) == r)).sum()) for r in RUSH}
                          for t in sorted(P.defensiveTeam.unique())},
                "paUse": {t: round(float(P[P.possessionTeam == t].pff_playAction.mean() * 100), 1)
                          for t in sorted(P.possessionTeam.unique())}}
# ---------- Flux « direct » : chaque match rejoué action par action ----------
games_df = pd.read_csv(f"{DATA}/games.csv")
L = P.merge(plays[["gameId", "playId", "gameClock", "preSnapHomeScore", "preSnapVisitorScore"]], on=["gameId", "playId"], how="left")
curves = pd.read_csv("plays_pocket_clock.csv", usecols=["gameId", "playId", "curve"])
L = L.drop(columns=[c for c in ["curve"] if c in L.columns]).merge(curves, on=["gameId", "playId"], how="left")
live = {}
for gid, g in L.sort_values(["gameId", "playId"]).groupby("gameId"):
    rows = []
    for _, r in g.iterrows():
        cv = json.loads(r.curve) if isinstance(r.curve, str) else []
        rows.append([r.possessionTeam, r.defensiveTeam, int(r.quarter), str(r.gameClock), int(r.down), int(r.yardsToGo),
                     r.passResult, int(r.playResult), float(r.pocket_time), int(bool(r.breached)), float(r.release_time),
                     names.get(r.first_rusher) if bool(r.breached) and not pd.isna(r.first_rusher) else "",
                     int(r.n_rush), int(r.pff_playAction) if not pd.isna(r.pff_playAction) else 0,
                     [round(min(v, 15), 1) for v in cv], str(r.playDescription)[:160],
                     int(r.preSnapHomeScore), int(r.preSnapVisitorScore)])
    gm = games_df[games_df.gameId == gid].iloc[0]
    live[str(gid)] = {"week": int(gm.week), "home": gm.homeTeamAbbr, "away": gm.visitorTeamAbbr, "plays": rows}
# ---------- Terrain en direct : positions image par image pour quelques matchs vedettes ----------
def featured_games():
    picks = []
    kc = games_df[((games_df.homeTeamAbbr == "KC") & (games_df.visitorTeamAbbr == "BUF")) | ((games_df.homeTeamAbbr == "BUF") & (games_df.visitorTeamAbbr == "KC"))]
    if len(kc): picks.append(int(kc.gameId.iloc[0]))
    for tm in ["TB", "CLE", "CIN", "LA", "PIT"]:
        g = games_df[((games_df.homeTeamAbbr == tm) | (games_df.visitorTeamAbbr == tm)) & ~games_df.gameId.isin(picks)]
        if len(g): picks.append(int(g.gameId.iloc[len(g) // 2]))
    return picks[:6]


ROLE = {"Pass": "Q", "Pass Rush": "R", "Pass Block": "K", "Coverage": "D", "Pass Route": "O"}
for gid in featured_games():
    t = pd.read_csv(f"{DATA}/tracking/tracking_{gid}.csv").merge(role, on=["gameId", "playId", "nflId"], how="left")
    meta = L[L.gameId == gid].set_index("playId")
    fields = {}
    for pid, p in t.groupby("playId"):
        if pid not in meta.index:
            continue
        ev = p[p.event.notna()][["frameId", "event"]].drop_duplicates()
        snap = ev[ev.event.isin({"ball_snap", "autoevent_ballsnap"})].frameId.min()
        if pd.isna(snap):
            continue
        rel = float(meta.loc[pid, "release_time"])
        p = p[(p.frameId >= snap - 5) & (p.frameId <= snap + round(rel * 10) + 8)]
        flip = p.playDirection.iloc[0] == "left"
        p = p.assign(nid=p.nflId.fillna(0).astype(int)).sort_values(["frameId", "nid"])
        ids = sorted(p.nid.unique())
        offteam = meta.loc[pid, "possessionTeam"]
        ents = []
        for n in ids:
            r = p[p.nid == n].iloc[0]
            code = "B" if n == 0 else ROLE.get(r.pff_role, "O" if r.team == offteam else "D")
            ents.append([code, "" if n == 0 or pd.isna(r.jerseyNumber) else int(r.jerseyNumber)])
        frames, last = [], {}
        for fid, fr_ in p.groupby("frameId"):
            pos = {int(a): (b, c) for a, b, c in zip(fr_.nid, fr_.x, fr_.y)}
            row = []
            for n in ids:
                x, y = pos.get(n, last.get(n, (0, 0)))
                last[n] = (x, y)
                if flip:
                    x, y = 120 - x, 53.3 - y
                row += [int(round(x * 10)), int(round(y * 10))]
            frames.append(row)
        fields[str(pid)] = {"e": ents, "f": frames}
    if str(gid) in live:
        order = [str(int(x)) for x in L[L.gameId == gid].sort_values("playId").playId]
        live[str(gid)]["field"] = [fields.get(o) for o in order]
        live[str(gid)]["featured"] = 1
out["live"] = live
if os.path.exists("validation.json"):
    out["validation"] = json.load(open("validation.json"))
json.dump(out, open("dashboard_data.json", "w"), separators=(",", ":"))
print(json.dumps(out["kpi"]), len(rush), len(replays))
print(fatigue_snap, fatigue_q)
print(oline[:5], len(oline))
print(json.dumps(context)[:900])
print(cards["offense"][:2])
print([(r["name"], r["firstInRate"], r["getThere"], r["arrival"]) for r in rush[:8]])
print([(t["team"], t["intact25"], t["median"]) for t in out["offense"][:5]])
