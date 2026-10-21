"""
Validation du Pocket Clock et du modèle de recommandation.

1. Hors échantillon : modèle entraîné sur les semaines 1 à 6, évalué sur les semaines 7 et 8,
   comparé à une prédiction naïve (taux moyen de la ligue).
2. Calibration : quand le modèle annonce 30 %, observe-t-on bien environ 30 % de poches percées ?
3. Sensibilité au seuil : classements recalculés avec un rayon de 1,5 / 2,5 / 3 yards.
4. Valeur d'une poche : yards par action selon que la poche tient ou casse.

Usage : python validate.py   (après pocket_clock.py)   ->  validation.json
"""
import json

import numpy as np
import pandas as pd

from model import fit_model, predict_df, outcomes, log_loss, brier, survival

P = pd.read_csv("plays_pocket_clock.csv")
R = pd.read_csv("rushers_pocket_clock.csv")
P["curve"] = P.curve.apply(json.loads)
y = P.breached.astype(float).values
res = {}

# ---------- 1. Hors échantillon ----------
train, test = P[P.week <= 6], P[P.week >= 7]
m = fit_model(train, k=100)
p_model = predict_df(m, test)
p_base = np.full(len(test), train.breached.mean())
yt = test.breached.astype(float).values
res["holdout"] = {
    "train_plays": int(len(train)), "test_plays": int(len(test)),
    "logloss_base": round(log_loss(yt, p_base), 4), "logloss_model": round(log_loss(yt, p_model), 4),
    "brier_base": round(brier(yt, p_base), 4), "brier_model": round(brier(yt, p_model), 4),
}
res["holdout"]["gain_brier_pct"] = round((1 - res["holdout"]["brier_model"] / res["holdout"]["brier_base"]) * 100, 1)

# AUC : probabilité qu'une poche percée reçoive un score plus élevé qu'une poche intacte
pos, neg = p_model[yt == 1], p_model[yt == 0]
rank = pd.Series(np.concatenate([pos, neg])).rank().values
auc = (rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
res["holdout"]["auc"] = round(float(auc), 3)

# ---------- 2. Calibration (quintiles de probabilité prédite) ----------
cal = pd.DataFrame({"p": p_model, "y": yt})
cal["bin"] = pd.qcut(cal.p, 5, labels=False, duplicates="drop")
res["calibration"] = [{"pred": round(g.p.mean() * 100, 1), "obs": round(g.y.mean() * 100, 1), "n": int(len(g))}
                      for _, g in cal.groupby("bin")]

# Le modèle classe-t-il bien les choix tactiques ? Yards attendus vs observés sur les semaines 7-8
outc = outcomes(train)
nr = test.n_rush.clip(3, 6).values
pa = test.pff_playAction.fillna(0).astype(int).values
sit = [f"{int(d >= 3)}{int(t >= 8)}" for d, t in zip(test.down, test.yardsToGo)]
ey = np.array([p * outc[f"1_{r}_{a}_{s}"]["yards"] + (1 - p) * outc[f"0_{r}_{a}_{s}"]["yards"] for p, r, a, s in zip(p_model, nr, pa, sit)])
es = np.array([p * outc[f"1_{r}_{a}_{s}"]["sack"] + (1 - p) * outc[f"0_{r}_{a}_{s}"]["sack"] for p, r, a, s in zip(p_model, nr, pa, sit)])
tv = test.assign(ey=ey, es=es, sack=(test.passResult == "S").astype(float))
tv["bin"] = pd.qcut(tv.es, 5, labels=False, duplicates="drop")
res["sack_check"] = [{"pred": round(g.es.mean() * 100, 1), "obs": round(g.sack.mean() * 100, 1), "n": int(len(g))}
                     for _, g in tv.groupby("bin")]
res["yards_check"] = {"pred_mean": round(float(tv.ey.mean()), 2), "obs_mean": round(float(tv.playResult.mean()), 2)}

# ---------- 3. Sensibilité au rayon ----------
def breach_at(curve, r):
    return any(d < r for d in curve)


sens = []
team_rank = {}
for r in (1.5, 2.0, 2.5, 3.0):
    b = P.curve.apply(lambda c: breach_at(c, r))
    tr = P.assign(b=b).groupby("possessionTeam").b.mean()
    team_rank[r] = tr
    sens.append({"radius": r, "breach": round(b.mean() * 100, 1)})
rr = {}
for r in (1.5, 2.0, 2.5, 3.0):
    g = R.assign(hit=R.min_d < r).groupby("nflId").agg(n=("hit", "size"), rate=("hit", "mean"))
    rr[r] = g[g.n >= 120].rate
for s in sens:
    r = s["radius"]
    s["team_spearman"] = round(float(team_rank[r].rank().corr(team_rank[2.0].rank())), 3)
    s["rusher_spearman"] = round(float(rr[r].rank().corr(rr[2.0].rank())), 3)
    top10 = set(rr[2.0].nlargest(10).index)
    s["rusher_top10_overlap"] = int(len(top10 & set(rr[r].nlargest(10).index)))
res["sensitivity"] = sens

# ---------- 4. Valeur d'une poche ----------
clean, br = P[~P.breached], P[P.breached]
res["value"] = {"yards_clean": round(float(clean.playResult.mean()), 2),
                "yards_breach": round(float(br.playResult.mean()), 2)}
res["value"]["yards_lost"] = round(res["value"]["yards_clean"] - res["value"]["yards_breach"], 2)
# Pente : parmi les poches percées, yards gagnés par demi-seconde supplémentaire de protection (jusqu'à 4 s)
b = br[(br.pocket_time >= 1.5) & (br.pocket_time <= 4)]
slope = np.polyfit(b.pocket_time, b.playResult, 1)[0]
res["value"]["yards_per_half_second"] = round(float(slope / 2), 2)
sl = np.polyfit(b.pocket_time, (b.passResult == "S").astype(float), 1)[0]
res["value"]["sack_pts_per_half_second"] = round(float(sl / 2 * 100), 2)

json.dump(res, open("validation.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
