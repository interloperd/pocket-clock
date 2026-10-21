"""
Modèle de recommandation du Pocket Clock.

1. Probabilité que la poche soit percée avant le lancer, par régression logistique régularisée :
   logit P(percée) = constante + effet attaque + effet défense + effet nombre de rushers
                     + effet play action + effet situation (3e/4e tentative, longue distance)
   La régularisation équivaut à ajouter k actions « moyennes » à chaque équipe : une équipe
   peu observée est ramenée vers la moyenne au lieu de produire des chiffres extrêmes.
2. Conséquences attendues (yards, sack, passe complète) selon que la poche tient ou casse,
   pour chaque nombre de rushers et avec ou sans play action, mesurées sur les données.
3. Valeur attendue d'un choix = P(percée) × issue si percée + (1 − P(percée)) × issue si intacte.
"""
import numpy as np
import pandas as pd

GRID = np.round(np.arange(0, 6.01, 0.1), 1)
RUSH = [3, 4, 5, 6]          # 6 = 6 rushers ou plus
RUSH_LABEL = {3: "3 rushers", 4: "4 rushers", 5: "5 rushers", 6: "6+ (blitz)"}


def logit(p):
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def survival(df):
    """Kaplan-Meier : % de poches encore intactes à t secondes (un lancer sans pression = censure)."""
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


def situation(down, togo):
    """Deux indicateurs de situation : tentative décisive (3e ou 4e) et longue distance (8 yards ou plus)."""
    return int(down >= 3), int(togo >= 8)


def _design(df, teams_o, teams_d):
    n = len(df)
    nr = df.n_rush.clip(3, 6).values
    late = (df.down.values >= 3).astype(float)
    long_ = (df.yardsToGo.values >= 8).astype(float)
    cols = (["const", "pa", "late", "long"] + [f"r{r}" for r in (3, 5, 6)]
            + [f"o_{t}" for t in teams_o] + [f"d_{t}" for t in teams_d])
    X = np.zeros((n, len(cols)))
    X[:, 0] = 1
    X[:, 1] = df.pff_playAction.fillna(0).values
    X[:, 2] = late
    X[:, 3] = long_
    for j, r in enumerate((3, 5, 6)):
        X[:, 4 + j] = (nr == r)
    io = {t: 7 + i for i, t in enumerate(teams_o)}
    idf = {t: 7 + len(teams_o) + i for i, t in enumerate(teams_d)}
    for i, (o, d) in enumerate(zip(df.possessionTeam.values, df.defensiveTeam.values)):
        if o in io:
            X[i, io[o]] = 1
        if d in idf:
            X[i, idf[d]] = 1
    return X, cols


def fit_model(df, k=100, iters=50):
    """Régression logistique régularisée (Newton). k = nombre d'actions « moyennes » ajoutées par équipe."""
    teams_o = sorted(df.possessionTeam.unique())
    teams_d = sorted(df.defensiveTeam.unique())
    X, cols = _design(df, teams_o, teams_d)
    y = df.breached.astype(float).values
    p0 = y.mean()
    lam = np.full(len(cols), 1.0)          # légère régularisation des effets tactiques
    lam[0] = 0.0                            # la constante n'est pas pénalisée
    lam[7:] = k * p0 * (1 - p0)             # effets équipes : équivalent à k actions moyennes
    w = np.zeros(len(cols)); w[0] = logit(p0)
    for _ in range(iters):
        p = expit(X @ w)
        g = X.T @ (y - p) - lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(lam)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    coef = dict(zip(cols, w))
    return {"coef": coef, "teams_o": teams_o, "teams_d": teams_d, "p0": p0, "k": k}


def predict(m, off, deff, n_rush, pa, down=1, togo=10):
    c = m["coef"]
    nr = min(max(int(n_rush), 3), 6)
    late, long_ = situation(down, togo)
    z = (c["const"] + c["pa"] * pa + c["late"] * late + c["long"] * long_
         + c.get(f"r{nr}", 0.0) + c.get(f"o_{off}", 0.0) + c.get(f"d_{deff}", 0.0))
    return float(expit(z))


def predict_df(m, df):
    return np.array([predict(m, o, d, r, pa, dn, tg) for o, d, r, pa, dn, tg in
                     zip(df.possessionTeam, df.defensiveTeam, df.n_rush, df.pff_playAction.fillna(0), df.down, df.yardsToGo)])


def outcomes(df, k=40):
    """Issue moyenne (yards, % sack, % complète) selon : poche percée ou non × nombre de rushers × play action
    × situation (tentative décisive, longue distance).
    Estimation hiérarchique : chaque case est ramenée vers sa case parente avec k actions fictives
    (situation -> tactique -> état de poche), pour que les cases peu remplies restent prudentes."""
    df = df.assign(nr=df.n_rush.clip(3, 6), pa=df.pff_playAction.fillna(0).astype(int),
                   late=(df.down >= 3).astype(int), lg=(df.yardsToGo >= 8).astype(int),
                   sack=(df.passResult == "S").astype(float), comp=(df.passResult == "C").astype(float))
    cols = ["playResult", "sack", "comp"]
    out = {}
    for b in (False, True):
        base = df[df.breached == b]
        mb = base[cols].mean()
        for r in RUSH:
            for pa in (0, 1):
                g = base[(base.nr == r) & (base.pa == pa)]
                mt = (g[cols].sum() + k * mb) / (len(g) + k)
                out[f"{int(b)}_{r}_{pa}"] = _cell(mt, len(g))
                for late in (0, 1):
                    for lg in (0, 1):
                        h = g[(g.late == late) & (g.lg == lg)]
                        ms = (h[cols].sum() + k * mt) / (len(h) + k)
                        out[f"{int(b)}_{r}_{pa}_{late}{lg}"] = _cell(ms, len(h))
    return out


def _cell(m, n):
    return {"yards": round(float(m.playResult), 3), "sack": round(float(m.sack), 4),
            "comp": round(float(m.comp), 4), "n": int(n)}


def expected(m, outc, off, deff, n_rush, pa, down=1, togo=10):
    p = predict(m, off, deff, n_rush, pa, down, togo)
    nr = min(max(int(n_rush), 3), 6)
    late, lg = situation(down, togo)
    ob, oc = outc[f"1_{nr}_{pa}_{late}{lg}"], outc[f"0_{nr}_{pa}_{late}{lg}"]
    return {"p_breach": p,
            "yards": p * ob["yards"] + (1 - p) * oc["yards"],
            "sack": p * ob["sack"] + (1 - p) * oc["sack"],
            "comp": p * ob["comp"] + (1 - p) * oc["comp"]}


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))
