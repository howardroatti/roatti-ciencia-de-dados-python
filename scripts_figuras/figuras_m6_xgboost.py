# -*- coding: utf-8 -*-
"""Gera figuras/fig_m6_xgboost_busca.png: amostra da busca em grade do XGBoost (Iris).

Executa exatamente o laço do Módulo VI (validação cruzada manual em 5 partições,
semente fixa) e desenha o DataFrame `errors` como o pandas o exibe (5 primeiras
e 5 últimas linhas). Rode com:  python figuras_m6_xgboost.py
"""
import os
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import datasets
from xgboost import XGBClassifier

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figuras")

iris = datasets.load_iris()
X, y = iris.data, iris.target

rng = np.random.default_rng(42)
idx = rng.integers(0, 5, size=len(X))
error = []
for eta, max_depth, n in product([0.1, 0.5, 0.9], [3, 6, 9, 12], list(range(10, 250, 10))):
    xgb = XGBClassifier(objective='multi:softprob', n_estimators=n,
                        max_depth=max_depth, learning_rate=eta,
                        eval_metric='merror', n_jobs=2, random_state=13)
    cv_error = []
    for k in range(5):
        fold_idx = idx == k
        train_X = pd.DataFrame(X).loc[~fold_idx]; train_y = y[~fold_idx]
        valid_X = pd.DataFrame(X).loc[fold_idx]; valid_y = y[fold_idx]
        xgb.fit(train_X, train_y)
        pred = xgb.predict(valid_X)
        cv_error.append(np.mean(pred != valid_y))
    error.append({'eta': eta, 'max_depth': max_depth, 'n_estimators': n,
                  'avg_error': np.mean(cv_error)})
errors = pd.DataFrame(error)

head, tail = errors.head(5), errors.tail(5)
cols = ["", "eta", "max_depth", "n_estimators", "avg_error"]
rows = [[str(i), f"{r.eta:g}", f"{int(r.max_depth)}", f"{int(r.n_estimators)}", f"{r.avg_error:.6f}"]
        for i, r in head.iterrows()]
rows.append(["...", "...", "...", "...", "..."])
rows += [[str(i), f"{r.eta:g}", f"{int(r.max_depth)}", f"{int(r.n_estimators)}", f"{r.avg_error:.6f}"]
         for i, r in tail.iterrows()]

fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.axis("off")
tb = ax.table(cellText=rows, colLabels=cols, cellLoc="right", loc="upper center",
              colWidths=[0.13, 0.15, 0.22, 0.27, 0.25])
tb.auto_set_font_size(False)
tb.set_fontsize(10)
tb.scale(1, 1.55)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("none")
    if r == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("white")
    else:
        cell.set_facecolor("#f3f3f3" if r % 2 == 0 else "white")
    if c == 0 and r > 0:
        cell.set_text_props(fontweight="bold")
ax.text(0.0, 0.02, f"{len(errors)} rows × {errors.shape[1]} columns", transform=ax.transAxes, fontsize=10)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_m6_xgboost_busca.png"), dpi=150, bbox_inches="tight", facecolor="white")
print(errors.head(5).to_string()); print(errors.tail(5).to_string()); print(errors.avg_error.min())
