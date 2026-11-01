# -*- coding: utf-8 -*-
"""Figuras do Módulo VI que dependem dos dados FICTÍCIOS (src/dados_sinteticos_m6.py).

Gera: fig_m6_rf_importancias_imoveis.png, fig_m6_rf_lista.png, fig_m6_rf_limiar5.png,
fig_m6_rf_acumulado95.png.   Rode com:  python figuras_dados_sinteticos_m6.py
Os textos impressos são exatamente os citados no livro.
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_aqui = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_aqui, "..", "src"))
from dados_sinteticos_m6 import gerar_imoveis  # noqa: E402

_book = os.path.join(_aqui, "..", "..", "figuras")
OUT = _book if os.path.isdir(_book) else os.path.join(_aqui, "figuras")
os.makedirs(OUT, exist_ok=True)
VERDE = "#2e8b6f"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("gerado:", nome)


def texto_em_figura(texto, nome, largura):
    linhas = texto.rstrip("\n").split("\n")
    fig = plt.figure(figsize=(largura, 0.26 * len(linhas) + 0.4))
    fig.text(0.02, 0.97, texto.rstrip("\n"), family="DejaVu Sans Mono", fontsize=8.5,
             va="top", ha="left", linespacing=1.45)
    fig.patch.set_facecolor("#f6f6f6")
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight",
                facecolor="#f6f6f6", pad_inches=0.12)
    plt.close(fig)
    print("gerado:", nome)


def main():
    df = gerar_imoveis()
    df["caro"] = (df["preco"] > df["preco"].median()).astype(int)
    cols = ["quartos", "distancia_km", "cep", "banheiros", "vagas", "idade_anos", "area_terreno",
            "latitude", "longitude", "imoveis_bairro"]
    X, y = df[cols], df["caro"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler().fit(X_train)
    std_X_train = pd.DataFrame(sc.transform(X_train), columns=cols, index=X_train.index)
    modelo = RandomForestClassifier(n_estimators=200, random_state=42).fit(std_X_train, y_train)
    importances = modelo.feature_importances_
    std = np.std([t.feature_importances_ for t in modelo.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    buf = io.StringIO()
    with redirect_stdout(buf):
        for f in range(X.shape[1]):
            print("%d. feature %s (%.3f%%)" % (f + 1, std_X_train.columns[indices[f]], importances[indices[f]] * 100))
    lista = buf.getvalue(); print(lista)
    texto_em_figura(lista, "fig_m6_rf_lista.png", 4.2)

    buf = io.StringIO(); sel5 = []
    with redirect_stdout(buf):
        for f in range(X.shape[1]):
            if importances[indices[f]] >= 0.05:
                print("%d. feature %s (%.3f%%)" % (f + 1, std_X_train.columns[indices[f]], importances[indices[f]] * 100))
                sel5.append(std_X_train.columns[indices[f]])
        print(); print("Features Based on Accumulative Threshold"); print(sel5)
    print(buf.getvalue())
    texto_em_figura(buf.getvalue(), "fig_m6_rf_limiar5.png", 6.4)

    buf = io.StringIO(); total = 0; acum = []
    with redirect_stdout(buf):
        for f in range(X.shape[1]):
            print("%d. feature %s (%.3f%%)" % (f + 1, std_X_train.columns[indices[f]], importances[indices[f]] * 100))
            total += importances[indices[f]]; acum.append(std_X_train.columns[indices[f]])
            if total >= 0.95:
                break
        print()
        print("Total: %.3f%%   Threshold Accumulative: %.3f%%" % (total * 100, 95.0))
        print(); print("Features Based on Accumulative Threshold"); print(acum)
    print(buf.getvalue())
    texto_em_figura(buf.getvalue(), "fig_m6_rf_acumulado95.png", 8.0)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(range(len(indices)), importances[indices], yerr=std[indices], color=VERDE,
           edgecolor="white", ecolor="#555555", capsize=3)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(std_X_train.columns[indices], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Importância", fontsize=9.5)
    ax.set_title("Importância das características — Random Forest (imóveis fictícios)", fontsize=10.5)
    ax.tick_params(labelsize=8); ax.grid(axis="y", color="#eeeeee"); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    salvar(fig, "fig_m6_rf_importancias_imoveis.png")
    print("acuracia teste: %.3f" % modelo.score(pd.DataFrame(sc.transform(X_test), columns=cols), y_test))


if __name__ == "__main__":
    main()
