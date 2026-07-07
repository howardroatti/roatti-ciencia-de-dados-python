# -*- coding: utf-8 -*-
"""Regenera, a partir de código reprodutível, as figuras de RESULTADO do Módulo VI.

Todas as figuras saem de execuções reais do scikit-learn sobre conjuntos públicos
(Iris, Breast Cancer, Diabetes). Rode com:  python figuras_resultado_m6.py

As imagens são salvas em ../../figuras (pasta de figuras do livro) e substituem
capturas/gráficos de terceiros ou do autor por versões reprodutíveis.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score

_book = os.path.join(os.path.dirname(__file__), "..", "..", "figuras")
OUT = _book if os.path.isdir(_book) else os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(OUT, exist_ok=True)
AZUL, VERDE, LARANJA = "#2c6fbf", "#2e8b6f", "#d9822b"
CM_FUNDO = ListedColormap(["#dce8f6", "#dcefe4", "#f8e6d2"])
CM_PONTO = ListedColormap([AZUL, VERDE, LARANJA])
plt.rcParams.update({"font.family": "DejaVu Sans"})


def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("gerado:", nome)


# --- dados 2D do Iris (comprimento e largura da pétala) para fronteiras de decisão ---
_iris = load_iris()
X2 = _iris.data[:, 2:4]
y = _iris.target
_nomes = _iris.target_names


def fronteira(modelo, titulo, nome, X=X2, yy=y):
    modelo.fit(X, yy)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy_ = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = modelo.predict(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.contourf(xx, yy_, Z, alpha=0.6, cmap=CM_FUNDO)
    for k, cor in enumerate([AZUL, VERDE, LARANJA]):
        m = yy == k
        ax.scatter(X[m, 0], X[m, 1], c=cor, s=28, edgecolor="white", lw=0.5, label=_nomes[k])
    ax.set_xlabel("Comprimento da pétala (cm)", fontsize=9.5)
    ax.set_ylabel("Largura da pétala (cm)", fontsize=9.5)
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True)
    ax.tick_params(labelsize=8)
    salvar(fig, nome)


def fronteiras_classificadores():
    fronteira(KNeighborsClassifier(n_neighbors=5),
              "Fronteiras de decisão — kNN (k = 5)", "fig_m6_knn_fronteira.png")
    fronteira(NearestCentroid(),
              "Fronteiras de decisão — Rocchio (centroides)", "fig_m6_rocchio_fronteira.png")
    fronteira(SVC(kernel="rbf", C=1.0, gamma="scale"),
              "Fronteiras de decisão — SVM (kernel RBF)", "fig_m6_svm_iris.png")
    fronteira(DecisionTreeClassifier(max_depth=4, random_state=0),
              "Fronteiras de decisão — Árvore de Decisão", "fig_m6_tree_fronteira.png")
    fronteira(RandomForestClassifier(n_estimators=100, random_state=0),
              "Fronteiras de decisão — Random Forest", "fig_m6_rf_fronteira.png")
    fronteira(LogisticRegression(max_iter=1000),
              "Fronteiras de decisão — Regressão Logística", "fig_m6_logistica_fronteira.png")
    fronteira(Perceptron(random_state=0),
              "Fronteiras de decisão — Perceptron", "fig_m6_perceptron_fronteira.png")
    fronteira(MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000, random_state=0),
              "Fronteiras de decisão — Perceptron Multicamadas (MLP)", "fig_m6_mlp_fronteira.png")


def knn_erro_por_k():
    Xtr, Xte, ytr, yte = train_test_split(_iris.data, y, test_size=0.3, random_state=1, stratify=y)
    ks = range(1, 41)
    erros = []
    for k in ks:
        m = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
        erros.append(1 - accuracy_score(yte, m.predict(Xte)))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(list(ks), erros, color=AZUL, marker="o", markersize=4, lw=1.6)
    ax.set_xlabel("Valor de k", fontsize=9.5)
    ax.set_ylabel("Taxa média de erro", fontsize=9.5)
    ax.set_title("kNN — taxa de erro por valor de k (1 a 40)", fontsize=11)
    ax.grid(color="#eeeeee"); ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    salvar(fig, "fig_m6_knn_erro_k.png")


def rf_importancias():
    bc = load_breast_cancer()
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(bc.data, bc.target)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(range(len(idx)), imp[idx], color=VERDE, edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([bc.feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importância", fontsize=9.5)
    ax.set_title("Importância das características — Random Forest (10 maiores)", fontsize=10.5)
    ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    salvar(fig, "fig_m6_rf_importancias.png")


def rf_erro_por_arvores():
    bc = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(bc.data, bc.target, test_size=0.3, random_state=1)
    ns = list(range(1, 101, 2))
    erros = []
    for n in ns:
        m = RandomForestClassifier(n_estimators=n, random_state=0).fit(Xtr, ytr)
        erros.append(1 - accuracy_score(yte, m.predict(Xte)))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ns, erros, color=LARANJA, lw=1.8)
    ax.set_xlabel("Número de árvores", fontsize=9.5)
    ax.set_ylabel("Taxa de erro", fontsize=9.5)
    ax.set_title("Random Forest — erro em relação ao número de árvores", fontsize=10.5)
    ax.grid(color="#eeeeee"); ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    salvar(fig, "fig_m6_rf_erro_arvores.png")


def arvore_plot():
    m = DecisionTreeClassifier(max_depth=3, random_state=0).fit(_iris.data, y)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    plot_tree(m, feature_names=_iris.feature_names, class_names=list(_nomes),
              filled=True, rounded=True, fontsize=7, ax=ax,
              impurity=False, proportion=True)
    salvar(fig, "fig_m6_tree_plot.png")


def bcancer_pca():
    bc = load_breast_cancer()
    Xs = StandardScaler().fit_transform(bc.data)
    P = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for k, cor, lab in [(0, LARANJA, bc.target_names[0]), (1, AZUL, bc.target_names[1])]:
        m = bc.target == k
        ax.scatter(P[m, 0], P[m, 1], c=cor, s=14, alpha=0.7, label=lab)
    ax.set_xlabel("1º componente principal", fontsize=9.5)
    ax.set_ylabel("2º componente principal", fontsize=9.5)
    ax.set_title("Câncer de mama em dois componentes principais", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    salvar(fig, "fig_m6_bcancer_pca.png")


def clusters():
    bc = load_breast_cancer()
    Xs = StandardScaler().fit_transform(bc.data)
    P = PCA(n_components=2).fit_transform(Xs)
    for algo, nome, tit in [
        (KMeans(n_clusters=2, n_init=10, random_state=0), "fig_m6_kmeans.png", "Agrupamentos gerados pelo k-Means"),
        (AgglomerativeClustering(n_clusters=2), "fig_m6_agglomerative.png", "Agrupamentos gerados pelo Agglomerative Clustering")]:
        lab = algo.fit_predict(P)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for k, cor in enumerate([AZUL, LARANJA]):
            m = lab == k
            ax.scatter(P[m, 0], P[m, 1], c=cor, s=14, alpha=0.7, label=f"grupo {k+1}")
        ax.set_xlabel("1º componente principal", fontsize=9.5)
        ax.set_ylabel("2º componente principal", fontsize=9.5)
        ax.set_title(tit, fontsize=10.5)
        ax.legend(fontsize=8.5); ax.tick_params(labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        salvar(fig, nome)


def regressao():
    d = load_diabetes()
    # regressão simples com 1 atributo (IMC) para visualizar resíduos
    x = d.data[:, 2].reshape(-1, 1)
    yv = d.target
    Xtr, Xte, ytr, yte = train_test_split(x, yv, test_size=0.3, random_state=1)
    reg = LinearRegression().fit(Xtr, ytr)
    pred = reg.predict(Xte)
    # resíduos
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    ax.plot(xs, reg.predict(xs), color=LARANJA, lw=2, zorder=3, label="reta ajustada")
    ax.scatter(Xte, yte, s=18, color=AZUL, alpha=0.7, zorder=2, label="valores reais")
    for xi, yi, pi in zip(Xte.ravel(), yte, pred):
        ax.plot([xi, xi], [yi, pi], color="#bbbbbb", lw=0.7, zorder=1)
    ax.set_xlabel("Índice de massa corporal (padronizado)", fontsize=9.5)
    ax.set_ylabel("Progressão da diabetes", fontsize=9.5)
    ax.set_title("Resíduos entre a reta da regressão e os valores reais", fontsize=10)
    ax.legend(fontsize=8.5); ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    salvar(fig, "fig_m6_reg_residuos.png")

    # regressão múltipla — real vs previsto
    Xtr, Xte, ytr, yte = train_test_split(d.data, yv, test_size=0.3, random_state=1)
    reg = LinearRegression().fit(Xtr, ytr)
    pred = reg.predict(Xte)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(yte, pred, s=20, color=VERDE, alpha=0.7)
    lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    ax.plot(lims, lims, color=CINZA if (CINZA := "#666666") else "#666", lw=1.2, ls=(0, (5, 3)))
    ax.set_xlabel("Valor real", fontsize=9.5)
    ax.set_ylabel("Valor previsto", fontsize=9.5)
    ax.set_title("Regressão Linear Múltipla — real vs. previsto", fontsize=10.5)
    ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    salvar(fig, "fig_m6_reg_real_prev.png")

    # saídas reais e previstas ordenadas pela 1ª componente principal
    pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(Xte)).ravel()
    ordem = np.argsort(pc1)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(pc1[ordem], yte[ordem], s=16, color=AZUL, alpha=0.6, label="valor real")
    ax.plot(pc1[ordem], pred[ordem], color=LARANJA, lw=2, label="valor previsto")
    ax.set_xlabel("1º componente principal", fontsize=9.5)
    ax.set_ylabel("Progressão da diabetes", fontsize=9.5)
    ax.set_title("Regressão Linear Múltipla vs. 1º componente principal", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.tick_params(labelsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    salvar(fig, "fig_m6_reg_multipla_pca.png")


if __name__ == "__main__":
    fronteiras_classificadores()
    knn_erro_por_k()
    rf_importancias()
    rf_erro_por_arvores()
    arvore_plot()
    bcancer_pca()
    clusters()
    regressao()
    print("OK - figuras de resultado do Modulo VI")
