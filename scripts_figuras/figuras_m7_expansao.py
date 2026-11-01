# -*- coding: utf-8 -*-
"""Gera as figuras novas do Módulo VII (dígitos do scikit-learn e séries FICTÍCIAS).

Rode a partir de qualquer pasta:  python figuras_m7_expansao.py
Requer PyTorch (CPU). Saída em ../../figuras:
fig_m7_digitos.png, fig_m7_overfit.png, fig_m7_otimizadores.png, fig_m7_mlp_cnn.png,
fig_m7_serie_lstm.png, fig_m7_autoencoder.png. Sementes fixas (42).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from src.dados_sinteticos_m7 import serie_senoide, janelas

OUT = os.path.join(AQUI, "..", "..", "figuras")
AZUL, LARANJA, VERDE, CINZA = "#2c6fbf", "#d9822b", "#3a9a5b", "#777777"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


class MLP(nn.Module):
    def __init__(self, entrada=64, oculta=64, saida=10, p_dropout=0.0):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(entrada, oculta), nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(oculta, oculta), nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(oculta, saida))

    def forward(self, x):
        return self.rede(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classificador = nn.Linear(64, 10)

    def forward(self, x):
        return self.classificador(self.conv(x).flatten(1))


def avaliar(modelo, X, y):
    modelo.eval()
    with torch.no_grad():
        lg = modelo(X)
        return nn.functional.cross_entropy(lg, y).item(), (lg.argmax(1) == y).float().mean().item()


def treinar(modelo, otimizador, X, y, X_val, y_val, epocas=100, batch=64, seed=42):
    carga = DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    h = {"perda_tr": [], "perda_val": [], "acc_tr": [], "acc_val": []}
    for _ in range(epocas):
        modelo.train()
        for xb, yb in carga:
            otimizador.zero_grad()
            nn.functional.cross_entropy(modelo(xb), yb).backward()
            otimizador.step()
        l_tr, a_tr = avaliar(modelo, X, y)
        l_v, a_v = avaliar(modelo, X_val, y_val)
        for k, v in zip(h, (l_tr, l_v, a_tr, a_v)):
            h[k].append(v)
    return h


digitos = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digitos.data, digitos.target, test_size=0.25,
                                          stratify=digitos.target, random_state=42)
esc = StandardScaler().fit(X_tr)
Xtr_t = torch.tensor(esc.transform(X_tr), dtype=torch.float32)
Xte_t = torch.tensor(esc.transform(X_te), dtype=torch.float32)
ytr_t, yte_t = torch.tensor(y_tr), torch.tensor(y_te)


def fig_digitos():
    fig, eixos = plt.subplots(2, 8, figsize=(9, 2.8))
    for ax, img, r in zip(eixos.ravel(), digitos.images, digitos.target):
        ax.imshow(img, cmap="gray_r"); ax.set_title(int(r), fontsize=9); ax.axis("off")
    fig.suptitle("Dígitos 8×8 do scikit-learn (valores de 0 a 16)", fontsize=10)
    salvar(fig, "fig_m7_digitos.png")


def fig_overfit():
    Xp, yp, Xv, yv = Xtr_t[:200], ytr_t[:200], Xtr_t[200:600], ytr_t[200:600]
    torch.manual_seed(42)
    a = MLP(oculta=256)
    h_sem = treinar(a, torch.optim.Adam(a.parameters(), lr=1e-3), Xp, yp, Xv, yv, 150)
    torch.manual_seed(42)
    b = MLP(oculta=256, p_dropout=0.5)
    h_reg = treinar(b, torch.optim.AdamW(b.parameters(), lr=1e-3, weight_decay=1e-2), Xp, yp, Xv, yv, 150)
    fig, eixos = plt.subplots(1, 2, figsize=(10, 3.8))
    eixos[0].plot(h_sem["perda_tr"], "--", color=AZUL, label="treino — sem regularização")
    eixos[0].plot(h_sem["perda_val"], color=AZUL, label="validação — sem regularização")
    eixos[0].plot(h_reg["perda_tr"], "--", color=LARANJA, label="treino — dropout + weight decay")
    eixos[0].plot(h_reg["perda_val"], color=LARANJA, label="validação — dropout + weight decay")
    eixos[0].axvline(int(np.argmin(h_sem["perda_val"])), color=CINZA, ls=":", lw=1)
    eixos[0].set_xlabel("época"); eixos[0].set_ylabel("perda"); eixos[0].set_yscale("log")
    eixos[0].set_title("Curvas de perda (200 imagens de treino)"); eixos[0].legend(fontsize=7)
    eixos[1].plot(h_sem["acc_tr"], "--", color=AZUL, label="treino — sem regularização")
    eixos[1].plot(h_sem["acc_val"], color=AZUL, label="validação — sem regularização")
    eixos[1].plot(h_reg["acc_val"], color=LARANJA, label="validação — dropout + weight decay")
    eixos[1].set_xlabel("época"); eixos[1].set_ylabel("acurácia"); eixos[1].legend(fontsize=7, loc="lower right")
    eixos[1].set_title("Acurácia");
    for ax in eixos: ax.grid(alpha=0.3)
    fig.tight_layout(); salvar(fig, "fig_m7_overfit.png")


def fig_otimizadores():
    configs = {
        "SGD lr=0,01": lambda p: torch.optim.SGD(p, lr=0.01),
        "SGD + momentum": lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
        "Adam lr=1e-3": lambda p: torch.optim.Adam(p, lr=1e-3),
        "Adam lr=1e-1 (alta)": lambda p: torch.optim.Adam(p, lr=1e-1),
        "Adam lr=1e-5 (baixa)": lambda p: torch.optim.Adam(p, lr=1e-5),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (nome, f), cor in zip(configs.items(), [CINZA, VERDE, AZUL, LARANJA, "#a04bb0"]):
        torch.manual_seed(42); m = MLP()
        h = treinar(m, f(m.parameters()), Xtr_t, ytr_t, Xte_t, yte_t, 15)
        ax.plot(range(1, 16), h["perda_tr"], color=cor, label=nome)
    ax.set_yscale("log"); ax.set_xlabel("época"); ax.set_ylabel("perda de treino")
    ax.set_title("Otimizador e taxa de aprendizado"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); salvar(fig, "fig_m7_otimizadores.png")


def fig_mlp_cnn():
    Xi_tr = torch.tensor(X_tr / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)
    Xi_te = torch.tensor(X_te / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)
    res = {}
    for nome, M, (A, B) in [("MLP", MLP, (Xi_tr.flatten(1), Xi_te.flatten(1))), ("CNN", CNN, (Xi_tr, Xi_te))]:
        torch.manual_seed(42); m = M()
        res[nome] = (treinar(m, torch.optim.Adam(m.parameters(), lr=1e-3), A, ytr_t, B, yte_t, 30),
                     sum(p.numel() for p in m.parameters()))
    fig, eixos = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for (nome, (h, n)), cor in zip(res.items(), [AZUL, LARANJA]):
        eixos[0].plot(range(1, 31), h["acc_val"], color=cor, label=f"{nome} ({n} parâmetros)")
        eixos[1].plot(range(1, 31), h["perda_val"], color=cor, label=nome)
    eixos[0].set_ylabel("acurácia de teste"); eixos[1].set_ylabel("perda de teste")
    for ax in eixos:
        ax.set_xlabel("época"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    eixos[0].set_ylim(0.8, 1.0)
    eixos[0].set_title("MLP × CNN nos dígitos"); eixos[1].set_title("Perda de teste")
    fig.tight_layout(); salvar(fig, "fig_m7_mlp_cnn.png")


def fig_serie_lstm():
    serie = serie_senoide(400, seed=42)
    X, y = janelas(serie, 20)
    Xtr, ytr, Xte, yte = (torch.tensor(a) for a in (X[:300], y[:300], X[300:], y[300:]))

    class Prev(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True); self.saida = nn.Linear(32, 1)

        def forward(self, x):
            h, _ = self.lstm(x); return self.saida(h[:, -1]).squeeze(-1)

    torch.manual_seed(42); p = Prev(); opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad(); nn.functional.mse_loss(p(Xtr), ytr).backward(); opt.step()
    p.eval()
    with torch.no_grad():
        pred = p(Xte)
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(range(len(serie)), serie, color=CINZA, lw=0.8, label="série fictícia (senoide + ruído)")
    ax.plot(range(320, 400), pred.numpy(), color=LARANJA, lw=1.6, label="previsão da LSTM (teste)")
    ax.axvline(320, color="k", ls=":", lw=1); ax.text(322, ax.get_ylim()[1] * 0.92, "teste", fontsize=8)
    ax.set_xlabel("tempo"); ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    ax.set_title("LSTM prevendo o próximo valor a partir de 20 passos")
    fig.tight_layout(); salvar(fig, "fig_m7_serie_lstm.png")


def fig_autoencoder():
    Xa_tr = torch.tensor(X_tr / 16.0, dtype=torch.float32)
    Xa_te = torch.tensor(X_te / 16.0, dtype=torch.float32)
    torch.manual_seed(42)
    cod = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
    dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
    ae = nn.Sequential(cod, dec)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    carga = DataLoader(TensorDataset(Xa_tr), batch_size=64, shuffle=True,
                       generator=torch.Generator().manual_seed(42))
    for _ in range(60):
        for (xb,) in carga:
            opt.zero_grad(); nn.functional.mse_loss(ae(xb), xb).backward(); opt.step()
    with torch.no_grad():
        rec = ae(Xa_te)
    fig, eixos = plt.subplots(2, 8, figsize=(9, 2.9))
    for i in range(8):
        eixos[0, i].imshow(Xa_te[i].reshape(8, 8), cmap="gray_r"); eixos[0, i].axis("off")
        eixos[1, i].imshow(rec[i].reshape(8, 8), cmap="gray_r"); eixos[1, i].axis("off")
    eixos[0, 0].set_title("original", fontsize=8, loc="left")
    eixos[1, 0].set_title("reconstruído (8 números)", fontsize=8, loc="left")
    fig.tight_layout(); salvar(fig, "fig_m7_autoencoder.png")


if __name__ == "__main__":
    fig_digitos(); fig_overfit(); fig_otimizadores(); fig_mlp_cnn(); fig_serie_lstm(); fig_autoencoder()
    print("figuras do Módulo VII geradas em", os.path.abspath(OUT))
