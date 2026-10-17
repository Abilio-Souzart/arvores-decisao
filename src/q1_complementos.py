"""Análises complementares à Questão 1.

1. Escolha da profundidade por validação cruzada no treino, sem usar o teste.
2. Variação da "melhor profundidade" entre diferentes divisões treino/teste.
3. Escolha do ponto de corte considerando o custo de cada tipo de erro.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import CUSTO_FN, CUSTO_FP, N_FOLDS, PROFUNDIDADES, SEED
from .data import Split, make_split
from .preprocessing import make_tree
from .utils import header, save_figure, save_table


def cv_profundidade(split: Split) -> pd.DataFrame:
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    linhas = []
    for p in PROFUNDIDADES:
        s = cross_val_score(make_tree(split.X_tr, p), split.X_tr, split.y_tr,
                            cv=folds, scoring="accuracy")
        linhas.append({"profundidade": p, "acc_cv": s.mean(),
                       "ep": s.std(ddof=1) / np.sqrt(len(s))})
    return pd.DataFrame(linhas).set_index("profundidade")


def escolher(cv: pd.DataFrame) -> tuple[int, int]:
    """Profundidade de maior acurácia de CV e a mais simples a até 1 erro padrão."""
    p_max = int(cv["acc_cv"].idxmax())
    limite = cv.loc[p_max, "acc_cv"] - cv.loc[p_max, "ep"]
    p_1ep = int(cv.index[cv["acc_cv"] >= limite].min())
    return p_max, p_1ep


def melhor_por_divisao(df: pd.DataFrame, n_rep: int = 50) -> pd.DataFrame:
    linhas = []
    for seed in range(n_rep):
        s = make_split(df, seed=seed)
        accs = [accuracy_score(s.y_te, make_tree(s.X_tr, p, seed).fit(s.X_tr, s.y_tr)
                               .predict(s.X_te)) for p in PROFUNDIDADES]
        linhas.append({"seed": seed, "melhor_profundidade": PROFUNDIDADES[int(np.argmax(accs))],
                       **{f"acc_p{p}": a for p, a in zip(PROFUNDIDADES, accs)}})
    return pd.DataFrame(linhas).set_index("seed")


def custo_por_corte(y, prob, cortes) -> pd.DataFrame:
    linhas = []
    for c in cortes:
        pred = (prob >= c).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        linhas.append({"corte": c, "FP (bom recusado)": fp, "FN (mau aprovado)": fn,
                       "acurácia": (tp + tn) / len(y),
                       "custo total": CUSTO_FP * fp + CUSTO_FN * fn})
    return pd.DataFrame(linhas).set_index("corte")


def plot_cv(cv: pd.DataFrame, curva_teste: pd.DataFrame, p_max: int, p_1ep: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(cv.index, cv["acc_cv"], yerr=cv["ep"], fmt="o-", color="#4CAF50",
                capsize=4, lw=2, label="CV 10 folds no treino (± 1 EP)")
    ax.plot(curva_teste.index, curva_teste["acc_teste"], "s--", color="#F44336",
            alpha=0.6, label="Teste (uma única divisão)")
    ax.axvline(p_max, color="#4CAF50", ls=":", label=f"Maior acurácia de CV: {p_max}")
    if p_1ep != p_max:
        ax.axvline(p_1ep, color="grey", ls=":", label=f"Regra de 1 EP: {p_1ep}")
    ax.set_xticks(cv.index)
    ax.set_xlabel("Profundidade máxima")
    ax.set_ylabel("Acurácia")
    ax.set_title("Escolha da profundidade: validação cruzada × conjunto de teste")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "comp_cv_profundidade")


def plot_divisoes(rep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cont = rep["melhor_profundidade"].value_counts().reindex(PROFUNDIDADES, fill_value=0)
    axes[0].bar(cont.index, cont.values, color="#90A4AE")
    axes[0].set_xticks(PROFUNDIDADES)
    axes[0].set_xlabel("Profundidade com maior acurácia no teste")
    axes[0].set_ylabel("Nº de divisões")
    axes[0].set_title(f"Melhor profundidade em {len(rep)} divisões treino/teste")
    accs = rep[[f"acc_p{p}" for p in PROFUNDIDADES]]
    axes[1].boxplot([accs[c] for c in accs], tick_labels=PROFUNDIDADES)
    axes[1].set_xlabel("Profundidade máxima")
    axes[1].set_ylabel("Acurácia no teste")
    axes[1].set_title("Distribuição da acurácia no teste")
    fig.tight_layout()
    save_figure(fig, "comp_variacao_divisoes")


def run(q1: dict) -> dict:
    header("ANÁLISES COMPLEMENTARES - Questão 1")
    split, df = q1["split"], q1["df"]

    # 1. Validação cruzada
    cv = cv_profundidade(split)
    p_max, p_1ep = escolher(cv)
    save_table(cv.round(4), "comp_cv_profundidade")
    print("1. Acurácia de CV (10 folds, só treino):")
    print((cv * 100).round(1).to_string())
    print(f"   Maior acurácia de CV: profundidade {p_max} | regra de 1 EP: {p_1ep}")
    plot_cv(cv, q1["curva"], p_max, p_1ep)
    for p in sorted({p_max, p_1ep}):
        acc = accuracy_score(split.y_te, make_tree(split.X_tr, p).fit(split.X_tr, split.y_tr)
                             .predict(split.X_te))
        print(f"   Profundidade {p}: acurácia no teste = {acc:.1%}")

    # 2. Variação entre divisões
    rep = melhor_por_divisao(df)
    save_table(rep.round(4), "comp_melhor_profundidade_por_divisao")
    print("\n2. Profundidade de maior acurácia no teste em 50 divisões:")
    print(rep["melhor_profundidade"].value_counts().sort_index().to_string())
    medias = rep[[f"acc_p{p}" for p in PROFUNDIDADES]].mean()
    print("   Acurácia média no teste por profundidade: " +
          ", ".join(f"{p}: {m:.1%}" for p, m in zip(PROFUNDIDADES, medias)))
    plot_divisoes(rep)

    # 3. Custo dos erros
    modelo = q1["modelo"]
    prob = modelo.predict_proba(split.X_te)[:, 1]
    corte_otimo = CUSTO_FP / (CUSTO_FP + CUSTO_FN)
    cortes = sorted({0.5, round(corte_otimo, 3), 0.25, 0.4, 0.6})
    custos = custo_por_corte(split.y_te.to_numpy(), prob, cortes)
    save_table(custos.round(4), "comp_custo_por_corte")
    print(f"\n3. Custo com FN = {CUSTO_FN} x FP (corte teórico = {corte_otimo:.3f}):")
    print(custos.round(3).to_string())

    return {"cv": cv, "p_max": p_max, "p_1ep": p_1ep, "rep": rep, "custos": custos}
