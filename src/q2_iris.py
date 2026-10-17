"""Questão 2: entendendo a árvore visualmente (Iris)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from .config import N_FOLDS, SEED
from .utils import header, save_figure, save_table


def plot_arvore(arvore, iris, nome: str, titulo: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(arvore, feature_names=iris.feature_names, class_names=list(iris.target_names),
              filled=True, rounded=True, proportion=True, precision=2, fontsize=11, ax=ax)
    ax.set_title(titulo, fontsize=13)
    save_figure(fig, nome)


def plot_dispersao(iris) -> None:
    X, y = iris.data, iris.target
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cores = ["#1b9e77", "#d95f02", "#7570b3"]
    for k, nome in enumerate(iris.target_names):
        m = y == k
        axes[0].scatter(X.loc[m, "petal length (cm)"], X.loc[m, "petal width (cm)"],
                        s=18, color=cores[k], label=nome, alpha=0.8)
        axes[1].hist(X.loc[m, "petal length (cm)"], bins=20, color=cores[k], alpha=0.7, label=nome)
    axes[0].axvline(2.45, color="k", ls="--", lw=1)
    axes[0].axhline(1.75, xmin=0.25, color="k", ls=":", lw=1)
    axes[0].set_xlabel("petal length (cm)")
    axes[0].set_ylabel("petal width (cm)")
    axes[0].set_title("As duas regras da árvore de profundidade 2")
    axes[0].legend()
    axes[1].axvline(2.45, color="k", ls="--", lw=1)
    axes[1].set_xlabel("petal length (cm)")
    axes[1].set_title("Setosa fica isolada abaixo de 2,45 cm")
    fig.tight_layout()
    save_figure(fig, "q2_iris_dispersao")


def raiz_por_semente(X, y, n=20) -> pd.Series:
    """Variável escolhida na raiz (profundidade 1) para diferentes random_state."""
    return pd.Series([X.columns[DecisionTreeClassifier(max_depth=1, random_state=s)
                                .fit(X, y).tree_.feature[0]] for s in range(n)]).value_counts()


def run() -> dict:
    header("QUESTÃO 2 - Árvore de decisão na base Iris")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    # (a)
    arv2 = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X, y)
    print("(a) Regras da árvore de profundidade 2:")
    print(export_text(arv2, feature_names=list(X.columns)))
    plot_arvore(arv2, iris, "q2a_iris_profundidade_2", "(a) Iris: profundidade 2")
    plot_dispersao(iris)

    # (b)
    arv1 = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X, y)
    print("(b) Profundidade 1:")
    print(export_text(arv1, feature_names=list(X.columns)))
    plot_arvore(arv1, iris, "q2b_iris_profundidade_1", "(b) Iris: profundidade 1")
    raizes = raiz_por_semente(X, y)
    save_table(raizes.rename("n_sementes").to_frame(), "q2b_variavel_raiz_por_semente")
    print("    Variável escolhida na raiz em 20 valores de random_state:")
    print(raizes.to_string())
    setosa = y == 0
    print(f"    Setosa: petal length máx = {X.loc[setosa, 'petal length (cm)'].max()}, "
          f"demais mín = {X.loc[~setosa, 'petal length (cm)'].min()}")
    print(f"    Setosa: petal width máx = {X.loc[setosa, 'petal width (cm)'].max()}, "
          f"demais mín = {X.loc[~setosa, 'petal width (cm)'].min()}")

    # (c)
    acc_res = accuracy_score(y, arv2.predict(X))
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    cv2 = cross_val_score(DecisionTreeClassifier(max_depth=2, random_state=SEED), X, y, cv=folds)
    cv_full = cross_val_score(DecisionTreeClassifier(random_state=SEED), X, y, cv=folds)
    acc_full = accuracy_score(y, DecisionTreeClassifier(random_state=SEED).fit(X, y).predict(X))
    tabela = pd.DataFrame({
        "acurácia nos próprios dados": [acc_res, acc_full],
        "acurácia CV 10 folds": [cv2.mean(), cv_full.mean()],
    }, index=["profundidade 2", "sem limite"]).round(4)
    tabela["otimismo (p.p.)"] = (100 * (tabela.iloc[:, 0] - tabela.iloc[:, 1])).round(1)
    save_table(tabela, "q2c_iris_acuracia")
    print("(c) Acurácia:")
    print(tabela.to_string())
    return {"tabela": tabela, "raizes": raizes}
