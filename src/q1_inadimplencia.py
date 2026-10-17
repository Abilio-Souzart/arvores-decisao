"""Questão 1: modelo de inadimplência com árvore de decisão (base PEJOTA)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.tree import export_text, plot_tree

from .config import MAX_DEPTH_B, PROFUNDIDADES, TABLE_DIR
from .data import Split, load_pejota, make_split
from .preprocessing import feature_names, make_tree
from .utils import header, save_figure, save_table

CLASSES = ["BOM (0)", "MAU (1)"]


# ---------------------------------------------------------------------------
# (a)
# ---------------------------------------------------------------------------
def resumo_base(df: pd.DataFrame) -> pd.Series:
    print(f"Observações: {len(df)}")
    print(f"Proporção de inadimplentes (DEFAULT = 1): {df['DEFAULT'].mean():.1%}")
    ausentes = df.isna().sum()
    ausentes = ausentes[ausentes > 0]
    print("Colunas com valores ausentes antes da imputação:")
    print((ausentes.to_frame("n_ausentes")
           .assign(pct=lambda d: (100 * d["n_ausentes"] / len(df)).round(1))).to_string())
    print(f"Observação: IDADE (em meses) tem máximo de {df['IDADE'].max():.0f} "
          f"(~{df['IDADE'].max() / 12:.0f} anos), provável erro de digitação. "
          "Árvores são pouco sensíveis a esse tipo de valor extremo.")
    save_table(ausentes.rename("n_ausentes").to_frame(), "q1a_ausentes")
    return ausentes


# ---------------------------------------------------------------------------
# (b)
# ---------------------------------------------------------------------------
def descrever_nos(pipe, X_tr: pd.DataFrame, niveis: int = 2) -> pd.DataFrame:
    """Resumo dos nós dos primeiros níveis: regra, tamanho e % de MAU no treino."""
    arvore = pipe.named_steps["tree"].tree_
    nomes = feature_names(pipe)
    linhas = []

    def visitar(no: int, nivel: int, caminho: str):
        n = int(arvore.n_node_samples[no])
        pct_mau = float(arvore.value[no][0][1] / arvore.value[no][0].sum())
        folha = arvore.children_left[no] == -1
        regra = "" if folha else f"{nomes[arvore.feature[no]]} <= {arvore.threshold[no]:.2f}"
        linhas.append({"nível": nivel, "caminho": caminho or "(raiz)", "n": n,
                       "% MAU": round(100 * pct_mau, 1), "próxima divisão": regra})
        if folha or nivel >= niveis:
            return
        visitar(arvore.children_left[no], nivel + 1,
                f"{caminho} & {regra}".strip(" &"))
        visitar(arvore.children_right[no], nivel + 1,
                f"{caminho} & {regra.replace('<=', '>')}".strip(" &"))

    visitar(0, 0, "")
    return pd.DataFrame(linhas)


def plot_arvore(pipe, nome: str, titulo: str) -> None:
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(pipe.named_steps["tree"], feature_names=feature_names(pipe),
              class_names=CLASSES, filled=True, rounded=True, proportion=True,
              precision=2, fontsize=9, ax=ax)
    ax.set_title(titulo, fontsize=15)
    save_figure(fig, nome, dpi=90)


# ---------------------------------------------------------------------------
# (c)
# ---------------------------------------------------------------------------
def curva_profundidade(split: Split) -> pd.DataFrame:
    linhas = []
    for p in PROFUNDIDADES:
        pipe = make_tree(split.X_tr, p).fit(split.X_tr, split.y_tr)
        linhas.append({
            "profundidade": p,
            "acc_treino": accuracy_score(split.y_tr, pipe.predict(split.X_tr)),
            "acc_teste": accuracy_score(split.y_te, pipe.predict(split.X_te)),
        })
    out = pd.DataFrame(linhas).set_index("profundidade")
    out["gap"] = out["acc_treino"] - out["acc_teste"]
    return out


def plot_curva(curva: pd.DataFrame, melhor: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curva.index, curva["acc_treino"], "o-", color="#2196F3", lw=2, label="Treino")
    ax.plot(curva.index, curva["acc_teste"], "s--", color="#F44336", lw=2, label="Teste")
    ax.axvline(melhor, color="grey", ls=":",
               label=f"Maior acurácia no teste: profundidade {melhor}")
    ax.fill_between(curva.index, curva["acc_teste"], curva["acc_treino"],
                    where=curva["gap"] > 0, color="grey", alpha=0.12, label="Gap treino - teste")
    ax.set_xticks(curva.index)
    ax.set_xlabel("Profundidade máxima")
    ax.set_ylabel("Acurácia")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("(c) Acurácia no treino e no teste por profundidade")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "q1c_curva_profundidade")


# ---------------------------------------------------------------------------
# (d)
# ---------------------------------------------------------------------------
def plot_matriz(cm: np.ndarray, acc: float, profundidade: int) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"(d) Matriz de confusão: profundidade {profundidade}\n"
                 f"acurácia no teste = {acc:.1%}")
    fig.tight_layout()
    save_figure(fig, "q1d_matriz_confusao")


# ---------------------------------------------------------------------------
# (e)
# ---------------------------------------------------------------------------
def importancias(pipe) -> pd.Series:
    return pd.Series(pipe.named_steps["tree"].feature_importances_,
                     index=feature_names(pipe)).sort_values(ascending=False)


def plot_importancias(imp: dict[str, pd.Series]) -> None:
    fig, axes = plt.subplots(1, len(imp), figsize=(6 * len(imp), 4.5))
    for ax, (titulo, s) in zip(np.atleast_1d(axes), imp.items()):
        top = s.head(8)[::-1]
        ax.barh(top.index, top.values, color="#4C72B0")
        for i, v in enumerate(top.values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_title(titulo)
        ax.set_xlim(0, top.max() * 1.2)
    fig.suptitle("(e) Importância das variáveis (redução de Gini)")
    fig.tight_layout()
    save_figure(fig, "q1e_importancias")


def taxa_mau(df: pd.DataFrame, cond: pd.Series) -> str:
    return f"{df.loc[cond, 'DEFAULT'].mean():.0%} (n={cond.sum()})"


# ---------------------------------------------------------------------------
def run() -> dict:
    header("QUESTÃO 1 - Inadimplência com árvore de decisão (PEJOTA)")
    df = load_pejota()

    print("(a) Pré-processamento")
    resumo_base(df)

    split = make_split(df)
    print(f"\n(b) Treino: {len(split.y_tr)} | Teste: {len(split.y_te)} | "
          f"% MAU treino = {split.y_tr.mean():.1%}, teste = {split.y_te.mean():.1%}")
    arvore4 = make_tree(split.X_tr, MAX_DEPTH_B).fit(split.X_tr, split.y_tr)
    print(f"Preditores ({len(feature_names(arvore4))}): {', '.join(feature_names(arvore4))}")
    plot_arvore(arvore4, "q1b_arvore_profundidade_4", "(b) Árvore de classificação, profundidade 4")
    (TABLE_DIR / "q1b_arvore_regras.txt").write_text(
        export_text(arvore4.named_steps["tree"], feature_names=feature_names(arvore4)),
        encoding="utf-8")
    nos = descrever_nos(arvore4, split.X_tr)
    save_table(nos, "q1b_primeiros_nos", index=False)
    print("Primeiros níveis da árvore (dados de treino):")
    print(nos.to_string(index=False))

    print("\n(c) Acurácia por profundidade")
    curva = curva_profundidade(split)
    save_table(curva.round(4), "q1c_acuracia_profundidade")
    print((curva * 100).round(1).to_string())
    # Empates: escolhe a árvore mais simples
    melhor = int(curva.index[curva["acc_teste"] == curva["acc_teste"].max()].min())
    plot_curva(curva, melhor)

    print(f"\n(d) Profundidade com maior acurácia no teste: {melhor}")
    modelo = make_tree(split.X_tr, melhor).fit(split.X_tr, split.y_tr)
    pred = modelo.predict(split.X_te)
    cm = confusion_matrix(split.y_te, pred)
    acc = accuracy_score(split.y_te, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Acurácia: {acc:.4f}")
    print(f"Matriz [[VN, FP], [FN, VP]] = {cm.tolist()}")
    print(f"Bons classificados como maus (FP): {fp} | Maus classificados como bons (FN): {fn}")
    print(f"Sensibilidade (maus detectados): {tp / (tp + fn):.1%} | "
          f"Especificidade (bons aprovados): {tn / (tn + fp):.1%}")
    plot_matriz(cm, acc, melhor)
    plot_arvore(modelo, "q1d_arvore_escolhida",
                f"Árvore escolhida no item (d): profundidade {melhor}")

    print("\n(e) Importância das variáveis")
    paineis = {f"Árvore final (profundidade {melhor})": importancias(modelo)}
    if melhor != MAX_DEPTH_B:
        paineis[f"Árvore do item (b) (profundidade {MAX_DEPTH_B})"] = importancias(arvore4)
    save_table(pd.DataFrame(paineis).round(4), "q1e_importancias")
    for titulo, imp in paineis.items():
        print(f"Top 5 - {titulo}:")
        print(imp.head(5).round(3).to_string())
    plot_importancias(paineis)

    print("\nTaxa de inadimplência (base completa) por faixa:")
    print(f"  IFIN3 <= 25,9: {taxa_mau(df, df['IFIN3'] <= 25.9)} | "
          f"IFIN3 > 25,9: {taxa_mau(df, df['IFIN3'] > 25.9)}")
    for k in [0, 1, 2]:
        print(f"  REST_TIPO_1 = {k}: {taxa_mau(df, df['REST_TIPO_1'] == k)}")
    print(f"  REST_TIPO_1 >= 3: {taxa_mau(df, df['REST_TIPO_1'] >= 3)}")
    print(f"  REST_TIPO_5 = 0: {taxa_mau(df, df['REST_TIPO_5'] == 0)} | "
          f"REST_TIPO_5 >= 3: {taxa_mau(df, df['REST_TIPO_5'] >= 3)}")

    return {"df": df, "split": split, "curva": curva, "melhor": melhor,
            "modelo": modelo, "cm": cm, "acc": acc}
