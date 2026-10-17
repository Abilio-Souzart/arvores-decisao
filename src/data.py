"""Leitura e limpeza da base PEJOTA (passos 1 a 4 do enunciado).

O passo 5 (imputação e one-hot) depende da divisão treino/teste e fica em
`preprocessing.py`, para que as medianas sejam calculadas só com o treino.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import COLS_COM_BRANCOS, COLS_EXCLUIDAS, DATA_PATH, SEED, TEST_SIZE


def load_pejota() -> pd.DataFrame:
    # 1. Aba PEJOTA
    df = pd.read_excel(DATA_PATH, sheet_name="PEJOTA")
    # 3. Espaços nos nomes ("IDADE " -> "IDADE", "RAMO " -> "RAMO")
    df.columns = df.columns.str.strip()
    # 2. Células com espaço em branco -> NaN, depois numérico
    df[COLS_COM_BRANCOS] = (df[COLS_COM_BRANCOS]
                            .replace(r"^\s*$", np.nan, regex=True)
                            .apply(pd.to_numeric, errors="coerce"))
    # 4. Alvo
    df["DEFAULT"] = (df["STATUS"] == "MAU").astype(int)
    return df


def features_e_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    faltando = [c for c in COLS_EXCLUIDAS if c not in df.columns]
    if faltando:
        # Sem errors="ignore": um nome errado deve gerar erro, e não manter a coluna.
        raise KeyError(f"Colunas não encontradas: {faltando}")
    return df.drop(columns=COLS_EXCLUIDAS), df["DEFAULT"]


@dataclass
class Split:
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: pd.Series
    y_te: pd.Series


def make_split(df: pd.DataFrame, seed: int = SEED) -> Split:
    X, y = features_e_alvo(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    return Split(X_tr, X_te, y_tr, y_te)
