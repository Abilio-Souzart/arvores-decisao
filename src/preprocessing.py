"""Passo 5: imputação pela mediana do treino e one-hot de UF, dentro de um Pipeline.

Com o Pipeline, a mediana e as categorias de UF são aprendidas apenas nos
dados de treino (ou nos folds de treino, durante a validação cruzada).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .config import COLS_CATEGORICAS, SEED


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numericas = [c for c in X.columns if c not in COLS_CATEGORICAS]
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), numericas),
         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), COLS_CATEGORICAS)],
        verbose_feature_names_out=False,
    )


def make_tree(X: pd.DataFrame, max_depth: int | None, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("tree", DecisionTreeClassifier(max_depth=max_depth, random_state=seed)),
    ])


def feature_names(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())
