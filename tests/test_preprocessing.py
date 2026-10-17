"""Testes do pré-processamento da base PEJOTA."""
import numpy as np
import pandas as pd
import pytest

from src.data import features_e_alvo, load_pejota, make_split
from src.preprocessing import feature_names, make_tree


@pytest.fixture(scope="module")
def df():
    return load_pejota()


def test_nomes_sem_espacos(df):
    assert "IDADE" in df.columns and "RAMO" in df.columns
    assert not any(c != c.strip() for c in df.columns)


def test_alvo(df):
    assert set(df["DEFAULT"].unique()) == {0, 1}
    assert (df.loc[df["STATUS"] == "MAU", "DEFAULT"] == 1).all()


def test_brancos_viram_nan(df):
    for c in ["IFIN1", "IFIN2", "IFIN4", "IFIN5", "IFIN7", "IFIN8"]:
        assert pd.api.types.is_numeric_dtype(df[c])
        assert df[c].isna().sum() > 0


def test_identificador_fora_dos_preditores(df):
    X, _ = features_e_alvo(df)
    assert "COD_CLI" not in X.columns
    assert "STATUS" not in X.columns and "DEFAULT" not in X.columns


def test_nome_errado_gera_erro(df):
    with pytest.raises(KeyError):
        features_e_alvo(df.rename(columns={"COD_CLI": "COD CLI"}))


def test_mediana_vem_so_do_treino(df):
    split = make_split(df)
    pipe = make_tree(split.X_tr, 2).fit(split.X_tr, split.y_tr)
    imputer = pipe.named_steps["prep"].named_transformers_["num"]
    nomes = [c for c in split.X_tr.columns if c != "UF"]
    medianas_treino = split.X_tr[nomes].median().to_numpy()
    assert np.allclose(imputer.statistics_, medianas_treino)


def test_one_hot_uf(df):
    split = make_split(df)
    pipe = make_tree(split.X_tr, 2).fit(split.X_tr, split.y_tr)
    nomes = feature_names(pipe)
    assert "UF" not in nomes
    assert {"UF_RJ", "UF_SP"} <= set(nomes)
