"""Caminhos e parâmetros globais."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "base_pejota.xls"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

SEED = 42
TEST_SIZE = 0.30
N_FOLDS = 10
PROFUNDIDADES = list(range(1, 11))
MAX_DEPTH_B = 4

COLS_COM_BRANCOS = ["IFIN1", "IFIN2", "IFIN4", "IFIN5", "IFIN7", "IFIN8"]
COLS_EXCLUIDAS = ["COD_CLI", "STATUS", "DEFAULT"]   # identificador e alvo
COLS_CATEGORICAS = ["UF"]

# Análise de custo (complementar): conceder crédito a um mau pagador custa
# CUSTO_FN vezes mais do que recusar um bom pagador.
CUSTO_FN = 5
CUSTO_FP = 1


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
