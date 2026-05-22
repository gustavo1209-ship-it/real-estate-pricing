"""
Configuração central do projeto.

Para adaptar ao dataset Jiff do Kaggle (elakiricoder/jiffs-house-price-prediction-dataset),
atualize os nomes das colunas abaixo conforme o CSV real.
"""

import os

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_CSV = os.path.join(DATA_RAW_DIR, "house_price.csv")
PROCESSED_CSV = os.path.join(DATA_PROCESSED_DIR, "house_price_processed.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")

# ── Mapeamento de colunas (ajuste se o CSV Kaggle tiver nomes diferentes) ─────
# Coluna alvo
TARGET_COL = "price"

# Colunas numéricas contínuas
NUM_FEATURES = [
    "house_size",       # área construída (sqft ou m²)
    "lot_size",         # área total do terreno
    "sqft_above",       # área acima do solo
    "sqft_basement",    # área do porão
    "lat",              # latitude
    "long",             # longitude
]

# Colunas inteiras (discretas)
INT_FEATURES = [
    "bedrooms",
    "bathrooms",
    "floors",
    "view",             # qualidade da vista (0-4)
    "condition",        # condição do imóvel (1-5)
    "grade",            # nota geral (1-13)
    "year_built",
    "year_renovated",   # 0 = nunca reformado
    "zipcode",
]

# Colunas binárias
BIN_FEATURES = [
    "waterfront",       # frente para água (0/1)
]

ALL_FEATURES = NUM_FEATURES + INT_FEATURES + BIN_FEATURES

# ── Parâmetros do modelo ───────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Bins para classificação de preço (USD) ────────────────────────────────────
PRICE_BINS = [0, 250_000, 500_000, 750_000, 1_000_000, float("inf")]
PRICE_LABELS = ["Econômico", "Médio", "Alto", "Premium", "Luxo"]
