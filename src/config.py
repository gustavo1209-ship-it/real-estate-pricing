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

# ── Bins para classificação de preço (BRL) ────────────────────────────────────
PRICE_BINS = [0, 500_000, 1_000_000, 2_000_000, 4_000_000, float("inf")]
PRICE_LABELS = ["Econômico", "Médio", "Alto", "Premium", "Luxo"]

# ── Colunas extras (não usadas pelo modelo, apenas para visualização) ──────────
CITY_COL          = "city"
REGION_COL        = "region"
BAIRRO_COL        = "bairro"
PROPERTY_TYPE_COL = "property_type"
LOTEAMENTO_COL    = "loteamento"

PROPERTY_TYPES = ["Casa", "Apartamento", "Condomínio Residencial", "Terreno", "Comercial"]

BAIRROS = {
    "Gramado":         ["Centro", "Várzea Grande", "Mato Queimado", "Três Pinheiros", "Morada da Serra", "Canavial"],
    "Canela":          ["Centro", "Fontes do Ipê", "Jardim", "Reserva da Serra", "São José", "Industrial"],
    "Caxias do Sul":   ["Centro", "Sanvitto", "São Pelegrino", "Exposição", "Santa Catarina", "Marechal Floriano", "Sagrada Família", "Desvio Rizzo"],
    "Bento Gonçalves": ["Centro", "São Bento", "Planalto", "Charqueadas", "Fátima", "São Roque"],
    "Farroupilha":     ["Centro", "São Caetano", "Cinquentenário", "Pio X", "Desvio Rizzo"],
    "Garibaldi":       ["Centro", "Santa Marta", "Progresso", "São João", "Bela Vista"],
    "Nova Petrópolis": ["Centro", "Linha Imperial", "Vila Nova", "Recanto Suíço", "Pinhal Alto"],
    "Nova Prata":      ["Centro", "São Jorge", "Planalto", "Vila Nova"],
    "Veranópolis":     ["Centro", "São João", "Bela Vista", "Santo Antônio"],
    "Vila Flores":     ["Centro", "Bairro Alto", "São José"],
    "Porto Alegre":    ["Moinhos de Vento", "Bela Vista", "Mont'Serrat", "Três Figueiras",
                        "Menino Deus", "Petrópolis", "Cidade Baixa", "Boa Vista",
                        "Chácara das Pedras", "Cristal", "Partenon", "Centro Histórico"],
}

CITY_COORDS = {
    "Gramado":         (-29.3789, -50.8761),
    "Canela":          (-29.3601, -50.8129),
    "Caxias do Sul":   (-29.1678, -51.1794),
    "Bento Gonçalves": (-29.1707, -51.5185),
    "Farroupilha":     (-29.2240, -51.3468),
    "Garibaldi":       (-29.2564, -51.5328),
    "Nova Petrópolis": (-29.3743, -51.1135),
    "Nova Prata":      (-28.7839, -51.6134),
    "Veranópolis":     (-28.9378, -51.5581),
    "Vila Flores":     (-28.8617, -51.6017),
    "Porto Alegre":    (-30.0346, -51.2177),
}
