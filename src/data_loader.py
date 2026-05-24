"""
Carregamento e limpeza inicial do dataset de preços imobiliários.
Compatível com o dataset Jiff (Kaggle) e o KC House Dataset.
"""

import os
import pandas as pd
import numpy as np
from src.config import (
    RAW_CSV, PROCESSED_CSV, TARGET_COL, ALL_FEATURES,
    PRICE_BINS, PRICE_LABELS, CITY_COL, REGION_COL, BAIRRO_COL, PROPERTY_TYPE_COL, LOTEAMENTO_COL
)


def _generate_sample_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Gera dados sintéticos no formato do dataset Jiff para demo."""
    rng = np.random.default_rng(seed)

    year_built = rng.integers(1900, 2023, n)
    has_renovation = rng.random(n) < 0.25
    year_renovated = np.where(
        has_renovation,
        rng.integers(1990, 2023, n),
        0
    )
    bedrooms = rng.integers(1, 8, n)
    bathrooms = (rng.integers(1, 5, n) + rng.integers(0, 2, n) * 0.5)
    floors = rng.choice([1, 1.5, 2, 2.5, 3], n, p=[0.4, 0.1, 0.35, 0.05, 0.1])
    house_size = rng.integers(500, 8000, n)
    sqft_basement = np.where(rng.random(n) < 0.6, 0, rng.integers(200, 2000, n))
    sqft_above = house_size - sqft_basement
    lot_size = house_size + rng.integers(500, 5000, n)
    grade = rng.integers(4, 13, n)
    condition = rng.integers(1, 6, n)
    view = rng.integers(0, 5, n)
    waterfront = (rng.random(n) < 0.07).astype(int)
    zipcode = rng.choice(range(98001, 98200), n)
    lat = rng.uniform(47.1, 47.8, n)
    long_ = rng.uniform(-122.5, -121.3, n)

    base_price = (
        house_size * 150
        + bedrooms * 15_000
        + bathrooms * 12_000
        + (2023 - year_built) * (-300)
        + waterfront * 200_000
        + grade * 20_000
        + view * 30_000
        + rng.normal(0, 50_000, n)
    )
    price = np.clip(base_price, 75_000, 3_000_000).astype(int)

    return pd.DataFrame({
        "price": price,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_size": house_size,
        "lot_size": lot_size,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "year_built": year_built,
        "year_renovated": year_renovated,
        "zipcode": zipcode,
        "lat": lat,
        "long": long_,
    })


def load_raw(path: str | None = None) -> pd.DataFrame:
    """
    Carrega o CSV bruto. Se não encontrado, gera dados sintéticos de demo.

    Args:
        path: caminho para o CSV. Padrão: data/raw/house_price.csv
    Returns:
        DataFrame com os dados brutos
    """
    csv_path = path or RAW_CSV

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[data_loader] Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    else:
        print(
            f"[data_loader] CSV não encontrado em '{csv_path}'.\n"
            "  → Baixe o dataset em: https://www.kaggle.com/datasets/elakiricoder/jiffs-house-price-prediction-dataset\n"
            "  → Salve como data/raw/house_price.csv\n"
            "  → Usando dados sintéticos para demo."
        )
        df = _generate_sample_data()

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza básica: remove duplicatas, trata missings, filtra outliers extremos.
    """
    df = df.copy()

    # Remove duplicatas
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        print(f"[data_loader] Removidas {before - len(df)} duplicatas")

    # Remove linhas sem target
    df = df.dropna(subset=[TARGET_COL])

    # Colunas numéricas: preenche NaN com mediana
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Garante colunas extras como string mesmo que venham de CSV sem elas
    for col in [CITY_COL, REGION_COL, BAIRRO_COL, PROPERTY_TYPE_COL, LOTEAMENTO_COL]:
        if col not in df.columns:
            df[col] = "—"

    # Remove bedrooms = 0, exceto terrenos (que legitimamente têm bedrooms=0)
    if "bedrooms" in df.columns:
        is_terreno = df[PROPERTY_TYPE_COL] == "Terreno" if PROPERTY_TYPE_COL in df.columns else pd.Series(False, index=df.index)
        df = df[(df["bedrooms"] > 0) | is_terreno]
    df = df[df[TARGET_COL] > 0]

    # Remove outliers extremos de preço (> 99.9th percentile)
    upper = df[TARGET_COL].quantile(0.999)
    df = df[df[TARGET_COL] <= upper]

    print(f"[data_loader] Dataset limpo: {df.shape[0]:,} linhas")
    return df.reset_index(drop=True)


def add_price_category(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna categórica de faixa de preço."""
    df = df.copy()
    df["price_category"] = pd.cut(
        df[TARGET_COL],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        right=True
    )
    return df


def load_and_prepare(path: str | None = None) -> pd.DataFrame:
    """Pipeline completo: load → clean → categorize."""
    df = load_raw(path)
    df = clean(df)
    df = add_price_category(df)
    return df
