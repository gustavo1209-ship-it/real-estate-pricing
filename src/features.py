"""
Pipeline de feature engineering para preços imobiliários.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.config import NUM_FEATURES, INT_FEATURES, BIN_FEATURES, ALL_FEATURES


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Cria features derivadas antes do pipeline sklearn.
    Aplicado no DataFrame antes do ColumnTransformer.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Idade do imóvel
        if "year_built" in df.columns:
            df["house_age"] = 2024 - df["year_built"]

        # Imóvel foi reformado?
        if "year_renovated" in df.columns:
            df["was_renovated"] = (df["year_renovated"] > 0).astype(int)
            df["years_since_renovation"] = np.where(
                df["year_renovated"] > 0,
                2024 - df["year_renovated"],
                df.get("house_age", 0)
            )

        # Razão banheiro/quarto
        if "bathrooms" in df.columns and "bedrooms" in df.columns:
            df["bath_per_bed"] = df["bathrooms"] / (df["bedrooms"] + 1)

        # Porão existe?
        if "sqft_basement" in df.columns:
            df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

        # Tamanho relativo da casa vs lote
        if "house_size" in df.columns and "lot_size" in df.columns:
            df["house_lot_ratio"] = df["house_size"] / (df["lot_size"] + 1)

        # Score combinado de qualidade
        grade_col = "grade" if "grade" in df.columns else None
        cond_col = "condition" if "condition" in df.columns else None
        if grade_col and cond_col:
            df["quality_score"] = df[grade_col] * df[cond_col]

        return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Detecta colunas disponíveis no DataFrame e as divide em
    numéricas (para scaling) e inteiras/binárias (sem scaling).
    """
    base_num = [c for c in NUM_FEATURES if c in df.columns]
    base_int = [c for c in INT_FEATURES + BIN_FEATURES if c in df.columns]

    # Features derivadas pelo HouseFeatureEngineer
    derived_num = [
        c for c in [
            "house_age", "years_since_renovation", "bath_per_bed",
            "house_lot_ratio", "quality_score"
        ]
        if c in df.columns
    ]
    derived_int = [
        c for c in ["was_renovated", "has_basement"]
        if c in df.columns
    ]

    return base_num + derived_num, base_int + derived_int


def build_preprocessing_pipeline(num_cols: list[str], int_cols: list[str]) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer:
    - Colunas numéricas contínuas: imputer + RobustScaler
    - Colunas inteiras/binárias: imputer (sem scaling)
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    int_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("int", int_pipeline, int_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retorna nomes das features após o ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        names = []
        for _, _, cols in preprocessor.transformers_:
            names.extend(cols)
        return names
