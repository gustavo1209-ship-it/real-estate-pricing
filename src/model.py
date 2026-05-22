"""
Treino, avaliação e persistência do modelo Random Forest.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    TARGET_COL, RF_PARAMS, TEST_SIZE, RANDOM_STATE, MODEL_PATH,
    PRICE_LABELS
)
from src.features import (
    HouseFeatureEngineer,
    get_feature_columns,
    build_preprocessing_pipeline,
)


def build_full_pipeline(df_sample: pd.DataFrame, mode: str = "classifier") -> Pipeline:
    """
    Constrói o pipeline completo: feature engineering → preprocessing → modelo.

    Args:
        df_sample: DataFrame com as colunas do dataset (para detectar features).
        mode: "classifier" (faixa de preço) ou "regressor" (preço numérico).
    """
    # Aplica feature engineering uma vez para detectar colunas resultantes
    engineer = HouseFeatureEngineer()
    df_engineered = engineer.transform(df_sample)
    num_cols, int_cols = get_feature_columns(df_engineered)

    preprocessor = build_preprocessing_pipeline(num_cols, int_cols)

    if mode == "classifier":
        estimator = RandomForestClassifier(**RF_PARAMS)
    else:
        params = {k: v for k, v in RF_PARAMS.items()}
        estimator = RandomForestRegressor(**params)

    return Pipeline([
        ("engineer", engineer),
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def train(
    df: pd.DataFrame,
    mode: str = "classifier",
    save: bool = True,
) -> dict:
    """
    Treina o modelo e retorna métricas.

    Args:
        df: DataFrame com todas as features + target + price_category.
        mode: "classifier" ou "regressor".
        save: salvar modelo em disco.
    Returns:
        dict com métricas, pipeline treinado e splits.
    """
    if mode == "classifier":
        y = df["price_category"].astype(str)
    else:
        y = df[TARGET_COL]

    # Remove colunas não-feature antes de passar para o pipeline
    drop_cols = [TARGET_COL, "price_category"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=(y if mode == "classifier" else None)
    )

    pipeline = build_full_pipeline(X_train, mode=mode)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if mode == "classifier":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=PRICE_LABELS),
            "labels": PRICE_LABELS,
        }
    else:
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({
            "pipeline": pipeline,
            "mode": mode,
            "feature_cols": X_train.columns.tolist(),
            "metrics": metrics,
        }, MODEL_PATH)
        print(f"[model] Modelo salvo em: {MODEL_PATH}")

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def load_model(path: str | None = None) -> dict:
    """Carrega o modelo salvo em disco."""
    p = path or MODEL_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{p}'. Execute train.py primeiro."
        )
    return joblib.load(p)


def get_feature_importances(pipeline: Pipeline) -> pd.Series:
    """Retorna importância das features do RandomForest."""
    rf = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        names = [f"feature_{i}" for i in range(len(rf.feature_importances_))]

    return (
        pd.Series(rf.feature_importances_, index=names)
        .sort_values(ascending=False)
    )
