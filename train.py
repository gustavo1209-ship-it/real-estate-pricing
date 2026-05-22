"""
Script de treino: carrega dados, treina o modelo e exibe métricas.

Uso:
    python train.py                    # classifica faixa de preço
    python train.py --mode regressor   # prevê preço numérico
    python train.py --csv data/raw/house_price.csv
"""

import argparse
import sys

import pandas as pd

from src.data_loader import load_and_prepare
from src.model import train, get_feature_importances


def main(csv_path: str | None, mode: str) -> None:
    print(f"\n{'='*55}")
    print(f"  Real Estate Pricing — Treino ({mode})")
    print(f"{'='*55}\n")

    df = load_and_prepare(csv_path)

    print(f"\nDistribuição de faixas de preço:")
    print(df["price_category"].value_counts().to_string())
    print()

    result = train(df, mode=mode, save=True)
    metrics = result["metrics"]

    if mode == "classifier":
        print(f"\nAcurácia no teste : {metrics['accuracy']:.4f}")
        print("\nRelatório de classificação:")
        report_df = pd.DataFrame(metrics["report"]).T
        print(report_df.round(3).to_string())
    else:
        print(f"\nMAE  : R$ {metrics['mae']:,.0f}")
        print(f"RMSE : R$ {metrics['rmse']:,.0f}")
        print(f"R²   : {metrics['r2']:.4f}")

    print("\nTop-10 features mais importantes:")
    importances = get_feature_importances(result["pipeline"])
    print(importances.head(10).to_string())

    print("\nTreino concluído!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o modelo de preços imobiliários")
    parser.add_argument("--csv", default=None, help="Caminho para o CSV bruto")
    parser.add_argument(
        "--mode",
        choices=["classifier", "regressor"],
        default="classifier",
        help="Modo de predição (default: classifier)"
    )
    args = parser.parse_args()
    main(args.csv, args.mode)
