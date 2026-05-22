# 🏠 Real Estate Pricing — ML com Random Forest

Classificação de faixas de preço de imóveis usando **scikit-learn**, **pandas** e **Streamlit**.

Dataset: [Jiff's House Price Prediction Dataset](https://www.kaggle.com/datasets/elakiricoder/jiffs-house-price-prediction-dataset) (Kaggle)

---

## Estrutura do projeto

```
real-estate-pricing/
├── data/
│   ├── raw/            ← coloque house_price.csv aqui (baixar do Kaggle)
│   └── processed/      ← dados processados (gerado automaticamente)
├── models/             ← modelo treinado (.joblib)
├── notebooks/
│   └── eda.ipynb       ← análise exploratória completa
├── src/
│   ├── config.py       ← configurações e mapeamento de colunas
│   ├── data_loader.py  ← carregamento e limpeza
│   ├── features.py     ← feature engineering + pipeline sklearn
│   └── model.py        ← treino, avaliação e persistência
├── app.py              ← interface Streamlit
├── train.py            ← script de treino via CLI
└── requirements.txt
```

---

## Instalação

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset

1. Acesse: https://www.kaggle.com/datasets/elakiricoder/jiffs-house-price-prediction-dataset
2. Baixe o CSV e salve em `data/raw/house_price.csv`

> Sem o CSV, o projeto roda com **dados sintéticos** para demonstração.

### Colunas esperadas

| Coluna | Tipo | Descrição |
|---|---|---|
| `price` | float | Preço de venda (target) |
| `house_size` | int | Área construída (sqft) |
| `lot_size` | int | Área do terreno (sqft) |
| `bedrooms` | int | Número de quartos |
| `bathrooms` | float | Número de banheiros |
| `floors` | float | Andares |
| `waterfront` | int | Frente para água (0/1) |
| `view` | int | Qualidade da vista (0–4) |
| `condition` | int | Condição (1–5) |
| `grade` | int | Nota da construção (1–13) |
| `sqft_above` | int | Área acima do solo (sqft) |
| `sqft_basement` | int | Área do porão (sqft) |
| `year_built` | int | Ano de construção |
| `year_renovated` | int | Ano da reforma (0 = nunca) |
| `zipcode` | int | Código postal |
| `lat` | float | Latitude |
| `long` | float | Longitude |

> Se o CSV do Kaggle usar outros nomes, atualize `src/config.py`.

---

## Uso

### Treinar o modelo

```bash
# Classificação de faixa de preço (padrão)
python train.py

# Regressão (preço numérico)
python train.py --mode regressor

# CSV personalizado
python train.py --csv data/raw/house_price.csv
```

### Iniciar a interface visual

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

### Notebook de EDA

```bash
cd notebooks
jupyter notebook eda.ipynb
```

---

## Faixas de preço

| Categoria | Intervalo |
|---|---|
| Econômico | até US$ 250k |
| Médio | US$ 250k – 500k |
| Alto | US$ 500k – 750k |
| Premium | US$ 750k – 1M |
| Luxo | acima de US$ 1M |

---

## Pipeline de features

```
raw data
  └─ HouseFeatureEngineer
       ├─ house_age, was_renovated, years_since_renovation
       ├─ bath_per_bed, has_basement, house_lot_ratio
       └─ quality_score (grade × condition)
  └─ ColumnTransformer
       ├─ numéricas  → Imputer(median) + RobustScaler
       └─ inteiras   → Imputer(mode)
  └─ RandomForestClassifier(n_estimators=200)
```

---

## Tecnologias

- Python 3.11+
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, RandomForest)
- Streamlit + Plotly (interface)
- Matplotlib + Seaborn (notebook)
