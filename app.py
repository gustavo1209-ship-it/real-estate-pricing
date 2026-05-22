"""
Interface Streamlit — Classificação de Preços Imobiliários
Inclui EDA interativo, previsão e análise de importância de features.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import TARGET_COL, PRICE_LABELS, PRICE_BINS, MODEL_PATH
from src.data_loader import load_and_prepare
from src.model import train, load_model, get_feature_importances

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Pricing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Econômico": "#2ecc71",
    "Médio":     "#3498db",
    "Alto":      "#9b59b6",
    "Premium":   "#e67e22",
    "Luxo":      "#e74c3c",
}


@st.cache_data(show_spinner="Carregando dados...")
def get_data(csv_path: str | None = None) -> pd.DataFrame:
    return load_and_prepare(csv_path)


@st.cache_resource(show_spinner="Treinando modelo...")
def get_model(df_hash: int):
    df = get_data()
    return train(df, mode="classifier", save=True)


def fmt_price(v: float) -> str:
    return f"US$ {v:,.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏠 Real Estate Pricing")
    st.caption("Classificação de faixas de preço com Random Forest")

    st.divider()
    csv_file = st.file_uploader(
        "Carregar CSV do Kaggle",
        type=["csv"],
        help="Baixe em kaggle.com/datasets/elakiricoder/jiffs-house-price-prediction-dataset"
    )
    st.caption("Sem arquivo → dados sintéticos para demo")

    st.divider()
    page = st.radio(
        "Navegação",
        ["Visão Geral", "EDA", "Modelo & Métricas", "Previsão Individual"],
        label_visibility="collapsed"
    )

# ── Carrega dados ─────────────────────────────────────────────────────────────
if csv_file is not None:
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(csv_file.read())
    tmp.close()
    df = get_data(tmp.path)
    os.unlink(tmp.path)
else:
    df = get_data()

# ── Treina modelo ─────────────────────────────────────────────────────────────
result = get_model(hash(df.shape))
pipeline = result["pipeline"]
metrics = result["metrics"]

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — Visão Geral
# ══════════════════════════════════════════════════════════════════════════════
if page == "Visão Geral":
    st.header("Visão Geral do Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de imóveis", f"{len(df):,}")
    col2.metric("Preço médio", fmt_price(df[TARGET_COL].mean()))
    col3.metric("Preço mediano", fmt_price(df[TARGET_COL].median()))
    col4.metric("Acurácia do modelo", f"{metrics['accuracy']:.1%}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribuição por faixa de preço")
        counts = df["price_category"].value_counts().reindex(PRICE_LABELS).dropna()
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map=CATEGORY_COLORS,
            labels={"x": "Faixa", "y": "Imóveis"},
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Histograma de preços")
        fig2 = px.histogram(
            df, x=TARGET_COL, nbins=60,
            color_discrete_sequence=["#3498db"],
            labels={TARGET_COL: "Preço (USD)"},
        )
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Amostra do dataset")
    st.dataframe(
        df[[TARGET_COL, "price_category", "bedrooms", "bathrooms",
            "house_size", "grade", "year_built"]].head(50),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.header("Análise Exploratória (EDA)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlação", "Dispersão", "Box Plots", "Estatísticas"
    ])

    num_cols_eda = df.select_dtypes(include=[np.number]).columns.tolist()

    with tab1:
        st.subheader("Mapa de Correlação")
        corr_df = df[num_cols_eda].corr()
        fig = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            height=600,
        )
        fig.update_layout(font=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Dispersão: feature vs. preço")
        feat_options = [c for c in num_cols_eda if c != TARGET_COL]
        selected = st.selectbox("Escolha a feature:", feat_options, index=0)
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig = px.scatter(
            sample, x=selected, y=TARGET_COL,
            color="price_category",
            color_discrete_map=CATEGORY_COLORS,
            opacity=0.5,
            labels={TARGET_COL: "Preço (USD)", selected: selected},
            trendline="ols",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Box Plot por faixa de preço")
        feat_box = st.selectbox(
            "Feature:", feat_options, index=feat_options.index("house_size")
            if "house_size" in feat_options else 0,
            key="box_feat"
        )
        fig = px.box(
            df, x="price_category", y=feat_box,
            color="price_category",
            category_orders={"price_category": PRICE_LABELS},
            color_discrete_map=CATEGORY_COLORS,
            labels={feat_box: feat_box, "price_category": "Faixa"},
            height=420,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Estatísticas descritivas")
        st.dataframe(df[num_cols_eda].describe().T.round(2), use_container_width=True)

        st.subheader("Valores ausentes")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("Nenhum valor ausente encontrado.")
        else:
            st.dataframe(missing.rename("valores nulos").to_frame(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — Modelo & Métricas
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Modelo & Métricas":
    st.header("Modelo Random Forest — Métricas")

    st.metric("Acurácia (teste)", f"{metrics['accuracy']:.4f}")

    tab_cm, tab_report, tab_imp = st.tabs(
        ["Matriz de Confusão", "Relatório", "Importância das Features"]
    )

    with tab_cm:
        cm = metrics["confusion_matrix"]
        labels = [l for l in PRICE_LABELS if l in metrics.get("labels", PRICE_LABELS)]
        fig = px.imshow(
            cm,
            x=labels, y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Previsto", "y": "Real"},
            height=450,
        )
        fig.update_layout(xaxis_title="Previsto", yaxis_title="Real")
        st.plotly_chart(fig, use_container_width=True)

    with tab_report:
        report = metrics["report"]
        rows = []
        for label in PRICE_LABELS:
            if label in report:
                r = report[label]
                rows.append({
                    "Faixa": label,
                    "Precision": round(r["precision"], 3),
                    "Recall": round(r["recall"], 3),
                    "F1-Score": round(r["f1-score"], 3),
                    "Suporte": int(r["support"]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_imp:
        importances = get_feature_importances(pipeline)
        top_n = st.slider("Top N features", 5, min(30, len(importances)), 15)
        top = importances.head(top_n)
        fig = px.bar(
            x=top.values, y=top.index,
            orientation="h",
            labels={"x": "Importância", "y": "Feature"},
            color=top.values,
            color_continuous_scale="Viridis",
            height=max(350, top_n * 22),
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Parâmetros do modelo"):
        rf = pipeline.named_steps["model"]
        params = rf.get_params()
        st.json({k: str(v) for k, v in params.items()})

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — Previsão Individual
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Previsão Individual":
    st.header("Previsão de Faixa de Preço")
    st.caption("Preencha as características do imóvel para obter a classificação de preço.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Tamanho & Estrutura")
        house_size = st.number_input("Área construída (sqft)", 300, 15000, 1800, step=50)
        lot_size = st.number_input("Área do terreno (sqft)", 500, 50000, 5000, step=100)
        sqft_basement = st.number_input("Área do porão (sqft)", 0, 5000, 0, step=50)
        sqft_above = house_size - sqft_basement
        floors = st.select_slider("Andares", [1, 1.5, 2, 2.5, 3], value=1)
        bedrooms = st.slider("Quartos", 1, 10, 3)
        bathrooms = st.select_slider("Banheiros", [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], value=2)

    with col_b:
        st.subheader("Qualidade & Localização")
        grade = st.slider("Nota da construção (1-13)", 1, 13, 7)
        condition = st.slider("Condição (1-5)", 1, 5, 3)
        view = st.slider("Qualidade da vista (0-4)", 0, 4, 0)
        waterfront = st.checkbox("Frente para água")
        year_built = st.number_input("Ano de construção", 1900, 2023, 1990)
        year_renovated = st.number_input("Ano da última reforma (0 = nunca)", 0, 2023, 0)

    with col_c:
        st.subheader("Geolocalização")
        lat = st.number_input("Latitude", 47.0, 48.0, 47.5, format="%.4f")
        long_ = st.number_input("Longitude", -123.0, -121.0, -122.2, format="%.4f")
        zipcode = st.number_input("CEP (ZIP)", 98001, 98199, 98052)

    input_data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_size": house_size,
        "lot_size": lot_size,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "floors": floors,
        "waterfront": int(waterfront),
        "view": view,
        "condition": condition,
        "grade": grade,
        "year_built": year_built,
        "year_renovated": year_renovated,
        "zipcode": zipcode,
        "lat": lat,
        "long": long_,
    }])

    st.divider()

    if st.button("Classificar imóvel", type="primary", use_container_width=True):
        prediction = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0]
        classes = pipeline.classes_

        color = CATEGORY_COLORS.get(prediction, "#3498db")

        st.markdown(
            f"""
            <div style="background:{color}22; border-left:6px solid {color};
                        padding:20px; border-radius:8px; margin:12px 0;">
                <h2 style="color:{color}; margin:0;">Faixa de Preço: {prediction}</h2>
                <p style="margin:6px 0 0 0; color:#555;">
                    Intervalo estimado:
                    <b>{fmt_price(PRICE_BINS[PRICE_LABELS.index(prediction)])}</b>
                    —
                    <b>{fmt_price(PRICE_BINS[PRICE_LABELS.index(prediction)+1])
                        if PRICE_LABELS.index(prediction)+1 < len(PRICE_BINS)
                        else "acima de US$ 1M"}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Probabilidade por faixa")
        proba_df = pd.DataFrame({
            "Faixa": classes,
            "Probabilidade": proba,
        }).sort_values("Probabilidade", ascending=False)

        fig = px.bar(
            proba_df, x="Faixa", y="Probabilidade",
            color="Faixa",
            color_discrete_map=CATEGORY_COLORS,
            text=proba_df["Probabilidade"].apply(lambda v: f"{v:.1%}"),
            height=320,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
