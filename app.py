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

from src.config import (
    TARGET_COL, PRICE_LABELS, PRICE_BINS, MODEL_PATH,
    CITY_COL, REGION_COL, BAIRRO_COL, PROPERTY_TYPE_COL, LOTEAMENTO_COL,
    PROPERTY_TYPES, BAIRROS, CITY_COORDS,
)
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


def _csv_mtime() -> int:
    """Retorna mtime do CSV para invalidar cache quando o arquivo muda."""
    p = os.path.join("data", "raw", "house_price.csv")
    return int(os.path.getmtime(p)) if os.path.exists(p) else 0


@st.cache_data(show_spinner="Carregando dados...")
def get_data(csv_path: str | None = None, mtime: int = 0) -> pd.DataFrame:
    return load_and_prepare(csv_path)


@st.cache_resource(show_spinner="Treinando modelo...")
def get_model(df_hash: int):
    df = get_data()
    return train(df, mode="classifier", save=True)


def fmt_price(v: float) -> str:
    return f"R$ {v:,.0f}"


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
        ["Visão Geral", "Comparativo por Cidade", "Comparar Cidades", "EDA", "Modelo & Métricas", "Previsão Individual"],
        label_visibility="collapsed"
    )

# ── Carrega dados ─────────────────────────────────────────────────────────────
if csv_file is not None:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(csv_file.read())
    tmp.close()
    df = get_data(tmp.name, mtime=0)
    os.unlink(tmp.name)
else:
    df = get_data(mtime=_csv_mtime())

# ── Treina modelo ─────────────────────────────────────────────────────────────
result = get_model(hash(df.shape))
pipeline = result["pipeline"]
metrics = result["metrics"]

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — Visão Geral
# ══════════════════════════════════════════════════════════════════════════════
if page == "Visão Geral":
    st.header("Visão Geral do Dataset")

    # ── Filtros ────────────────────────────────────────────────────────────────
    all_cities_vg = sorted(df[CITY_COL].dropna().unique().tolist()) if CITY_COL in df.columns else []
    all_types_vg  = sorted(df[PROPERTY_TYPE_COL].dropna().unique().tolist()) if PROPERTY_TYPE_COL in df.columns else []

    fv1, fv2 = st.columns(2)
    with fv1:
        cidades_vg = st.multiselect("Filtrar por cidade", all_cities_vg, default=all_cities_vg, key="vg_city")
    with fv2:
        tipos_vg = st.multiselect("Filtrar por tipo", all_types_vg, default=all_types_vg, key="vg_type")

    df_vg = df.copy()
    if cidades_vg:
        df_vg = df_vg[df_vg[CITY_COL].isin(cidades_vg)]
    if tipos_vg:
        df_vg = df_vg[df_vg[PROPERTY_TYPE_COL].isin(tipos_vg)]

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de imóveis", f"{len(df_vg):,}")
    col2.metric("Preço médio",   fmt_price(df_vg[TARGET_COL].mean())   if not df_vg.empty else "—")
    col3.metric("Preço mediano", fmt_price(df_vg[TARGET_COL].median()) if not df_vg.empty else "—")
    col4.metric("Acurácia do modelo", f"{metrics['accuracy']:.1%}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribuição por faixa de preço")
        counts = df_vg["price_category"].value_counts().reindex(PRICE_LABELS).dropna()
        fig = px.bar(
            x=counts.index, y=counts.values,
            color=counts.index,
            color_discrete_map=CATEGORY_COLORS,
            labels={"x": "Faixa", "y": "Imóveis"},
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Histograma de preços")
        fig2 = px.histogram(
            df_vg, x=TARGET_COL, nbins=60,
            color_discrete_sequence=["#3498db"],
            labels={TARGET_COL: "Preço (R$)"},
        )
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    if CITY_COL in df_vg.columns:
        st.subheader("Preço médio por cidade e tipo")
        city_type = df_vg.groupby([CITY_COL, PROPERTY_TYPE_COL])[TARGET_COL].mean().reset_index()
        city_order_vg = df_vg.groupby(CITY_COL)[TARGET_COL].median().sort_values(ascending=False).index.tolist()
        fig3 = px.bar(
            city_type, x=CITY_COL, y=TARGET_COL, color=PROPERTY_TYPE_COL,
            barmode="group",
            category_orders={CITY_COL: city_order_vg},
            labels={TARGET_COL: "Preço Médio (R$)", CITY_COL: "Cidade", PROPERTY_TYPE_COL: "Tipo"},
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Amostra do dataset")
    cols_show = [c for c in [CITY_COL, PROPERTY_TYPE_COL, LOTEAMENTO_COL, TARGET_COL,
                              "price_category", "bedrooms", "house_size", "lot_size", "grade", "year_built"]
                 if c in df_vg.columns]
    st.dataframe(df_vg[cols_show].head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — Comparativo por Cidade
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Comparativo por Cidade":
    st.header("Comparativo por Cidade — Serra Gaúcha vs Porto Alegre")

    has_city = CITY_COL in df.columns and df[CITY_COL].nunique() > 1

    if not has_city:
        st.warning("O dataset atual não possui coluna 'city'. Faça upload do CSV gerado com os dados da Serra Gaúcha.")
        st.stop()

    regions = sorted(df[REGION_COL].dropna().unique()) if REGION_COL in df.columns else []
    selected_regions = st.multiselect("Filtrar por região:", regions, default=regions)
    df_f = df[df[REGION_COL].isin(selected_regions)] if selected_regions else df

    city_order = (
        df_f.groupby(CITY_COL)[TARGET_COL].median()
        .sort_values(ascending=False).index.tolist()
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Preço Médio", "Distribuição", "Terrenos", "Tabela Resumo"
    ])

    with tab1:
        st.subheader("Preço médio por cidade")
        avg = df_f.groupby(CITY_COL)[TARGET_COL].mean().reindex(city_order)
        fig = px.bar(
            x=avg.index, y=avg.values,
            color=avg.values,
            color_continuous_scale="Plasma",
            labels={"x": "Cidade", "y": "Preço Médio (R$)"},
            height=420,
            text=[fmt_price(v) for v in avg.values],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Preço médio por m² construído")
            pm2 = (df_f.groupby(CITY_COL).apply(
                lambda g: (g[TARGET_COL] / g["house_size"]).mean()
            )).reindex(city_order)
            fig2 = px.bar(
                x=pm2.index, y=pm2.values,
                color=pm2.values,
                color_continuous_scale="Viridis",
                labels={"x": "Cidade", "y": "R$/m²"},
                height=360,
                text=[f"R$ {v:,.0f}" for v in pm2.values],
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.subheader("Preço médio por m² de terreno")
            pm2_lot = (df_f.groupby(CITY_COL).apply(
                lambda g: (g[TARGET_COL] / g["lot_size"]).mean()
            )).reindex(city_order)
            fig3 = px.bar(
                x=pm2_lot.index, y=pm2_lot.values,
                color=pm2_lot.values,
                color_continuous_scale="Cividis",
                labels={"x": "Cidade", "y": "R$/m² terreno"},
                height=360,
                text=[f"R$ {v:,.0f}" for v in pm2_lot.values],
            )
            fig3.update_traces(textposition="outside")
            fig3.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("Distribuição de preços por cidade")
        fig = px.box(
            df_f, x=CITY_COL, y=TARGET_COL,
            color=REGION_COL if REGION_COL in df_f.columns else None,
            category_orders={CITY_COL: city_order},
            labels={TARGET_COL: "Preço (R$)", CITY_COL: "Cidade"},
            height=480,
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Faixas de preço por cidade (%)")
        cat_pct = (
            df_f.groupby([CITY_COL, "price_category"])
            .size()
            .unstack(fill_value=0)
            .reindex(city_order)
            .reindex(columns=PRICE_LABELS, fill_value=0)
        )
        cat_pct = cat_pct.div(cat_pct.sum(axis=1), axis=0) * 100
        fig4 = px.bar(
            cat_pct.reset_index().melt(id_vars=CITY_COL, var_name="Faixa", value_name="Percentual"),
            x=CITY_COL, y="Percentual", color="Faixa",
            color_discrete_map=CATEGORY_COLORS,
            category_orders={CITY_COL: city_order, "Faixa": PRICE_LABELS},
            labels={"Percentual": "%", CITY_COL: "Cidade"},
            height=420,
        )
        fig4.update_layout(barmode="stack")
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.subheader("Perfil dos terrenos por cidade")
        c1, c2 = st.columns(2)
        with c1:
            lot_avg = df_f.groupby(CITY_COL)["lot_size"].mean().reindex(city_order)
            fig5 = px.bar(
                x=lot_avg.index, y=lot_avg.values,
                labels={"x": "Cidade", "y": "Área média terreno (m²)"},
                color=lot_avg.values, color_continuous_scale="Blues",
                height=360, text=[f"{v:.0f} m²" for v in lot_avg.values],
            )
            fig5.update_traces(textposition="outside")
            fig5.update_layout(coloraxis_showscale=False, title="Área média do terreno")
            st.plotly_chart(fig5, use_container_width=True)

        with c2:
            house_avg = df_f.groupby(CITY_COL)["house_size"].mean().reindex(city_order)
            fig6 = px.bar(
                x=house_avg.index, y=house_avg.values,
                labels={"x": "Cidade", "y": "Área média construída (m²)"},
                color=house_avg.values, color_continuous_scale="Greens",
                height=360, text=[f"{v:.0f} m²" for v in house_avg.values],
            )
            fig6.update_traces(textposition="outside")
            fig6.update_layout(coloraxis_showscale=False, title="Área média construída")
            st.plotly_chart(fig6, use_container_width=True)

    with tab4:
        st.subheader("Resumo por cidade")
        summary = df_f.groupby(CITY_COL).agg(
            Imóveis=(TARGET_COL, "count"),
            Preço_Médio=(TARGET_COL, "mean"),
            Preço_Mediano=(TARGET_COL, "median"),
            Preço_Mínimo=(TARGET_COL, "min"),
            Preço_Máximo=(TARGET_COL, "max"),
            m2_médio=("house_size", "mean"),
            terreno_médio=("lot_size", "mean"),
            pm2_construído=(TARGET_COL, lambda x: (x / df_f.loc[x.index, "house_size"]).mean()),
        ).reindex(city_order).reset_index()

        for col in ["Preço_Médio", "Preço_Mediano", "Preço_Mínimo", "Preço_Máximo", "pm2_construído"]:
            summary[col] = summary[col].apply(lambda v: f"R$ {v:,.0f}")
        summary["m2_médio"] = summary["m2_médio"].apply(lambda v: f"{v:.0f} m²")
        summary["terreno_médio"] = summary["terreno_médio"].apply(lambda v: f"{v:.0f} m²")

        summary.columns = [
            "Cidade", "Imóveis", "Preço Médio", "Preço Mediano",
            "Mínimo", "Máximo", "Área Média", "Terreno Médio", "R$/m² constr."
        ]
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — Comparar Cidades
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Comparar Cidades":
    st.header("Comparar Cidades")

    has_city = CITY_COL in df.columns and df[CITY_COL].nunique() > 1
    if not has_city:
        st.warning("Dataset sem coluna 'city'. Faça upload do CSV gerado com dados da Serra Gaúcha.")
        st.stop()

    all_cities = sorted(df[CITY_COL].dropna().unique().tolist())

    # ── Filtros globais ────────────────────────────────────────────────────────
    with st.expander("Filtros", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            tipos_sel = st.multiselect(
                "Tipo de imóvel",
                PROPERTY_TYPES,
                default=PROPERTY_TYPES,
            )
        with fc2:
            area_min_df = int(df["house_size"].min())
            area_max_df = int(df["house_size"].max())
            area_range = st.slider(
                "Área construída (m²)",
                area_min_df, area_max_df,
                (area_min_df, area_max_df),
            )
        with fc3:
            lot_min_df = int(df["lot_size"].min())
            lot_max_df = int(df["lot_size"].max())
            lot_range = st.slider(
                "Área do terreno (m²)",
                lot_min_df, lot_max_df,
                (lot_min_df, lot_max_df),
            )

    st.divider()

    # ── Seleção de cidades ─────────────────────────────────────────────────────
    col_a, col_sep, col_b = st.columns([5, 1, 5])

    with col_a:
        st.subheader("Cidade A")
        cidade_a  = st.selectbox("Selecione a cidade A", all_cities, index=all_cities.index("Vila Flores") if "Vila Flores" in all_cities else 0, key="city_a")
        bairros_a = ["Todos"] + sorted(df[df[CITY_COL] == cidade_a][BAIRRO_COL].dropna().unique().tolist())
        bairro_a  = st.selectbox("Bairro", bairros_a, key="bairro_a")
        lots_a_opts = ["Todos"] + sorted(df[df[CITY_COL] == cidade_a][LOTEAMENTO_COL].dropna().unique().tolist())
        lote_a    = st.selectbox("Loteamento / Empreendimento", lots_a_opts, key="lote_a")

    with col_sep:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:3rem'>⇄</div>", unsafe_allow_html=True)

    with col_b:
        st.subheader("Cidade B")
        default_b = all_cities[1] if len(all_cities) > 1 else all_cities[0]
        cidade_b  = st.selectbox("Selecione a cidade B", all_cities, index=all_cities.index(default_b), key="city_b")
        bairros_b = ["Todos"] + sorted(df[df[CITY_COL] == cidade_b][BAIRRO_COL].dropna().unique().tolist())
        bairro_b  = st.selectbox("Bairro", bairros_b, key="bairro_b")
        lots_b_opts = ["Todos"] + sorted(df[df[CITY_COL] == cidade_b][LOTEAMENTO_COL].dropna().unique().tolist())
        lote_b    = st.selectbox("Loteamento / Empreendimento", lots_b_opts, key="lote_b")

    # ── Filtragem ──────────────────────────────────────────────────────────────
    def filtrar(cidade, bairro, lote):
        mask = (
            (df[CITY_COL] == cidade)
            & (df[PROPERTY_TYPE_COL].isin(tipos_sel))
            & (df["house_size"].between(*area_range))
            & (df["lot_size"].between(*lot_range))
        )
        if bairro != "Todos":
            mask &= df[BAIRRO_COL] == bairro
        if lote != "Todos":
            mask &= df[LOTEAMENTO_COL] == lote
        return df[mask].copy()

    dfa = filtrar(cidade_a, bairro_a, lote_a)
    dfb = filtrar(cidade_b, bairro_b, lote_b)

    def montar_label(cidade, bairro, lote):
        parts = [cidade]
        if bairro != "Todos": parts.append(bairro)
        if lote   != "Todos": parts.append(lote)
        return " — ".join(parts)

    label_a = montar_label(cidade_a, bairro_a, lote_a)
    label_b = montar_label(cidade_b, bairro_b, lote_b)

    if dfa.empty or dfb.empty:
        st.warning("Nenhum imóvel encontrado com os filtros selecionados. Ajuste os critérios.")
        st.stop()

    st.divider()

    # ── Métricas lado a lado ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Imóveis — {cidade_a}", f"{len(dfa):,}")
    m2.metric("Preço médio", fmt_price(dfa[TARGET_COL].mean()),
              delta=f"{((dfa[TARGET_COL].mean() / dfb[TARGET_COL].mean()) - 1):+.1%} vs {cidade_b}")
    m3.metric(f"Imóveis — {cidade_b}", f"{len(dfb):,}")
    m4.metric("Preço médio", fmt_price(dfb[TARGET_COL].mean()))

    st.divider()

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["🏡 Florescer vs Mercado", "Preços", "Por Tipo", "Metragem", "Tabela"])

    with tab0:
        st.subheader("Florescer Parque Residencial — Posicionamento Competitivo")
        st.caption("Comparativo de terrenos por R$/m² entre loteamentos das 3 cidades da região.")

        cidades_alvo = ["Vila Flores", "Nova Prata", "Veranópolis"]
        df_terra = df[
            (df[CITY_COL].isin(cidades_alvo)) &
            (df[PROPERTY_TYPE_COL] == "Terreno") &
            (df[LOTEAMENTO_COL] != "—")
        ].copy()
        df_terra["pm2"] = df_terra[TARGET_COL] / df_terra["lot_size"].replace(0, np.nan)

        # R$/m² por loteamento
        pm2_lot = df_terra.groupby([CITY_COL, LOTEAMENTO_COL])["pm2"].median().reset_index()
        pm2_lot.columns = ["Cidade", "Loteamento", "R$/m²"]
        pm2_lot = pm2_lot.sort_values("R$/m²")
        pm2_lot["cor"] = pm2_lot["Loteamento"].apply(
            lambda x: "#e74c3c" if x == "Florescer Parque Residencial" else "#95a5a6"
        )

        fig = px.bar(
            pm2_lot, x="R$/m²", y="Loteamento", orientation="h",
            color="Cidade",
            text=pm2_lot["R$/m²"].apply(lambda v: f"R$ {v:,.0f}"),
            height=max(400, len(pm2_lot) * 32),
            labels={"R$/m²": "Preço mediano por m² (R$)", "Loteamento": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Tabela comparativa detalhada
        st.subheader("Tabela: R$/m², área e preço médio por loteamento")
        tab_comp = df_terra.groupby([CITY_COL, LOTEAMENTO_COL]).agg(
            Lotes=("price", "count"),
            Área_Média=("lot_size", "mean"),
            Preço_Médio=("price", "mean"),
            PM2_Mín=("pm2", "min"),
            PM2_Mediano=("pm2", "median"),
            PM2_Máx=("pm2", "max"),
        ).reset_index().sort_values(["Cidade", "PM2_Mediano"])
        tab_comp.columns = ["Cidade", "Loteamento", "Lotes", "Área Média (m²)", "Preço Médio", "R$/m² Mín", "R$/m² Mediano", "R$/m² Máx"]
        tab_comp["Preço Médio"]   = tab_comp["Preço Médio"].apply(fmt_price)
        tab_comp["Área Média (m²)"] = tab_comp["Área Média (m²)"].apply(lambda v: f"{v:.0f} m²")
        for c in ["R$/m² Mín", "R$/m² Mediano", "R$/m² Máx"]:
            tab_comp[c] = tab_comp[c].apply(lambda v: f"R$ {v:,.0f}")

        def highlight_florescer(row):
            if row["Loteamento"] == "Florescer Parque Residencial":
                return ["background-color: #fdecea; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            tab_comp.style.apply(highlight_florescer, axis=1),
            use_container_width=True, hide_index=True
        )

        # Insight rápido
        florescer_pm2 = df_terra[df_terra[LOTEAMENTO_COL] == "Florescer Parque Residencial"]["pm2"].median()
        mercado_pm2   = df_terra[df_terra[LOTEAMENTO_COL] != "Florescer Parque Residencial"]["pm2"].median()
        if not np.isnan(florescer_pm2) and not np.isnan(mercado_pm2):
            delta_pct = (florescer_pm2 / mercado_pm2 - 1) * 100
            sinal = "acima" if delta_pct > 0 else "abaixo"
            cor   = "#e74c3c" if delta_pct > 15 else "#2ecc71"
            st.markdown(
                f"""<div style="border-left:5px solid {cor}; padding:12px 16px; border-radius:6px; background:{cor}18; margin-top:12px">
                <b>Diagnóstico:</b> O Florescer está precificado em <b>R$ {florescer_pm2:,.0f}/m²</b>,
                <b>{abs(delta_pct):.1f}% {sinal}</b> da mediana dos concorrentes da região
                (R$ {mercado_pm2:,.0f}/m²).
                </div>""",
                unsafe_allow_html=True
            )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Distribuição — {label_a}")
            fig = px.histogram(dfa, x=TARGET_COL, nbins=40,
                               color_discrete_sequence=["#3498db"],
                               labels={TARGET_COL: "Preço (R$)"}, height=320)
            fig.add_vline(x=dfa[TARGET_COL].median(), line_dash="dash", line_color="red",
                          annotation_text="mediana")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader(f"Distribuição — {label_b}")
            fig = px.histogram(dfb, x=TARGET_COL, nbins=40,
                               color_discrete_sequence=["#e67e22"],
                               labels={TARGET_COL: "Preço (R$)"}, height=320)
            fig.add_vline(x=dfb[TARGET_COL].median(), line_dash="dash", line_color="red",
                          annotation_text="mediana")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Box Plot comparativo")
        combined = pd.concat([
            dfa.assign(Cidade=label_a),
            dfb.assign(Cidade=label_b),
        ])
        fig = px.box(combined, x="Cidade", y=TARGET_COL, color="Cidade",
                     color_discrete_sequence=["#3498db", "#e67e22"],
                     labels={TARGET_COL: "Preço (R$)"}, height=380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Preço médio por tipo de imóvel")
        avg_a = dfa.groupby(PROPERTY_TYPE_COL)[TARGET_COL].mean().reset_index()
        avg_b = dfb.groupby(PROPERTY_TYPE_COL)[TARGET_COL].mean().reset_index()
        avg_a["Cidade"] = label_a
        avg_b["Cidade"] = label_b
        combined_type = pd.concat([avg_a, avg_b])
        fig = px.bar(combined_type, x=PROPERTY_TYPE_COL, y=TARGET_COL, color="Cidade",
                     barmode="group",
                     color_discrete_sequence=["#3498db", "#e67e22"],
                     labels={TARGET_COL: "Preço Médio (R$)", PROPERTY_TYPE_COL: "Tipo"},
                     height=380,
                     text=combined_type[TARGET_COL].apply(fmt_price))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("R$/m² por tipo de imóvel")
        pm2_a = dfa[dfa["house_size"] > 0].groupby(PROPERTY_TYPE_COL).apply(
            lambda g: (g[TARGET_COL] / g["house_size"]).mean()).reset_index(name="pm2")
        pm2_b = dfb[dfb["house_size"] > 0].groupby(PROPERTY_TYPE_COL).apply(
            lambda g: (g[TARGET_COL] / g["house_size"]).mean()).reset_index(name="pm2")
        pm2_a["Cidade"] = label_a
        pm2_b["Cidade"] = label_b
        combined_pm2 = pd.concat([pm2_a, pm2_b])
        fig2 = px.bar(combined_pm2, x=PROPERTY_TYPE_COL, y="pm2", color="Cidade",
                      barmode="group",
                      color_discrete_sequence=["#3498db", "#e67e22"],
                      labels={"pm2": "R$/m²", PROPERTY_TYPE_COL: "Tipo"},
                      height=360,
                      text=combined_pm2["pm2"].apply(lambda v: f"R$ {v:,.0f}"))
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Área construída vs Preço")
        _sc_parts = pd.concat([
            dfa[dfa["house_size"] > 0].assign(Cidade=label_a),
            dfb[dfb["house_size"] > 0].assign(Cidade=label_b),
        ])
        if _sc_parts.empty:
            st.info("Nenhum imóvel com área construída nos filtros selecionados (apenas terrenos).")
        else:
            combined_sc = _sc_parts.sample(min(1500, len(_sc_parts)), random_state=42)
            fig = px.scatter(combined_sc, x="house_size", y=TARGET_COL, color="Cidade",
                             color_discrete_sequence=["#3498db", "#e67e22"],
                             opacity=0.55,
                             labels={"house_size": "Área construída (m²)", TARGET_COL: "Preço (R$)"},
                             trendline="ols", height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Área do terreno vs Preço")
        _lot_parts = pd.concat([
            dfa.assign(Cidade=label_a),
            dfb.assign(Cidade=label_b),
        ])
        if _lot_parts.empty:
            st.info("Sem dados para exibir.")
        else:
            combined_lot = _lot_parts.sample(min(1500, len(_lot_parts)), random_state=42)
            fig2 = px.scatter(combined_lot, x="lot_size", y=TARGET_COL, color="Cidade",
                              color_discrete_sequence=["#3498db", "#e67e22"],
                              opacity=0.55,
                              labels={"lot_size": "Área do terreno (m²)", TARGET_COL: "Preço (R$)"},
                              trendline="ols", height=400)
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        def resumo(d, label):
            _com_area = d[d["house_size"] > 0]
            _pm2_val = (_com_area[TARGET_COL] / _com_area["house_size"]).mean()
            _pm2_str = f"R$ {_pm2_val:,.0f}" if not pd.isna(_pm2_val) else "—"
            return pd.DataFrame([{
                "Local": label,
                "Imóveis": len(d),
                "Preço Médio": fmt_price(d[TARGET_COL].mean()),
                "Preço Mediano": fmt_price(d[TARGET_COL].median()),
                "Mínimo": fmt_price(d[TARGET_COL].min()),
                "Máximo": fmt_price(d[TARGET_COL].max()),
                "Área média (m²)": f"{d['house_size'].mean():.0f}",
                "Terreno médio (m²)": f"{d['lot_size'].mean():.0f}",
                "R$/m² médio": _pm2_str,
            }])
        summary_tab = pd.concat([resumo(dfa, label_a), resumo(dfb, label_b)], ignore_index=True)
        st.dataframe(summary_tab, use_container_width=True, hide_index=True)

        st.subheader("Por bairro")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.caption(label_a)
            b_a = dfa.groupby(BAIRRO_COL)[TARGET_COL].agg(["count","mean","median"]).reset_index()
            b_a.columns = ["Bairro", "Imóveis", "Preço Médio", "Mediana"]
            b_a["Preço Médio"] = b_a["Preço Médio"].apply(fmt_price)
            b_a["Mediana"] = b_a["Mediana"].apply(fmt_price)
            st.dataframe(b_a.sort_values("Imóveis", ascending=False), use_container_width=True, hide_index=True)
        with bc2:
            st.caption(label_b)
            b_b = dfb.groupby(BAIRRO_COL)[TARGET_COL].agg(["count","mean","median"]).reset_index()
            b_b.columns = ["Bairro", "Imóveis", "Preço Médio", "Mediana"]
            b_b["Preço Médio"] = b_b["Preço Médio"].apply(fmt_price)
            b_b["Mediana"] = b_b["Mediana"].apply(fmt_price)
            st.dataframe(b_b.sort_values("Imóveis", ascending=False), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.header("Análise Exploratória (EDA)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlação", "Dispersão", "Box Plots", "Estatísticas"
    ])

    num_cols_eda = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_options = [c for c in num_cols_eda if c != TARGET_COL]

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
# PÁGINA 4 — Modelo & Métricas
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
# PÁGINA 5 — Previsão Individual
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Previsão Individual":
    st.header("Previsão de Faixa de Preço")
    st.caption("Preencha as características do imóvel para obter a classificação de preço.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Tamanho & Estrutura")
        prop_type_sel = st.selectbox("Tipo de imóvel", PROPERTY_TYPES)
        house_size = st.number_input("Área construída (m²)", 0, 2000, 120, step=10)
        lot_size   = st.number_input("Área do terreno (m²)", 10, 10000, 300, step=10)
        sqft_basement = st.number_input("Área do porão (m²)", 0, 500, 0, step=5)
        sqft_above = max(0, house_size - sqft_basement)
        floors   = st.select_slider("Andares", [1, 1.5, 2, 2.5, 3], value=1)
        bedrooms = st.slider("Quartos", 0, 8, 3)
        bathrooms = st.select_slider("Banheiros", [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], value=2)

    with col_b:
        st.subheader("Qualidade")
        grade     = st.slider("Padrão da construção (1–13)", 1, 13, 7)
        condition = st.slider("Condição (1–5)", 1, 5, 3)
        view      = st.slider("Qualidade da vista (0–4)", 0, 4, 0)
        waterfront    = st.checkbox("Frente para água / lago")
        year_built    = st.number_input("Ano de construção", 1900, 2025, 2000)
        year_renovated = st.number_input("Ano da última reforma (0 = nunca)", 0, 2025, 0)

    with col_c:
        st.subheader("Localização")
        all_cities_pred = sorted(CITY_COORDS.keys())
        city_sel  = st.selectbox("Cidade", all_cities_pred)
        bairros_pred = BAIRROS.get(city_sel, ["Centro"])
        bairro_sel = st.selectbox("Bairro", bairros_pred)
        lat_base, long_base = CITY_COORDS[city_sel]
        zipcode = int(df[df[CITY_COL] == city_sel]["zipcode"].median()) if city_sel in df[CITY_COL].values else 95000

    input_data = pd.DataFrame([{
        "bedrooms":       bedrooms,
        "bathrooms":      bathrooms,
        "house_size":     house_size,
        "lot_size":       lot_size,
        "sqft_above":     sqft_above,
        "sqft_basement":  sqft_basement,
        "floors":         floors,
        "waterfront":     int(waterfront),
        "view":           view,
        "condition":      condition,
        "grade":          grade,
        "year_built":     year_built,
        "year_renovated": year_renovated,
        "zipcode":        zipcode,
        "lat":            lat_base,
        "long":           long_base,
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
                        else "acima de R$ 4M"}</b>
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
