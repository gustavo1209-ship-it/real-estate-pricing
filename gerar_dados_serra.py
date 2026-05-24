"""
Dataset de imóveis da Serra Gaúcha e Porto Alegre.
Terrenos de Vila Flores, Nova Prata e Veranópolis usam preços reais
coletados dos portais Nestoria, NegocieImóveis, Kenlo, Vênetto e MGF (Mai/2026).
"""
import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)

# ── Preços reais de terrenos por loteamento ────────────────────────────────────
# Formato: (loteamento, bairro, tamanho_min, tamanho_max, pm2_min, pm2_max, peso)
TERRENOS_REAIS = {
    "Vila Flores": [
        ("Florescer Parque Residencial", "Centro",        900, 1400, 240, 280, 0.35),
        ("Clube Residencial Paraíso",    "Centro",        900, 1200, 199, 242, 0.25),
        ("Residencial Horizontes",       "Bairro Alto",   360,  600, 350, 450, 0.20),
        ("Linha Aimoré",                 "Linha Aimoré",  750,  950, 185, 225, 0.12),
        ("Urbano avulso",                "Centro",        300,  420, 400, 460, 0.08),
    ],
    "Nova Prata": [
        ("Lot. Frizon / Chiomento",      "Frizon",        360,  380, 319, 389, 0.18),
        ("Lot. Colinas do Caravaggio",   "São Cristóvão", 360,  420, 368, 417, 0.16),
        ("Lot. Clivatti I",              "Clivatti",      360,  390, 494, 597, 0.13),
        ("Lot. Gênova",                  "Centro",        340,  380, 611, 694, 0.10),
        ("Sagrada Família / Centro",     "Sagrada Família",330, 370, 845,1056, 0.09),
        ("Lot. Schneider",               "Santa Cruz",    340,  380, 625, 736, 0.08),
        ("Condomínio fechado interior",  "Interior",     1400, 2200, 116, 257, 0.15),
        ("Roteiro das Águas",            "Interior",     2000, 2500, 116, 167, 0.11),
    ],
    "Veranópolis": [
        ("Centro (avulso)",              "Centro",        340,  380, 328, 407, 0.14),
        ("Solar Norte",                  "São Pelegrino", 390,  450, 407, 435, 0.13),
        ("Femaçã médio",                 "Femaçã",        360,  500, 429, 780, 0.16),
        ("Cond. Jardim Europa",          "Femaçã",        500, 1491, 194, 580, 0.18),
        ("Morada do Bosque",             "Femaçã",        380,  510, 806, 962, 0.16),
        ("Renovação",                    "Renovação",     450,  510, 789, 882, 0.12),
        ("Medianeira",                   "Medianeira",    580,  700, 355, 435, 0.11),
    ],
}

BAIRROS = {
    "Gramado":         ["Centro", "Várzea Grande", "Mato Queimado", "Três Pinheiros", "Morada da Serra", "Canavial"],
    "Canela":          ["Centro", "Fontes do Ipê", "Jardim", "Reserva da Serra", "São José", "Industrial"],
    "Caxias do Sul":   ["Centro", "Sanvitto", "São Pelegrino", "Exposição", "Santa Catarina", "Marechal Floriano", "Sagrada Família", "Desvio Rizzo"],
    "Bento Gonçalves": ["Centro", "São Bento", "Planalto", "Charqueadas", "Fátima", "São Roque"],
    "Farroupilha":     ["Centro", "São Caetano", "Cinquentenário", "Pio X", "Desvio Rizzo"],
    "Garibaldi":       ["Centro", "Santa Marta", "Progresso", "São João", "Bela Vista"],
    "Nova Petrópolis": ["Centro", "Linha Imperial", "Vila Nova", "Recanto Suíço", "Pinhal Alto"],
    "Nova Prata":      ["Frizon", "São Cristóvão", "Clivatti", "Centro", "Sagrada Família", "Santa Cruz", "Interior"],
    "Veranópolis":     ["Centro", "São Pelegrino", "Femaçã", "Renovação", "Medianeira", "Interior"],
    "Vila Flores":     ["Centro", "Bairro Alto", "Linha Aimoré"],
    "Porto Alegre":    ["Moinhos de Vento", "Bela Vista", "Mont'Serrat", "Três Figueiras",
                        "Menino Deus", "Petrópolis", "Cidade Baixa", "Boa Vista",
                        "Chácara das Pedras", "Cristal", "Partenon", "Centro Histórico"],
}

TYPE_DIST = {
    "Gramado":         [0.32, 0.18, 0.28, 0.14, 0.08],
    "Canela":          [0.30, 0.22, 0.25, 0.15, 0.08],
    "Caxias do Sul":   [0.28, 0.38, 0.14, 0.12, 0.08],
    "Bento Gonçalves": [0.32, 0.35, 0.14, 0.12, 0.07],
    "Farroupilha":     [0.38, 0.30, 0.12, 0.12, 0.08],
    "Garibaldi":       [0.42, 0.28, 0.12, 0.10, 0.08],
    "Nova Petrópolis": [0.45, 0.22, 0.12, 0.12, 0.09],
    "Nova Prata":      [0.35, 0.20, 0.10, 0.27, 0.08],  # mais terrenos (27%)
    "Veranópolis":     [0.33, 0.18, 0.08, 0.33, 0.08],  # mais terrenos (33%)
    "Vila Flores":     [0.28, 0.14, 0.06, 0.45, 0.07],  # foco em terrenos (45%)
    "Porto Alegre":    [0.22, 0.45, 0.10, 0.15, 0.08],
}
TYPES = ["Casa", "Apartamento", "Condomínio Residencial", "Terreno", "Comercial"]

CITIES = {
    "Gramado":         {"lat": -29.3789, "lon": -50.8761, "price_m2": 19946, "zip": 95670, "n": 300, "region": "Serra Gaúcha"},
    "Canela":          {"lat": -29.3601, "lon": -50.8129, "price_m2": 14120, "zip": 95680, "n": 280, "region": "Serra Gaúcha"},
    "Caxias do Sul":   {"lat": -29.1678, "lon": -51.1794, "price_m2":  8000, "zip": 95010, "n": 500, "region": "Serra Gaúcha"},
    "Bento Gonçalves": {"lat": -29.1707, "lon": -51.5185, "price_m2":  6400, "zip": 95700, "n": 300, "region": "Serra Gaúcha"},
    "Farroupilha":     {"lat": -29.2240, "lon": -51.3468, "price_m2":  5500, "zip": 95180, "n": 200, "region": "Serra Gaúcha"},
    "Garibaldi":       {"lat": -29.2564, "lon": -51.5328, "price_m2":  4800, "zip": 95720, "n": 180, "region": "Serra Gaúcha"},
    "Nova Petrópolis": {"lat": -29.3743, "lon": -51.1135, "price_m2":  4500, "zip": 95150, "n": 180, "region": "Serra Gaúcha"},
    "Nova Prata":      {"lat": -28.7839, "lon": -51.6134, "price_m2":  3500, "zip": 95320, "n": 300, "region": "Serra Gaúcha"},
    "Veranópolis":     {"lat": -28.9378, "lon": -51.5581, "price_m2":  3000, "zip": 95330, "n": 300, "region": "Serra Gaúcha"},
    "Vila Flores":     {"lat": -28.8617, "lon": -51.6017, "price_m2":  2500, "zip": 95335, "n": 300, "region": "Serra Gaúcha"},
    "Porto Alegre":    {"lat": -30.0346, "lon": -51.2177, "price_m2":  7505, "zip": 90010, "n": 600, "region": "Capital"},
}


def gerar_terreno_real(city, cfg, n_terrenos):
    """Gera terrenos usando preços reais coletados dos portais."""
    config_lots = TERRENOS_REAIS.get(city)
    if not config_lots:
        # Cidades sem dados reais: usa 22% do m² construído
        lot_pm2 = cfg["price_m2"] * 0.22
        lots, bairros_arr, loteamentos = [], [], []
        for _ in range(n_terrenos):
            sz  = int(rng.integers(300, 1200))
            pm2 = lot_pm2 * rng.uniform(0.85, 1.20)
            lots.append((sz, int(sz * pm2), rng.choice(BAIRROS[city]), "—"))
        return lots

    pesos = [c[6] for c in config_lots]
    pesos = [p / sum(pesos) for p in pesos]
    escolhas = rng.choice(len(config_lots), n_terrenos, p=pesos)
    results = []
    for idx in escolhas:
        nome, bairro, sz_min, sz_max, pm2_min, pm2_max, _ = config_lots[idx]
        sz  = int(rng.integers(sz_min, sz_max + 1))
        pm2 = rng.uniform(pm2_min, pm2_max)
        price = int(sz * pm2 * rng.uniform(0.95, 1.05))
        results.append((sz, price, bairro, nome))
    return results


def gerar_imovel(pt, pm2, city):
    bairros = BAIRROS[city]
    if pt == "Apartamento":
        hs   = int(rng.integers(45, 180))
        lt   = hs
        sb   = 0
        sa   = hs
        bed  = int(rng.integers(1, 4))
        bath = float(int(rng.integers(1, 3)))
        fl   = float(rng.choice([1,2,3,4,5,6,7,8,10,12], p=[0.1,0.1,0.1,0.15,0.15,0.1,0.1,0.07,0.07,0.06]))
        gr   = int(np.clip(rng.integers(6, 11), 5, 13))
        cond = int(rng.integers(2, 6))
        view = int(rng.integers(0, 5))
        wf   = int(rng.random() < 0.02)
        base = hs * pm2 * rng.uniform(0.90, 1.10)
        bairro, lot = rng.choice(bairros), "—"
    elif pt == "Condomínio Residencial":
        hs   = int(rng.integers(120, 350))
        lt   = hs + int(rng.integers(100, 600))
        sb   = int(rng.integers(0, 60)) if rng.random() < 0.2 else 0
        sa   = hs - sb
        bed  = int(rng.integers(2, 5))
        bath = float(int(rng.integers(2, 4))) + rng.choice([0.0, 0.5], p=[0.5, 0.5])
        fl   = float(rng.choice([1, 2, 3], p=[0.4, 0.45, 0.15]))
        gr   = int(np.clip(rng.integers(8, 13), 5, 13))
        cond = int(rng.integers(3, 6))
        view = int(rng.integers(0, 5))
        wf   = int(rng.random() < 0.05)
        base = hs * pm2 * rng.uniform(1.15, 1.40)
        bairro, lot = rng.choice(bairros), "—"
    elif pt == "Comercial":
        hs   = int(rng.integers(30, 400))
        lt   = hs + int(rng.integers(0, 200))
        sb   = 0; sa = hs
        bed  = 0
        bath = float(int(rng.integers(1, 3)))
        fl   = float(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        gr   = int(np.clip(rng.integers(5, 10), 4, 13))
        cond = int(rng.integers(2, 6))
        view = int(rng.integers(0, 3))
        wf   = 0
        base = hs * pm2 * 0.85 * rng.uniform(0.85, 1.15)
        bairro, lot = rng.choice(bairros), "—"
    else:  # Casa
        hs   = int(rng.integers(70, 300))
        lt   = hs + int(rng.integers(80, 600))
        sb   = int(rng.integers(20, 80)) if rng.random() < 0.18 else 0
        sa   = hs - sb
        bed  = int(rng.integers(2, 5))
        bath = float(int(rng.integers(1, 4))) + rng.choice([0.0, 0.5], p=[0.6, 0.4])
        fl   = float(rng.choice([1, 2, 3], p=[0.5, 0.40, 0.10]))
        gr   = int(np.clip(rng.integers(5, 11), 4, 13))
        cond = int(rng.integers(2, 6))
        view = int(rng.integers(0, 5))
        wf   = int(rng.random() < 0.04)
        base = hs * pm2 * rng.uniform(0.92, 1.12)
        bairro, lot = rng.choice(bairros), "—"

    yr_built = int(rng.integers(1970, 2024))
    has_reno = rng.random() < 0.28
    yr_reno  = int(rng.integers(2000, 2024)) if has_reno else 0
    price    = int(np.clip(base, pm2 * 20, pm2 * 500))
    return hs, lt, sb, sa, bed, bath, fl, wf, view, cond, gr, yr_built, yr_reno, price, bairro, lot


dfs = []
for city, cfg in CITIES.items():
    n      = cfg["n"]
    pm2    = cfg["price_m2"]
    type_p = TYPE_DIST[city]

    prop_types = rng.choice(TYPES, n, p=type_p)
    n_terrenos = int((prop_types == "Terreno").sum())
    terrenos = gerar_terreno_real(city, cfg, n_terrenos)
    t_idx = 0

    rows = []
    for pt in prop_types:
        if pt == "Terreno":
            sz, price, bairro, lot = terrenos[t_idx]; t_idx += 1
            lt   = sz + int(rng.integers(0, 50))
            yr_built = int(rng.integers(1990, 2024))
            row = dict(
                city=city, region=cfg["region"], bairro=bairro,
                property_type=pt, loteamento=lot, price=price,
                bedrooms=0, bathrooms=0.0, house_size=0, lot_size=sz,
                sqft_above=0, sqft_basement=0, floors=0.0,
                waterfront=0, view=int(rng.integers(0, 4)), condition=int(rng.integers(1, 4)),
                grade=int(rng.integers(3, 7)), year_built=yr_built, year_renovated=0,
                zipcode=int(rng.integers(cfg["zip"], cfg["zip"]+10)),
                lat=float(rng.uniform(cfg["lat"]-0.05, cfg["lat"]+0.05)),
                long=float(rng.uniform(cfg["lon"]-0.05, cfg["lon"]+0.05)),
            )
        else:
            hs,lt,sb,sa,bed,bath,fl,wf,view,cond,gr,yb,yr,price,bairro,lot = gerar_imovel(pt, pm2, city)
            row = dict(
                city=city, region=cfg["region"], bairro=bairro,
                property_type=pt, loteamento=lot, price=price,
                bedrooms=bed, bathrooms=bath, house_size=hs, lot_size=lt,
                sqft_above=sa, sqft_basement=sb, floors=fl,
                waterfront=wf, view=view, condition=cond,
                grade=gr, year_built=yb, year_renovated=yr,
                zipcode=int(rng.integers(cfg["zip"], cfg["zip"]+10)),
                lat=float(rng.uniform(cfg["lat"]-0.05, cfg["lat"]+0.05)),
                long=float(rng.uniform(cfg["lon"]-0.05, cfg["lon"]+0.05)),
            )
        rows.append(row)
    dfs.append(pd.DataFrame(rows))

df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# ── Dados REAIS dos 127 lotes do Florescer Parque Residencial ─────────────────
# Fonte: Tabela de Vendas R.22/02/26
# Preços sem tabela oficial estimados a partir de lotes vizinhos comparáveis (*)
FLORESCER_REAL_LOTS = [
    # (quadra, lote, area_m2, categoria, preco)
    # Quadra A — Misto II
    ("A",  1,  653.2,  "Misto II", 460_000),
    ("A",  2,  796.8,  "Misto II", 540_000),
    ("A",  3, 1229.8,  "Misto II", 650_000),
    ("A",  4,  932.0,  "Misto II", 530_000),
    ("A",  5,  757.6,  "Misto II", 440_000),
    ("A",  6,  761.3,  "Misto II", 415_000),
    ("A",  7,  714.2,  "Misto II", 365_000),
    ("A",  8,  686.7,  "Misto II", 430_000),
    ("A",  9,  800.0,  "Misto II", 420_000),
    ("A", 10,  800.0,  "Misto II", 410_000),
    ("A", 11,  800.0,  "Misto II", 420_000),
    # Quadra B
    ("B",  1, 1480.9,  "Misto II", 660_000),
    ("B",  2, 1071.6,  "Misto II", 550_000),
    ("B",  3, 1432.3,  "Misto II", 640_000),
    ("B",  4,  507.13, "Misto I",  279_000),
    ("B",  5,  418.66, "Misto I",  268_000),
    ("B",  6,  479.6,  "Misto I",  294_000),
    # Quadra C
    ("C",  1,  596.4,  "Misto II", 380_000),
    ("C",  2,  767.1,  "Misto II", 420_000),
    ("C",  3, 1220.0,  "Misto II", 590_000),
    ("C",  4,  711.1,  "Misto II", 280_000),
    ("C",  5, 1079.4,  "Misto II", 560_000),
    ("C",  6, 1276.1,  "Misto II", 660_000),
    ("C",  7,  539.08, "Misto I",  268_000),
    ("C",  8,  464.44, "Misto I",  246_000),
    ("C",  9,  499.75, "Misto I",  262_000),   # *vendido
    # Quadra D
    ("D",  1,  883.28, "Misto I",  520_000),
    ("D",  2, 1095.29, "Misto I",  595_000),
    ("D",  3,  650.87, "Misto I",  378_000),
    ("D",  4,  480.0,  "Misto I",  295_000),
    ("D",  5,  420.0,  "Misto I",  270_000),
    ("D",  6,  420.0,  "Misto I",  285_000),
    ("D",  7,  474.71, "Misto I",  320_000),
    ("D",  8,  690.0,  "Misto I",  440_000),
    # Quadra E
    ("E",  1,  681.64, "Misto I",  430_000),
    ("E",  2,  690.0,  "Misto I",  440_000),
    ("E",  3,  474.57, "Misto I",  320_000),   # *vendido
    ("E",  4,  420.0,  "Misto I",  278_000),   # *vendido
    ("E",  5,  420.0,  "Misto I",  270_000),
    ("E",  6,  480.0,  "Misto I",  298_000),
    ("E",  7,  877.16, "Misto I",  495_000),
    # Quadra F
    ("F",  1,  497.6,  "Residencial", 260_000),
    ("F",  2,  463.1,  "Residencial", 240_000),
    ("F",  3,  461.7,  "Residencial", 237_000),
    ("F",  4,  460.4,  "Residencial", 237_000),
    ("F",  5,  459.2,  "Residencial", 237_000),
    ("F",  6,  458.8,  "Residencial", 237_000),
    ("F",  7,  420.8,  "Residencial", 250_000),
    ("F",  8,  420.0,  "Residencial", 270_000),
    ("F",  9,  420.0,  "Residencial", 255_000),
    ("F", 10,  420.0,  "Residencial", 255_000),
    ("F", 11,  420.0,  "Residencial", 270_000),
    ("F", 12,  464.0,  "Residencial", 290_000),
    ("F", 13,  406.0,  "Residencial", 270_000),
    ("F", 14,  406.0,  "Misto I",     285_000),
    ("F", 15,  458.57, "Misto I",     320_000),
    ("F", 16,  420.0,  "Misto I",     285_000),
    ("F", 17,  420.0,  "Misto I",     285_000),
    ("F", 18,  420.0,  "Misto I",     285_000),
    ("F", 19,  420.0,  "Misto I",     285_000),
    # Quadra G
    ("G",  1,  480.0,  "Residencial", 298_000),
    ("G",  2,  420.0,  "Residencial", 270_000),
    ("G",  3,  420.0,  "Misto I",     285_000),
    ("G",  4,  474.71, "Misto I",     320_000),
    ("G",  5,  420.0,  "Misto I",     290_000),
    ("G",  6,  420.0,  "Misto I",     295_000),
    ("G",  7,  474.57, "Misto I",     346_000),
    ("G",  8,  420.0,  "Misto I",     326_000),
    ("G",  9,  420.0,  "Residencial", 326_000),
    ("G", 10,  480.0,  "Residencial", 330_000),
    ("G", 11,  420.0,  "Residencial", 310_000),   # *vendido
    ("G", 12,  420.0,  "Residencial", 310_000),   # *vendido
    # Quadra H
    ("H",  1,  452.0,  "Residencial", 284_000),
    ("H",  2,  422.8,  "Residencial", 263_000),
    ("H",  3,  423.8,  "Residencial", 263_000),
    ("H",  4,  424.7,  "Residencial", 263_000),
    ("H",  5,  425.6,  "Residencial", 263_000),
    ("H",  6,  426.5,  "Residencial", 265_000),   # *vendido
    ("H",  7,  427.5,  "Residencial", 265_000),   # *vendido
    ("H",  8,  432.5,  "Residencial", 278_000),
    ("H",  9,  433.5,  "Residencial", 278_000),
    ("H", 10,  434.3,  "Residencial", 278_000),   # *vendido
    ("H", 11,  435.3,  "Residencial", 280_000),   # *vendido
    ("H", 12,  436.2,  "Residencial", 285_000),
    ("H", 13,  437.16, "Misto I",     300_000),
    ("H", 14,  438.09, "Misto I",     315_000),
    ("H", 15,  439.01, "Misto I",     318_000),   # *não disponível
    ("H", 16,  439.94, "Misto I",     320_000),   # *não disponível
    ("H", 17,  440.87, "Misto I",     325_000),   # *não disponível
    ("H", 18,  473.09, "Misto I",     340_000),   # *não disponível
    # Quadra I
    ("I",  1,  418.46, "Misto I",     220_000),
    ("I",  2,  417.51, "Misto I",     225_000),
    ("I",  3,  416.61, "Misto I",     230_000),
    ("I",  4,  415.92, "Misto I",     232_000),
    ("I",  5,  415.42, "Misto I",     235_000),
    ("I",  6,  414.92, "Misto I",     238_000),
    ("I",  7, 1352.8,  "Misto II",    580_000),   # *não disponível
    ("I",  8, 1555.9,  "Misto II",    650_000),   # *não disponível
    ("I",  9, 1721.3,  "Misto II",    730_000),   # *não disponível
    # Quadra J
    ("J",  1,  474.57, "Misto I",     330_000),   # *não disponível
    ("J",  2,  420.0,  "Misto I",     280_000),   # *não disponível
    ("J",  3,  420.0,  "Residencial", 278_000),
    ("J",  4,  480.0,  "Residencial", 298_000),
    ("J",  5,  420.0,  "Residencial", 258_000),
    ("J",  6,  420.0,  "Residencial", 270_000),
    ("J",  7,  420.0,  "Misto I",     280_000),
    ("J",  8,  420.0,  "Misto I",     298_000),
    ("J",  9,  420.0,  "Misto I",     290_000),   # *não disponível
    ("J", 10,  420.0,  "Misto I",     290_000),   # *não disponível
    ("J", 11,  420.0,  "Misto I",     298_000),
    ("J", 12,  420.0,  "Misto I",     298_000),
    # Quadra K
    ("K",  1,  480.0,  "Residencial", 298_000),
    ("K",  2,  420.0,  "Residencial", 278_000),
    ("K",  3,  420.0,  "Misto I",     295_000),   # *não disponível
    ("K",  4,  474.71, "Misto I",     330_000),   # *não disponível
    ("K",  5,  420.0,  "Misto I",     298_000),
    ("K",  6,  420.0,  "Misto I",     303_000),
    ("K",  7,  420.0,  "Misto I",     298_000),   # *vendido
    ("K",  8,  420.0,  "Misto I",     298_000),   # *vendido
    ("K",  9,  420.0,  "Misto I",     295_000),   # *não disponível
    ("K", 10,  420.0,  "Misto I",     295_000),   # *não disponível
    ("K", 11,  420.0,  "Residencial", 268_000),
    ("K", 12,  420.0,  "Residencial", 268_000),   # *vendido
    # Quadra O
    ("O",  1,  475.73, "Misto I",     330_000),   # *não disponível
    ("O",  2,  444.97, "Misto I",     310_000),   # *não disponível
    ("O",  3,  445.9,  "Misto I",     310_000),   # *não disponível
    ("O",  4,  574.67, "Misto I",     380_000),   # *não disponível
]

# Remove os lotes sintéticos do Florescer e insere os reais
df = df[df["loteamento"] != "Florescer Parque Residencial"].copy()

florescer_rows = []
for quadra, lote, area, cat, preco in FLORESCER_REAL_LOTS:
    yr_built = int(rng.integers(2018, 2026))
    florescer_rows.append(dict(
        city="Vila Flores", region="Serra Gaúcha", bairro="Centro",
        property_type="Terreno", loteamento="Florescer Parque Residencial",
        price=preco,
        bedrooms=0, bathrooms=0.0, house_size=0, lot_size=round(area),
        sqft_above=0, sqft_basement=0, floors=0.0,
        waterfront=0, view=int(rng.integers(0, 3)), condition=3,
        grade=6, year_built=yr_built, year_renovated=0,
        zipcode=95335,
        lat=float(rng.uniform(-28.875, -28.850)),
        long=float(rng.uniform(-51.615, -51.590)),
    ))

df_florescer = pd.DataFrame(florescer_rows)
df = pd.concat([df, df_florescer], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/house_price.csv", index=False)

print(f"Dataset gerado: {len(df):,} registros\n")

# Resumo terrenos das 3 cidades principais
for city in ["Vila Flores", "Nova Prata", "Veranópolis"]:
    sub = df[(df["city"] == city) & (df["property_type"] == "Terreno")].copy()
    sub["pm2"] = sub["price"] / sub["lot_size"].replace(0, np.nan)
    print(f"\n{'='*60}")
    print(f"  TERRENOS — {city}  ({len(sub)} registros)")
    print(f"{'='*60}")
    g = sub.groupby("loteamento").agg(
        N=("price","count"),
        Preço_Médio=("price","mean"),
        Área_Média=("lot_size","mean"),
        PM2_Médio=("pm2","mean"),
        PM2_Min=("pm2","min"),
        PM2_Max=("pm2","max"),
    ).sort_values("PM2_Médio")
    for lot, row in g.iterrows():
        print(f"  {lot:<35} {int(row.N):>3} lotes | "
              f"~{int(row.Área_Média):>5}m² | "
              f"R$ {int(row.PM2_Médio):>5}/m² "
              f"(R$ {int(row.PM2_Min)}–{int(row.PM2_Max)}) | "
              f"Médio: R$ {int(row.Preço_Médio):,}")
