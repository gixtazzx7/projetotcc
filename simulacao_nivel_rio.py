"""
SIMULAÇÃO COMPUTACIONAL - DADOS FICTÍCIOS
=========================================
TCC: IoT na medição e monitoramento do nível de água em rios da Amazônia.

Demonstra a etapa de PROCESSAMENTO NO SERVIDOR da arquitetura proposta
(sensor ultrassônico -> LoRaWAN -> servidor -> painel). Não há hardware,
LoRaWAN real, API nem banco de dados: as leituras são geradas por código.

Executar:  streamlit run simulacao_nivel_rio.py
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd

PASTA_IMAGENS = Path(__file__).parent / "imagens"

DIAS = 3
INTERVALO_MIN = 10
FUSO = "America/Manaus"


# ---------------------------------------------------------------------------
# 1. Leituras fictícias (o que o sensor enviaria a cada 10 minutos)
# ---------------------------------------------------------------------------
def gerar_leituras(semente: int = 42) -> pd.DataFrame:
    """Gera 3 dias de nível (cm): subida lenta + oscilação suave + ruído."""
    rng = np.random.default_rng(semente)
    n = DIAS * 24 * 60 // INTERVALO_MIN  # 432 leituras
    tempo = pd.date_range("2026-03-10 00:00", periods=n, freq=f"{INTERVALO_MIN}min", tz=FUSO)
    horas = np.arange(n) * INTERVALO_MIN / 60

    nivel = (
        1800.0                                   # nível de referência fictício
        + 0.25 * horas                           # subida lenta (~6 cm/dia)
        + 5.0 * np.sin(2 * np.pi * horas / 36)   # oscilação suave
        + rng.normal(0, 0.7, n)                  # ruído do sensor
    )
    return pd.DataFrame({"tempo": tempo, "nivel_cm": np.round(nivel, 1)})


def inserir_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """Insere 3 leituras anormais típicas de sensor ultrassônico."""
    df = df.copy()
    df["anomalia_inserida"] = ""
    anomalias = [
        (100, +60.0, "pico (eco em objeto flutuante)"),
        (230, -250.0, "queda brusca (eco perdido)"),
        (350, +25.0, "média de ocorrência (respingo/inseto)"),
    ]
    for i, desvio, descricao in anomalias:
        df.loc[i, "nivel_cm"] += desvio
        df.loc[i, "anomalia_inserida"] = descricao
    return df


# ---------------------------------------------------------------------------
# 2. Processamento no servidor: mediana móvel + MAD
# ---------------------------------------------------------------------------
def detectar_suspeitas(df: pd.DataFrame, w: int = 13, k: float = 5.0) -> pd.DataFrame:
    """Marca como suspeita a leitura em que |x_t - m_t| > k * MAD_t.

    m_t   = mediana das w leituras em torno de t
    MAD_t = mediana de |x_i - m_t| nessa mesma janela
    Um piso de 0,5 cm no MAD evita marcar ruído em trechos muito estáveis.
    """
    df = df.copy()
    janela = df["nivel_cm"].rolling(w, center=True, min_periods=3)
    df["mediana_movel"] = janela.median()
    df["mad"] = janela.apply(lambda v: np.median(np.abs(v - np.median(v))), raw=True)
    df["limite"] = k * df["mad"].clip(lower=0.5)
    df["desvio"] = (df["nivel_cm"] - df["mediana_movel"]).abs()
    df["suspeita"] = df["desvio"] > df["limite"]
    return df


# ---------------------------------------------------------------------------
# 3. Painel
# ---------------------------------------------------------------------------
# Paleta amazônica
VERDE_FLORESTA = "#1B5E20"
VERDE_FOLHA = "#2E7D32"
VERDE_CLARO = "#81C784"
AGUA_RIO = "#00796B"      # verde-água do rio
TERRA = "#8D6E63"         # marrom de águas barrentas
URUCUM = "#D84315"        # laranja-avermelhado (alerta)

CSS_AMAZONICO = f"""
<style>
/* Barra lateral: degradê de mata */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0B2E10 0%, #12391A 60%, #1A4422 100%);
}}
[data-testid="stSidebar"] * {{ color: #F1F8E9 !important; }}
/* Logo colada no topo: o cabeçalho (botão de recolher) flutua por cima, à direita */
[data-testid="stSidebarHeader"] {{
    position: absolute; top: 0; right: 0; z-index: 10;
    height: auto; padding: .25rem;
}}
[data-testid="stSidebarUserContent"] {{ padding-top: .75rem; }}
/* Logo um pouco menor e centralizada */
[data-testid="stSidebar"] [data-testid="stImage"] {{ margin: 0 auto; }}
[data-testid="stSidebar"] [data-testid="stImage"] img {{
    width: 90% !important; display: block; margin: 0 auto;
}}

/* Título */
h1 {{
    color: {VERDE_FLORESTA} !important;
    border-bottom: 4px solid {VERDE_CLARO};
    padding-bottom: .3rem;
}}
h2, h3 {{ color: {VERDE_FOLHA} !important; }}

/* Cartões dos indicadores */
[data-testid="stMetric"] {{
    background: #FFFFFF;
    border-left: 6px solid {VERDE_FOLHA};
    border-radius: 10px;
    padding: .8rem 1rem;
    box-shadow: 0 2px 6px rgba(27, 94, 32, .15);
}}
[data-testid="stMetricValue"] {{ color: {VERDE_FLORESTA}; }}

/* Botões */
.stButton > button, .stDownloadButton > button {{
    background-color: {VERDE_FOLHA};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: {VERDE_FLORESTA};
    color: #FFFFFF;
}}
</style>
"""


def encontrar_imagem(nome: str) -> Path | None:
    """Procura imagens/<nome>.png|.jpg|.jpeg|.webp e devolve o primeiro que existir."""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        caminho = PASTA_IMAGENS / f"{nome}{ext}"
        if caminho.exists():
            return caminho
    return None


def recortar_logo(caminho: Path):
    """Recorta só a parte colorida da logo (descarta fundo branco/xadrez embutido)
    e arredonda os cantos com transparência real."""
    from PIL import Image, ImageDraw

    img = Image.open(caminho).convert("RGBA")
    rgb = np.asarray(img.convert("RGB")).astype(int)
    saturacao = rgb.max(axis=2) - rgb.min(axis=2)       # cinza/branco ≈ 0
    linhas, colunas = np.where(saturacao > 60)
    if linhas.size == 0:
        return img
    img = img.crop((colunas.min(), linhas.min(), colunas.max() + 1, linhas.max() + 1))

    mascara = Image.new("L", img.size, 0)
    raio = max(4, img.size[1] // 16)
    ImageDraw.Draw(mascara).rounded_rectangle((0, 0, *img.size), radius=raio, fill=255)
    img.putalpha(mascara)
    return img


def css_fundo(caminho: Path | None) -> str:
    """Fundo com a foto da floresta/rio + véu claro para manter o texto legível."""
    if caminho is None:
        return ""
    mime = "jpeg" if caminho.suffix.lower() in (".jpg", ".jpeg") else caminho.suffix[1:].lower()
    dados = base64.b64encode(caminho.read_bytes()).decode()
    return f"""
<style>
[data-testid="stApp"] {{
    background:
        linear-gradient(rgba(243, 247, 236, .20), rgba(243, 247, 236, .20)),
        url("data:image/{mime};base64,{dados}") center / cover fixed no-repeat;
}}
[data-testid="stHeader"] {{ background: transparent; }}
/* Conteúdo principal num painel translúcido sobre a foto */
[data-testid="stMainBlockContainer"], .block-container {{
    background: rgba(255, 255, 255, .75);
    border-radius: 16px;
    padding: 2rem 2.5rem !important;
    margin-top: 3rem;
    box-shadow: 0 6px 24px rgba(0, 0, 0, .25);
}}
</style>
"""


def main() -> None:
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="Simulação - nível do rio", page_icon="🌿", layout="wide")
    st.markdown(CSS_AMAZONICO, unsafe_allow_html=True)
    st.markdown(css_fundo(encontrar_imagem("fundo_amazonia")), unsafe_allow_html=True)

    # Logo da FAMETRO no topo da barra lateral, ocupando toda a largura
    logo = encontrar_imagem("logo_fametro")
    if logo is not None:
        st.sidebar.image(recortar_logo(logo), width="stretch")
    st.error("**SIMULAÇÃO COMPUTACIONAL - DADOS FICTÍCIOS**")
    st.title("Monitoramento do nível do rio")
    st.caption(
        "Demonstração da etapa de processamento no servidor da arquitetura IoT proposta "
        "(sensor ultrassônico → LoRaWAN → servidor → painel). "
        f"{DIAS} dias de leituras fictícias a cada {INTERVALO_MIN} minutos."
    )

    st.sidebar.header("Parâmetros do filtro")
    w = st.sidebar.slider("Janela da mediana móvel (w, leituras)", 5, 31, 13, 2)
    k = st.sidebar.slider("Multiplicador do MAD (k)", 2.0, 10.0, 5.0, 0.5)

    df = detectar_suspeitas(inserir_anomalias(gerar_leituras()), w, k)
    validas = df[~df["suspeita"]]
    suspeitas = df[df["suspeita"]]

    # Indicadores (calculados sem as leituras suspeitas)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nível atual", f"{validas['nivel_cm'].iloc[-1]:.1f} cm")
    c2.metric("Nível máximo", f"{validas['nivel_cm'].max():.1f} cm")
    c3.metric("Nível mínimo", f"{validas['nivel_cm'].min():.1f} cm")
    c4.metric("Leituras suspeitas", len(suspeitas))
    st.caption("Atual, máximo e mínimo desconsideram as leituras suspeitas.")

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["tempo"], y=df["nivel_cm"], mode="lines",
                             name="Nível lido", line=dict(color=AGUA_RIO, width=1.5)))
    fig.add_trace(go.Scatter(x=df["tempo"], y=df["mediana_movel"], mode="lines",
                             name="Mediana móvel", line=dict(color=TERRA, dash="dot")))
    fig.add_trace(go.Scatter(x=suspeitas["tempo"], y=suspeitas["nivel_cm"], mode="markers",
                             name="Leitura suspeita",
                             marker=dict(color=URUCUM, size=10, line=dict(color="white", width=1))))
    fig.update_layout(
        title="Variação do nível da água (dados fictícios)",
        xaxis_title="Data/hora (Manaus, UTC-4)", yaxis_title="Nível (cm)",
        height=450, legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F3F7EC",
        font=dict(color=VERDE_FLORESTA), title_font=dict(color=VERDE_FLORESTA),
    )
    fig.update_xaxes(gridcolor="#E3EED8")
    fig.update_yaxes(gridcolor="#E3EED8")
    st.plotly_chart(fig, width="stretch")

    # Tabela de ocorrências
    st.subheader("Ocorrências detectadas")
    if suspeitas.empty:
        st.info("Nenhuma leitura suspeita com os parâmetros atuais.")
    else:
        tabela = pd.DataFrame({
            "Data/hora": suspeitas["tempo"].dt.strftime("%d/%m/%Y %H:%M"),
            "Nível lido (cm)": suspeitas["nivel_cm"].round(1),
            "Mediana móvel (cm)": suspeitas["mediana_movel"].round(1),
            "Desvio (cm)": suspeitas["desvio"].round(1),
            "Limite k·MAD (cm)": suspeitas["limite"].round(1),
            "Anomalia inserida": suspeitas["anomalia_inserida"].replace("", "— (ruído)"),
        })
        st.dataframe(tabela, hide_index=True, width="stretch")

    st.caption("SIMULAÇÃO COMPUTACIONAL - DADOS FICTÍCIOS. Não representa medições de campo.")


if __name__ == "__main__":
    main()
