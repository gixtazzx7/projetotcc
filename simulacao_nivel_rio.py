"""
DADOS HISTÓRICOS - RIO NEGRO EM MANAUS (2026)
=============================================
TCC: IoT na medição e monitoramento do nível de água em rios da Amazônia.

Usa as leituras reais de telemetria da ANA (estação 14990000, Rio Negro em
Manaus, uma leitura a cada 15 minutos) para demonstrar a etapa de
PROCESSAMENTO NO SERVIDOR da arquitetura proposta (sensor -> LoRaWAN ->
servidor -> painel): filtro de leituras suspeitas, resumo por trimestre e
previsão de curto prazo com aprendizado de máquina.

São dados históricos, não medições em tempo real. Para atualizar o arquivo
de dados:  python atualizar_dados_ana.py

Executar:  streamlit run simulacao_nivel_rio.py
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd

PASTA_IMAGENS = Path(__file__).parent / "imagens"
ARQUIVO_DADOS = Path(__file__).parent / "dados" / "rio_negro_manaus_2026.csv"
# Os dados reais param aqui: leituras posteriores no arquivo são ignoradas.
DATA_FINAL = pd.Timestamp("2026-11-17 23:59:59")
ESTACAO = "ANA 14990000 - Rio Negro em Manaus"

TRIMESTRES = {1: "1º trimestre (jan–mar)", 2: "2º trimestre (abr–jun)",
              3: "3º trimestre (jul–set)", 4: "4º trimestre (out–dez)"}
PREVISAO = "Previsão (até 31/12)"

# Registros do Porto de Manaus informados para o projeto (régua diferente da
# estação da ANA; mostrados só para comparação).
REGISTROS_PORTO = [
    ("2026-01-01", 22.03), ("2026-01-31", 22.77),
    ("2026-06-01", 27.95), ("2026-06-21", 28.51), ("2026-06-28", 28.51),
    ("2026-10-01", 18.73), ("2026-10-27", 16.59),
]


def metros(cm: float, sinal: bool = False) -> str:
    """Converte cm para texto em metros com vírgula: 2203 -> '22,03 m'."""
    texto = f"{cm / 100:+.2f}" if sinal else f"{cm / 100:.2f}"
    return texto.replace(".", ",") + " m"


# ---------------------------------------------------------------------------
# 1. Leituras reais (telemetria da ANA, a cada 15 minutos)
# ---------------------------------------------------------------------------
def carregar_leituras() -> pd.DataFrame:
    df = pd.read_csv(ARQUIVO_DADOS, parse_dates=["data_hora"])
    df = df[df["data_hora"] <= DATA_FINAL].reset_index(drop=True)
    df["trimestre"] = df["data_hora"].dt.quarter
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


def niveis_diarios(df: pd.DataFrame) -> pd.Series:
    """Média de cada dia (cm) usando só as leituras válidas."""
    validas = df[~df["suspeita"]]
    diario = validas.groupby(validas["data_hora"].dt.normalize())["nivel_cm"].mean()
    return diario.asfreq("D").interpolate()   # cobre um dia sem leitura válida, se houver


def resumo_trimestres(diario: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Início, fim, máximo, mínimo e variação de cada trimestre (médias diárias)."""
    linhas = []
    for t, s in diario.groupby(diario.index.quarter):
        fim_trimestre = pd.Timestamp(year=s.index[0].year, month=3 * t, day=1) + pd.offsets.MonthEnd()
        linhas.append({
            "trimestre": t,
            "data_ini": s.index[0], "data_fim": s.index[-1],
            "parcial": s.index[-1] < fim_trimestre,
            "inicio": s.iloc[0], "fim": s.iloc[-1],
            "maximo": s.max(), "data_max": s.idxmax(),
            "minimo": s.min(), "data_min": s.idxmin(),
            "variacao": s.iloc[-1] - s.iloc[0],
            "por_dia": (s.iloc[-1] - s.iloc[0]) / max(len(s) - 1, 1),
            "suspeitas": int(df.loc[df["trimestre"] == t, "suspeita"].sum()),
        })
    return pd.DataFrame(linhas).set_index("trimestre")


# ---------------------------------------------------------------------------
# 3. Previsão de curto prazo (aprendizado de máquina)
# ---------------------------------------------------------------------------
def prever_nivel(serie: pd.Series, passos: int, lags: int = 7) -> np.ndarray:
    """Modelo autorregressivo com regressão linear (scikit-learn).

    Treino: cada exemplo usa as médias dos `lags` dias anteriores (7 = uma
    semana) como entrada e a média do dia seguinte como saída.
    Previsão recursiva: prevê o próximo dia, acrescenta essa previsão ao
    histórico e repete até completar o horizonte.
    """
    from sklearn.linear_model import LinearRegression

    x = serie.to_numpy()
    X = np.column_stack([x[i : len(x) - lags + i] for i in range(lags)])
    y = x[lags:]
    modelo = LinearRegression().fit(X, y)

    historico = list(x[-lags:])
    previsoes = []
    for _ in range(passos):
        p = modelo.predict(np.array(historico[-lags:]).reshape(1, -1))[0]
        previsoes.append(p)
        historico.append(p)
    return np.array(previsoes)


def erro_da_previsao(serie: pd.Series, passos: int) -> float:
    """Teste simples: esconde os últimos `passos` dias, prevê esse trecho
    com o restante e mede o erro médio absoluto (cm)."""
    previsto = prever_nivel(serie.iloc[:-passos], passos)
    real = serie.iloc[-passos:].to_numpy()
    return float(np.mean(np.abs(previsto - real)))


# ---------------------------------------------------------------------------
# 4. Painel
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

    st.set_page_config(page_title="Nível do Rio Negro - Manaus", page_icon="🌿", layout="wide")
    st.markdown(CSS_AMAZONICO, unsafe_allow_html=True)
    st.markdown(css_fundo(encontrar_imagem("fundo_amazonia")), unsafe_allow_html=True)

    # Logo da FAMETRO no topo da barra lateral, ocupando toda a largura
    logo = encontrar_imagem("logo_fametro")
    if logo is not None:
        st.sidebar.image(recortar_logo(logo), width="stretch")

    st.sidebar.header("Período")
    opcoes = ["Ano inteiro"] + list(TRIMESTRES.values()) + [PREVISAO]
    escolha = st.sidebar.radio("Mostrar", opcoes, index=0)
    if escolha == PREVISAO:
        from projecao_nivel_rio import mostrar_previsao
        mostrar_previsao()
        return
    st.sidebar.header("Parâmetros do filtro")
    w = st.sidebar.slider("Janela da mediana móvel (w, leituras)", 5, 31, 13, 2)
    k = st.sidebar.slider("Multiplicador do MAD (k)", 2.0, 10.0, 5.0, 0.5)
    st.sidebar.header("Previsão")
    dias_prev = st.sidebar.slider("Horizonte da previsão (dias)", 1, 15, 7)

    @st.cache_data
    def processar(w: int, k: float) -> pd.DataFrame:
        return detectar_suspeitas(carregar_leituras(), w, k)

    df = processar(w, k)
    diario = niveis_diarios(df)
    resumo = resumo_trimestres(diario, df)
    inicio_dados, fim_dados = df["data_hora"].iloc[0], df["data_hora"].iloc[-1]

    st.info(f"**DADOS HISTÓRICOS REAIS - {ESTACAO}**, telemetria de "
            f"{inicio_dados:%d/%m/%Y} a {fim_dados:%d/%m/%Y}. Não são medições em tempo real.")
    st.title("Monitoramento do nível do rio")
    st.caption(
        "Leituras reais (a cada 15 minutos) processadas como na etapa de servidor da arquitetura "
        "IoT proposta (sensor ultrassônico → LoRaWAN → servidor → painel): filtro de leituras "
        "suspeitas, resumo por trimestre e previsão dos próximos dias."
    )

    # Recorte do período escolhido
    trimestre = next((t for t, nome in TRIMESTRES.items() if nome == escolha), None)
    if trimestre is not None and trimestre not in resumo.index:
        st.warning(f"Ainda não há dados do {escolha}.")
        return
    if trimestre is None:
        dfp, diap = df, diario
    else:
        dfp = df[df["trimestre"] == trimestre]
        diap = diario[diario.index.quarter == trimestre]
    suspeitas = dfp[dfp["suspeita"]]

    # Previsão sempre a partir do último dia com dados
    previsao = prever_nivel(diario, dias_prev)
    erro = erro_da_previsao(diario, dias_prev)
    datas_prev = pd.date_range(diario.index[-1], periods=dias_prev + 1, freq="D")[1:]
    mostra_previsao = diap.index[-1] == diario.index[-1]

    # Indicadores do período (médias diárias, sem as leituras suspeitas)
    st.subheader(escolha + (" — parcial" if trimestre and resumo.loc[trimestre, "parcial"] else ""))
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(f"Início ({diap.index[0]:%d/%m})", metros(diap.iloc[0]))
    c2.metric(f"Fim ({diap.index[-1]:%d/%m})", metros(diap.iloc[-1]),
              delta=metros(diap.iloc[-1] - diap.iloc[0], sinal=True), delta_color="off")
    c3.metric(f"Máximo ({diap.idxmax():%d/%m})", metros(diap.max()))
    c4.metric(f"Mínimo ({diap.idxmin():%d/%m})", metros(diap.min()))
    c5.metric("Leituras suspeitas", len(suspeitas))
    c6.metric(f"Previsão {datas_prev[-1]:%d/%m}", metros(previsao[-1]),
              delta=metros(previsao[-1] - diario.iloc[-1], sinal=True), delta_color="off")
    st.caption(
        "Início, fim, máximo e mínimo usam a média diária das leituras válidas; a seta do fim é a "
        f"variação no período. A previsão parte do último dia com dados ({diario.index[-1]:%d/%m}); "
        f"erro médio no teste: ±{erro:.1f} cm."
    )

    # Gráfico principal (em metros)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=dfp["data_hora"], y=dfp["nivel_cm"] / 100, mode="lines",
                               name="Leituras (15 min)",
                               line=dict(color=AGUA_RIO, width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=diap.index + pd.Timedelta(hours=12), y=diap / 100, mode="lines",
                             name="Média diária", line=dict(color=VERDE_FLORESTA, width=2)))
    fig.add_trace(go.Scatter(x=suspeitas["data_hora"], y=suspeitas["nivel_cm"] / 100,
                             mode="markers", name="Leitura suspeita",
                             marker=dict(color=URUCUM, size=10, line=dict(color="white", width=1))))
    porto = pd.DataFrame(REGISTROS_PORTO, columns=["data", "nivel_m"])
    porto["data"] = pd.to_datetime(porto["data"]) + pd.Timedelta(hours=12)
    porto = porto[(porto["data"] >= diap.index[0]) & (porto["data"] <= diap.index[-1] + pd.Timedelta(days=1))]
    if not porto.empty:
        fig.add_trace(go.Scatter(x=porto["data"], y=porto["nivel_m"], mode="markers",
                                 name="Registro do Porto de Manaus",
                                 marker=dict(color=TERRA, size=9, symbol="diamond",
                                             line=dict(color="white", width=1))))
    if mostra_previsao:
        x_prev = datas_prev + pd.Timedelta(hours=12)
        fig.add_trace(go.Scatter(
            x=list(x_prev) + list(x_prev[::-1]),
            y=list((previsao + erro) / 100) + list(((previsao - erro) / 100)[::-1]),
            fill="toself", fillcolor="rgba(46,125,50,0.18)", line=dict(width=0),
            name=f"Margem (±{erro:.1f} cm)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_prev, y=previsao / 100, mode="lines",
                                 name=f"Previsão ({dias_prev} dias)",
                                 line=dict(color=VERDE_FOLHA, width=2, dash="dash")))
        fig.add_vline(x=fim_dados, line=dict(color=VERDE_CLARO, dash="dot"))
    if trimestre is None:
        # Faixas destacando cada trimestre no ano inteiro
        for t in resumo.index:
            ini = pd.Timestamp(year=inicio_dados.year, month=3 * t - 2, day=1)
            fim_faixa = min(ini + pd.DateOffset(months=3), datas_prev[-1] + pd.Timedelta(days=1))
            fig.add_vrect(x0=ini, x1=fim_faixa,
                          fillcolor=VERDE_CLARO if t % 2 else TERRA, opacity=0.08, line_width=0,
                          annotation_text=f"{t}º tri", annotation_position="top left")
    fig.update_layout(
        title=f"Variação do nível da água - {escolha.lower()} (dados históricos)",
        xaxis_title="Data", yaxis_title="Nível (m)", separators=",.",
        height=450, legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F3F7EC",
        font=dict(color=VERDE_FLORESTA), title_font=dict(color=VERDE_FLORESTA),
    )
    fig.update_xaxes(gridcolor="#E3EED8")
    fig.update_yaxes(gridcolor="#E3EED8")
    st.plotly_chart(fig, width="stretch")

    # Comparativo entre trimestres
    st.subheader("Comparativo entre trimestres")
    nomes = [TRIMESTRES[t].split(" (")[0] + (" *" if resumo.loc[t, "parcial"] else "")
             for t in resumo.index]
    opac = [1.0 if trimestre in (None, t) else 0.35 for t in resumo.index]
    layout_comp = dict(
        height=380, legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F3F7EC", separators=",.",
        font=dict(color=VERDE_FLORESTA), title_font=dict(color=VERDE_FLORESTA),
    )
    g1, g2 = st.columns(2)

    comp = go.Figure()
    for coluna, rotulo, cor in [("inicio", "Início", VERDE_CLARO), ("fim", "Fim", AGUA_RIO),
                                ("maximo", "Máximo", VERDE_FLORESTA), ("minimo", "Mínimo", TERRA)]:
        comp.add_trace(go.Bar(x=nomes, y=resumo[coluna] / 100, name=rotulo,
                              marker=dict(color=cor, opacity=opac),
                              hovertext=[metros(v) for v in resumo[coluna]], hoverinfo="text+name"))
    comp.update_layout(title="Níveis por trimestre", yaxis_title="Nível (m)", barmode="group",
                       **layout_comp)
    comp.update_yaxes(range=[resumo["minimo"].min() / 100 * 0.85, resumo["maximo"].max() / 100 * 1.05],
                      gridcolor="#E3EED8")
    g1.plotly_chart(comp, width="stretch")

    var = go.Figure(go.Bar(
        x=nomes, y=resumo["variacao"] / 100, showlegend=False,
        marker=dict(color=[VERDE_FOLHA if v >= 0 else URUCUM for v in resumo["variacao"]],
                    opacity=opac),
        text=[f"{metros(v, sinal=True)}<br>({p:+.1f} cm/dia)".replace(".", ",")
              for v, p in zip(resumo["variacao"], resumo["por_dia"])],
        textposition="outside",
    ))
    var.update_layout(title="Variação no trimestre (subida × descida)", yaxis_title="Variação (m)",
                      **layout_comp)
    var.update_yaxes(range=[min(resumo["variacao"].min(), 0) / 100 * 1.4 - 0.5,
                            max(resumo["variacao"].max(), 0) / 100 * 1.4 + 0.5],
                     zeroline=True, zerolinecolor=VERDE_FLORESTA, gridcolor="#E3EED8")
    g2.plotly_chart(var, width="stretch")
    if resumo["parcial"].any():
        st.caption("* Trimestre parcial: dados até "
                   f"{resumo.loc[resumo['parcial'], 'data_fim'].iloc[-1]:%d/%m/%Y}.")

    tabela = pd.DataFrame({
        "Trimestre": [TRIMESTRES[t] for t in resumo.index],
        "Período": [f"{a:%d/%m} a {b:%d/%m}" for a, b in zip(resumo["data_ini"], resumo["data_fim"])],
        "Início": [metros(v) for v in resumo["inicio"]],
        "Fim": [metros(v) for v in resumo["fim"]],
        "Variação": [metros(v, sinal=True) for v in resumo["variacao"]],
        "Máximo": [f"{metros(v)} ({d:%d/%m})" for v, d in zip(resumo["maximo"], resumo["data_max"])],
        "Mínimo": [f"{metros(v)} ({d:%d/%m})" for v, d in zip(resumo["minimo"], resumo["data_min"])],
        "Ritmo médio": [f"{p:+.1f} cm/dia".replace(".", ",") for p in resumo["por_dia"]],
        "Leituras suspeitas": resumo["suspeitas"],
    })
    st.dataframe(tabela, hide_index=True, width="stretch")

    # Tabela de ocorrências do período
    st.subheader("Ocorrências detectadas")
    if suspeitas.empty:
        st.info("Nenhuma leitura suspeita no período com os parâmetros atuais.")
    else:
        st.dataframe(pd.DataFrame({
            "Data/hora": suspeitas["data_hora"].dt.strftime("%d/%m/%Y %H:%M"),
            "Nível lido (cm)": suspeitas["nivel_cm"].round(1),
            "Mediana móvel (cm)": suspeitas["mediana_movel"].round(1),
            "Desvio (cm)": suspeitas["desvio"].round(1),
            "Limite k·MAD (cm)": suspeitas["limite"].round(1),
        }), hide_index=True, width="stretch")

    st.caption(f"DADOS HISTÓRICOS - {ESTACAO} (telemetria bruta, não consistida). "
               "Não representam medições em tempo real do sistema proposto. Os registros do "
               "Porto de Manaus usam outra régua e aparecem só para comparação.")


if __name__ == "__main__":
    main()
