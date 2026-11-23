"""
PREVISÃO DO NÍVEL DO RIO NEGRO EM MANAUS ATÉ O FIM DE 2026
==========================================================
TCC: IoT na medição e monitoramento do nível de água em rios da Amazônia.

Seção "Previsão" do painel (streamlit run simulacao_nivel_rio.py).

Usa só os dados reais de 2026 (telemetria da ANA, estação 14990000), que
vão até 17/11/2026, e projeta dia a dia até 31/12/2026 com o mesmo modelo
do painel: regressão linear sobre as médias dos últimos 7 dias, aplicada
de forma recursiva.

É uma PROJEÇÃO, não uma medição.
"""

import pandas as pd

from simulacao_nivel_rio import (
    AGUA_RIO, ESTACAO, URUCUM, VERDE_CLARO, VERDE_FLORESTA, VERDE_FOLHA,
    carregar_leituras, detectar_suspeitas, metros, niveis_diarios, prever_nivel,
)

FIM_PROJECAO = pd.Timestamp("2026-12-31")


def calcular_previsao() -> tuple[pd.Series, pd.Series]:
    """Devolve a série diária real de 2026 e a projeção até 31/12."""
    serie = niveis_diarios(detectar_suspeitas(carregar_leituras()))
    passos = (FIM_PROJECAO - serie.index[-1]).days
    datas = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), FIM_PROJECAO, freq="D")
    return serie, pd.Series(prever_nivel(serie, passos), index=datas)


def mostrar_previsao() -> None:
    import plotly.graph_objects as go
    import streamlit as st

    serie, proj = st.cache_data(calcular_previsao)()
    inicio, atual = serie.index[-1], serie.iloc[-1]

    st.sidebar.header("Previsão")
    dia = st.sidebar.slider("Consultar o dia", min_value=proj.index[0].date(),
                            max_value=proj.index[-1].date(), value=proj.index[-1].date(),
                            format="DD/MM/YYYY")
    dia = pd.Timestamp(dia)
    desde = st.sidebar.slider("Mostrar 2026 desde (dias antes do último dado)", 30, 320, 90, 10)

    st.warning(f"**PROJEÇÃO - não é medição.** Os dados reais da {ESTACAO} vão até "
               f"{inicio:%d/%m/%Y}; daí até {FIM_PROJECAO:%d/%m/%Y} os valores são projetados.")
    st.title("Monitoramento do nível do rio")
    st.subheader(f"Previsão: {proj.index[0]:%d/%m} a {FIM_PROJECAO:%d/%m/%Y}")
    st.caption(
        "Modelo de aprendizado de máquina treinado só com os dados de 2026: regressão linear que "
        "prevê a média de um dia a partir das médias dos 7 dias anteriores. Cada dia previsto entra "
        "como entrada do seguinte, até 31/12."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Último dado real ({inicio:%d/%m})", metros(atual))
    c2.metric(f"Previsão para {dia:%d/%m}", metros(proj.loc[dia]),
              delta=metros(proj.loc[dia] - atual, sinal=True), delta_color="off")
    c3.metric(f"Menor nível previsto ({proj.idxmin():%d/%m})", metros(proj.min()))

    layout = dict(
        xaxis_title="Data", yaxis_title="Nível (m)", separators=",.",
        legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F3F7EC",
        font=dict(color=VERDE_FLORESTA), title_font=dict(color=VERDE_FLORESTA),
    )

    # Gráfico da previsão
    obs = serie.loc[inicio - pd.Timedelta(days=desde):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs.index, y=obs / 100, mode="lines", name="2026 (dados reais)",
                             line=dict(color=AGUA_RIO, width=2.5)))
    fig.add_trace(go.Scatter(x=proj.index, y=proj / 100, mode="lines", name="Previsão",
                             line=dict(color=VERDE_FOLHA, width=2.5, dash="dash")))
    fig.add_vline(x=inicio, line=dict(color=VERDE_CLARO, dash="dot"))
    fig.add_vline(x=dia, line=dict(color=URUCUM, width=1))
    fig.update_layout(title=f"Dados reais até {inicio:%d/%m} e previsão até {FIM_PROJECAO:%d/%m}",
                      height=460, **layout)
    fig.update_xaxes(gridcolor="#E3EED8")
    fig.update_yaxes(gridcolor="#E3EED8")
    st.plotly_chart(fig, width="stretch")

    # Tabela semanal
    st.subheader("Previsão semana a semana")
    semanas = proj.index[(proj.index - proj.index[0]).days % 7 == 6].union([proj.index[-1]])
    st.dataframe(pd.DataFrame({
        "Data": semanas.strftime("%d/%m/%Y"),
        "Previsão": [metros(v) for v in proj.loc[semanas]],
        f"Variação desde {inicio:%d/%m}": [metros(v - atual, sinal=True) for v in proj.loc[semanas]],
    }), hide_index=True, width="stretch")

    st.info("Limitação: com apenas um ano de dados, o modelo segue a tendência recente e não "
            "conhece o ciclo anual completo do rio (por exemplo, quando a vazante termina e a "
            "enchente começa). Quanto mais longe do último dado real, menos confiável é a previsão.")
    st.caption(f"PROJEÇÃO a partir de dados históricos da {ESTACAO} (telemetria bruta, não "
               "consistida). Não representa medições nem previsão oficial.")
