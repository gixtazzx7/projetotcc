# seção de previsão até 31/12 (dados de 2026)

import pandas as pd

from projeto import (
    URUCUM, VERDE_FLORESTA, VERDE_FOLHA,
    carregar_leituras, detectar_suspeitas, metros, niveis_diarios, prever_nivel,
)

FIM_PROJECAO = pd.Timestamp("2026-12-31")


# série real e previsão até 31/12
def calcular_previsao() -> tuple[pd.Series, pd.Series]:
    serie = niveis_diarios(detectar_suspeitas(carregar_leituras()))
    passos = (FIM_PROJECAO - serie.index[-1]).days
    datas = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), FIM_PROJECAO, freq="D")
    return serie, pd.Series(prever_nivel(serie, passos), index=datas)


# mostra a seção de previsão
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

    # gráfico só com a previsão de dezembro
    ini_dez = pd.Timestamp(year=FIM_PROJECAO.year, month=12, day=1)
    dez = proj.loc[ini_dez:FIM_PROJECAO]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dez.index, y=dez / 100, mode="lines+markers", name="Previsão",
                             line=dict(color=VERDE_FOLHA, width=2.5, dash="dash"),
                             marker=dict(size=5)))
    if ini_dez <= dia <= FIM_PROJECAO:
        fig.add_vline(x=dia, line=dict(color=URUCUM, width=1))
    fig.update_layout(title=f"Previsão para dezembro de {FIM_PROJECAO.year}", height=460, **layout)
    fig.update_xaxes(gridcolor="#E3EED8", dtick="D2", tickformat="%d/%m")
    fig.update_yaxes(gridcolor="#E3EED8")
    from rolagem import mostrar_rolavel
    mostrar_rolavel(fig, ini_dez, FIM_PROJECAO + pd.Timedelta(days=1), 15, abrir_no_fim=False)

    # tabela semanal
    st.subheader("Previsão semana a semana")
    semanas = proj.index[(proj.index - proj.index[0]).days % 7 == 6].union([proj.index[-1]])
    st.dataframe(pd.DataFrame({
        "Data": semanas.strftime("%d/%m/%Y"),
        "Previsão": [metros(v) for v in proj.loc[semanas]],
        f"Variação desde {inicio:%d/%m}": [metros(v - atual, sinal=True) for v in proj.loc[semanas]],
    }), hide_index=True, width="stretch")
