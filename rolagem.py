# gráfico com barra de rolagem horizontal (estilo windows), para celular e navegador
# o eixo y fica parado à esquerda e só o gráfico rola para os lados

import json
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

MARGEM_TOPO, MARGEM_BAIXO = 10, 45

HTML = """
<style>
  body {{ margin: 0; font-family: "Source Sans Pro", sans-serif; background: {fundo}; }}
  .titulo {{ color: {cor}; font-weight: 700; font-size: 16px; padding: 6px 4px 2px 4px; }}
  .linha {{ display: flex; }}
  #eixo {{ flex: 0 0 60px; }}
  #rolo {{ flex: 1; overflow-x: auto; overflow-y: hidden; scrollbar-width: none; }}
  #rolo::-webkit-scrollbar {{ display: none; }}
  .barra {{ display: flex; height: 18px; margin-left: 60px; user-select: none;
            touch-action: none; }}
  #legenda {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 4px 16px;
              padding: 8px 4px; font-size: 12px; color: {cor}; }}
  .item {{ display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }}
  .traco {{ display: inline-block; width: 22px; }}
  .ponto {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; }}
  .amostra {{ display: inline-block; width: 22px; height: 9px; }}
  .seta {{ width: 18px; height: 18px; background: #D4D0C8; box-sizing: border-box;
           border: 2px outset #FFFFFF; display: flex; align-items: center;
           justify-content: center; cursor: pointer; }}
  .seta:active {{ border-style: inset; }}
  .seta svg {{ width: 8px; height: 8px; }}
  #trilho {{ flex: 1; position: relative;
             background: repeating-conic-gradient(#D4D0C8 0 25%, #FFFFFF 0 50%) 0 0 / 2px 2px; }}
  #polegar {{ position: absolute; top: 0; height: 18px; background: #D4D0C8;
              box-sizing: border-box; border: 2px outset #FFFFFF; cursor: grab; }}
</style>
<div class="titulo">{titulo}</div>
<div class="linha"><div id="eixo"></div><div id="rolo"><div id="grafico"></div></div></div>
<div class="barra">
  <div class="seta" id="esq"><svg viewBox="0 0 8 8"><path d="M6 0 L2 4 L6 8 Z"/></svg></div>
  <div id="trilho"><div id="polegar"></div></div>
  <div class="seta" id="dir"><svg viewBox="0 0 8 8"><path d="M2 0 L6 4 L2 8 Z"/></svg></div>
</div>
<div id="legenda">{legenda}</div>
<script src="https://cdn.plot.ly/plotly-{versao}.min.js"></script>
<script>
const principal = {principal}, eixo = {eixo};
const config = {{displayModeBar: false, scrollZoom: false, doubleClick: false, responsive: false}};
const rolo = document.getElementById("rolo"), trilho = document.getElementById("trilho"),
      polegar = document.getElementById("polegar");
const proporcao = {proporcao};

function desenhar() {{
  // no celular a tela é estreita: o gráfico fica mais baixo e a legenda ganha espaço
  const estreito = window.innerWidth < 600;
  const altura = estreito ? {altura} - 70 : {altura};
  const largura = Math.max(rolo.clientWidth, Math.round(rolo.clientWidth * proporcao),
                           Math.round({total_dias} * 7));
  principal.layout.width = largura;
  principal.layout.height = altura; eixo.layout.height = altura;
  Plotly.newPlot("grafico", principal.data, principal.layout, config);
  Plotly.newPlot("eixo", eixo.data, eixo.layout, {{staticPlot: true}});
  rolo.scrollLeft = {abrir_no_fim} ? rolo.scrollWidth : 0;
  atualizar();
}}

// posição e tamanho do retângulo da barra
function atualizar() {{
  const total = rolo.scrollWidth, visivel = rolo.clientWidth, t = trilho.clientWidth;
  const w = Math.max(24, t * visivel / total);
  const livre = Math.max(total - visivel, 1);
  polegar.style.width = w + "px";
  polegar.style.left = ((t - w) * rolo.scrollLeft / livre) + "px";
}}
rolo.addEventListener("scroll", atualizar);

// arrastar o retângulo (mouse e dedo)
let inicioX = null, inicioScroll = 0;
polegar.addEventListener("pointerdown", e => {{
  inicioX = e.clientX; inicioScroll = rolo.scrollLeft;
  polegar.setPointerCapture(e.pointerId); e.preventDefault();
}});
polegar.addEventListener("pointermove", e => {{
  if (inicioX === null) return;
  const t = trilho.clientWidth - polegar.clientWidth;
  const livre = rolo.scrollWidth - rolo.clientWidth;
  rolo.scrollLeft = inicioScroll + (e.clientX - inicioX) * livre / Math.max(t, 1);
}});
polegar.addEventListener("pointerup", () => {{ inicioX = null; }});

// clique no trilho anda uma tela
trilho.addEventListener("pointerdown", e => {{
  if (e.target !== trilho) return;
  const lado = e.offsetX < polegar.offsetLeft ? -1 : 1;
  rolo.scrollBy({{left: lado * rolo.clientWidth * 0.9}});
}});

// setas: andam um pouco e repetem enquanto estiverem apertadas
function seta(id, lado) {{
  let repetir = null;
  const passo = () => rolo.scrollBy({{left: lado * rolo.clientWidth * 0.1}});
  const el = document.getElementById(id);
  el.addEventListener("pointerdown", e => {{
    e.preventDefault(); passo(); repetir = setInterval(passo, 80);
  }});
  ["pointerup", "pointerleave", "pointercancel"].forEach(ev =>
    el.addEventListener(ev, () => clearInterval(repetir)));
}}
seta("esq", -1); seta("dir", 1);

window.addEventListener("load", desenhar);
let espera = null;
window.addEventListener("resize", () => {{ clearTimeout(espera); espera = setTimeout(desenhar, 200); }});
</script>
"""


# menor e maior valor de y entre os traços, para travar o eixo
def faixa_y(fig) -> tuple[float, float]:
    valores = []
    for tr in fig.data:
        if getattr(tr, "y", None) is None:
            continue
        valores += [float(v) for v in tr.y if v is not None and not math.isnan(float(v))]
    lo, hi = min(valores), max(valores)
    folga = (hi - lo) * 0.06 or 0.5
    return lo - folga, hi + folga


# mostra o gráfico com dias_visiveis na tela e a barra de rolagem embaixo
def mostrar_rolavel(fig, ini: pd.Timestamp, fim: pd.Timestamp, dias_visiveis: int,
                    abrir_no_fim: bool = True, altura: int = 400) -> None:
    import streamlit.components.v1 as components

    titulo = fig.layout.title.text or ""
    fundo = fig.layout.paper_bgcolor or "#FFFFFF"
    cor = fig.layout.title.font.color or "#1B5E20"
    y0, y1 = faixa_y(fig)
    titulo_y = fig.layout.yaxis.title.text or ""

    principal = go.Figure(fig)
    principal.update_layout(title=None, showlegend=False, height=altura, dragmode=False,
                            margin=dict(l=20, r=20, t=MARGEM_TOPO, b=MARGEM_BAIXO))
    passo = max(1, round(dias_visiveis / 8))
    principal.update_xaxes(range=[ini, fim], fixedrange=True, automargin=False, title=None,
                           tick0=ini, dtick=passo * 86_400_000, tickformat="%d/%m",
                           tickangle=0)
    principal.update_yaxes(range=[y0, y1], fixedrange=True, automargin=False,
                           showticklabels=False, title=None)

    eixo = go.Figure(layout=dict(
        height=altura, width=60, margin=dict(l=45, r=0, t=MARGEM_TOPO, b=MARGEM_BAIXO),
        paper_bgcolor=fundo, plot_bgcolor=fundo, separators=fig.layout.separators,
        font=dict(color=cor, size=11), showlegend=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(range=[y0, y1], fixedrange=True, automargin=False, showgrid=False,
                   zeroline=False, title=dict(text=titulo_y, standoff=4)),
    ))

    # legenda em html, que quebra linha sozinha no celular
    itens = []
    for tr in fig.data:
        if tr.showlegend is False or not tr.name:
            continue
        cor_tr = (tr.line.color if tr.line and tr.line.color else None) or \
            (tr.marker.color if tr.marker and isinstance(tr.marker.color, str) else None) or \
            (tr.fillcolor or "#888888")
        modo = tr.mode or "lines"
        if tr.fill:
            simbolo = f'<span class="amostra" style="background:{tr.fillcolor}"></span>'
        elif "markers" in modo and "lines" not in modo:
            forma = "transform:rotate(45deg);border-radius:0" if tr.marker.symbol == "diamond" else ""
            simbolo = f'<span class="ponto" style="background:{cor_tr};{forma}"></span>'
        else:
            estilo = "dashed" if tr.line and tr.line.dash else "solid"
            simbolo = f'<span class="traco" style="border-top:3px {estilo} {cor_tr}"></span>'
        itens.append(f'<span class="item">{simbolo}{tr.name}</span>')
    legenda = "".join(itens)

    html = HTML.format(
        fundo=fundo, cor=cor, titulo=titulo, versao=get_plotlyjs_version(),
        principal=principal.to_json(), eixo=eixo.to_json(), legenda=legenda,
        proporcao=json.dumps((fim - ini).days / dias_visiveis), altura=altura,
        total_dias=(fim - ini).days,
        abrir_no_fim="true" if abrir_no_fim else "false",
    )
    components.html(html, height=altura + 30 + 18 + 50)
