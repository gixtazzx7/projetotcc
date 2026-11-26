# assistente simples: responde perguntas em português sobre o nível do rio
# usando os dados reais da ana e a previsão até 31/12 (sem ia externa, só regras)

import difflib
import re
import unicodedata

import pandas as pd

from projeto import ESTACAO, REGISTROS_PORTO, VERDE_FLORESTA, VERDE_FOLHA, metros

ANO = 2026

MESES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}
NOMES_MESES = {n: m for m, n in MESES.items()}
NOMES_MESES[3] = "março"

TRIMESTRES = {"primeiro": 1, "1": 1, "segundo": 2, "2": 2, "terceiro": 3, "3": 3, "quarto": 4, "4": 4}

PALAVRAS_MAXIMO = ("maior", "maximo", "mais alto", "cheia", "pico", "subir mais")
PALAVRAS_MINIMO = ("menor", "minimo", "mais baixo", "seca", "vazante", "baixar mais")
PALAVRAS_FUTURO = ("vai ", "previsao", "previsto", "proxim", "futuro", "sera", "ira ", "daqui",
                   "deve ", "vao ")

EXEMPLOS = [
    "Como vai estar o rio em dezembro de 2026?",
    "Qual foi o maior nível do ano?",
    "Como estava o rio em 15/09?",
    "O rio vai subir até o fim do ano?",
    "Compare junho com dezembro",
    "Quando o rio vai estar mais baixo?",
]


ABREVIACOES = {"jan": "janeiro", "fev": "fevereiro", "abr": "abril", "jun": "junho",
               "jul": "julho", "ago": "agosto", "set": "setembro", "out": "outubro",
               "nov": "novembro", "dez": "dezembro"}
NUMEROS = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5,
           "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10, "quinze": 15, "trinta": 30}


# corrige uma palavra parecida com nome de mês (ex.: "dezenbro", "fevereiro")
def corrigir_palavra(p: str) -> str:
    if p in ABREVIACOES:
        return ABREVIACOES[p]
    if len(p) >= 5 and p not in MESES:
        parecida = difflib.get_close_matches(p, MESES, n=1, cutoff=0.8)
        if parecida:
            return parecida[0]
    return p


# tira acentos, deixa em minúsculo e corrige os meses
def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    sem_acento = re.sub(r"[^\w/ ]", " ", sem_acento.lower())
    palavras = [corrigir_palavra(p) for p in sem_acento.split()]
    # "dez dias" é número, não dezembro
    palavras = ["dez" if p == "dezembro" and o == "dez" and i + 1 < len(palavras)
                and palavras[i + 1].startswith(("dia", "semana")) else p
                for i, (p, o) in enumerate(zip(palavras, sem_acento.split()))]
    return " ".join(palavras)


def numero(texto: str) -> int:
    return int(texto) if texto.isdigit() else NUMEROS.get(texto, 1)


def data_br(d: pd.Timestamp) -> str:
    return f"{d:%d/%m/%Y}"


def tem(texto: str, palavras) -> bool:
    return any(p in texto for p in palavras)


# descobre na pergunta qual período e o que a pessoa quer saber
# devolve (inicio, fim, rótulo, foco) ou um texto de erro
def interpretar(pergunta: str, ultimo_real: pd.Timestamp, fim_previsao: pd.Timestamp,
                hoje: pd.Timestamp | None = None):
    t = normalizar(pergunta)
    foco = "maximo" if tem(t, PALAVRAS_MAXIMO) else "minimo" if tem(t, PALAVRAS_MINIMO) else "resumo"
    if re.search(r"\b(subi|desc|baix|tendencia|enche|seca)", t) and foco == "resumo":
        foco = "tendencia"

    anos = [int(a) for a in re.findall(r"\b(19\d\d|20\d\d)\b", t)]
    outros = [a for a in anos if a != ANO]
    if outros:
        return (f"O painel só tem dados de {ANO} (dados reais até {data_br(ultimo_real)} e "
                f"previsão até {data_br(fim_previsao)}). Não consigo responder sobre {outros[0]}.")

    # data numérica: 15/09 ou 15/09/2026
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        if m.group(3) and int(m.group(3)) % 100 != ANO % 100:
            return f"O painel só tem dados de {ANO}."
        try:
            d = pd.Timestamp(year=ANO, month=mes, day=dia)
        except ValueError:
            return f"Não entendi a data {m.group(0)}."
        return d, d, data_br(d), foco

    # "25 de dezembro"
    nomes = "|".join(MESES)
    m = re.search(rf"\b(\d{{1,2}})\s*(?:de\s+)?({nomes})\b", t)
    if m:
        try:
            d = pd.Timestamp(year=ANO, month=MESES[m.group(2)], day=int(m.group(1)))
        except ValueError:
            return f"Não entendi a data {m.group(0)}."
        return d, d, data_br(d), foco

    if "natal" in t:
        d = pd.Timestamp(year=ANO, month=12, day=25)
        return d, d, f"o Natal ({data_br(d)})", foco
    if re.search(r"\b(fim|final|virada) do ano\b|reveillon|ano novo|31 de dezembro", t):
        return fim_previsao, fim_previsao, data_br(fim_previsao), foco

    if re.search(r"\b(hoje|agora|atual|atualmente|neste momento)\b", t):
        hoje = (hoje or pd.Timestamp.today()).normalize()
        d = hoje if ultimo_real < hoje <= fim_previsao else ultimo_real
        return d, d, f"hoje ({data_br(d)})" if d == hoje else data_br(d), "hoje"
    if re.search(r"\bultim[oa] (dado|leitura|medicao|registro)\b", t):
        return ultimo_real, ultimo_real, data_br(ultimo_real), foco

    hoje = (hoje or pd.Timestamp.today()).normalize()
    if not (pd.Timestamp(year=ANO, month=1, day=1) <= hoje <= fim_previsao):
        hoje = ultimo_real
    for palavra, desloc in (("anteontem", -2), ("ontem", -1), ("amanha", 1)):
        if re.search(rf"\b{palavra}\b", t) and not re.search(r"\bdepois de amanha\b", t):
            d = hoje + pd.Timedelta(days=desloc)
            return d, d, f"{palavra.replace('amanha', 'amanhã')} ({data_br(d)})", foco
    if re.search(r"\bdepois de amanha\b", t):
        d = hoje + pd.Timedelta(days=2)
        return d, d, f"depois de amanhã ({data_br(d)})", foco

    m = re.search(r"\b(?:daqui a|daqui|em|dentro de) (\d+|\w+) (dia|semana|mes)", t)
    if m and m.group(1) in NUMEROS or (m and m.group(1).isdigit()):
        n = numero(m.group(1)) * {"dia": 1, "semana": 7, "mes": 30}[m.group(2)]
        d = hoje + pd.Timedelta(days=n)
        return d, d, f"daqui a {n} dias ({data_br(d)})", foco

    m = re.search(r"\bultim[oa]s (\d+|\w+) (dias|semanas)", t)
    if m:
        n = numero(m.group(1)) * (7 if m.group(2) == "semanas" else 1)
        return (ultimo_real - pd.Timedelta(days=n - 1), ultimo_real,
                f"os últimos {n} dias com dados reais", foco)
    if re.search(r"\bsemana passada\b", t):
        return (hoje - pd.Timedelta(days=7), hoje - pd.Timedelta(days=1),
                "a semana passada", foco)

    mes_hoje = pd.Timestamp(year=hoje.year, month=hoje.month, day=1)
    relativos = {r"\b(mes que vem|proximo mes)\b": 1, r"\b(mes passado|ultimo mes)\b": -1,
                 r"\b(este|esse|neste|nesse) mes\b|\bmes atual\b": 0}
    for padrao, desloc in relativos.items():
        if re.search(padrao, t):
            ini = mes_hoje + pd.DateOffset(months=desloc)
            if ini.year != ANO:
                return f"O painel só tem dados de {ANO}."
            return (ini, ini + pd.offsets.MonthEnd(),
                    f"{NOMES_MESES[ini.month]} de {ANO}", foco)

    m = re.search(r"\b(primeiro|segundo|1|2)[oº]? semestre", t)
    if m:
        sem = 1 if m.group(1) in ("primeiro", "1") else 2
        ini = pd.Timestamp(year=ANO, month=6 * sem - 5, day=1)
        return ini, ini + pd.DateOffset(months=6) - pd.Timedelta(days=1), \
            f"o {sem}º semestre de {ANO}", foco

    # dois meses: comparação
    nomes_achados = list(dict.fromkeys(re.findall(rf"\b({'|'.join(MESES)})\b", t)))
    if len(nomes_achados) >= 2:
        return "comparar", MESES[nomes_achados[0]], MESES[nomes_achados[1]]

    if re.search(r"\bproxim[oa]s? (\w+ )?(dias|semanas?)\b|semana que vem", t):
        n = 7
        m = re.search(r"proxim[oa]s? (\w+) (dias|semanas)", t)
        if m:
            n = numero(m.group(1)) * (7 if m.group(2) == "semanas" else 1)
        ini = ultimo_real + pd.Timedelta(days=1)
        fim = min(ultimo_real + pd.Timedelta(days=n), fim_previsao)
        return ini, fim, f"os próximos {(fim - ini).days + 1} dias após o último dado real", foco

    # mês, com "início/meio/fim de"
    m = re.search(rf"\b(?:(inicio|comeco|meio|metade|fim|final) (?:de |do mes de )?)?({nomes})\b", t)
    if m:
        mes = MESES[m.group(2)]
        ini = pd.Timestamp(year=ANO, month=mes, day=1)
        fim = ini + pd.offsets.MonthEnd()
        rotulo = f"{NOMES_MESES[mes]} de {ANO}"
        parte = m.group(1)
        if parte in ("inicio", "comeco"):
            fim = ini + pd.Timedelta(days=9)
            rotulo = f"o início de {rotulo}"
        elif parte in ("meio", "metade"):
            ini, fim = ini + pd.Timedelta(days=10), ini + pd.Timedelta(days=19)
            rotulo = f"o meio de {rotulo}"
        elif parte in ("fim", "final"):
            ini = ini + pd.Timedelta(days=20)
            rotulo = f"o fim de {rotulo}"
        return ini, fim, rotulo, foco

    m = re.search(r"\b(primeiro|segundo|terceiro|quarto|[1-4])[oº]? trimestre", t)
    if m:
        tri = TRIMESTRES[m.group(1)]
        ini = pd.Timestamp(year=ANO, month=3 * tri - 2, day=1)
        fim = ini + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        return ini, fim, f"o {tri}º trimestre de {ANO}", foco

    explicacao = explicar(t, ultimo_real, fim_previsao)
    if explicacao:
        return explicacao

    # sem período: pergunta sobre o futuro usa a previsão; o resto usa o ano todo
    if foco == "tendencia" and re.search(r"\b(esta|ta|anda|continua)\b", t):
        ini, fim = ultimo_real - pd.Timedelta(days=13), ultimo_real + pd.Timedelta(days=14)
        return ini, fim, "as duas semanas antes e as duas depois do último dado real", foco
    if tem(t, PALAVRAS_FUTURO):
        ini = ultimo_real + pd.Timedelta(days=1)
        return ini, fim_previsao, f"o período previsto ({data_br(ini)} a {data_br(fim_previsao)})", foco
    if foco != "resumo" or re.search(r"\b(rio|nivel|ano|2026|cota)\b", t):
        return pd.Timestamp(year=ANO, month=1, day=1), fim_previsao, f"o ano de {ANO}", foco

    return None


# respostas fixas sobre o painel
def explicar(t: str, ultimo_real: pd.Timestamp, fim_previsao: pd.Timestamp) -> str | None:
    if re.search(r"\b(fonte|de onde|origem|quem mede|ana|estacao|sensor|telemetria)\b", t):
        return (f"Os dados são da estação {ESTACAO}, da Agência Nacional de Águas (ANA). São "
                f"leituras de telemetria a cada 15 minutos, de 01/01 a {data_br(ultimo_real)}, "
                "sem consistência oficial. O painel tira as leituras suspeitas e usa a média de "
                "cada dia.")
    if re.search(r"\b(como funciona|modelo|regressao|calcul|confiavel|confianca|erro|precis)", t):
        return ("A previsão usa regressão linear: o modelo estima a média de um dia a partir das "
                "médias dos 7 dias anteriores, e cada dia previsto vira entrada do dia seguinte, "
                f"até {data_br(fim_previsao)}. No teste com os últimos 7 dias reais o erro médio "
                "ficou em torno de 13 cm. Como só há um ano de dados, o modelo segue a tendência "
                "recente e não conhece o ciclo anual do rio, então o erro cresce com a distância.")
    if re.search(r"\b(suspeit|filtro|mad|mediana|outlier|erro de leitura)", t):
        return ("Uma leitura é marcada como suspeita quando fica longe demais da mediana das "
                "leituras vizinhas (janela de 13 leituras). O limite é k vezes o MAD, o desvio "
                "mediano da janela, com k = 5 por padrão. Essas leituras saem das médias diárias.")
    if re.search(r"\b(porto|regua)\b", t):
        return ("Os registros do Porto de Manaus aparecem só para comparação. A régua do porto é "
                "diferente da régua da estação da ANA, então os valores ficam próximos mas não "
                "iguais.")
    if re.search(r"^(oi|ola|bom dia|boa tarde|boa noite|e ai|opa)\b", t):
        return ("Olá. Pergunte sobre o nível do Rio Negro em Manaus em 2026, citando um mês, uma "
                "data ou um período. Por exemplo: \"" + EXEMPLOS[0] + "\"")
    if re.search(r"\b(obrigad|valeu)", t):
        return "De nada."
    return None


# compara a média de dois meses
def comparar_meses(m1: int, m2: int, completa: pd.Series, ultimo_real: pd.Timestamp) -> str:
    frases, medias = [], []
    for m in (m1, m2):
        ini = pd.Timestamp(year=ANO, month=m, day=1)
        s = completa.loc[ini:ini + pd.offsets.MonthEnd()]
        if s.empty:
            return f"Não há dados nem previsão para {NOMES_MESES[m]} de {ANO}."
        tipo = ("previsão" if s.index[0] > ultimo_real
                else "dados reais" if s.index[-1] <= ultimo_real else "dados reais e previsão")
        medias.append(s.mean())
        frases.append(f"{NOMES_MESES[m]} tem média de {metros(s.mean())}, entre "
                      f"{metros(s.min())} e {metros(s.max())} ({tipo})")
    dif = medias[1] - medias[0]
    comp = "mais alto" if dif > 0 else "mais baixo"
    return (f"Em {ANO}, {frases[0]}. Já {frases[1]}. Na média, o rio em {NOMES_MESES[m2]} fica "
            f"{metros(abs(dif))} {comp} que em {NOMES_MESES[m1]}.")


# registros do porto no período (outra régua, só para comparação)
def comparar_porto(ini: pd.Timestamp, fim: pd.Timestamp) -> str:
    regs = [(pd.Timestamp(d), v) for d, v in REGISTROS_PORTO
            if ini <= pd.Timestamp(d) <= fim]
    if not regs:
        return ""
    itens = ", ".join(f"{v:.2f} m em {d:%d/%m}".replace(".", ",") for d, v in regs)
    return (f" Para comparação, o Porto de Manaus registrou {itens} (régua diferente, "
            "então os valores não batem exatamente com os da ANA).")


# frase sobre subida ou descida (passado para dados reais, futuro para previsão)
def tendencia(s: pd.Series, previsto: bool) -> str:
    var = s.iloc[-1] - s.iloc[0]
    if abs(var) < 5:
        return "deve ficar praticamente estável" if previsto else "ficou praticamente estável"
    if previsto:
        verbo = "deve subir" if var > 0 else "deve descer"
    else:
        verbo = "subiu" if var > 0 else "desceu"
    por_dia = abs(var) / max(len(s) - 1, 1)
    return f"{verbo} {metros(abs(var))}, cerca de {por_dia:.1f} cm por dia".replace(".", ",")


# monta o texto da resposta
def responder(pergunta: str, serie: pd.Series, proj: pd.Series, erro_cm: float,
              hoje: pd.Timestamp | None = None) -> str:
    ultimo_real, fim_previsao = serie.index[-1], proj.index[-1]
    inicio_dados = serie.index[0]
    r = interpretar(pergunta, ultimo_real, fim_previsao, hoje)

    if r is None:
        return ("Não entendi a pergunta. Eu respondo sobre o nível do Rio Negro em 2026: cite um "
                "mês, uma data (15/09, amanhã, daqui a 10 dias), um trimestre ou semestre, ou "
                "dois meses para comparar. Também explico de onde vêm os dados e como funciona a "
                "previsão.")
    if isinstance(r, str):
        return r
    if r[0] == "comparar":
        return comparar_meses(r[1], r[2], pd.concat([serie, proj]), ultimo_real)

    ini, fim, rotulo, foco = r
    if fim < inicio_dados or ini > fim_previsao:
        return (f"Só tenho dados reais de {data_br(inicio_dados)} a {data_br(ultimo_real)} e "
                f"previsão até {data_br(fim_previsao)}. O período pedido fica fora disso.")
    ini, fim = max(ini, inicio_dados), min(fim, fim_previsao)

    completa = pd.concat([serie, proj])
    s = completa.loc[ini:fim]
    real = s.loc[:ultimo_real]
    prev = s.loc[ultimo_real + pd.Timedelta(days=1):]
    aviso = (f" Os valores depois de {data_br(ultimo_real)} são previsão de um modelo de "
             f"regressão linear (erro médio de ±{erro_cm:.0f} cm no teste de 7 dias) e ficam menos "
             "confiáveis quanto mais longe do último dado real.")
    porto = comparar_porto(ini, fim + pd.Timedelta(days=1)) if (fim - ini).days <= 92 else ""

    # um dia só
    if ini == fim:
        v = s.iloc[0]
        if prev.empty:
            return (f"Em {data_br(ini)} o nível médio do rio foi {metros(v)} "
                    f"(dado real da {ESTACAO})." + porto)
        texto = (f"A previsão para {rotulo} é {metros(v)}. O último dado real é de "
                 f"{data_br(ultimo_real)}, com {metros(serie.iloc[-1])}, então a diferença "
                 f"prevista é de {metros(v - serie.iloc[-1], sinal=True)}.")
        if foco == "hoje":
            texto = "Não há medição de hoje no painel. " + texto
        return texto + aviso + porto

    # período com vários dias
    def quando(d: pd.Timestamp) -> str:
        return f"em {d:%d/%m}" + (" (previsto)" if d > ultimo_real else "")

    partes = []
    if prev.empty:
        partes.append(f"Para {rotulo} há só dados reais. O rio começou em {metros(s.iloc[0])} "
                      f"e terminou em {metros(s.iloc[-1])}, ou seja, {tendencia(s, False)}.")
    elif real.empty:
        partes.append(f"Esse período fica todo depois do último dado real "
                      f"({data_br(ultimo_real)}), então a resposta vem da previsão. "
                      f"Para {rotulo}, o rio deve começar em {metros(s.iloc[0])} e terminar em "
                      f"{metros(s.iloc[-1])}, ou seja, {tendencia(s, True)}.")
    else:
        partes.append(f"Para {rotulo} há dados reais até {data_br(ultimo_real)} e previsão "
                      f"depois disso. Nos dados reais o rio foi de {metros(real.iloc[0])} a "
                      f"{metros(real.iloc[-1])}. Pela previsão, deve chegar a "
                      f"{metros(prev.iloc[-1])} em {prev.index[-1]:%d/%m}.")

    frase_max = f"o maior nível é {metros(s.max())} {quando(s.idxmax())}"
    frase_min = f"o menor é {metros(s.min())} {quando(s.idxmin())}"
    if foco == "maximo":
        partes.append(f"No período, {frase_max}. Para comparar, {frase_min}.")
    elif foco == "minimo":
        partes.append(f"No período, o menor nível é {metros(s.min())} {quando(s.idxmin())}. "
                      f"Para comparar, o maior é {metros(s.max())} {quando(s.idxmax())}.")
    else:
        partes.append(f"No período, {frase_max} e {frase_min}. "
                      f"A média diária fica em {metros(s.mean())}.")
    if foco == "tendencia" and not prev.empty and not real.empty:
        partes.append(f"Do último dado real até {prev.index[-1]:%d/%m}, o rio "
                      f"{tendencia(pd.concat([real.iloc[-1:], prev]), True)}.")

    # posição em relação ao ano
    if (s.index[0], s.index[-1]) != (completa.index[0], completa.index[-1]):
        partes.append(f"No ano inteiro, a cheia chegou a {metros(completa.max())} "
                      f"{quando(completa.idxmax())} e o nível mais baixo é "
                      f"{metros(completa.min())} {quando(completa.idxmin())}.")

    texto = " ".join(partes)
    if not prev.empty:
        texto += aviso
    return texto + porto


# caixa de pergunta em verde escuro
CSS_CAIXA_PERGUNTA = f"""
<style>
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {{
    background-color: {VERDE_FLORESTA} !important;
    border-color: {VERDE_FLORESTA} !important;
}}
[data-testid="stChatInput"] textarea {{
    background-color: {VERDE_FLORESTA} !important;
    color: #FFFFFF !important;
    caret-color: #FFFFFF;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: #C8E6C9 !important; }}
[data-testid="stChatInput"] button {{ background-color: {VERDE_FOLHA} !important; }}
[data-testid="stChatInput"] button svg {{ fill: #FFFFFF; color: #FFFFFF; }}
</style>
"""


# tela do assistente no streamlit
def mostrar_assistente() -> None:
    import streamlit as st

    from projecao_nivel_rio import calcular_previsao
    from projeto import erro_da_previsao

    serie, proj = st.cache_data(calcular_previsao)()
    erro = st.cache_data(erro_da_previsao)(serie, 7)

    st.title("pergunte sobre o nivel do rio")

    if "conversa" not in st.session_state:
        st.session_state.conversa = []

    def apagar_ultima():
        st.session_state.conversa = st.session_state.conversa[:-2]

    def limpar():
        st.session_state.conversa = []

    st.write("Exemplos:")
    colunas = st.columns(3)
    clicada = None
    for i, ex in enumerate(EXEMPLOS):
        if colunas[i % 3].button(ex, key=f"ex{i}", width="stretch"):
            clicada = ex

    st.markdown(CSS_CAIXA_PERGUNTA, unsafe_allow_html=True)
    pergunta = st.chat_input("como vai estar o rio em dezembro de 2026?") or clicada
    if pergunta:
        resposta = responder(pergunta, serie, proj, erro)
        st.session_state.conversa += [("user", pergunta), ("assistant", resposta)]

    for papel, texto in st.session_state.conversa:
        with st.chat_message(papel):
            st.write(texto)

    if st.session_state.conversa:
        b1, b2, _ = st.columns([1, 1, 2])
        b1.button("Apagar última pergunta", on_click=apagar_ultima, width="stretch")
        b2.button("Limpar conversa", on_click=limpar, width="stretch")
