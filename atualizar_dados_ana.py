# baixa as leituras da ana e salva em dados/rio_negro_manaus_<ano>.csv
# uso: python atualizar_dados_ana.py 2026

import calendar
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd

ESTACAO = "14990000"  # rio negro - manaus
URL = ("https://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos"
       "?codEstacao={est}&dataInicio={ini}&dataFim={fim}")
PASTA_DADOS = Path(__file__).parent / "dados"


# baixa um mês
def baixar_mes(ano: int, mes: int, ate: date) -> list[tuple[str, str]]:
    ultimo = min(date(ano, mes, calendar.monthrange(ano, mes)[1]), ate)
    url = URL.format(est=ESTACAO, ini=f"01/{mes:02d}/{ano}", fim=ultimo.strftime("%d/%m/%Y"))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        raiz = ET.fromstring(resp.read())
    return [(r.findtext("DataHora", "").strip(), (r.findtext("Nivel") or "").strip())
            for r in raiz.iter("DadosHidrometereologicos")]


# baixa o ano até a data dada
def baixar_ano(ano: int, ate: date) -> pd.DataFrame:
    linhas = []
    for mes in range(1, ate.month + 1 if ate.year == ano else 13):
        lidas = baixar_mes(ano, mes, ate)
        print(f"{mes:02d}/{ano}: {len(lidas)} leituras")
        linhas += lidas
    df = pd.DataFrame(linhas, columns=["data_hora", "nivel_cm"])
    df["data_hora"] = pd.to_datetime(df["data_hora"])
    df["nivel_cm"] = pd.to_numeric(df["nivel_cm"], errors="coerce")
    return df.dropna().drop_duplicates("data_hora").sort_values("data_hora")


# salva o csv
def main() -> None:
    hoje = date.today()
    ano = int(sys.argv[1]) if len(sys.argv) > 1 else hoje.year
    ate = min(hoje, date(ano, 12, 31))
    df = baixar_ano(ano, ate)

    PASTA_DADOS.mkdir(exist_ok=True)
    destino = PASTA_DADOS / f"rio_negro_manaus_{ano}.csv"
    df.to_csv(destino, index=False, date_format="%Y-%m-%d %H:%M")
    print(f"{len(df)} leituras salvas em {destino} "
          f"({df['data_hora'].min():%d/%m/%Y} a {df['data_hora'].max():%d/%m/%Y})")


if __name__ == "__main__":
    main()
