# Monitoramento do nível do Rio Negro (TCC)

Painel web que mostra, filtra e prevê o nível do Rio Negro em Manaus com dados
reais de 2026 da ANA (estação 14990000, de 01/01 a 17/11/2026). Demonstra a etapa
de servidor e painel da arquitetura IoT proposta (sensor ultrassônico → LoRaWAN →
servidor → painel).

## Como rodar em qualquer computador (Windows)

1. Instale o **Python 3.10 ou mais novo** em <https://www.python.org/downloads/>.
   Na instalação, marque **"Add Python to PATH"**.
2. Baixe o projeto: no GitHub, clique em **Code → Download ZIP** e extraia a pasta
   (ou use `git clone https://github.com/gixtazzx7/projetotcc.git`).
3. Dê dois cliques em **`abrir_painel.bat`**.
   - Na primeira vez, ele instala as bibliotecas (precisa de internet, leva alguns minutos).
   - Depois, o painel abre sozinho no navegador em <http://localhost:8501>.
4. Para fechar, feche a janela preta (ou aperte Ctrl+C nela).

Depois da primeira instalação, o painel funciona **sem internet**: os dados estão
no arquivo `dados/rio_negro_manaus_2026.csv`.

### Pelo terminal (Windows, Mac ou Linux)

```bash
python -m pip install -r requirements.txt
python -m streamlit run simulacao_nivel_rio.py
```

## Arquivos

| Arquivo ou pasta | Papel |
| --- | --- |
| `simulacao_nivel_rio.py` | Painel principal (filtro, trimestres, previsão curta, telas) |
| `projecao_nivel_rio.py` | Seção "Previsão (até 31/12)" do painel |
| `atualizar_dados_ana.py` | Baixa de novo os dados da ANA (`python atualizar_dados_ana.py 2026`) |
| `dados/` | Leituras reais de 2026 usadas pelo painel |
| `imagens/` | Logo e foto de fundo |
| `.streamlit/config.toml` | Cores do tema |
| `abrir_painel.bat` | Abre o painel com dois cliques no Windows |

Os dados são históricos, não medições em tempo real.
