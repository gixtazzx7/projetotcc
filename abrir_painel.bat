@echo off
title Monitoramento do Nivel do Rio (TCC)

REM Entra na pasta onde este arquivo esta (funciona em qualquer PC e qualquer pasta).
cd /d "%~dp0"

REM Instala as bibliotecas na primeira vez (ou se faltar alguma).
python -c "import streamlit, pandas, numpy, plotly, sklearn" 2>nul
if errorlevel 1 (
    echo Instalando as bibliotecas necessarias... isso so acontece na primeira vez.
    python -m pip install -r requirements.txt
)

echo.
echo Iniciando o painel... o navegador abrira automaticamente.
echo Para encerrar, feche esta janela ou pressione Ctrl+C.
echo.

python -m streamlit run simulacao_nivel_rio.py

REM Se der erro, mantem a janela aberta para ler a mensagem
if errorlevel 1 (
    echo.
    echo Ocorreu um erro. Verifique se o Python esta instalado e marcado no PATH
    echo e tente instalar as bibliotecas manualmente:
    echo     python -m pip install -r requirements.txt
    pause
)
