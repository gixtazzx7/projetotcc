@echo off
title Simulacao - Nivel do Rio (TCC)

REM Pasta do projeto (necessario para o Streamlit achar .streamlit e imagens).
REM Se mover o projeto de lugar, altere este caminho.
cd /d "C:\trabalhotcc"

echo Iniciando o painel... o navegador abrira automaticamente.
echo Para encerrar, feche esta janela ou pressione Ctrl+C.
echo.

python -m streamlit run simulacao_nivel_rio.py

REM Se der erro, mantem a janela aberta para ler a mensagem
if errorlevel 1 (
    echo.
    echo Ocorreu um erro. Verifique se o Python e as dependencias estao instalados:
    echo     python -m pip install -r requirements.txt
    pause
)
