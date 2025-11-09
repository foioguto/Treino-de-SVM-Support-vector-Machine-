# Treino de SVM (Support Vector Machine)  
Projeto de classificação de risco cardíaco usando SVM  

## 📋 Visão Geral  
Este repositório contém o notebook de análise exploratória, pré-processamento, treinamento e avaliação de modelos de machine learning — em especial o algoritmo scikit‑learn `SVC` — aplicado a um problema de previsão de presença de doença cardíaca.

## 🧰 Estrutura do Repositório  
- `analysis.ipynb` : Notebook com todo o fluxo — importação dos dados, limpeza, normalização, treino, avaliação e conclusões.  
- `analysis.py` : Versão em script Python (exportada a partir do notebook) para facilitar execução automatizada ou produção.  
- `requirements.txt` : Dependências do projeto.  
- `README.md` : Esse arquivo de documentação.  
- `.gitignore` : Itens a ignorar no controle de versão (ex.: `.ipynb_checkpoints/`, datasets brutos, caches).

## 🚀 Como rodar  
1. Crie e ative um ambiente virtual (recomendado):  
   ```bash
   python -m venv .venv  
   source .venv/bin/activate  # Linux/macOS  
   .venv\Scripts\activate     # Windows  
