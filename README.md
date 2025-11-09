# MLOps - Modelo de Predição de Evasão de Clientes (Churn)

## 📋 Descrição do Projeto

Este projeto implementa um pipeline completo de Machine Learning para predição de evasão de clientes bancários (churn). O sistema utiliza Random Forest com balanceamento SMOTE e otimização de limiar de decisão para classificar o risco de evasão de clientes.

## 🎯 Objetivo

Desenvolver um modelo preditivo que identifica clientes com maior probabilidade de deixar o banco, permitindo ações preventivas de retenção.

## 📁 Estrutura do Projeto

```
mlops-docker-template/
├── data/                           # Dados do projeto
│   ├── raw/                        # Dados brutos
│   │   ├── dados_treino.csv        # Dataset inicial de treinamento
│   │   ├── dados_novos_1.csv       # Novos dados para predição/retreino
│   │   └── dados_novos_2.csv       # Novos dados para predição/retreino
│   └── docs/                       # Documentação dos dados
│       └── bank_churn_dict.csv     # Dicionário de dados
│
├── src/                            # Código fonte
│   ├── treinamento.py              # Script de treinamento do modelo
│   ├── predicao.py                 # Script de predição
│   └── retreinamento.py            # Script auxiliar de retreinamento
│
├── models/                         # Modelos treinados
│   └── pipeline_modelo_treinado.joblib
│
├── outputs/                        # Resultados e métricas
│   ├── metricas_desempenho_evasao.csv
│   └── predicoes.csv
│
├── .gitignore                      # Arquivos ignorados pelo Git
├── requirements.txt                # Dependências do projeto
├── README.md                       # Este arquivo
└── README.txt                      # Documentação original
```

## 🔧 Tecnologias Utilizadas

- **Python 3.x**
- **pandas** - Manipulação de dados
- **numpy** - Operações numéricas
- **scikit-learn** - Algoritmos de ML e pré-processamento
- **imbalanced-learn** - Tratamento de classes desbalanceadas (SMOTE)
- **joblib** - Serialização de modelos

## 📊 Variáveis do Dataset

Consulte o arquivo `data/docs/bank_churn_dict.csv` para descrição detalhada das variáveis.

**Principais features:**
- Variáveis numéricas: `idade`, `saldo_conta`, `salario_estimado`, `escore_credito`
- Variáveis categóricas: `pais`, `genero`, `cartao_credito`
- Variáveis ordinais: `anos_cliente`, `numero_produtos`
- Target: `saiu` (0 = não saiu, 1 = saiu)

## 🚀 Como Usar

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/jaisonschmidt/mlops-docker-template.git
cd mlops-docker-template
```

2. Crie e ative um ambiente virtual Python:

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

> **Nota:** Para desativar o ambiente virtual, use o comando `deactivate`

### Treinamento do Modelo

Execute o script de treinamento para criar o modelo inicial:

```bash
python src/treinamento.py
```

**Entradas:**
- `data/raw/dados_treino.csv`

**Saídas:**
- `models/pipeline_modelo_treinado.joblib` - Pipeline completo do modelo
- `outputs/metricas_desempenho_evasao.csv` - Métricas de desempenho

### Predição em Novos Dados

Execute o script de predição para classificar novos clientes:

```bash
python src/predicao.py
```

**Entradas:**
- `data/raw/dados_novos_1.csv` (ou `dados_novos_2.csv`)
- `models/pipeline_modelo_treinado.joblib`

**Saídas:**
- `outputs/predicoes.csv` - Probabilidades e classificação de risco

**Classificação de Risco:**
- 🟢 **Risco muito alto**: Probabilidade > 90%
- 🟡 **Risco alto**: Probabilidade > 70%
- 🟠 **Risco moderado**: Probabilidade > 50%
- 🔴 **Risco baixo**: Probabilidade < 50%

## 🔄 Retreinamento do Modelo

Para retreinar o modelo com novos dados:

### Opção 1: Usando o script de retreinamento (Recomendado)

```bash
# Retreinar com dados_novos_1.csv
python src/retreinamento.py

# Ou especificar outro arquivo
python src/retreinamento.py data/raw/dados_novos_2.csv

# Depois executar o treinamento
python src/treinamento.py
```

### Opção 2: Retreino manual

```bash
# No terminal Python ou script
import pandas as pd

# Combinar datasets
dados_treino = pd.read_csv("data/raw/dados_treino.csv")
dados_novos = pd.read_csv("data/raw/dados_novos_1.csv")
dados_combinados = pd.concat([dados_treino, dados_novos], ignore_index=True)
dados_combinados.to_csv("data/raw/dados_treino.csv", index=False)

# Executar treinamento
```

```bash
python src/treinamento.py
```

### Opção 3: Retreino incremental (em lotes)

É possível fazer retreino em lotes menores (ex: 32 em 32) para simular aprendizado contínuo.

## 📈 Pipeline de ML

O modelo implementa o seguinte pipeline:

1. **Imputação de valores ausentes**
   - KNN Imputer para variáveis numéricas
   - Most Frequent para categóricas

2. **Transformações**
   - Power Transform + Standard Scaler (numéricas)
   - One-Hot Encoding (categóricas)
   - Target Encoding (ordinais)
   - Polynomial Features (interações)

3. **Balanceamento**
   - SMOTE para classes desbalanceadas

4. **Modelo**
   - Random Forest Classifier (1000 árvores)
   - Tuned Threshold com otimização F2-score

## 📊 Métricas de Avaliação

O modelo é avaliado usando:
- **F1-Score** e **F2-Score** (weighted)
- **Precisão** e **Recall** (weighted)
- **AUC-ROC**

Classificação de desempenho:
- ✅ Excelente: > 0.90
- 👍 Bom: > 0.80
- ⚠️ Aceitável: > 0.70
- ⚡ Fraco: > 0.60
- ❌ Ruim: < 0.60

## 🐳 Docker

Este projeto inclui um tutorial completo sobre como conteinerizar e executar a aplicação com Docker.

📖 **Consulte o guia detalhado:** [`tutorial/README.md`](tutorial/README.md)

O tutorial cobre:
- Por que usar Docker neste projeto
- Criação de Dockerfile simplificado
- Comandos para build e execução de containers
- Boas práticas e próximos passos

## 👥 Contribuindo

Contribuições são bem-vindas! Sinta-se livre para abrir issues ou pull requests.

## 📝 Licença

Este projeto está sob licença MIT.

## 📧 Contato

Para dúvidas ou sugestões, abra uma issue no repositório.

---

**Desenvolvido com ❤️ para aprendizado em MLOps**