# Tech Challenge Fase 2 — Sistema de Recomendação com MLOps

## Objetivo

Este projeto implementa um pipeline completo de Machine Learning para um sistema de recomendação utilizando o dataset **MovieLens 100K**.

Além do treinamento de modelos, o projeto demonstra boas práticas de **MLOps**, incluindo:

- versionamento do pipeline com DVC;
- rastreamento de experimentos com MLflow;
- containerização com Docker;
- comparação entre modelos baseline e Deep Learning;
- avaliação automática com métricas de classificação e de ranking;
- registro do melhor modelo;
- comparação controlada e mensurável entre uma versão baseline e uma versão otimizada do pipeline (ver seção [Evolução do projeto](#evolução-do-projeto-baseline-v1-vs-melhorado-v2)).

O objetivo não é apenas construir um modelo, mas disponibilizar um fluxo reproduzível de treinamento, com cada melhoria de engenharia rastreada e justificável por evidências no MLflow.

---

# Arquitetura

```
                 MovieLens 100K
                        │
                        ▼
              Pré-processamento
                        │
                        ▼
             Engenharia de Features
        (agregações + índices para embeddings)
                        │
                        ▼
                 Dataset Final
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
 Logistic Regression Random Forest   MLP (embeddings +
         │              │         BatchNorm + Dropout)
         │              │       Busca de hiperparâmetros
         └──────────────┼──────────────┘
                        ▼
             Comparação de Métricas
        (classificação + ranking, no MLflow)
                        │
                        ▼
               Melhor modelo (maior F1)
                        │
                        ▼
             Registro no MLflow Model Registry
```

---

# Tecnologias utilizadas

| Ferramenta | Finalidade |
|------------|------------|
| Python | Linguagem principal |
| PyTorch | Implementação da MLP |
| Scikit-Learn | Modelos baseline, StandardScaler e métricas |
| MLflow | Tracking e registro de modelos |
| DVC | Versionamento do pipeline |
| Docker | Ambiente reproduzível |
| Poetry | Gerenciamento de dependências |
| Pandas | Manipulação dos dados |
| NumPy | Operações numéricas |
| Joblib | Persistência do StandardScaler treinado |
| Matplotlib | Geração dos gráficos de avaliação |

---

# Estrutura do projeto

```
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│   ├── model.pt
│   ├── model_config.json
│   ├── scaler.pkl
│   └── embedding_config.json
│
├── reports/
│   ├── evaluation_metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall.png
│   └── classification_report.txt
│
├── scripts/
│
├── src/
│   ├── configs/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── pyproject.toml
└── README.md
```

---

# Organização dos módulos

## configs

Centraliza todas as configurações da aplicação.

Exemplo:

- caminhos dos arquivos;
- seed;
- URI do MLflow.

---

## data

Responsável pela preparação dos dados.

### preprocess.py

- realiza download automático do MovieLens 100K;
- extrai o dataset;
- converte para CSV;
- cria a variável alvo (`target`, `rating >= 4`).

Saída:

```
data/processed/dataset.csv
```

---

### feature_engine.py

Cria atributos utilizados durante o treinamento.

Exemplos:

- índices contíguos de usuário/item (`user_idx`, `item_idx`), necessários para as camadas de embedding do MLP;
- número de interações por usuário;
- popularidade do item;
- média das avaliações do usuário;
- média das avaliações do item;
- hora da interação;
- dia da semana;
- indicador de fim de semana;
- timestamp em segundos (sem normalização manual — o escalonamento acontece em `data.py`, ajustado apenas no conjunto de treino, para evitar vazamento de dados).

Também persiste `models/embedding_config.json`, com o número de usuários/itens únicos, usado para dimensionar corretamente as camadas `nn.Embedding` do MLP.

Saída:

```
data/features/dataset.csv
```

---

## models

Implementação dos modelos utilizados.

### Baseline

Modelos clássicos:

- Logistic Regression
- Random Forest

Utilizados para comparação.

---

### MLP

Rede neural implementada em PyTorch, com:

- **embeddings de usuário e item**: em vez de alimentar `user_id`/`item_id` como valores numéricos crus (o que induziria relações de ordem sem sentido, ex. `435 > 92`), a rede aprende uma representação vetorial para cada usuário/item;
- **BatchNorm1d + Dropout** entre as camadas ocultas, para estabilizar o treinamento e reduzir overfitting;
- saída em **logits** (sem `Sigmoid` na última camada), treinada com `BCEWithLogitsLoss` — a implementação numericamente mais estável recomendada pelo PyTorch. A probabilidade é obtida aplicando `torch.sigmoid` sobre a saída na inferência.

A arquitetura pode ser facilmente modificada para testar diferentes números de camadas, neurônios e dimensão de embedding.

---

### Model Factory

Implementa o padrão Factory para instanciar qualquer modelo da aplicação (`"mlp"` ou `"baseline"`), repassando os hiperparâmetros de forma genérica via `**kwargs`.

---

## training

Contém toda a lógica de treinamento.

### data.py

Carrega os dados e realiza:

- leitura do CSV;
- separação treino/teste (estratificada, com seed fixa para reprodutibilidade);
- escalonamento com `StandardScaler`, **ajustado apenas no conjunto de treino** e aplicado ao teste, evitando vazamento de dados. As colunas `user_idx`/`item_idx` não são escalonadas, pois alimentam diretamente as camadas de embedding;
- persistência do scaler ajustado em `models/scaler.pkl`, para reuso na avaliação/inferência.

---

### mlp_trainer.py

Responsável pelo treinamento da rede neural. Possui:

- **treinamento em mini-batch** via `DataLoader` (em vez de treinar com o dataset inteiro de uma vez);
- **Adam Optimizer** com `weight_decay` (regularização L2);
- **BCEWithLogitsLoss**, compatível com a saída em logits do MLP;
- **ReduceLROnPlateau**: reduz o learning rate automaticamente quando a loss estagna;
- **Early Stopping real**: interrompe o treinamento após um número configurável de épocas sem melhora na loss, restaurando os melhores pesos encontrados.

---

### train.py

Pipeline principal. Executa:

1. Logistic Regression;
2. Random Forest;
3. MLP, com **busca de hiperparâmetros** (grid search sobre `hidden_dim`, `learning_rate`, `dropout` e `batch_size` — cada combinação vira um run próprio no MLflow);
4. comparação das métricas de todos os candidatos;
5. seleção do melhor modelo por F1-score;
6. registro do melhor MLP no MLflow Model Registry, em estágio Production;
7. salvamento do modelo (`models/model.pt`) e da configuração vencedora (`models/model_config.json`);
8. registro de `reports/` e `models/` como artefatos do MLflow ao final da execução.

---

### evaluate.py

Executa a avaliação final do modelo salvo, no conjunto de holdout. Gera:

**Métricas de classificação:**
- Accuracy, Precision, Recall, F1-score, ROC-AUC, RMSE.

**Métricas de ranking**, calculadas por usuário (ver seção [Métricas avaliadas](#métricas-avaliadas)):
- Precision@10, Recall@10, NDCG@10, MAP@10.

**Artefatos visuais**, salvos em `reports/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall.png`
- `classification_report.txt`

Os resultados numéricos são armazenados em:

```
reports/evaluation_metrics.json
```

---

### metrics.py

Centraliza todas as métricas utilizadas durante treinamento e avaliação: as métricas de classificação (`calculate_binary_metrics`) e as métricas de ranking orientadas a recomendação (`calculate_ranking_metrics`).

---

# Pipeline DVC

O pipeline é definido em:

```
dvc.yaml
```

Fluxo:

```
Preprocess
      │
      ▼
Feature Engineering
      │
      ▼
Training (com busca de hiperparâmetros)
      │
      ▼
Evaluation (métricas + gráficos)
```

Executado com:

```bash
dvc repro
```

---

# MLflow

Todos os experimentos são registrados automaticamente.

São armazenados:

- parâmetros (incluindo os hiperparâmetros de cada combinação testada);
- métricas (classificação e, na versão otimizada, ranking);
- modelos;
- artefatos (relatórios e configuração do melhor modelo).

A interface pode ser acessada em:

```
http://localhost:5001
```

---

# Modelos avaliados

O projeto compara três abordagens.

## Logistic Regression

Modelo linear.

Vantagens:

- rápido;
- simples;
- boa interpretabilidade.

---

## Random Forest

Modelo baseado em árvores de decisão.

Vantagens:

- robusto;
- excelente baseline;
- captura relações não lineares.

---

## Multi Layer Perceptron (MLP)

Rede neural totalmente conectada, com embeddings de usuário/item, BatchNorm e Dropout.

Vantagens:

- maior capacidade de representação;
- captura relações complexas entre usuários e itens, inclusive de similaridade via embeddings;
- se beneficia diretamente do escalonamento de features e da busca de hiperparâmetros — diferente do Random Forest, que é praticamente insensível a essas melhorias.

---

# Métricas avaliadas

## Accuracy

Percentual de previsões corretas. Quanto maior, melhor.

## Precision

Entre os itens recomendados, quantos realmente eram relevantes.

## Recall

Entre todos os itens relevantes, quantos o modelo conseguiu recuperar.

## F1-score

Média harmônica entre Precision e Recall. Principal métrica utilizada para seleção do melhor modelo.

## ROC-AUC

Capacidade do modelo em separar as classes. Valores próximos de 1 indicam excelente separação.

## RMSE

Erro médio entre probabilidades previstas e valores reais. Quanto menor, melhor.

## Métricas de ranking (Precision@K, Recall@K, NDCG@K, MAP@K)

Métricas de classificação tratam cada par usuário-item de forma independente, o que não reflete bem um cenário de recomendação real, onde o que importa é o **ranking** de itens sugerido a cada usuário. Por isso, na versão otimizada do pipeline, essas métricas são calculadas por usuário sobre o conjunto de holdout:

- **Precision@K**: entre os K itens mais bem ranqueados para um usuário, qual fração é de fato relevante;
- **Recall@K**: da totalidade de itens relevantes para o usuário, qual fração aparece entre os top-K;
- **NDCG@K**: penaliza itens relevantes ranqueados mais abaixo, considerando a posição no ranking;
- **MAP@K**: precisão média considerando a posição de cada acerto dentro do top-K.

---

# Execução

## 1. Clonar o projeto

```bash
git clone <repositorio>
cd tech-challenge-f2
```

## 2. Instalar dependências (fora do Docker, opcional para dev local)

```bash
poetry install
```

## 3. Construir as imagens

```bash
docker compose build
```

## 4. Executar

```bash
docker compose up
```

O pipeline DVC será executado automaticamente.

## 5. Acessar o MLflow

```
http://localhost:5001
```

---

# Evolução do projeto: Baseline (v1) vs. Melhorado (v2)

Este projeto foi estruturado para permitir comparação controlada entre uma implementação simples e uma implementação otimizada, com cada melhoria rastreada e mensurável no MLflow — em vez de simplesmente substituir o pipeline por uma "versão melhor".

## v1-baseline

- `user_id`/`item_id` tratados como valores numéricos crus (sem embeddings);
- normalização manual (min-max) do timestamp;
- MLP treinada em batch único (sem mini-batch), sem BatchNorm/Dropout;
- sem busca de hiperparâmetros;
- resultado esperado: a MLP não supera necessariamente os modelos clássicos — comportamento comum em dados tabulares.

## v2-melhorado

- embeddings de usuário e item;
- `StandardScaler` ajustado apenas no conjunto de treino;
- MLP com `BatchNorm1d` + `Dropout`;
- treinamento em mini-batch, `Adam` com `weight_decay`, `ReduceLROnPlateau`, early stopping real;
- busca de hiperparâmetros (`hidden_dim`, `learning_rate`, `dropout`, `batch_size`);
- gráficos de avaliação (matriz de confusão, curva ROC, curva precision-recall);
- métricas de ranking (Precision@K, Recall@K, NDCG@K, MAP@K).

Para comparar as duas versões, execute o treino em cada branch/tag com a variável de ambiente `PIPELINE_VERSION` definida. Cada execução cai em um experimento MLflow próprio (`recommender-v1-baseline` / `recommender-v2-melhorado`), permitindo comparar ambas na mesma UI sem misturar os runs:

```bash
git checkout v1-baseline
PIPELINE_VERSION=v1-baseline dvc repro

git checkout v2-melhorado
PIPELINE_VERSION=v2-melhorado dvc repro
```

Se `PIPELINE_VERSION` não for definida, os runs caem em um experimento `recommender-default`.

Essa evolução permite demonstrar, na prática, o impacto de técnicas de engenharia de Machine Learning sobre a qualidade das recomendações — com rastreabilidade completa de qual alteração produziu qual ganho.

---

# Solução de Problemas

Se `docker compose up` falhar com um erro de conexão do Docker Desktop no Windows (ex.: `open //./pipe/dockerDesktopLinuxEngine`), verifique se o Docker Desktop está rodando e se o engine WSL2 está saudável:

```powershell
wsl --shutdown
wsl -l -v
```

Se o container do MLflow falhar ao subir com um erro do Alembic do tipo `Can't locate revision`, o volume `mlflow_data` provavelmente foi criado por uma versão diferente do MLflow. Reinicie-o com:

```bash
docker compose down -v
docker compose up --build
```

Isso apaga os runs/experimentos armazenados, então só faça isso se não precisar manter o histórico atual do MLflow.

---

# Autor

João Pedro

Projeto desenvolvido para o **Tech Challenge - Fase 2**.
