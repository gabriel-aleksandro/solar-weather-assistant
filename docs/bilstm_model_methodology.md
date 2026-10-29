# Modelo biLSTM com Atenção para Previsão de Flares e CMEs

## Visão Geral do Projeto

Este documento descreve a metodologia do projeto de Iniciação Científica (PIBIC/CNPq) desenvolvido na UNICAMP que resultou no modelo de previsão de flares solares e CMEs baseado em redes neurais biLSTM com mecanismo de atenção de Bahdanau.

O modelo está publicado no Hugging Face: `gabriel-aleksandro/solar-flare-bilstm`

---

## Problema e Justificativa

A previsão de flares solares e CMEs é um problema crítico de **séries temporais multivariadas** com:

1. **Desbalanceamento severo de classes**: flares M/X representam cerca de 1/60 das amostras; eventos X são 1/800
2. **Correlação temporal**: janelas consecutivas (passo de 1h) compartilham 11h das 12h de dados
3. **Dependência intra-região**: múltiplas janelas de um mesmo HARPNUM são correlacionadas
4. **Sinal preditivo temporal**: a evolução da topologia magnética ao longo do tempo carrega informação sobre a probabilidade de erupção — não apenas o estado instantâneo

A abordagem de classificar séries temporais completas (em vez de pontos individuais) captura essa dinâmica evolutiva.

---

## Arquitetura: biLSTM + Atenção de Bahdanau (Seq2Seq)

### Por que biLSTM?

**LSTM** (*Long Short-Term Memory*) é uma variante de Redes Neurais Recorrentes (RNNs) projetada para aprender dependências de longo prazo em sequências. As células LSTM têm portas (gate) de entrada, esquecimento e saída que controlam o fluxo de informação:

- **Porta de esquecimento (forget gate)**: decide quais informações do estado anterior descartar
- **Porta de entrada (input gate)**: decide quais novas informações adicionar ao estado
- **Porta de saída (output gate)**: decide qual parte do estado interno expor como saída

**biLSTM** (LSTM bidirecional) processa a sequência em duas direções simultaneamente — frente e verso — e concatena as representações. Para séries temporais de parâmetros magnéticos:
- Direção **forward**: captura como o campo evolui progressivamente para o evento
- Direção **backward**: captura padrões que ficam mais claros "de trás para frente"

A combinação captura contexto temporal completo de cada passo de tempo.

### Arquitetura completa

```
Entrada: (batch_size, T=60, n_features=18)
            │
            ▼
    Encoder biLSTM
    ├── 2 camadas empilhadas
    ├── hidden_size = 128 por direção (256 bidirecional)
    └── dropout = 0.4 entre camadas
            │
            ▼  saídas de todos os T passos: (batch, 60, 256)
    Atenção de Bahdanau
    ├── Context vector: soma ponderada das saídas do encoder
    └── Pesos de atenção: quais passos de tempo são mais relevantes
            │
            ▼  context vector: (batch, 256)
    Decoder Linear
    └── Camada linear 256 → 1 + Sigmoid
            │
            ▼
    Probabilidade de flare [0, 1]
            │
            ▼
    Threshold ótimo (0.45) → classe FL (1) ou NF (0)
```

### Atenção de Bahdanau

O mecanismo de atenção aprende automaticamente quais passos de tempo na sequência de 60 horas são mais informativos para a previsão. Para cada passo de tempo `t`, calcula um peso `α_t`:

```
e_t = tanh(W_a * h_t + b_a)         # score de energia
α_t = softmax(v_a^T * e_t)           # peso normalizado
context = Σ α_t * h_t                 # média ponderada
```

Os pesos `α_t` formam a **distribuição de atenção** — visualizável como um mapa de calor temporal que mostra quais horas antes do evento o modelo considera mais importantes.

---

## Pré-processamento e Representação dos Dados

### Formato de entrada

Cada amostra é uma série temporal de dimensões **T=60 × 18**:
- **T=60**: 60 observações consecutivas (60 horas de dados, uma por hora)
- **18**: os 18 parâmetros SHARP do HMI

Fisicamente, isso representa a evolução temporal da topologia magnética de uma região ativa nas 60 horas anteriores ao horizonte de previsão.

### Imputação de valores faltantes

Parâmetros SHARP podem ter valores faltantes por nuvens, falhas instrumentais ou eclipses do satélite. O projeto usa **imputação KNN com seleção de vizinhos por correlação de Pearson**:

1. Para cada parâmetro com valor faltante no passo de tempo `t`, encontra os `k` parâmetros com maior correlação de Pearson com o faltante
2. Usa os k vizinhos mais correlacionados para estimar o valor faltante via média ponderada pela correlação
3. Preserva a estrutura de correlação entre parâmetros SHARP

### Normalização Z-score

Cada parâmetro é normalizado independentemente com **Z-score** (média zero, variância unitária):

```
x_norm = (x - μ_train) / σ_train
```

**Crítico**: os parâmetros `μ` e `σ` são calculados **apenas no conjunto de treino** de cada fold. Aplicar a normalização usando estatísticas do conjunto de teste seria data leakage (o modelo "veria" informação estatística do futuro).

Após normalização, valores são **clampados** para o intervalo `[-10, 10]` para eliminar outliers extremos.

---

## Validação Cruzada e Avaliação

### GroupKFold por HARPNUM

```
Total de HARPNUMs: ~4075
        │
        ▼  isolamento estratificado
Hold-out (15% dos HARPNUMs, nunca vistos)
        │
        ▼  dos 85% restantes
    GroupKFold 5 folds por HARPNUM
    ├── Fold 1: treino=folds 2,3,4,5 / val=fold 1
    ├── Fold 2: treino=folds 1,3,4,5 / val=fold 2
    ├── ...
    └── Fold 5: treino=folds 1,2,3,4 / val=fold 5
```

**Garantia**: nenhum HARPNUM aparece simultaneamente em treino e validação. Isso simula rigorosamente o cenário de prever flares em regiões ativas novas.

### Ensemble dos 5 modelos

O modelo final é um **ensemble** dos 5 modelos treinados em cada fold:

1. Cada modelo `i` produz probabilidade `p_i(x)` para uma amostra `x`
2. Probabilidade de ensemble: `p_ensemble(x) = mean(p_1, p_2, p_3, p_4, p_5)`
3. Classificação final: `FL` se `p_ensemble > 0.45` (threshold ótimo)

O ensemble é mais robusto que qualquer modelo individual — erros de um fold são compensados pelos outros.

### Threshold ótimo

Com desbalanceamento severo (FL≪NF), o threshold de 0.5 é subótimo. A busca por grade em `[0.05, 0.95]` com passo 0.05 no conjunto de validação maximiza o TSS:

```
TSS(t) = Recall(t) - FAR(t) = TP/(TP+FN) - FP/(FP+TN)
```

O threshold ótimo encontrado foi **0.45** — ligeiramente abaixo de 0.5, favorecendo mais detecções de flares (maior Recall) ao custo de algumas falsas alarmes adicionais.

---

## Métricas de Avaliação

### TSS — True Skill Statistic

O TSS (também chamado Hanssen-Kuipers score) é a métrica principal:

```
TSS = Recall − FAR = TP/(TP+FN) − FP/(FP+TN)
```

**Por que TSS e não acurácia?**

Com 1% de flares na amostra, um modelo que prevê "nunca há flare" tem 99% de acurácia. TSS penaliza igualmente erros nos dois sentidos:
- Perder um flare real (FN): reduz Recall
- Falso alarme (FP): aumenta FAR

TSS = 0 → modelo sem habilidade (equivalente a adivinhar aleatoriamente)
TSS = 1 → modelo perfeito
TSS = -1 → modelo sistematicamente errado

Adicionalmente, o TSS é matematicamente invariante à razão de desbalanceamento — não é artificialmente inflado por datasets com poucos positivos.

### Matriz de Confusão

Para o modelo de flares (ensemble no hold-out):
- **TSS ≈ 0.78** com threshold 0.45
- Recall ≈ 0.88 (detecta ~88% dos flares reais)
- FAR ≈ 0.10 (10% de falsas alarmes no conjunto negativo)

Para o modelo de CMEs (hold-out, threshold 0.45):
- **TSS = 0.5132**
- TP=67, FN=18, FP=22, TN=58 (N=165)

---

## Transfer Learning para CMEs

### Motivação

O dataset de CMEs é muito menor que o de flares:
- Flares (FL/NF): ~316.000 séries temporais
- CMEs (cmesrc v2): ~1.137 séries temporais

Treinar a rede do zero no dataset de CMEs levaria a overfitting severo.

### Estratégia

O **encoder biLSTM** treinado na tarefa de flares aprendeu representações ricas de padrões magnéticos. Esses padrões são relevantes também para CMEs (afinal, CMEs frequentemente ocorrem junto com flares).

**Transfer learning**: congelar os pesos do encoder biLSTM treinado para flares; treinar apenas o mecanismo de atenção e o decoder linear para a tarefa de CME.

```python
# Encoder congelado (pesos fixos do treino de flares)
for param in model.encoder.parameters():
    param.requires_grad = False

# Apenas atenção + decoder são treinados para CME
for param in model.attention.parameters():
    param.requires_grad = True
for param in model.decoder.parameters():
    param.requires_grad = True
```

Isso permite aproveitar o dataset grande de flares para aprender representações gerais de campo magnético, enquanto o dataset pequeno de CMEs ajusta apenas a "cabeça" de decisão do modelo.

---

## Treinamento e Hiperparâmetros

### Função de perda

**BCE ponderada** (*Binary Cross-Entropy* com pesos de classe):

```
BCE_weighted = -[w_pos * y * log(p) + (1-y) * log(1-p)]
```

O peso `w_pos` compensa o desbalanceamento: se há 30x mais NF que FL, `w_pos = 30`. O peso é calculado dinamicamente em cada fold com base no ratio de classes no conjunto de treino, com cap em 12 para evitar instabilidade numérica.

### Otimizador e schedule

- **AdamW** com `lr=3e-4` e `weight_decay=1e-4`
- Sem scheduler de learning rate (LR fixo durante treino)

### Checkpoints intra-época

Com ~316k amostras e batch de 256, cada época tem ~1234 batches. O TSS de validação pode atingir pico no meio de uma época. Checkpoints são salvos a cada 100 batches, e o melhor checkpoint por val loss é selecionado ao final do treino.

### Critério de seleção

Modelo selecionado por **val loss** (não TSS@0.5), pois val loss é mais suave e estável como sinal de treinamento. O TSS final é calculado com o threshold ótimo após seleção do modelo.

---

## Parâmetros de Configuração Final

| Parâmetro | Valor |
|-----------|-------|
| n_features | 18 |
| window_size | 60 (horas) |
| hidden_size | 128 |
| num_layers | 2 |
| dropout | 0.4 |
| learning_rate | 3e-4 |
| weight_decay | 1e-4 |
| batch_size | 256 |
| w_pos_cap | 12 |
| n_folds | 5 |
| holdout_frac | 0.15 |
| threshold | 0.45 |
| ensemble | média das 5 probabilidades |

---

## Reprodutibilidade

O modelo está publicado no Hugging Face como `gabriel-aleksandro/solar-flare-bilstm` com todos os 5 checkpoints dos folds. O predict.py inclui pipeline de inferência completo, incluindo normalização Z-score e ensemble.

---

*Projeto de Iniciação Científica — UNICAMP/TADS — PIBIC/CNPq*
*Referências: Bahdanau et al. (2015); Hochreiter & Schmidhuber (1997); Liu et al. (2020); Angryk et al. (2020)*
