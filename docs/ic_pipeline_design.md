# Pipeline de decisões do projeto

## Visão geral

O projeto tem dois tracks paralelos:

```
SWAN-SF dataset (FL/NF)
        │
        ▼
  preprocess.py ──────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
   train.py (v5)                              cmesrc_crossmatch.py
  [Flare prediction]                          [CME labeling - 168h]
        │                                                  │
        │ models/flare/ (encoder treinado)                 ▼
        │                                          train_cme.py
        └──── transfer learning ──────────────► train_cme_finetune.py
                                                           │
                                                           ▼
                                               statistical_tests.py
```

---

## Track 1 — Predição de Flares (FL/NF)

### Por que GroupKFold por HARPNUM?

Cada região ativa (HARPNUM) gera múltiplas janelas temporais sequencialmente correlacionadas. Colocar janelas do mesmo HARPNUM em treino e validação seria **data leakage**. O GroupKFold garante que todos os dados de um HARPNUM ficam no mesmo fold.

### Por que hold-out isolado antes do CV?

O CV 5-fold escolhe hiperparâmetros e seleciona modelos — o hold-out dá a estimativa **verdadeiramente não-vista** do desempenho. Sem essa separação, a performance reportada é otimista.

### Por que checkpoints intra-epoch?

Com datasets grandes (~316k séries) e batches de 256, cada época tem ~1234 batches. O TSS pode atingir seu pico no meio de uma época e descer antes do fim. Os checkpoints a cada 100 batches capturam esses picos.

### Por que selecionar por val loss em vez de TSS@0.5?

TSS@0.5 é instável em batches menores de validação — pequenas flutuações de threshold mudam drasticamente o score. Val loss é mais suave e mais confiável como critério de seleção.

### Por que threshold ótimo?

Com desbalanceamento de classes severo (FL≪NF), o threshold de 0.5 é arbitrário. A busca em grade [0.05, 0.95] no conjunto de validação maximiza o TSS.

---

## Track 2 — Predição de CMEs

### Evolução do labeling

**DONKI API** foi o primeiro approach: simples, mas o catálogo da NASA tem 0% de informação sobre qual região ativa gerou o CME → impossível associar ao HARPNUM correto.

**LASCO/CDAW** tem associações flare↔CME explícitas, mas apenas para flares ≥M1 — perde CMEs de flares fracos (classe B/C).

**cmesrc (Hernandez et al., 2025)** é o único catálogo público que cruza CMEs com regiões ativas *incluindo* flares fracos. Usa janela de 72h antes do CME, com lead time mínimo de 6h (evita usar dados durante/após o evento).

### Por que 72h?

A janela de 72h é suficiente para capturar a maioria dos eventos no SWAN-SF com informação pré-eruptiva significativa. Uma versão experimental com 168h (`experiments/labeling/cmesrc_crossmatch_v3.py`) recupera mais eventos em regiões com cobertura esparsa, mas não foi adotada como final.

### Por que o dataset CME é pequeno (~1137 séries)?

O cmesrc cruza CMEs com regiões ativas específicas (HARPNUM), o que é rigoroso mas restritivo. A maioria dos CMEs no catálogo não tem correspondência direta com um HARPNUM no SWAN-SF.

### Transfer learning

O encoder biLSTM treinado na tarefa FL/NF aprende representações de padrões magnéticos gerais. Congelar o encoder e treinar apenas a atenção + decoder para CME permite:
1. Aproveitar o dataset grande de flares (316k séries)
2. Evitar overfitting no dataset pequeno de CME (~1137 séries)

---

## Hiperparâmetros finais (flare)

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| HIDDEN_SIZE | 128 | Capacidade suficiente sem overfitting |
| NUM_LAYERS | 2 | Profundidade mínima para padrões temporais |
| DROPOUT | 0.4 | Regularização para dataset grande |
| LR | 3e-4 | AdamW padrão |
| WEIGHT_DECAY | 1e-4 | L2 regularization |
| BATCH_SIZE | 256 | Memória vs. gradiente estável |
| W_POS_CAP | 12 | Cap no peso da classe positiva |
| PATIENCE_CKPT | 8 | Paciência em checkpoints (não épocas) |
| N_FOLDS | 5 | GroupKFold padrão |
| HOLDOUT_FRAC | 0.15 | 15% dos HARPNUMs isolados |

---

## Testes estatísticos (CME ensemble)

Resultado do hold-out: TP=67, FN=18, FP=22, TN=58 (N=165, TSS=0.5132, threshold=0.45)

1. **Bootstrap CI 95%** [10k reamostras]: intervalo de confiança para TSS
2. **Binomial test**: Recall > 0.5 (hipótese alternativa unilateral)
3. **Permutation test**: TSS observado vs. distribuição nula (rótulos embaralhados)
