# CME & Solar Flare Prediction with biLSTM + Attention

> **PIBIC/CNPq Scientific Initiation Project**  
> Gabriel Aleksandro de Paula — TADS, UNICAMP  
> Advisor: [nome do orientador]

Predição de **Ejeções de Massa Coronal (CMEs)** e **flares solares** a partir de parâmetros magnéticos de regiões ativas do Sol, usando redes neurais **biLSTM bidirecionais com mecanismo de atenção (Bahdanau)** e validação cruzada por agrupamento (GroupKFold por HARPNUM).

---

## Resultados principais

| Tarefa | Conjunto | TSS | Threshold |
|--------|----------|-----|-----------|
| Flare (FL/NF) | Hold-out isolado (15%) | **~0.78** | ótimo |
| CME (cmesrc v2) | Hold-out isolado (15%) | **0.5132** | 0.45 |

> TSS = Recall − FAR (True Skill Statistic) — métrica principal, robusta ao desbalanceamento de classes.

---

## Dataset

**SWAN-SF** — Solar Wind ANalysis – Solar Flare (5 partições, NASA/NOAA)  
- Janelas de parâmetros SHARP do instrumento SDO/HMI  
- 18 features por passo de tempo (USFLUX, MEANGAM, MEANGBT, ...)  
- Identificador único por região ativa: **HARPNUM** (impede vazamento de dados entre folds)

**Labeling CME** — catálogo **cmesrc** (Hernandez et al., 2025, doi:[10.3847/1538-4357/ad9b27](https://doi.org/10.3847/1538-4357/ad9b27))  
- Janela temporal: [-2h, +6h] em relação ao início do CME  
- Janela de série temporal: **168h** antes do evento (v3 final)

---

## Arquitetura do modelo

```
Entrada: (batch, T, 18)  →  Encoder biLSTM (hidden=128, 2 camadas)
                         →  Atenção de Bahdanau  →  context vector
                         →  Decoder Linear  →  logit  →  BCE ponderado
```

- **Validação**: 5-fold GroupKFold estratificado por HARPNUM  
- **Hold-out isolado**: 15% dos HARPNUMs reservados antes de qualquer CV  
- **Ensemble**: média das probabilidades dos 5 modelos no hold-out  
- **Threshold ótimo**: busca em grade [0.05, 0.95] sobre validação  
- **Imputação**: KNN com seleção de vizinhos por correlação de Pearson  
- **Normalização**: Z-score ajustado apenas no conjunto de treino

---

## Estrutura do repositório

```
cme-flare-prediction/
│
├── src/                          ← código de produção
│   ├── labeling/
│   │   └── cmesrc_crossmatch.py      # FINAL: janela 72h (v2 — usado)
│   │
│   ├── preprocessing/
│   │   └── preprocess.py             # pipeline FL/NF completo
│   │
│   ├── training/
│   │   ├── flare/
│   │   │   └── train.py              # FINAL: biLSTM+Atenção, 5-fold, hold-out
│   │   └── cme/
│   │       ├── train_cme.py          # treino CME (dataset v2)
│   │       └── train_cme_finetune.py # transfer learning: encoder FL → CME
│   │
│   └── evaluation/
│       └── statistical_tests.py      # Bootstrap CI, Binomial, Permutation
│
├── experiments/                  ← versões intermediárias (histórico)
│   ├── labeling/
│   │   ├── donki_crossmatch.py       # ABANDONADO: API DONKI sem info de AR
│   │   ├── lasco_crossmatch.py       # catálogo LASCO/CDAW
│   │   ├── cmesrc_crossmatch_v1.py   # cmesrc v1 (sem controle de janela)
│   │   └── cmesrc_crossmatch_v3.py   # janela 168h (mais cobertura, não usado)
│   └── training/
│       ├── train_flare_v1.py         # biLSTM básico, sem CV
│       ├── train_flare_v2.py         # + Atenção + GroupKFold
│       ├── train_flare_v3.py         # + checkpoints intra-epoch
│       ├── train_flare_v4.py         # + seleção por val loss + threshold ótimo
│       ├── train_cme_v3.py           # CME com dataset 168h
│       └── train_cme_v4.py           # + Focal Loss + seleção de features
│
├── data/
│   ├── raw/          ← SWAN-SF partitions (não versionado, ver abaixo)
│   ├── processed/    ← saída do preprocess.py
│   └── cme_labeled/  ← saída dos scripts de labeling
│
├── models/           ← checkpoints .pt (não versionados)
├── results/          ← CSVs e relatórios de treino
├── docs/
│   └── pipeline.md   ← diagrama e decisões de design
│
├── requirements.txt
└── README.md
```

---

## Como reproduzir

### 1. Pré-requisitos

```bash
pip install -r requirements.txt
```

Baixe o dataset **SWAN-SF** (Harvard Dataverse) e extraia em `data/raw/`.  
Baixe o catálogo **cmesrc** (Zenodo, Hernandez et al. 2025) e coloque em `data/raw/cmesrc/`.

### 2. Pré-processamento FL/NF

```bash
python src/preprocessing/preprocess.py
# Saída: data/processed/{train,val,test}.parquet + scaler.pkl
```

### 3. Labeling CME

```bash
python src/labeling/cmesrc_crossmatch.py   # janela 72h
# Saída: data/cme_labeled/cme_labeled_v2.parquet + events_v2.csv
```

### 4. Treino — Flare (FL/NF)

```bash
python src/training/flare/train.py
# Saída: models/flare/fold_{1..5}_best.pt
#        results/cv_results_v5.csv
#        results/ensemble_report_v5.txt
```

### 5. Treino — CME

**Opção A — treinamento direto:**
```bash
python src/training/cme/train_cme.py
# Saída: models/cme/fold_{1..5}_best.pt + ensemble_report_cme.txt
```

**Opção B — transfer learning (encoder FL → CME):**
```bash
# Requer treino flare concluído (models/flare/)
python src/training/cme/train_cme_finetune.py
# Saída: models/cme_finetune/fold_{1..5}_best.pt
```

### 6. Testes estatísticos

```bash
python src/evaluation/statistical_tests.py
# Bootstrap CI 95%, Binomial test, Permutation test sobre TSS do ensemble
```

---

## Evolução do pipeline de labeling CME

| Script | Catálogo | Janela | Status |
|--------|----------|--------|--------|
| `donki_crossmatch.py` | NASA DONKI API | ±6h | Abandonado — 0% info de AR |
| `lasco_crossmatch.py` | LASCO/CDAW | ±3h | Experimental |
| `cmesrc_crossmatch_v1.py` | cmesrc (Hernandez 2025) | ±2h/+6h | Sem controle de janela |
| `cmesrc_crossmatch_v3.py` | cmesrc | 168h | Alternativa (mais cobertura) |
| **`cmesrc_crossmatch.py`** | **cmesrc** | **72h** | **Final (usado)** |

## Evolução do treino de flares

| Script | Novidade principal | TSS médio CV |
|--------|--------------------|-------------|
| `train_flare_v1.py` | biLSTM básico, split único | — |
| `train_flare_v2.py` | + Atenção + GroupKFold 5-fold | ~0.788 |
| `train_flare_v3.py` | + checkpoints intra-epoch | ~0.778 |
| `train_flare_v4.py` | + seleção por val loss + threshold ótimo | — |
| **`train.py` (v5)** | **+ hold-out isolado 15% + ensemble** | **~0.78** |

---

## Referências

- Hernandez et al. (2025). cmesrc catalog. *ApJ*, doi:[10.3847/1538-4357/ad9b27](https://doi.org/10.3847/1538-4357/ad9b27)
- Liu et al. (2020). Predicting Coronal Mass Ejections Using SDO/HMI Vector Magnetic Data Products and Recurrent Neural Networks. *ApJ*, 890, 12. doi:10.3847/1538-4357/ab6850
- EskandariNasab et al. (2024). KNN imputation with Pearson correlation for solar magnetic data.
- Bahdanau et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
- SWAN-SF dataset: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM)
- cmesrc catalog: [Zenodo](https://zenodo.org/records/[id])

---

## Licença

MIT — veja `LICENSE` para detalhes.

---

*Projeto de Iniciação Científica financiado pelo CNPq (PIBIC).*
