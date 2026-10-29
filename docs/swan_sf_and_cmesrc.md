# Dataset SWAN-SF e Catálogo cmesrc

## SWAN-SF — Space Weather ANalytics for Solar Flares

### Descrição Geral

O SWAN-SF (*Space Weather ANalytics for Solar Flares*) é o benchmark científico mais utilizado para previsão de flares solares a partir de séries temporais. Foi criado por Angryk et al. (2020) e está disponível publicamente no Harvard Dataverse (DOI: 10.7910/DVN/EBCFKM).

**Escopo do dataset:**
- 4.075 regiões ativas (HARPNUMs) rastreadas pelo HMI
- Cobertura temporal: maio de 2010 a dezembro de 2018 (mais de 9 anos de SDO)
- Mais de 316.000 séries temporais de 60 horas
- 18 parâmetros SHARP por passo de tempo

### Estrutura do Dataset

O SWAN-SF é organizado em **5 partições temporais**, cada uma contendo dados de uma época diferente do ciclo solar 24. Essa divisão temporal garante que cada partição tenha uma distribuição de flares similar:

| Partição | Período aproximado | Conteúdo |
|----------|--------------------|----------|
| 1 | 2010–2011 | Mínimo → subida do ciclo |
| 2 | 2012–2013 | Máximo do ciclo 24 (fase ascendente) |
| 3 | 2013–2014 | Máximo do ciclo 24 (fase descendente) |
| 4 | 2014–2016 | Descida do ciclo |
| 5 | 2016–2018 | Mínimo do ciclo → início ciclo 25 |

Cada partição tem aproximadamente o mesmo número de flares X e M, tornando a divisão 5-fold temporalmente equilibrada.

### Estrutura de Diretórios

```
SWAN-SF/
├── partition1/
│   ├── FL/     ← séries temporais com flare M/X em 24h
│   └── NF/     ← séries temporais sem flare (ou apenas B/C)
├── partition2/
│   ├── FL/
│   └── NF/
...
└── partition5/
    ├── FL/
    └── NF/
```

Cada arquivo `.csv` contém a série temporal de uma janela: 60 linhas (horas) × 18 colunas (parâmetros SHARP) + metadados (HARPNUM, timestamp).

### Janelas de Observação e Previsão

O SWAN-SF usa um esquema deslizante com:
- **Janela de observação**: 12 horas de parâmetros SHARP (entrada do modelo)
- **Janela de previsão**: 24 horas à frente do fim da observação
- **Passo**: 1 hora (janelas adjacentes compartilham 11h de dados)

**No projeto de IC**, a janela foi expandida para **60 horas de observação** (usando 60 steps do SWAN-SF), mantendo a previsão em 24h. Isso captura mais contexto evolutivo da região ativa.

O rótulo de cada janela é determinado pelo **flare mais forte** na janela de previsão de 24h:
- FL (flare): se houver ao menos um flare ≥M1.0
- NF (no flare): se apenas flares B/C ou nenhum flare

### Desbalanceamento de Classes

O desbalanceamento é o principal desafio do SWAN-SF:
- **MX vs. CBN**: ~1:60 (60 amostras sem flare para cada com flare M ou X)
- **X-only vs. resto**: ~1:800

Isso significa que um modelo que simplesmente prevê "sem flare" teria ~98% de acurácia mas TSS = 0 (nenhuma habilidade real de previsão).

---

## O Identificador HARPNUM e seu Papel Crítico

### O que é HARPNUM

O HARPNUM (*HMI Active Region Patch Number*) é o identificador numérico único atribuído pelo pipeline HMI a cada região ativa rastreada. Equivale ao NOAA AR number mas com rastreamento mais sistemático.

### Uma região ativa = múltiplas janelas

Uma região ativa pode existir por dias a semanas (uma rotação solar dura 27 dias). Isso gera dezenas a centenas de janelas temporais para o mesmo HARPNUM no SWAN-SF.

Por exemplo, HARPNUM 3722 existiu por 12 dias → ~288 janelas no dataset, todas altamente correlacionadas entre si.

### Data Leakage sem GroupKFold

Se janelas do HARPNUM 3722 aparecem em treino E validação:
- O modelo "aprende" os quirks específicos dessa região ativa
- A validação mede desempenho num HARPNUM já visto, não em regiões novas
- O modelo parece mais capaz do que realmente é (overfitting por HARPNUM)

### GroupKFold como solução

```python
# Cada HARPNUM vai inteiramente para um único fold
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=harpnums)):
    # Garantido: nenhum HARPNUM em train_idx aparece em val_idx
    ...
```

O GroupKFold estratificado por HARPNUM garante que o modelo aprende padrões magnéticos **gerais**, não específicos de regiões já vistas.

---

## Catálogo cmesrc — CME Source Region Associations

### Motivação e Contexto

Associar CMEs a regiões ativas específicas (HARPNUMs) é um problema não trivial. Histórica e operacionalmente, o processo era manual, realizado por especialistas que analisavam imagens da coroa solar.

O catálogo **cmesrc** (Hernandez et al., 2025, *ApJ*, doi:10.3847/1538-4357/ad9b27) é o primeiro catálogo público e sistemático que:
1. Associa automaticamente CMEs (catálogo LASCO/CDAW) a regiões ativas específicas (HARPNUMs)
2. Inclui CMEs de flares de todas as classes (não apenas M/X)
3. Usa janelas temporais bem definidas para controlar a associação

### Abordagens anteriores e seus problemas

**NASA DONKI API:**
- Catálogo online em tempo real
- **Problema**: 0% das entradas contém informação sobre qual região ativa causou o CME
- **Resultado**: impossível associar ao HARPNUM → descartado

**LASCO/CDAW catalog (manual):**
- Posição e velocidade de CMEs observados pelo coronógrafo LASCO
- Associações flare↔CME apenas para flares ≥M1
- **Problema**: perde CMEs de flares fracos (B/C), muito frequentes

**cmesrc (Hernandez 2025):**
- Cruza CMEs com regiões ativas usando coordenadas solares heliocêntricas
- Inclui associações para todos os flares (B, C, M, X)
- **Solução escolhida** para o projeto de IC

### Janela de 72h do cmesrc

**Por que 72h?**

A janela temporal define quais observações do SWAN-SF são rotuladas como "precursores de CME". A janela de 72h antes do início do CME foi escolhida porque:

1. Captura a maioria dos eventos com informação pré-eruptiva significativa no SWAN-SF
2. Janelas mais curtas (24h) perdem muitos eventos onde os dados do SWAN-SF têm lacunas
3. Janelas mais longas (168h — versão experimental do projeto) recuperam mais eventos mas introduzem mais ruído (70h antes pode não ser precursor do CME)
4. Lead time mínimo de 6h: não usa dados nas 6h imediatamente antes do CME (para simular previsão realista)

**Dataset CME resultante:**
- ~1.137 séries temporais (com e sem CME)
- Muito menor que o dataset de flares (~316k)
- Justifica o uso de transfer learning (encoder de flares → CME)

### Comparação dos catálogos de CME

| Catálogo | Cobertura | Associação AR | Status no projeto |
|----------|-----------|---------------|-------------------|
| NASA DONKI | 2010–presente | 0% | Abandonado |
| LASCO/CDAW (manual) | 1996–presente | Apenas flares ≥M1 | Experimental |
| cmesrc_v1 | 2010–2018 | Todos os flares | Sem controle de janela |
| cmesrc_v3 | 2010–2018 | Todos os flares | Janela 168h (não adotado) |
| **cmesrc_v2** | **2010–2018** | **Todos os flares** | **Final (72h, adotado)** |

---

## Evolução do Pipeline de Labeling de CME

O projeto passou por várias iterações de labeling:

### Versão 1 — DONKI
Script `donki_crossmatch.py`: tentou usar a API da NASA. Descoberto que o campo de região ativa (`sourceLocation`) está vazio em praticamente todas as entradas → CMEs sem HARPNUM → impossível cruzar com SWAN-SF.

### Versão 2 — LASCO/CDAW
Script `lasco_crossmatch.py`: usa o catálogo LASCO/CDAW que tem associações explícitas flare↔CME. Funciona para flares ≥M1 mas perde a maioria dos CMEs.

### Versão 3 — cmesrc sem controle de janela (`cmesrc_crossmatch_v1.py`)
Primeiro uso do catálogo cmesrc. Problema: janela temporal não controlada — rotula como precursor de CME séries muito distantes do evento, introduzindo ruído.

### Versão 4 — cmesrc com janela 72h (`cmesrc_crossmatch.py` — FINAL)
Janela explícita de 72h com lead time de 6h. Dataset resultante:
- TP: 67 CMEs detectados pelo ensemble
- FN: 18 CMEs perdidos
- FP: 22 falsas alarmes
- TN: 58 verdadeiros negativos
- **TSS = 0.5132** com threshold 0.45

### Versão experimental — 168h (`cmesrc_crossmatch_v3.py`)
Testada por capturar mais eventos em regiões com cobertura esparsa do SWAN-SF. Recupera mais eventos mas não foi adotada como final por introduzir mais séries distantes do evento.

---

## Testes Estatísticos dos Resultados (CME)

Para validar que o TSS = 0.5132 é estatisticamente significativo (e não resultado do acaso), foram aplicados três testes:

### 1. Bootstrap CI 95% (10.000 reamostras)
Reamostragem com reposição do hold-out 10.000 vezes. Intervalo de confiança para o TSS do ensemble de CME. Confirma que o resultado não é flutuação aleatória.

### 2. Teste Binomial (Recall > 0.5)
Hipótese alternativa unilateral: o modelo detecta mais de 50% dos CMEs reais (Recall > 0.5). Com TP=67 e total_positivo=85, p-valor calculado pela distribuição binomial.

### 3. Permutation Test
Embaralha os rótulos do hold-out 1.000 vezes e calcula o TSS em cada permutação. O TSS observado (0.5132) é comparado com a distribuição nula para calcular p-valor empírico. Rejeitar a hipótese nula confirma que o modelo tem habilidade real de previsão.

---

*Fontes: Angryk et al. (2020) Scientific Data 7:227; Hernandez et al. (2025) "Identifying Coronal Mass Ejection Active Region Sources: An Automated Approach" ApJ 979:63, doi:10.3847/1538-4357/ad9b27; Liu et al. (2020) "Predicting Coronal Mass Ejections Using SDO/HMI Vector Magnetic Data Products and Recurrent Neural Networks" ApJ 890:12, doi:10.3847/1538-4357/ab6850*
