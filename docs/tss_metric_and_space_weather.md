# TSS, Métricas de Previsão e Clima Espacial Operacional

## Por que métricas especializadas em previsão solar?

A previsão de clima espacial enfrenta dois desafios estatísticos simultâneos:
1. **Desbalanceamento extremo**: eventos raros (flares M/X: ~1%, CMEs de impacto: <0.1%)
2. **Consequências assimétricas**: perder um flare X tem consequências muito mais graves que um falso alarme

Métricas tradicionais de machine learning como **acurácia** são completamente inadequadas. Um modelo que prevê "nunca há flare" alcança 99% de acurácia — mas tem utilidade operacional zero.

---

## TSS — True Skill Statistic

### Definição Formal

```
TSS = Recall − FAR
    = TP/(TP+FN) − FP/(FP+TN)
    = (TP·TN − FP·FN) / ((TP+FN)·(FP+TN))
```

Onde:
- **TP** (True Positives): flares previstos corretamente
- **FN** (False Negatives): flares perdidos (não previstos)
- **FP** (False Positives): falsas alarmes (previsto flare, mas não ocorreu)
- **TN** (True Negatives): não-flares previstos corretamente

### Interpretação

| TSS | Interpretação |
|-----|--------------|
| 1.0 | Previsão perfeita |
| 0.78 | Resultado do projeto (flares) — excelente |
| 0.51 | Resultado do projeto (CMEs) — habilidade real comprovada |
| 0.0 | Sem habilidade (equivale a adivinhar aleatoriamente) |
| -1.0 | Sistematicamente errado (pior que aleatório) |

### Por que TSS é ideal para dados desbalanceados

TSS é matematicamente invariante ao ratio de desbalanceamento. Prova:

Se multiplicar todos os negativos por k (aumentar k vezes as amostras negativas):
- TP e FN não mudam
- FP → k·FP, TN → k·TN

```
Recall = TP/(TP+FN)               # não muda com k
FAR = k·FP/(k·FP + k·TN) = FP/(FP+TN)  # não muda com k
TSS = Recall - FAR                 # não muda com k
```

Portanto, um dataset com 100 positivos e 1 negativo, ou com 100 positivos e 10.000 negativos, terão o mesmo TSS para o mesmo modelo — apenas a acurácia (sem utilidade aqui) mudaria.

### TSS vs. outras métricas

| Métrica | Sensível ao desbalanceamento? | Ideal para flares? |
|---------|-------------------------------|-------------------|
| Acurácia | Sim — inflada por TN | Não |
| F1 | Menos que acurácia, mas ainda sim | Razoável |
| AUC-ROC | Não | Bom para ranking |
| **TSS** | **Não** | **Sim — padrão da área** |
| HSS2 | Sim — varia com imbalance | Não ideal |

O TSS é o padrão adotado em previsão operacional de clima espacial (NOAA/SWPC, centros europeus) e na literatura científica desde Bloomfield et al. (2012).

---

## Componentes do TSS

### Recall (Sensibilidade / POD — Probability of Detection)

```
Recall = TP / (TP + FN)
```

Fração dos eventos reais que foram detectados. Para clima espacial, Recall alto é crucial: um flare X não detectado pode causar danos enormes.

No projeto: Recall_flares ≈ 0.88 (detecta 88% dos flares M/X)

### FAR — False Alarm Rate

```
FAR = FP / (FP + TN)
```

Fração das previsões negativas verdadeiras que foram erroneamente classificadas como positivas. FAR alto significa muitos falsos alertas — operacionalmente custoso (agências tomam ações preventivas para cada alerta).

No projeto: FAR_flares ≈ 0.10 (10% de falsas alarmes)

### Trade-off Recall vs. FAR

O threshold controla o trade-off:
- Threshold baixo → mais FP (mais falsas alarmes) mas mais TP (detecta mais flares) → Recall↑, FAR↑
- Threshold alto → menos FP mas mais FN (perde flares) → Recall↓, FAR↓

A curva ROC (Recall vs. FAR em todos os thresholds) mostra esse trade-off. O threshold ótimo de TSS maximiza a diferença Recall - FAR.

---

## Previsão Operacional de Clima Espacial

### Agências envolvidas

**NOAA/SWPC** (*Space Weather Prediction Center*, EUA): principal agência operacional. Emite alertas de flares e CMEs 24/7. Usa modelos empíricos e automatizados baseados nos parâmetros SHARP do HMI.

**ESA Space Weather Service Network** (Europa): similar ao SWPC mas para a infraestrutura europeia.

**ISEP** (Brasil — INPE): monitoramento e previsão de clima espacial para o território brasileiro.

### Escala de impacto geomagnético (Kp)

O índice **Kp** (0–9) mede perturbações no campo magnético terrestre:
- Kp < 4: atividade baixa
- Kp 4–5: atividade moderada (menor distúrbio)
- Kp 6–7: **tempestade geomagnética moderada** (G2-G3)
- Kp 8–9: **tempestade geomagnética severa/extrema** (G4-G5)

Eventos G4-G5 podem derrubar redes elétricas e satélites de comunicação.

### Horizonte de previsão

Diferentes partes do processo CME têm diferentes horizontes de previsão possíveis:
- **Alerta de flare**: 24–48h de antecedência (baseado em parâmetros SHARP)
- **Alerta de CME**: 1–3 dias (após detecção visual do CME no coronógrafo)
- **Impacto preciso na Terra**: 15–30 min (satélite L1 a 1.5M km da Terra)

O projeto foca em **previsão de flares com 24h de antecedência** — o mais desafiador e mais valioso para preparação antecipada.

---

## Ensemble e Incerteza

### Por que ensemble de 5 modelos?

Cada fold do GroupKFold treina um modelo diferente com diferentes subconjuntos de HARPNUMs. Os modelos diferem em:
- Quais padrões específicos de campo magnético foram vistos
- Como o dropout aleatório afetou a convergência
- Qual ponto no espaço de parâmetros foi encontrado

A **média das probabilidades dos 5 modelos** cancela erros idiossincráticos de cada fold. Empiricamente, ensembles de modelos moderados superam modelos individuais mais ajustados.

### Calibração da incerteza

A probabilidade ensemble `p_ensemble` também fornece uma medida de incerteza:
- `p_ensemble ≈ 0.5`: modelo muito incerto (os 5 folds discordam)
- `p_ensemble ≈ 0.9`: modelo muito confiante em "flare"
- `p_ensemble ≈ 0.1`: modelo muito confiante em "sem flare"

Alta variância entre os 5 folds → incerteza alta → usuário deve ser mais cauteloso com a previsão.

---

## Limitações do modelo atual

1. **Previsão binária**: apenas "flare M/X" ou "sem flare" — não prevê a intensidade exata
2. **Janela fixa**: 60 horas de observação → não captura eventos que se desenvolvem mais rapidamente
3. **Apenas parâmetros SHARP**: não usa imagens diretamente (EUV, raios-X, coronógrafo)
4. **Dataset histórico**: SWAN-SF vai até ~2018 — não captura atividade do ciclo solar 25 (2019–atual)
5. **CME vs. flare**: distinguir *qual* região ativa causa um CME ainda é desafiador mesmo com o cmesrc

---

*Fontes: Bloomfield et al. (2012) ApJL 747:L41; Barnes et al. (2016) ApJ 829:89; Leka et al. (2019) ApJ 881:101; Liu et al. (2020) ApJ 890:12*
