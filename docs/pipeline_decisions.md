# Pipeline de Decisões do Projeto — Perguntas e Respostas

Este documento descreve as principais decisões de design do projeto de IC, em formato de perguntas e respostas, útil para entender o raciocínio por trás de cada escolha metodológica.

---

## Por que usar séries temporais e não ponto-no-tempo?

A maioria dos trabalhos de previsão de flares usa apenas o estado instantâneo da região ativa (um único conjunto de parâmetros SHARP em um momento). Porém, flares são processos dinâmicos — a *evolução* do campo magnético nas horas/dias anteriores contém informação preditiva crucial.

Usar séries temporais de 60 horas captura:
- Tendências de crescimento da helicidade e energia livre
- Padrões de cisalhamento progressivo
- Acúmulo de correntes magnéticas

Modelos baseados em séries temporais consistentemente superam modelos de ponto-no-tempo na literatura.

---

## Por que GroupKFold em vez de KFold simples?

**KFold simples** dividiria as amostras aleatoriamente, colocando diferentes janelas do mesmo HARPNUM em treino e validação.

**Problema**: janelas consecutivas do mesmo HARPNUM são altamente correlacionadas (compartilham 11/12 horas de dados). Se a janela `t` de um HARPNUM está no treino e a janela `t+1` está na validação, o modelo efetivamente "viu" quase toda a informação de validação.

**GroupKFold** garante que todos os dados de um HARPNUM ficam no mesmo fold — o modelo nunca vê dados de uma região ativa durante validação se já viu dados dessa mesma região no treino.

O resultado é uma avaliação mais realista: mede se o modelo generaliza para **regiões ativas completamente novas**, não apenas para janelas diferentes de regiões já conhecidas.

---

## Por que isolar 15% como hold-out antes do CV?

O Cross-Validation 5-fold por si só seleciona modelos e hiperparâmetros. Mas o conjunto de validação do CV é "visto" indiretamente — o processo de seleção cria um viés otimista.

O **hold-out isolado** (15% dos HARPNUMs, reservados antes de qualquer CV) representa regiões ativas que nunca participaram de nenhuma decisão de modelagem. A performance no hold-out é a estimativa mais confiável do desempenho operacional real.

Sem o hold-out, a performance reportada seria otimista (inflada pelo processo de seleção). Com ele, o TSS ~0.78 é uma estimativa válida para regiões ativas nunca vistas.

---

## Por que Bahdanau Attention e não atenção simples ou transformer?

**Atenção de Bahdanau** (ou *additive attention*) foi escolhida pela:

1. **Interpretabilidade**: os pesos de atenção `α_t` são diretamente interpretáveis — mostram quais horas antes do evento o modelo considera mais relevantes
2. **Integração natural com biLSTM**: o mecanismo de atenção sobre as saídas do encoder é o design Seq2Seq clássico
3. **Eficiência**: para sequências de T=60, a atenção quadrática (O(T²)) não é um gargalo computacional
4. **Tamanho do dataset**: transformers completos requerem muito mais dados — com ~316k amostras, o biLSTM+Atenção é mais eficiente

---

## Por que BCE ponderada e não Focal Loss ou oversampling?

**Binary Cross-Entropy com pesos de classe** (BCE ponderada) foi escolhida por simplicidade e estabilidade. O peso `w_pos` é calculado automaticamente como `n_neg/n_pos` no conjunto de treino de cada fold, com cap em 12 para estabilidade numérica.

**Focal Loss** foi testada em experimentos com CME mas não apresentou melhoria consistente sobre BCE ponderada.

**Oversampling (SMOTE)** cria amostras sintéticas interpolando séries temporais. Para séries temporais de parâmetros físicos, interpolação pode gerar padrões fisicamente impossíveis — optou-se por abordagem mais conservadora.

**Undersampling** foi descartado por descartar informação real de amostras NF, que totalizam ~98% do dataset.

---

## Por que threshold 0.45 e não 0.5?

O threshold de 0.5 (padrão para classificação binária) é arbitrário para problemas desbalanceados. Com poucos positivos (FL), a distribuição de probabilidades do modelo tende a ser conservadora — a maioria das probabilidades fica abaixo de 0.5 mesmo para casos verdadeiramente positivos.

**Busca em grade [0.05, 0.95] no conjunto de validação** maximiza o TSS:

```
TSS(t) = max_t [TP(t)/(TP+FN) - FP(t)/(FP+TN)]
```

O threshold 0.45 representa o ponto ótimo entre capturar mais flares reais (Recall) e evitar falsas alarmes (FAR).

---

## Por que checkpoints a cada 100 batches?

Com ~316k amostras e batch de 256, cada época tem ~1234 batches. Experimentos mostraram que o TSS de validação pode atingir pico no meio de uma época e decair antes do fim (regularização por dropout, gradientes ruidosos).

Salvar checkpoints a cada 100 batches (~8% de uma época) e selecionar o de melhor val loss evita perder esses picos transitórios. O checkpoint selecionado é o com menor val loss no conjunto de validação ao longo de todo o treino.

---

## Por que val loss e não TSS para seleção de checkpoint?

**TSS@0.5** no conjunto de validação tem alta variância em batches menores — pequenas mudanças no número de TP ou FP causam saltos grandes no TSS, especialmente com classes desbalanceadas.

**Val loss** é mais suave e captura a qualidade probabilística do modelo de forma mais estável. O TSS final é calculado separadamente com o threshold ótimo após seleção do modelo por val loss.

---

## Por que transfer learning para CME e não treino direto?

**Dataset de CME** (~1137 séries) é ~280x menor que o dataset de flares (~316k séries). Treinar biLSTM+Atenção do zero no dataset de CME levaria a overfitting severo.

**Transfer learning** aproveita o conhecimento aprendido na tarefa de flares:
- O encoder biLSTM aprende representações de padrões de evolução de campo magnético em 316k exemplos
- Esses padrões são relevantes para CMEs (eventos relacionados ao mesmo campo magnético)
- Apenas o mecanismo de atenção e o decoder (much fewer parameters) são treinados do zero para CME

Resultado: TSS_CME = 0.5132 com transfer learning vs. performance muito inferior sem (overfitting).

---

## Por que a janela CME é 72h e não 24h ou 168h?

**24h**: muito restrita — exige que exista uma série no SWAN-SF nas 24h imediatamente antes do CME. Para muitos CMEs no catálogo, as regiões ativas têm cobertura esparsa nessa janela.

**72h**: janela suficientemente ampla para capturar a maioria dos eventos com dados do SWAN-SF disponíveis. Lead time mínimo de 6h: não usa dados nas últimas 6h antes do CME (simula previsão realista — você não sabe que o CME vai acontecer).

**168h**: captura mais eventos em regiões com cobertura esparsa mas introduz séries de 7 dias antes do CME, quando a região pode ter estado completamente inativa — mais ruído que sinal.

O trade-off foi: **quantidade de eventos vs. qualidade do sinal preditivo**. A versão de 72h foi escolhida como equilíbrio.

---

## Por que KNN com correlação de Pearson para imputação?

Parâmetros SHARP têm correlações físicas fortes:
- USFLUX correlaciona com TOTPOT (ambos medem quantidade de campo magnético)
- MEANJZH correlaciona com TOTUSJH (ambos medem helicidade de corrente)

Imputação KNN com seleção de vizinhos por correlação de Pearson usa os parâmetros mais correlacionados para estimar valores faltantes, preservando a estrutura de covariância física.

Alternativas descartadas:
- **Interpolação linear**: boa para lacunas curtas mas ruim para lacunas longas ou valores extremos
- **Média global**: ignora a correlação entre parâmetros
- **Zero-filling**: distorce a distribuição e as correlações

---

## Por que Z-score e não Min-Max normalization?

**Min-Max normalization** é sensível a outliers — um único valor extremo comprime toda a distribuição para um intervalo estreito. Parâmetros SHARP têm outliers físicos reais (regiões ativas excepcionalmente intensas).

**Z-score** (µ=0, σ=1) é mais robusto a outliers. O **clamping em [-10, 10]** após Z-score remove outliers extremos que poderiam dominar as ativações da rede.

Crítico: Z-score calculado **apenas no treino** de cada fold. Usar estatísticas do conjunto de teste seria data leakage (o modelo "veria" informação estatística do futuro).

---

## Resultados Finais

| Tarefa | Dataset | TSS | Threshold | Método |
|--------|---------|-----|-----------|--------|
| Flare (FL/NF) | Hold-out isolado 15% | ~0.78 | 0.45 | Ensemble 5-fold biLSTM+Atenção |
| CME | Hold-out isolado 15% | 0.5132 | 0.45 | Transfer learning + ensemble |

**Métricas do ensemble CME no hold-out:**
- TP=67, FN=18, FP=22, TN=58
- Recall = 0.788 (detecta 78.8% dos CMEs)
- FAR = 0.275 (27.5% de falsas alarmes no conjunto negativo)
- TSS = 0.788 - 0.275 = 0.5132

---

*Projeto de Iniciação Científica (PIBIC/CNPq) — UNICAMP/TADS*
