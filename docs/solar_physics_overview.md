# Física Solar e Clima Espacial — Visão Geral

## O Sol e sua Atividade Magnética

O Sol é uma estrela de tipo G2V com campo magnético complexo e dinâmico. A atividade magnética solar é o principal motor dos fenômenos de clima espacial que afetam a Terra e a infraestrutura tecnológica humana. As estruturas magnéticas mais importantes para previsão de clima espacial são as **regiões ativas** — áreas de campo magnético forte e concentrado visíveis como manchas solares.

O ciclo solar dura aproximadamente 11 anos, alternando entre períodos de mínima e máxima atividade. Durante o máximo solar, aumentam a frequência e intensidade de flares e ejeções de massa coronal.

---

## Flares Solares

### O que são

Flares solares são liberações súbitas e intensas de energia eletromagnética na atmosfera solar, causadas pela reconexão de linhas de campo magnético em regiões ativas. Um flare pode liberar energia equivalente a bilhões de bombas atômicas em questão de minutos.

A emissão de radiação de um flare abrange todo o espectro eletromagnético: rádio, ultravioleta extremo (EUV), raios-X e raios gama. Os efeitos mais imediatos na Terra incluem perturbações nas comunicações de rádio de alta frequência e degradação de sinais GPS.

### Classificação GOES

A classificação de flares usa medições de pico de fluxo de raios-X moles (banda 1–8 Å) dos satélites GOES (Geostationary Operational Environmental Satellite). A escala é logarítmica:

| Classe | Fluxo de pico (W/m²) |
|--------|----------------------|
| A      | < 10⁻⁷              |
| B      | 10⁻⁷ a 10⁻⁶         |
| C      | 10⁻⁶ a 10⁻⁵         |
| M      | 10⁻⁵ a 10⁻⁴         |
| X      | > 10⁻⁴              |

Cada classe tem subclasses de 1 a 9. Um flare X10 é 10 vezes mais intenso que um X1. A previsão operacional foca em flares de classe M e X, que têm impacto significativo no clima espacial.

O **desbalanceamento de classes** é um desafio central: eventos M e X são raros (cerca de 1:60 para flares MX vs. instâncias sem flare, e 1:800 para X-class).

### Janela de previsão

O padrão operacional usa uma **janela de observação de 12 horas** (os parâmetros magnéticos usados como entrada) e uma **janela de previsão de 24 horas** (quando o flare pode ocorrer). No dataset SWAN-SF, janelas são criadas com passo de 1 hora, resultando em alta correlação temporal entre amostras consecutivas.

---

## Ejeções de Massa Coronal (CMEs)

### O que são

Ejeções de Massa Coronal (CMEs, do inglês *Coronal Mass Ejections*) são expulsões de plasma e campo magnético da coroa solar com massa de 10¹²–10¹³ kg e velocidades de 200 a 3000 km/s. Diferentemente dos flares (radiação eletromagnética), as CMEs são eventos de partículas e campo magnético — viajam no espaço e chegam à Terra em 1 a 3 dias.

### Impactos geomagnéticos

Quando uma CME atinge a magnetosfera terrestre, pode causar:
- **Tempestades geomagnéticas** que danificam satélites e redes elétricas
- **Auroras boreais e austrais** em latitudes mais baixas
- **Interrupção de comunicações via satélite** e rádio
- **Aumento de radiação** para astronautas e passageiros de voos polares

O evento de Carrington (1859) — o maior já registrado — derrubou sistemas de telégrafo em toda a Europa e América do Norte. Um evento equivalente hoje causaria danos estimados em trilhões de dólares.

### Relação flare-CME

Flares e CMEs frequentemente ocorrem juntos, mas nem sempre:
- ~15% dos flares X produzem CMEs
- ~50% das CMEs rápidas são associadas a flares M/X
- CMEs podem ocorrer sem flares observados (CMEs "silenciosas")

A associação entre CMEs e regiões ativas é complexa. O catálogo **cmesrc (Hernandez et al., 2025)** é o primeiro a sistematicamente associar CMEs com HARPNUMs específicos, incluindo CMEs de flares fracos (B/C), usando janela temporal de 72 horas antes do início do CME.

---

## O Instrumento SDO/HMI e os Parâmetros SHARP

### Solar Dynamics Observatory (SDO)

O SDO (NASA) foi lançado em fevereiro de 2010 e opera em órbita geoestacionária. O instrumento **HMI (Helioseismic and Magnetic Imager)** produz magnetogramas vetoriais de disco completo com cadência de 12 minutos, cobrindo toda a superfície solar visível continuamente desde maio de 2010.

### HARPs e SHARPs

**HARP** (*HMI Active Region Patch*) é a denominação de regiões de campo magnético forte identificadas automaticamente pelo pipeline de processamento do HMI. Cada HARP recebe um identificador único chamado **HARPNUM**.

**SHARP** (*Space-weather HMI Active Region Patches*) são os parâmetros magnéticos derivados de cada HARP — escalares calculados a partir dos campos magnéticos vetoriais do HMI, atualizados a cada 12 minutos.

### Os 18 parâmetros SHARP usados no SWAN-SF

O dataset SWAN-SF usa um subconjunto de 18 parâmetros SHARP como features de entrada para os modelos de previsão:

| Parâmetro | Descrição Física |
|-----------|-----------------|
| USFLUX | Fluxo magnético total não-assinado (Mx) — medida da quantidade total de campo magnético na região |
| MEANGAM | Ângulo médio do campo em relação à radial — indica inclinação do campo magnético |
| MEANGBT | Gradiente horizontal médio do campo total (G/Mm) — mede variação espacial do campo |
| MEANGBZ | Gradiente horizontal médio do campo vertical — componente vertical do gradiente |
| MEANGBH | Gradiente horizontal médio do campo horizontal — componente horizontal do gradiente |
| MEANJZD | Densidade de corrente vertical média (mA/m²) — correntes associadas a torção do campo |
| TOTUSJZ | Corrente elétrica vertical total não-assinada (A) — medida total das correntes verticais |
| MEANALP | Parâmetro de torção característico α — descreve helicidade por unidade de fluxo |
| MEANJZH | Helicidade de corrente média — torção do campo magnético |
| TOTUSJH | Helicidade de corrente total não-assinada (G²/m) — soma do módulo da helicidade |
| ABSNJZH | Valor absoluto da helicidade de corrente líquida — helicidade resultante sem cancelamentos |
| SAVNCPP | Soma do módulo da corrente líquida por polaridade (A) — balanço de correntes |
| MEANPOT | Energia livre magnética média (erg/cm³) — energia disponível para liberação em flares |
| TOTPOT | Energia livre fotosférica total não-assinada — potencial total para explosão |
| MEANSHR | Ângulo de cisalhamento médio — quanto as linhas de campo desviam do potencial mínimo |
| SHRGT45 | Fração da área com cisalhamento > 45° — regiões de alta torção indicam instabilidade |
| NACR | Número de pixels de campo forte na linha de inversão de polaridade |
| SIZE_ACR | Área de pixels de campo vertical forte (micro-hemisférios) |

### Por que esses parâmetros?

Parâmetros de helicidade de corrente (TOTUSJH, MEANJZH, ABSNJZH) e energia livre (MEANPOT, TOTPOT) são os mais preditivos para flares, pois medem o quanto o campo magnético está "tensionado" acima de seu estado de energia mínima (potencial). Quando essa energia acumulada é liberada subitamente, ocorre um flare.

O parâmetro de cisalhamento (MEANSHR, SHRGT45) indica o quão distorcidas estão as linhas de campo em relação à configuração de potencial. Alto cisalhamento é precursor de reconexão magnética.

---

## HARPNUM e Data Leakage

O **HARPNUM** é o identificador único de cada região ativa rastreada pelo HMI. Uma mesma região ativa pode existir por 1 a 27 dias (duração de rotação solar), gerando dezenas a centenas de janelas temporais no SWAN-SF.

Se janelas do mesmo HARPNUM aparecem em treino e validação simultâneamente, o modelo "viu" a mesma região ativa durante o treinamento — isso é **data leakage** (vazamento de dados). O modelo aprende características específicas daquela região, não padrões magnéticos gerais.

A solução é **GroupKFold estratificado por HARPNUM**: todos os dados de um HARPNUM ficam no mesmo fold, garantindo que treino e validação nunca compartilham a mesma região ativa. Isso garante que o modelo generaliza para regiões ativas novas, nunca vistas durante o treinamento.

Adicionalmente, **15% dos HARPNUMs são isolados como hold-out** antes de qualquer cross-validation. Esses HARPNUMs nunca são vistos durante a seleção de hiperparâmetros ou escolha de modelos, fornecendo estimativa não-viesada do desempenho real.

---

## O Dataset SWAN-SF

**SWAN-SF** (*Space Weather ANalytics for Solar Flares*) é o benchmark padrão da área, criado por Angryk et al. (2020) e disponível no Harvard Dataverse.

**Características principais:**
- **4.075 regiões ativas** cobrindo mais de 9 anos de operação do SDO (2010–2018+)
- **5 partições temporais** — cada partição contém janelas de uma época diferente do ciclo solar, com número semelhante de eventos X e M
- **316.000+ séries temporais** de 60 passos de tempo (60 horas de observação cada)
- **Representação temporal**: séries de T=60 observações × 18 parâmetros SHARP
- **Classes**: FL (flare M ou X em 24h) e NF (sem flare ou apenas B/C)

As 5 partições permitem validação cruzada temporal: modelos treinados em partições passadas são testados em partições futuras, simulando o cenário operacional real de previsão.

---

*Fontes: Liu et al. (2020) ApJ 890:12; Angryk et al. (2020) Scientific Data; Bobra & Couvidat (2015) ApJ 798; Nishizuka et al. (2018); Hoeksema et al. (2014) Solar Physics*
