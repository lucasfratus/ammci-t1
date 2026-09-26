# AED e E11 — Mudança de distribuição D0 → D1

Pergunta: os dados recentes diferem dos históricos? Dados: somente D0/D1,
verificados pelos hashes do manifesto. D2 não foi carregado.

D0: 42,800 linhas; D1: 14,200 linhas.
Proporção positiva: 0.3616 → 0.3130, variação de -4.86 pontos percentuais.

## Mudanças numéricas (ordenadas por PSI)

| Atributo | PSI | KS | Wasserstein / IQR de D0 |
|---|---:|---:|---:|
| digital_rank | 0.0937 | 0.1418 | 0.1437 |
| album_rank | 0.0833 | 0.0853 | 0.0267 |
| distancia_pico | 0.0766 | 0.1199 | 0.2910 |
| variacao_1s | 0.0752 | 0.0708 | 0.4833 |
| peak_pos | 0.0458 | 0.1029 | 0.1084 |

Maiores divergências categóricas: digital_presente (0.0159 bits), semana_ano (0.0103 bits), mes (0.0087 bits).

## Método e interpretação

KS mede a maior diferença entre distribuições acumuladas. PSI usa quantis
de D0 (até dez intervalos), caudas infinitas e pseudocontagem 0,5 por intervalo.
Wasserstein está nas unidades do atributo; sua razão pelo IQR de D0 é deixada
vazia quando o IQR é zero. Jensen–Shannon é a divergência em bits, não a distância.
PSI e JS não devem ser comparados na mesma escala. Não usamos limiares automáticos
para declarar drift nem p-valores iid: músicas se repetem e semanas são dependentes.

As diferenças medem mudança de distribuição dos atributos e do alvo. Não provam
mudança da relação condicional entre atributos e alvo (concept drift), nem causalidade.
Mudanças de cobertura dos rankings auxiliares também podem refletir a coleta.
Variáveis de calendário refletem sazonalidade e composição dos períodos.
Correlação entre posições e indicadores pode ser induzida pelas regras de construção.

## Ausências, censuras e passagens

Os atributos processados não têm nulos. Ausência no Hot 100 anterior usa posição
101; ranking secundário ausente usa 51; álbum ausente usa 201, acompanhado de
indicadores de presença. Esses valores são convenções, não posições observadas.
`estreia` indica início de uma passagem; inclui retorno após hiato superior a 28 dias.
`reentrada` identifica os inícios de passagem de música já observada. Portanto,
esses indicadores não representam necessariamente primeira estreia da carreira.

## Decisão e próximos passos

Há mudança mensurável entre D0 e D1, a ser relacionada ao desempenho de M0.
Manter a avaliação temporal e as quatro estratégias previstas; nenhuma variável
foi descartada com base nesta exploração. A avaliação futura exige protocolo
congelado. Artefatos quantitativos e seis figuras acompanham este relatório.
