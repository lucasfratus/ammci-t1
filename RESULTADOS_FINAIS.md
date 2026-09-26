# Avaliação final em D2 — 26/09/2026

O protocolo experimental foi congelado antes da avaliação original.
Foram treinados os 16 modelos antes da leitura de D2 pelo avaliador. A execução
concluiu sem falhas e os hashes dos três conjuntos coincidiram com o protocolo.
D2 contém 14.200 observações música/semana, de 20/12/2023 a 02/09/2026.
As configurações permaneceram fixas após a abertura do teste.

## Comparação

MCC é a métrica principal: quanto maior, melhor; zero indica ausência de
correlação entre as classes previstas e reais. Não representa porcentagem de acerto.
Médias e desvios amostrais entre três sementes, exceto LR determinística.

| Modelo | MCC médio ± desvio | F1 | Precisão | Recall | Acurácia |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting, treinado em D0 | 0,1978 ± 0,0000 | 0,4557 | 45,61% | 45,52% | 65,01% |
| MRT: nova MLP em D0+D1 | 0,1789 ± 0,0064 | 0,3655 | 49,58% | 29,01% | 67,62% |
| MREC: nova MLP em D1 | 0,1715 ± 0,0053 | 0,3345 | 50,64% | 25,01% | 68,01% |
| MFT: M0 atualizado em D1 | 0,1681 ± 0,0069 | 0,3308 | 50,33% | 24,72% | 67,92% |
| M0: MLP original de D0 | 0,1623 ± 0,0021 | 0,3470 | 48,37% | 27,10% | 67,21% |
| Regressão logística do zero, D0 | 0,1309; uma execução | 0,3945 | 41,80% | 37,34% | 63,11% |

GB produziu resultados idênticos nas três sementes desta configuração. Desvio
zero entre sementes não significa ausência de incerteza sobre outros dados.

## Interpretação

- GB obteve o maior MCC e F1 entre todos os modelos avaliados. Entre as estratégias
  da MLP, MRT obteve o maior MCC: ganho absoluto de 0,0166, ou 10,24% relativo a M0.
- Fine-tuning trouxe ganho pequeno de MCC: 0,0058, ou 3,60% relativo. Sua precisão
  aumentou, mas recall e F1 caíram. Portanto, não melhorou todas as métricas.
- A atualização MFT custou em média 0,0207 s, aproveitando M0 já treinado.
  Incluindo M0, o custo foi 4,2194 s, contra 5,1295 s de MRT. São tempos desta
  máquina e execução, não um benchmark universal.
- O MCC de M0 caiu de 0,2152 em D1 para 0,1623 em D2: queda de 0,0529.
  O desempenho permanece modesto. As diferenças observadas entre estratégias
  não são uma demonstração de significância estatística; três sementes medem
  variação do treinamento, não incerteza amostral sobre a população.
- Apenas 32,18% das observações de D2 tiveram melhora na semana seguinte.
  Prever sempre "não melhora" daria 67,82% de acurácia e MCC zero. Isso explica
  por que acurácia isolada é pouco informativa: GB tem menor acurácia, mas
  encontra mais melhoras e apresenta maior MCC.

## Drift, importância e erros

A frequência de melhora foi 36,16% em D0, 31,30% em D1 e 32,18% em D2.
A maior mudança numérica D0→D2 foi em `digital_rank` (PSI 1,2124), seguida
por `album_rank` (0,1442). Isso evidencia mudança de distribuição, mas não
prova concept drift nem estabelece a causa da queda de desempenho.

Para MRT, as maiores quedas de MCC após permutação foram `album_rank` (0,0518),
`variacao_1s` (0,0442) e `peak_pos` (0,0400). Atributos correlacionados podem
dividir importância; esses resultados não são efeitos causais.

MRT teve dificuldade em identificar melhoras no top 10: recall médio 3,65%
nesse grupo, contra 35,21% nas posições 41–100. Para músicas com 13 ou mais
semanas na parada, recall de 8,56%, contra 52,55% no grupo de 1–4 semanas.
Esses recortes são descritivos e foram definidos antes da avaliação.

## Artefatos e verificação

- `protocolo/REVISAO_FINAL.md`: revisão, autorização e hash do congelamento.
- `resultados/final/`: modelos, previsões, métricas por semente, resumo, tempos,
  curvas de treinamento e eventos com manifesto de integridade.
- `resultados/analises_finais/`: drift, importâncias, erros, figuras e manifesto.
- `notebooks/03_experimento_final_drift.ipynb` e
  `notebooks/04_explicabilidade_erros.ipynb`: executados com os resultados reais.

As nove MLPs treinadas do zero e a regressão logística registraram convergência.
MFT usa uma época fixa, sem critério de convergência; GB usa 200 iterações fixas.
Nenhum ajuste foi feito em resposta às métricas de D2.

## Correção de reprodutibilidade R2

A política de hashes da versão 2 dependia das terminações CRLF do Windows e
falhava depois de um checkout com LF. A versão 3 usa hashes canônicos para
texto e bytes exatos para binários, sem mudar nenhuma decisão experimental.

Uma reprodução completa posterior à correção treinou novamente os 16 modelos
e regenerou as análises. As 227.200 previsões, métricas, matrizes, curvas,
drift, importâncias e erros coincidiram exatamente com os resultados acima;
somente tempos de execução variaram. A integridade do pacote real passou a ser
verificada pela suíte automatizada.
