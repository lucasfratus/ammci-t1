# E12 — Avaliacao final

| Modelo | MCC | Desvio entre seeds | Diferenca vs M0 |
|---|---:|---:|---:|
| GB | 0.1978 | 0.0000 | +0.0356 |
| LR | 0.1309 | nan | -0.0314 |
| M0 | 0.1623 | 0.0021 | +0.0000 |
| MFT | 0.1681 | 0.0069 | +0.0058 |
| MREC | 0.1715 | 0.0053 | +0.0093 |
| MRT | 0.1789 | 0.0064 | +0.0166 |

Desvio amostral ddof=1; LR deterministica tem uma execucao (seed=-1) e desvio indefinido. PR-AUC e average precision; ROC-AUC indefinida se houver uma unica classe. Tempos de MFT separam atualizacao e custo total desde M0. Complexidade de GB e um limite superior de folhas, nao contagem de pesos. A comparacao central usa M0/MFT/MRT/MREC; GB e LR sao baselines historicos. Os desvios entre sementes nao sao intervalos de confianca para a populacao.
