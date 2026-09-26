# E09-R1 — Regressao logistica com embargo

Reexecucao da grade original em D0, quatro folds expansivos com embargo de sete dias. Padronizacao ajustada somente no treino.

Configuracao: `{"nome": "lr_1_balanceado", "taxa_aprendizado": 1.0, "class_weight": "balanced", "l2": 0.0, "max_iter": 6000, "tolerancia": 1e-11, "criterio_selecao": "maior MCC medio arredondado a 4 casas; desempate por menor tempo medio; exige convergencia em todos os folds", "mcc_medio": 0.29493847216088753, "mcc_desvio": 0.05114322941164527}`.

MCC: 0.2949; desvio amostral entre folds: 0.0511.

Concordancia de classes com scikit-learn: 1.000000; correlacao de probabilidades: 0.99999990.

Decisao: usar esta configuracao na comparacao corrigida. Resultados antigos preservados em resultados/historico_sem_embargo/. D1 e D2 nao utilizados.
