# Revisão temporal R1

## Correções

Todos os folds agora exigem t+7 dias < início da validação. As semanas de
validação foram mantidas; foram excluídas 100 linhas de treino por fold.
Os scripts das três famílias compartilham a mesma função de divisão.
As saídas anteriores foram preservadas em `resultados/historico_sem_embargo/`.

MLP selecionada: 32 neurônios, alpha=0.01, taxa=0.003.
Teto de 800 épocas, com parada automática pela perda de treino.
Todos os 12 ajustes revisados convergiram; máximo observado: 171 épocas.
O teto 100 deixou cinco ajustes truncados nessa configuração. O teto maior
resolve a truncagem, mas não aumenta necessariamente o MCC: 100 épocas tiveram
MCC médio 0.3143, contra 0.3134 com teto 800.

## Comparação corrigida em D0

| Família | MCC médio | Desvio temporal (ddof=0) |
|---|---:|---:|
| Gradient Boosting | 0.3542 | 0.0541 |
| MLP | 0.3134 | 0.0604 |
| Logistica do zero | 0.2949 | 0.0443 |

MLP/GB: média entre três sementes em cada fold e depois média entre folds.
A regressão logística é determinística. Não confundir o MCC da busca da MLP
(0.3213, seed 42) com a média posterior entre três sementes.
As medidas são de seleção/validação, não de teste independente.

## Atualização em D1

M0 em D1: MCC médio 0.2152.
Fine-tuning escolhido: B_lr_original, 1 época(s), alpha=0.01.
MCC em D1b: 0.2120, ganho 0.0218 sobre M0 no mesmo D1b.
A seleção anterior de 11 épocas foi substituída porque M0 mudou.
Isso ainda não demonstra que o fine-tuning será superior em D2.

## Continuação do trabalho

AED e drift D0→D1 estão em `resultados/aed_drift/` e no notebook 01.
A proporção positiva mudou de 0.3616 para 0.3130.
O diário distingue execuções atuais de registros herdados incompletos.
O rascunho do protocolo final deverá ser revisado pela equipe antes de congelar
as decisões e avaliar D2, conforme a etapa 13 do plano recebido.
Não foram feitos commit, push ou avaliação em D2.
