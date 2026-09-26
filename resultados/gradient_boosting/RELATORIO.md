# E10 — Gradient Boosting em D0

Hipotese: arvores impulsionadas capturam relacoes nao lineares e podem superar
MLP e regressao logistica nos mesmos quatro folds temporais de D0.

Busca: 16 configuracoes x quatro folds, seed 42, maior MCC medio.
Escolha: GB01, `{"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 0.0, "class_weight": "balanced"}`.
Early stopping desativado; sem padronizacao para arvores. Pesos balanceados
sao calculados pelo estimador somente no treino de cada fold.

| Familia | MCC medio ± desvio temporal | F1 | PR-AUC (AP) | Treino s/fold |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.3542 ± 0.0541 | 0.5989 | 0.5774 | 1.117 |
| MLP | 0.3134 ± 0.0604 | 0.5345 | 0.5614 | 6.847 |
| Logistica do zero | 0.2949 ± 0.0443 | 0.5660 | 0.5331 | 3.499 |

MLP e Gradient Boosting usam seeds 42, 1337 e 2024; logistica deterministica
usa uma execucao. As configuracoes anteriores da MLP e logistica foram
reavaliadas neste ambiente, sem nova busca. Tempo inclui padronizacao quando
aplicavel. Desvio temporal: ddof=0 sobre medias por fold entre seeds;
variabilidade entre seeds esta registrada separadamente no CSV.

Interpretacao: a tabela compara validacao usada na selecao, nao desempenho
futuro independente. O vencedor de Gradient Boosting fica selecionado para
baseline; a MLP permanece modelo principal dos experimentos de atualizacao.
MCC de treino e validacao por fold permitem examinar sobreajuste.
Maior MCC nesta comparacao: Gradient Boosting.
MLP atingiu o limite de epocas em 0 de 12 ajustes.
O CSV registra iteracoes: atingir o limite de 800 na MLP exige cautela
quanto a convergencia, sem alterar sua configuracao durante esta comparacao.

Correcao R1: embargo de sete dias em todos os folds. Nenhum alvo de treino
alcanca a validacao. Os periodos de validacao foram preservados e a semana
imediatamente anterior foi retirada do treino. As tres selecoes foram refeitas.
Resultados anteriores estao em resultados/historico_sem_embargo/.

D1 e D2 nao foram carregados. Artefatos: busca_folds.csv, comparacao_folds.csv,
comparacao_familias.csv, configuracao_escolhida.json e protocolo.json.
O protocolo registra data, hash, grade, ambiente, sementes e tempo total.
