# E13–E15 — Análises após avaliação final

Melhor estratégia atualizada por MCC médio observado em D2: MRT.
Essa identificação é descritiva: todos os modelos foram avaliados antes e
nenhum é reajustado ou reavaliado com parâmetros escolhidos pelo resultado.

Drift usa as mesmas medidas de E11. Intervalos de PSI são definidos no período
de referência de cada comparação, incluindo D1 na comparação D1→D2. Alteração
do alvo não prova concept drift. Não são usados p-valores iid.

Importâncias foram calculadas para as quatro estratégias, em todas as seeds,
com 10 permutações e limiar 0.5.
Valores negativos são mantidos. Atributos correlacionados podem dividir ou
ocultar importância; permutação não mede causalidade. Desvios entre seeds
e entre repetições são registrados separadamente, sem interpretação como IC.

As matrizes mostram contagens médias entre seeds, não observações independentes
adicionais. Erros por grupo incluem tamanho da amostra; cortes pequenos exigem
cautela. As tabelas de exemplos contêm os falsos positivos e falsos negativos
mais confiantes de cada modelo/seed. D0 em m0_por_periodo.csv é desempenho de
treino, portanto não deve ser comparado a D1/D2 como se fosse teste independente.

Artefatos: drift.csv, intervalos_psi.csv, alvo_periodos.csv,
importancias_repeticoes.csv, importancias_seeds.csv, importancias_resumo.csv,
erros_grupos.csv, erros_exemplos.csv, m0_por_periodo.csv e três figuras.
