# Diário experimental consolidado

Este diário organiza as decisões e os resultados oficiais do projeto. R1 corrige a validação temporal e substitui os resultados anteriores. Datas e tempos ausentes não foram inventados. As hipóteses abaixo explicitam as perguntas investigadas a partir dos scripts e artefatos; não constituem um registro prévio datado de hipóteses.

**Critério de contagem:** execuções de configurações distintas são identificadas por seus parâmetros e resultados. A tabela de E05-R1 documenta as 12 configurações efetivamente treinadas em quatro folds, fornecendo mais de oito execuções relevantes verificáveis. Essas configurações são partes de E05, não experimentos adicionais à mesma busca. E02 é uma decisão herdada, E03 não possui evidência quantitativa e E08 é uma análise de E07: nenhum desses três é contado como execução independente. E04 e E11 compartilham a análise de D0/D1; E13–E15 são análises diferentes produzidas pelo mesmo script, não três novos treinamentos.

Os IDs históricos foram preservados para permitir relacionar o diário aos artefatos. As métricas de seleção em D0/D1 não são métricas de teste independente. Os resultados finais são os de E12.

## E01

- **hipótese investigada:** O pico histórico deve usar apenas informações disponíveis até a semana da previsão.
- **pergunta:** peak_pos antecipa informacao futura?
- **dados:** D0+D1 brutos
- **alteracao:** auditoria de causalidade
- **resultado:** 0 inconsistencias nas 50.855 continuacoes auditadas
- **decisao:** manter construcao causal
- **artefato:** resultados/auditoria_integridade/resumo.json
- **origem:** historico anterior a R1
- **interpretação:** A auditoria não encontrou inconsistências nas continuações verificadas; isso sustenta a construção causal de peak_pos, sem provar ausência de todo tipo de vazamento.


## E02

- **hipótese investigada:** Separar passagens pode distinguir uma reentrada de uma estreia e representar melhor o histórico.
- **pergunta:** Como tratar reentradas?
- **dados:** D0+D1
- **alteracao:** passagens e indicador de reentrada
- **resultado:** regra de hiato implementada e auditada
- **decisao:** manter regra e distinguir passagem de estreia da carreira
- **artefato:** construir_base.py; plano recebido (não incluído nesta cópia; evidência quantitativa de E03 pendente)
- **origem:** registro herdado do plano
- **interpretação:** A regra está implementada; este registro herdado não demonstra, isoladamente, ganho preditivo.


## E03

- **hipótese investigada:** O limiar de hiato pode modificar a segmentação das passagens e afetar os atributos.
- **pergunta:** Qual limiar de hiato usar?
- **dados:** D0+D1 conforme plano recebido
- **alteracao:** 14/28/56 dias
- **resultado:** 28 dias mantidos no plano anterior
- **decisao:** preservar; recuperar artefato quantitativo original para o diario final
- **artefato:** plano recebido (não incluído nesta cópia; evidência quantitativa de E03 pendente)
- **origem:** herdado; artefato bruto nao localizado
- **interpretação:** Sem o artefato original de 14/28/56 dias, não é possível comprovar a comparação nem afirmar superioridade quantitativa de 28 dias. Não contar como execução comprovada.


## E04

- **hipótese investigada:** A frequência de melhora pode variar entre períodos e afetar a avaliação dos modelos.
- **pergunta:** A proporcao do alvo muda?
- **dados:** D0/D1
- **alteracao:** comparacao temporal descritiva
- **resultado:** 0.3616 -> 0.3130
- **decisao:** priorizar MCC/F1/PR-AUC
- **artefato:** resultados/aed_drift/resumo_base.csv
- **origem:** recalculado em R1
- **interpretação:** A proporção caiu de 36,16% para 31,30%, cerca de 4,86 pontos percentuais; acurácia isolada não basta para avaliar as classes.


## E05-R1

- **hipótese investigada:** Arquitetura, regularização e taxa de aprendizado podem alterar a generalização temporal da MLP.
- **pergunta:** Qual MLP generaliza melhor nos folds corrigidos?
- **dados:** D0
- **alteracao:** 12 configuracoes; embargo sete dias; teto 800; seed 42
- **resultado:** 32 neuronios, alpha=0.01, lr=0.003; MCC=0.3213
- **decisao:** usar configuracao revisada
- **artefato:** modelos/busca_completa.json
- **origem:** executado em R1
- **interpretação:** A configuração selecionada teve o maior MCC médio nesta grade em D0. Isso justifica sua seleção, mas não garante superioridade em dados futuros.


### Configurações executadas e verificáveis de E05-R1

Fonte: `modelos/busca_completa.json`. Identificadores C01–C12 atribuídos aqui para localizar cada configuração no arquivo, na ordem salva. Cada linha corresponde a quatro treinamentos/validações temporais com seed 42, teto de 800 épocas e embargo de sete dias. Os desvios desta tabela são entre folds (ddof=0), não entre sementes.

A hipótese comum é a de E05: variar capacidade, penalização dos pesos e tamanho das atualizações pode melhorar a generalização temporal. As alterações exatas e a interpretação da seleção estão em cada linha. Não são 12 repetições do mesmo modelo nem 12 testes em D2.

| Execução | Neurônios por camada | Alpha | Taxa | MCC médio | Desvio entre folds | Interpretação/decisão |
|---|---|---:|---:|---:|---:|---|
| E05-C01 | 32 | 0.01 | 0.003 | 0.321277 | 0.064884 | Selecionada: maior MCC médio |
| E05-C02 | 32 | 0.01 | 0.001 | 0.320735 | 0.063404 | Não selecionada: MCC médio menor |
| E05-C03 | 32 | 0.001 | 0.003 | 0.319719 | 0.065125 | Não selecionada: MCC médio menor |
| E05-C04 | 32 | 0.001 | 0.001 | 0.319367 | 0.063976 | Não selecionada: MCC médio menor |
| E05-C05 | 32 | 0.0001 | 0.003 | 0.315428 | 0.062361 | Não selecionada: MCC médio menor |
| E05-C06 | 32 | 0.0001 | 0.001 | 0.315161 | 0.063339 | Não selecionada: MCC médio menor |
| E05-C07 | 64/32 | 0.01 | 0.001 | 0.293768 | 0.046622 | Não selecionada: MCC médio menor |
| E05-C08 | 64/32 | 0.001 | 0.001 | 0.291368 | 0.040217 | Não selecionada: MCC médio menor |
| E05-C09 | 64/32 | 0.0001 | 0.001 | 0.291162 | 0.040956 | Não selecionada: MCC médio menor |
| E05-C10 | 64/32 | 0.01 | 0.003 | 0.286847 | 0.038130 | Não selecionada: MCC médio menor |
| E05-C11 | 64/32 | 0.001 | 0.003 | 0.277932 | 0.026527 | Não selecionada: MCC médio menor |
| E05-C12 | 64/32 | 0.0001 | 0.003 | 0.272272 | 0.034957 | Não selecionada: MCC médio menor |

## E06-R1

- **hipótese investigada:** O teto de 100 épocas pode interromper o treino antes do critério de convergência.
- **pergunta:** 100 epocas truncam o treinamento?
- **dados:** D0
- **alteracao:** 100 vs 800; quatro folds e tres seeds
- **resultado:** 7/12 convergiram com 100; 12/12 com 800; maximo observado 171
- **decisao:** manter teto revisado e parada pela perda; convergencia nao garante melhor MCC
- **artefato:** resultados/convergencia/diagnostico.csv
- **origem:** executado em R1
- **interpretação:** O teto de 800 eliminou os avisos nas 12 combinações fold/semente verificadas, sem obrigar 800 épocas. Convergência numérica não garante generalização.


## E07-R1

- **hipótese investigada:** Continuar o treinamento com dados recentes pode melhorar M0; taxa, regularização e duração podem mudar o ganho.
- **pergunta:** Como atualizar a MLP revisada?
- **dados:** D0 para M0; D1a/D1b para FT
- **alteracao:** tres taxas/regularizacoes relativas a M0; 0-30 epocas; tres seeds
- **resultado:** B_lr_original, 1 epoca(s), MCC=0.2120, ganho=0.0218
- **decisao:** substituir as 11 epocas antigas pela escolha revisada
- **artefato:** modelos/curvas_finetuning.csv
- **origem:** executado em R1
- **interpretação:** B por uma época teve maior MCC médio em D1b entre as configurações avaliadas. É resultado de seleção em D1b, não evidência independente de melhora em D2.


## E08-R1

- **hipótese investigada:** Regularização forte durante a atualização pode reduzir sobreajuste aos dados recentes.
- **pergunta:** L2 forte melhora a atualizacao?
- **dados:** D1a/D1b
- **alteracao:** configuracao C, alpha=1.0; integrada ao E07-R1
- **resultado:** melhor MCC medio C=0.2061
- **decisao:** nao selecionada; nao contar como execucao independente de E07-R1
- **artefato:** modelos/curvas_finetuning.csv
- **origem:** analise de E07-R1
- **interpretação:** O melhor MCC da configuração C ficou abaixo de B. A hipótese não foi favorecida nessa comparação; análise complementar de E07, sem nova execução independente.


## E09-R1

- **hipótese investigada:** A regressão manual deve produzir previsões próximas às da biblioteca sob condições equivalentes.
- **pergunta:** A regressao propria continua equivalente com embargo?
- **dados:** D0
- **alteracao:** repetir seis configuracoes e comparacao NumPy/sklearn
- **resultado:** MCC=0.2949; concordancia de classes 100%
- **decisao:** manter taxa 1, pesos balanceados
- **artefato:** resultados/regressao_logistica/resumo.json
- **origem:** executado em R1
- **interpretação:** A concordância de classes foi de 100% e a correlação média de probabilidades de 0,99999990. Isso sustenta a coerência da implementação nas condições testadas.


## E10-R1

- **hipótese investigada:** Árvores combinadas podem capturar relações que diferem das aprendidas pela MLP e pela regressão logística.
- **pergunta:** Como Gradient Boosting se compara as outras familias?
- **dados:** D0
- **alteracao:** 16 configuracoes com embargo; comparacao tres seeds para MLP/GB
- **resultado:** Gradient Boosting: MCC=0.3542; MLP: MCC=0.3134; Logistica do zero: MCC=0.2949
- **decisao:** GB como baseline; MLP permanece principal para atualizacao
- **artefato:** resultados/gradient_boosting/protocolo.json
- **origem:** executado em R1
- **interpretação:** GB teve maior MCC de validação, mas MLP permanece principal por permitir continuidade do treinamento por gradiente. Esses resultados não substituem o teste final.


## E11

- **hipótese investigada:** Os atributos e a frequência do alvo podem apresentar mudanças de distribuição entre D0 e D1.
- **pergunta:** Quais atributos mudaram entre os periodos?
- **dados:** D0/D1
- **alteracao:** PSI, KS, Wasserstein e Jensen-Shannon; estatisticas descritivas
- **resultado:** maior PSI em digital_rank; queda de 4,86 pontos percentuais no alvo
- **decisao:** mudanca de distribuicao documentada; nao afirmar concept drift ou causalidade
- **artefato:** resultados/aed_drift/drift_d0_d1.csv
- **origem:** executado em R1

Seeds atuais: 42 na busca; 42, 1337 e 2024 nos experimentos estocásticos principais. Regressão logística determinística. Tempos estão nas tabelas por fold; E11 registra a data em seu protocolo.
- **interpretação:** As medidas documentam mudança de distribuição, mas não identificam por si sós concept drift ou causalidade.


## E12 — Avaliação real

- **hipótese investigada:** Atualizar M0, retreinar com D0+D1 ou usar apenas D1 pode melhorar a generalização futura, com custos diferentes.
- **pergunta:** Atualizar a MLP melhora a previsão no período futuro?
- **dados:** D0/D1 para treino; D2 com 14.200 observações para avaliação final.
- **alteração realizada:** comparar M0 (D0), MFT (M0 atualizado em D1 por uma época), MRT (nova MLP em D0+D1) e MREC (nova MLP em D1), além de GB e regressão logística treinados em D0.
- **protocolo:** revisão e congelamento autorizados explicitamente pelo usuário;
  todos os 16 modelos treinados antes da leitura de D2 pelo avaliador.
- **resultado:** MCC médio GB=0,1978; MRT=0,1789; MREC=0,1715;
  MFT=0,1681; M0=0,1623; LR=0,1309.
- **decisão:** MRT tem maior MCC entre as estratégias MLP; GB tem maior MCC geral.
  Ganho de MFT é pequeno e acompanhado de queda de recall/F1. Nenhum reajuste.
- **artefatos:** `resultados/final/manifesto_resultados.json`, `RESULTADOS_FINAIS.md`.
- **interpretação:** MRT obteve o maior MCC entre as MLPs e GB o maior MCC geral. MFT ganhou pouco em MCC e perdeu recall/F1; atualização não melhorou todas as métricas. Desvios entre sementes não demonstram significância estatística.


## E13 — Drift entre os três períodos

- **hipótese investigada:** as distribuições dos atributos e do alvo podem mudar em D2 em relação a D0/D1.
- **dados:** D0, D1 e D2; análise descritiva posterior à avaliação oficial.
- **alteração realizada:** comparar D0→D1, D0→D2 e D1→D2 com PSI, KS, Wasserstein e Jensen–Shannon. Os intervalos de PSI vêm do período de referência de cada comparação.
- **resultado:** em D0→D2, maior PSI em digital_rank (1,2124), seguido por album_rank (0,1442). Proporção de melhora: D0=36,16%, D1=31,30%, D2=32,18%.
- **interpretação:** há mudanças de distribuição, mas os resultados não comprovam mudança na relação entre atributos e alvo nem explicam causalmente a queda de desempenho.
- **decisão:** reportar as mudanças e limitações, sem selecionar atributos ou reajustar modelos com D2.
- **artefatos:** `resultados/analises_finais/drift.csv`, `resultados/analises_finais/intervalos_psi.csv`, `resultados/analises_finais/alvo_periodos.csv`, `resultados/analises_finais/drift.png`.

## E14 — Importância dos atributos

- **hipótese investigada:** as estratégias de treinamento podem depender de atributos diferentes para prever melhoras.
- **dados:** modelos já treinados, avaliados em D2 após o teste oficial.
- **alteração realizada:** permutação de cada atributo para M0/MFT/MRT/MREC nas três sementes, com dez repetições por atributo e queda de MCC como importância.
- **resultado:** em MRT, maiores importâncias médias em album_rank (0,0518), variacao_1s (0,0442) e peak_pos (0,0400). As tabelas registram as quatro estratégias, incluindo valores negativos.
- **interpretação:** o modelo utiliza informações de histórico e rankings auxiliares. Importância por permutação não mede causalidade; correlações entre atributos podem dividir ou ocultar importância. Desvios entre sementes e entre permutações são distintos.
- **decisão:** usar os resultados para interpretação, sem remover atributos ou retreinar modelos com base em D2.
- **artefatos:** `resultados/analises_finais/importancias_repeticoes.csv`, `resultados/analises_finais/importancias_seeds.csv`, `resultados/analises_finais/importancias_resumo.csv`, `resultados/analises_finais/importancias.png`.

## E15 — Análise de erros

- **hipótese investigada:** métricas agregadas podem esconder dificuldades específicas por posição, tempo na parada ou período.
- **dados:** previsões salvas da avaliação oficial em D2.
- **alteração realizada:** calcular métricas nos recortes pré-definidos no protocolo e selecionar até dez falsos positivos e dez falsos negativos de maior confiança por modelo/semente.
- **resultado:** MRT teve recall médio de 3,65% no top 10 e 35,21% nas posições 41–100; por tempo na parada, 8,56% em 13+ semanas e 52,55% em 1–4 semanas.
- **interpretação:** o desempenho varia entre grupos e a rede deixa passar muitas melhoras em alguns deles. Os exemplos ilustram falhas, sem determinar suas causas; tamanhos e composição dos grupos precisam ser considerados.
- **decisão:** discutir as limitações por grupo e apresentar exemplos de erros; não alterar limiar ou modelo em função deles.
- **artefatos:** `resultados/analises_finais/erros_grupos.csv`, `resultados/analises_finais/erros_exemplos.csv`, `resultados/analises_finais/matrizes_confusao.png`.

## Reprodutibilidade e limites dos registros

- Busca MLP: seed 42. Experimentos estocásticos principais: seeds 42, 1337 e 2024. Regressão logística determinística.
- E05 usa desvio populacional entre folds (ddof=0); E09 registra desvio amostral entre folds (ddof=1). E10 usa ddof=0 sobre médias por fold e separa variação entre sementes. E12 usa desvio amostral entre sementes (ddof=1); LR tem apenas uma execução.
- Tempos de seleção e comparação constam dos CSVs por fold; os tempos finais estão em `resultados/final/tempos.csv`.
- Configurações e hashes da avaliação oficial: `protocolo/protocolo_final.congelado.json`. Eventos: `resultados/final/execucao.json`. Integridade: `resultados/final/manifesto_resultados.json` e `resultados/analises_finais/manifesto_analises.json`.
- E01 possui artefato de auditoria anterior à revisão. E02 é decisão herdada. E03 depende de evidência não localizada. E08 reutiliza resultados de E07. Não apresentar esses registros como novos treinamentos independentes.
