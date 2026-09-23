# E09 — regressão logística implementada do zero

## Objetivo

Implementar um algoritmo de classificação em nível didático sem recorrer à
implementação pronta e verificar sua coerência por comparação com a regressão
logística do scikit-learn.

O experimento abriu exclusivamente D0. D1 e D2 não participaram da escolha da
configuração nem da comparação.

## Implementação

A probabilidade da classe positiva é calculada por:

```text
p(y=1|x) = sigmoide(xw + b)
```

O treinamento minimiza binary cross-entropy ponderada, com suporte opcional a
regularização L2. Os parâmetros são atualizados por gradiente descendente em
lote:

```text
w <- w - learning_rate * gradiente_w
b <- b - learning_rate * gradiente_b
```

A classe própria inclui:

- inicialização explícita dos pesos e do viés;
- sigmoide numericamente estável;
- custo por `logaddexp`, evitando `log(0)` e overflow;
- cálculo do gradiente e atualização dos parâmetros;
- balanceamento opcional das classes;
- regularização L2 opcional;
- parada por variação relativa do custo;
- `decision_function`, `predict_proba` e `predict`;
- validação de dimensões, valores finitos e alvo binário;
- registro da curva de custo, iterações e convergência.

## Protocolo temporal

- base: 42.800 linhas e 428 semanas de D0;
- quatro folds expansivos, sempre treinando no passado e validando no futuro;
- `StandardScaler` ajustado novamente apenas no treino de cada fold;
- métrica de seleção: MCC médio;
- limiar de classificação fixado em 0,5;
- taxas testadas: 0,2, 0,5 e 1,0;
- balanceamento testado: ausente ou `balanced`;
- L2 igual a zero na comparação, para as duas implementações minimizarem a
  mesma função objetivo;
- limite de 6.000 iterações e tolerância de `1e-11`;
- configurações que não convergiram em todos os folds foram inelegíveis;
- diferenças de MCC abaixo da quarta casa foram tratadas como empate, com
  desempate por menor tempo médio.

O experimento é determinístico: o gradiente em lote e o solver L-BFGS não
possuem inicialização aleatória, portanto a repetição com três seeds não se
aplica.

## Configuração escolhida

- taxa de aprendizado: 1,0;
- pesos de classe: `balanced`;
- L2: 0;
- tolerância: `1e-11`;
- MCC temporal: `0,2945 ± 0,0506`.

O MCC caiu de `0,3631` no primeiro fold para aproximadamente `0,256` nos dois
folds finais, reproduzindo a tendência de degradação temporal já encontrada na
MLP.

## Comparação com a biblioteca

| Implementação | MCC | F1 | Precision | Recall | PR-AUC | Tempo/fold |
|---|---:|---:|---:|---:|---:|---:|
| NumPy, do zero | 0,2945 ± 0,0506 | 0,5659 | 0,5114 | 0,6352 | 0,5331 | 3,013 s |
| scikit-learn | 0,2945 ± 0,0506 | 0,5659 | 0,5114 | 0,6352 | 0,5331 | 0,494 s |

As duas implementações produziram:

- correlação média das probabilidades de `0,9999999`;
- diferença absoluta média de probabilidade de `0,000020`;
- 100% de concordância nas classes previstas;
- métricas de classificação idênticas nos quatro folds.

A implementação de biblioteca foi aproximadamente 6,1 vezes mais rápida. A
comparação de número de iterações não é direta: a versão própria usa gradiente
descendente em lote, enquanto a biblioteca usa L-BFGS, um método de otimização
de segunda ordem aproximada.

## Registro no diário

| Experimento | Alteração | Hipótese | Resultado | Interpretação |
|---|---|---|---|---|
| E09 | Regressão logística NumPy versus scikit-learn; seis configurações em quatro folds de D0 | A implementação própria deve convergir para previsões equivalentes, mas ser mais lenta | Mesmo MCC de 0,2945; correlação de probabilidades 0,9999999; biblioteca 6,1 vezes mais rápida | Hipótese confirmada. A implementação própria reproduz corretamente o algoritmo, e a diferença de tempo é compatível com os otimizadores usados |

## Limitações

O teste demonstra equivalência numérica no dataset e em dados sintéticos, mas
não pretende reproduzir todas as funcionalidades de uma biblioteca madura. A
implementação própria é restrita à classificação binária, gradiente em lote e
pesos `None` ou `balanced`.
