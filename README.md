# AMMCI T1 — mudança temporal na Billboard Hot 100

Trabalho prático da disciplina de Aprendizagem de Máquina e Modelagem de
Conhecimento Incerto. O projeto investiga a degradação temporal e diferentes
estratégias de atualização de modelos na predição do movimento de músicas na
Billboard Hot 100 entre 2013 e 2026.

## Problema

Cada linha representa uma música na semana `t`. O alvo vale `1` quando a música
aparece na semana `t+1` em uma posição melhor e `0` quando cai, permanece na
mesma posição ou deixa o Hot 100.

Os dados são separados cronologicamente:

| Bloco | Período processado | Linhas | Uso |
|---|---|---:|---|
| D0 | 02/01/2013–10/03/2021 | 42.800 | Treinamento histórico e ajuste de hiperparâmetros |
| D1 | 24/03/2021–06/12/2023 | 14.200 | Avaliação recente e escolha do fine-tuning |
| D2 | 20/12/2023–02/09/2026 | 14.200 | Teste final avaliado após congelamento do protocolo |

As semanas imediatamente anteriores a D1 e D2 são removidas por embargo, pois
seus alvos dependem da primeira semana do bloco seguinte. O mesmo embargo
de sete dias agora é aplicado dentro dos quatro folds de D0: exigimos
`data_treino + 7 dias < início_validação`.

> **Regra experimental:** não use D2 para escolher atributos, hiperparâmetros,
> limiar de classificação ou qualquer outra decisão. A avaliação oficial de D2
> ocorreu após o congelamento do protocolo.

## Ambiente

O ambiente original foi testado com Python 3.14.4 em Linux. A revisão R1 foi
executada com Python 3.12 em Windows; versões exatas estão no protocolo E10.
Na raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

As versões das dependências diretas estão fixadas em `requirements.txt`. O
ambiente inclui JupyterLab para os notebooks em `notebooks/`:

```bash
jupyter lab
```

## Dataset

Fonte: [Billboard Hot 100 & more, versão 175](https://www.kaggle.com/datasets/ludmin/billboard),
publicada em 16/09/2026. O dataset é público e o script não precisa de uma
chave da API do Kaggle.

Baixe os cinco CSVs e valide seus hashes com:

```bash
python baixar_dados.py
```

O script fixa explicitamente a versão 175, recusa sobrescrever arquivos
divergentes e valida tanto o ZIP quanto cada CSV. Os CSVs brutos e processados
são ignorados pelo Git devido ao tamanho. Seus hashes estão documentados em
`dados/MANIFESTO_FONTE.txt` e `dados/processados/manifesto.txt`.

## Ordem de execução

Execute sempre a partir da raiz do repositório.

### 1. Auditoria da fonte

```bash
python auditoria_billboard.py --data-dir dados --inicio 2013-01-01
python auditar_integridade.py
```

O primeiro comando apresenta schema, período, duplicidades, cobertura dos joins
e cortes temporais. O segundo confere os hashes, valida a causalidade de
`peak_pos` e procura colisões introduzidas pela normalização. Ele analisa apenas
D0+D1 e grava resultados em `resultados/auditoria_integridade/`.

### 2. Construção de D0, D1 e D2

Durante o desenvolvimento, reconstrua somente D0/D1, conferindo os hashes já
registrados e preservando o manifesto:

```bash
python preparar_desenvolvimento.py
```

O construtor original continua disponível para gerar os três blocos quando
necessário. Gerar D2 não autoriza analisá-lo antes do congelamento:

```bash
python construir_base.py
```

O comando gera `dados/processados/base_d0.csv`, `base_d1.csv` e `base_d2.csv`.
Os atributos de histórico são calculados antes do recorte de 2013, e as duas
fronteiras recebem embargo de uma semana para impedir vazamento pelo alvo.

### 3. Busca temporal da MLP

```bash
python ajustar_mlp.py
python diagnostico_convergencia.py
```

`ajustar_mlp.py` abre somente D0 e usa quatro folds cronológicos. O scaler é
reajustado dentro de cada fold. Os hiperparâmetros escolhidos são gravados em
`modelos/hiperparametros.json`. A busca mantém as 12 configurações originais,
com embargo e teto de 800 épocas. A parada automática depende da perda de
treino, sem validação aleatória interna. O diagnóstico compara tetos 100/800
nos quatro folds e três seeds: todos os 12 ajustes revisados convergiram,
com máximo observado de 171 épocas. Convergência não garante melhor MCC.

### 4. Escolha do fine-tuning

```bash
python selecionar_finetuning.py
```

Esse comando abre somente D0 e D1. Ele treina M0 em três seeds (`42`, `1337` e
`2024`), divide D1 cronologicamente em D1a/D1b e compara três configurações de
fine-tuning. As taxas/regularizações A e B acompanham a MLP escolhida em D0.
A escolha revisada é `B_lr_original`, com **uma época** e `alpha=0,01`.
As 11 épocas anteriores pertencem ao histórico substituído.

### 5. Regressão logística implementada do zero

```bash
python -m unittest -v test_regressao_logistica_zero.py
python comparar_regressao_logistica.py
```

O primeiro comando testa estabilidade numérica, validação de erros,
regularização L2 e equivalência com a biblioteca em dados controlados. O
segundo abre somente D0, escolhe a configuração própria em quatro folds
temporais e a compara com `sklearn.linear_model.LogisticRegression`. Os
resultados são gravados em `resultados/regressao_logistica/`.

### 6. Gradient Boosting e comparação das três famílias (E10)

```bash
python -m unittest -v test_gradient_boosting.py
python avaliar_gradient_boosting.py
```

O script confere o hash de D0 no manifesto e reutiliza exatamente os quatro
folds compartilhados, com embargo. Testa 16 configurações de `HistGradientBoostingClassifier`
com seed 42 e seleciona pelo maior MCC médio. O early stopping fica desativado
para evitar uma separação aleatória interna. Árvores não usam padronização.

A configuração escolhida e a MLP são comparadas com seeds 42, 1337 e 2024;
a regressão logística do zero, determinística, é executada uma vez por fold.
As configurações já escolhidas das duas famílias anteriores são reavaliadas
no mesmo ambiente, sem nova busca. São registradas métricas de validação,
MCC de treino, tempos e variação entre folds e entre seeds separadamente.
PR-AUC corresponde a average precision. O desvio entre folds usa `ddof=0`.

Resultados e diário E10: `resultados/gradient_boosting/RELATORIO.md`.
A pasta também contém a busca completa, comparação por fold e por família,
configuração selecionada e protocolo com hash, ambiente, grade e sementes.
D1 e D2 não são carregados. A MLP continua sendo o modelo principal.

**Correção R1:** o embargo agora impede que alvos de treino alcancem a
validação. As três buscas foram refeitas. Os resultados antigos sem embargo
foram excluídos desta cópia do projeto; não são os resultados atuais.
A validação usada para seleção continua não sendo teste independente.

### 7. Análise exploratória, drift, diário e notebooks

```bash
python analisar_dados_drift.py
python -m unittest -v test_regressao_logistica_zero.py test_gradient_boosting.py test_protocolo_drift.py
```

O notebook 01 chama o script de AED/drift e gera tabelas e seis figuras. O 02
apresenta os experimentos já executados pelos scripts, sem repetir buscas
demoradas. Os notebooks já estão salvos com suas saídas. Para executá-los,
ative o ambiente do projeto, rode `jupyter lab`, abra o notebook e escolha
“Executar todas as células”. Salve o notebook após a execução.

PSI usa quantis de D0; KS e Wasserstein comparam atributos numéricos;
Jensen–Shannon compara categorias e alvo. As medidas são descritivas: não
usamos p-valores iid, pois há dependência temporal e repetição de músicas.
Mudança de distribuição não demonstra concept drift nem causalidade.

Veja `DIARIO_EXPERIMENTOS.md` e `resultados/revisao_temporal/RELATORIO.md`.
O diário identifica os registros herdados cujo artefato bruto ainda falta,
sem contá-los como novas execuções. O plano recebido, anterior às correções,
não está incluído nesta cópia do projeto.

### 8. Protocolo e avaliação final

```bash
python preparar_protocolo_final.py
```

Esse comando cria `protocolo/protocolo_final.rascunho.json` e seu SHA-256,
sem abrir D2. O rascunho precisa da revisão da equipe e do congelamento
formal, conforme a etapa 13 do plano, antes de qualquer avaliação futura.

A implementação e a avaliação real foram concluídas em 26/09/2026, após revisão
e congelamento autorizados pelo usuário. Veja `RESULTADOS_FINAIS.md`.
Os comandos abaixo documentam a sequência usada; os destinos existentes
são protegidos contra sobrescrita:

```bash
python protocolo_final.py --responsavel "Nome de quem revisou" --confirmar-revisao
python construir_base.py
python experimento_final.py
python analisar_resultados_finais.py
```

`construir_base.py` materializa também o CSV de D2. O experimento treina os
16 modelos (quatro estratégias MLP e GB em três seeds, mais uma regressão
determinística), salva seus estados e somente então carrega D2 para avaliação.
As métricas, previsões, tempos, curvas e hashes ficam em `resultados/final/`.
Drift entre os três períodos, importância por permutação e erros por grupos
ficam em `resultados/analises_finais/`. Nenhuma análise reajusta modelos.

O congelamento verifica versões e hashes do código, configurações e dados.
Se código ou ambiente mudarem, a execução é bloqueada. Antes de congelar,
regenere o rascunho quando houver alterações. As pastas de saída não são
sobrescritas: uma falha preserva os artefatos e seu estado em `execucao.json`.
Se D2 já tiver sido aberto, documente qualquer recuperação como continuação
da mesma avaliação; não use seus resultados para escolher novos parâmetros.

Os notebooks 03 e 04 apenas apresentam artefatos: sem a execução final,
mostram a pendência explicitamente. Para atualizar suas saídas, abra-os no
Jupyter e execute todas as células após gerar os resultados finais.
Para executar os testes do projeto:

```bash
python -m unittest discover -v
```

Os testes finais usam conjuntos sintéticos temporários, sem acessar D2 real.

## Estrutura atual

```text
.
├── baixar_dados.py             # download imutável da versão 175
├── auditoria_billboard.py      # inventário inicial e cobertura dos joins
├── auditar_integridade.py      # peak_pos, hashes e normalização
├── construir_base.py           # engenharia de atributos e D0/D1/D2
├── ajustar_mlp.py              # busca temporal de hiperparâmetros
├── diagnostico_convergencia.py # número de épocas da MLP
├── selecionar_finetuning.py    # seleção em D1a/D1b
├── regressao_logistica_zero.py # algoritmo didático em NumPy
├── comparar_regressao_logistica.py
├── test_regressao_logistica_zero.py
├── avaliar_gradient_boosting.py # E10: seleção em D0 e três famílias
├── test_gradient_boosting.py    # protocolo temporal, integridade e métricas
├── preparar_desenvolvimento.py # reconstrói D0/D1 sem gerar D2
├── revisar_convergencia.py     # quatro folds e três seeds, tetos 100/800
├── analisar_dados_drift.py     # AED e drift D0 → D1
├── preparar_protocolo_final.py # rascunho para revisão
├── protocolo_final.py         # validação e congelamento explícito
├── experimento_final.py       # treinamento e avaliação final
├── analisar_resultados_finais.py # drift, importância e erros
├── test_experimento_final.py  # integração com dados sintéticos
├── notebooks/
├── protocolo/
├── dados/
│   ├── MANIFESTO_FONTE.txt
│   └── processados/manifesto.txt
├── modelos/                    # resultados da MLP e fine-tuning
└── resultados/
    ├── auditoria_integridade/
    └── regressao_logistica/
```

## Resultados revisados (R1)

- MLP: 32 neurônios, `alpha=0,01`, `learning_rate_init=0,003`, teto de 800 épocas;
- MCC da busca da MLP (seed 42): `0,3213 ± 0,0649`;
- comparação em quatro folds, médias entre seeds: GB `0,3542`, MLP `0,3134`,
  regressão logística `0,2949`; desvios e métricas completas nos CSVs E10;
- MCC médio de M0 em D1: `0,2152`;
- fine-tuning: taxa original durante uma época; MCC D1b `0,2120`,
  ganho de `0,0218` sobre M0 no mesmo D1b;
- proporção positiva: D0 `36,16%`, D1 `31,30%`; maior PSI em `digital_rank`;
- zero inconsistências causais de `peak_pos` nas 50.855 continuações de
  passagem auditadas em D0+D1;
- regressão própria e scikit-learn: 100% de concordância nas classes.

Os desvios temporais E10 usam `ddof=0` sobre médias por fold; E09 usa
desvio amostral `ddof=1`. Na tabela final futura, os desvios serão entre seeds.
Resultados de busca (seed 42) não devem ser confundidos com médias entre seeds.

## Reprodutibilidade e dados ignorados

As seeds, cortes, features e censuras são constantes explícitas nos scripts.
Arquivos grandes não são versionados; compare os hashes dos manifestos depois
de qualquer reconstrução. Mudanças em cortes, atributos, limiares ou seeds
devem ser registradas como um novo experimento antes da execução.

## Uso de IA generativa

Ferramentas de IA generativa foram usadas como apoio à auditoria, organização
do projeto, revisão metodológica e documentação. Todo código e todas as decisões
devem ser revisados pela equipe. O short paper deverá incluir o apêndice exigido
pelo enunciado com ferramenta, finalidade, conteúdo aproveitado e validação
realizada pelos integrantes.
