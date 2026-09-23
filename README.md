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
| D2 | 20/12/2023–02/09/2026 | 14.200 | Teste final, ainda congelado |

As semanas imediatamente anteriores a D1 e D2 são removidas por embargo, pois
seus alvos dependem da primeira semana do bloco seguinte.

> **Regra experimental:** não use D2 para escolher atributos, hiperparâmetros,
> limiar de classificação ou qualquer outra decisão. D2 só deve ser aberto pelo
> futuro script de avaliação final depois que o protocolo estiver congelado.

## Ambiente

O ambiente foi testado com Python 3.14.4 em Linux. Na raiz do repositório:

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
ambiente inclui JupyterLab para os notebooks que serão adicionados nas próximas
etapas:

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
`modelos/hiperparametros.json`; o diagnóstico confirma que 100 épocas são um
limite suficiente.

### 4. Escolha do fine-tuning

```bash
python selecionar_finetuning.py
```

Esse comando abre somente D0 e D1. Ele treina M0 em três seeds (`42`, `1337` e
`2024`), divide D1 cronologicamente em D1a/D1b e compara três configurações de
fine-tuning. A escolha atual é a configuração `B_lr_original`, com 11 épocas.

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

### 6. Avaliação final

Ainda não implementada. Ela deverá treinar M0, MFT, MRT e MREC com o protocolo
já congelado e somente então avaliar os quatro modelos em D2.

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
├── dados/
│   ├── MANIFESTO_FONTE.txt
│   └── processados/manifesto.txt
├── modelos/                    # resultados da MLP e fine-tuning
└── resultados/
    ├── auditoria_integridade/
    └── regressao_logistica/
```

## Resultados reproduzidos até aqui

- configuração da MLP: uma camada de 32 neurônios, `alpha=1e-4`,
  `learning_rate_init=3e-3` e `max_iter=100`;
- MCC na validação temporal de D0: `0,3205 ± 0,0620`;
- MCC médio de M0 em D1: `0,2206`;
- configuração de fine-tuning escolhida: taxa original durante 11 épocas;
- MCC médio em D1b após fine-tuning: `0,2116`, ganho de `0,0191` sobre M0;
- zero inconsistências causais de `peak_pos` nas 50.855 continuações de
  passagem auditadas em D0+D1;
- regressão logística do zero: MCC temporal `0,2945 ± 0,0506` em D0;
- implementação própria e scikit-learn: 100% de concordância nas classes e
  correlação `0,9999999` entre probabilidades.

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
