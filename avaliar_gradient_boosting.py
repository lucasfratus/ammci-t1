"""E10: busca de HistGradientBoosting e comparacao das tres familias em D0.

Execute da raiz: python avaliar_gradient_boosting.py
Nao carrega D1/D2. Folds compartilhados com embargo de sete dias.
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ajustar_mlp import folds_temporais
from comparar_regressao_logistica import metricas
from construir_base import ALVO, FEATURES, INICIO, CORTE_D1, SEMANA
from regressao_logistica_zero import RegressaoLogisticaZero

SEEDS = (42, 1337, 2024)
METRICAS = ('mcc', 'f1', 'precision', 'recall', 'pr_auc', 'roc_auc',
            'accuracy', 'tempo_treino_s', 'tempo_inferencia_s', 'mcc_treino')
# 16 configuracoes, fixadas antes de observar resultados. O par taxa/epocas
# compara passos pequenos/longos e grandes/curtos, com custo limitado.
GRADE = [dict(learning_rate=lr, max_iter=epocas, max_leaf_nodes=folhas,
              l2_regularization=l2, class_weight=peso)
         for (lr, epocas), folhas, l2, peso in product(
             [(0.05, 200), (0.1, 100)], [15, 31], [0.0, 1.0],
             [None, 'balanced'])]


def carregar_d0(caminho, manifesto):
    """Confere o hash registrado antes de interpretar qualquer dado."""
    linha = next(l for l in manifesto.read_text().splitlines()
                 if l.startswith('d0 '))
    esperado = linha.split('sha256=')[1].strip()
    observado = hashlib.sha256(caminho.read_bytes()).hexdigest()
    if observado != esperado:
        raise ValueError('Hash de D0 diferente do manifesto; execucao interrompida.')
    d0 = pd.read_csv(caminho, parse_dates=['date'])
    validar_d0(d0)
    return d0, observado


def validar_d0(d0):
    if d0.empty or not set(FEATURES + [ALVO, 'date']).issubset(d0.columns):
        raise ValueError('D0 vazio ou esquema incompleto.')
    if (d0.date.isna().any() or not d0.date.is_monotonic_increasing
            or not d0.date.between(INICIO, CORTE_D1 - 2 * SEMANA).all()):
        raise ValueError('Datas invalidas, fora de D0 ou fora de ordem.')
    if not np.isfinite(d0[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Atributos devem ser numericos e finitos.')
    if set(d0[ALVO].unique()) != {0, 1}:
        raise ValueError('O alvo deve conter as duas classes binarias.')


def obter_folds(d0):
    if d0.date.nunique() < 5:
        raise ValueError('Sao necessarias pelo menos cinco semanas.')
    folds = list(folds_temporais(d0.date, 4))
    for treino, val, _ in folds:
        if d0.loc[treino, 'date'].max() + SEMANA >= d0.loc[val, 'date'].min():
            raise ValueError('Sobreposicao temporal entre treino e validacao.')
        if any(d0.loc[mask, ALVO].nunique() != 2 for mask in (treino, val)):
            raise ValueError('Cada treino e validacao deve conter ambas as classes.')
    return folds


def avaliar(d0, folds, familia, params, seed, config):
    linhas = []
    for numero, (treino, val, inicio_val) in enumerate(folds, 1):
        X_t, X_v = d0.loc[treino, FEATURES], d0.loc[val, FEATURES]
        y_t, y_v = d0.loc[treino, ALVO].to_numpy(), d0.loc[val, ALVO].to_numpy()
        escala = None
        inicio = time.perf_counter()
        if familia == 'Gradient Boosting':
            modelo = HistGradientBoostingClassifier(
                **params, early_stopping=False, random_state=seed)
        else:
            escala = StandardScaler().fit(X_t)
            X_t = escala.transform(X_t)
            modelo = (MLPClassifier(**params, early_stopping=False, random_state=seed)
                      if familia == 'MLP' else RegressaoLogisticaZero(**params))
        modelo.fit(X_t, y_t)
        tempo_treino = time.perf_counter() - inicio
        if familia == 'Logistica do zero' and not modelo.convergiu_:
            raise RuntimeError('Regressao logistica nao convergiu.')
        inicio = time.perf_counter()
        if escala is not None:
            X_v = escala.transform(X_v)
        prob = modelo.predict_proba(X_v)[:, 1]
        tempo_inferencia = time.perf_counter() - inicio
        linhas.append(dict(
            familia=familia, config=config, seed=seed, fold=numero,
            inicio_validacao=inicio_val.date().isoformat(),
            fim_treino=d0.loc[treino, 'date'].max().date().isoformat(),
            n_treino=int(treino.sum()), n_validacao=int(val.sum()),
            iteracoes=int(modelo.n_iter_), tempo_treino_s=tempo_treino,
            tempo_inferencia_s=tempo_inferencia,
            mcc_treino=metricas(y_t, modelo.predict_proba(X_t)[:, 1])['mcc'],
            **metricas(y_v, prob)))
    return linhas


def selecionar(busca):
    # Desempate estavel pela ordem declarada, sem selecionar por ruido de tempo.
    medias = busca.groupby('config', sort=True).mcc.mean()
    return str(medias.idxmax())


def resumir(comparacao):
    linhas = []
    for familia, grupo in comparacao.groupby('familia', sort=False):
        # Media entre seeds em cada fold; desvio temporal calculado de maneira
        # uniforme (ddof=0), sem misturar variacao temporal e aleatoriedade.
        por_fold = grupo.groupby('fold')[list(METRICAS)].mean()
        linha = dict(familia=familia, seeds=grupo.seed.nunique())
        for metrica in METRICAS:
            linha[metrica + '_media'] = float(por_fold[metrica].mean())
            linha[metrica + '_desvio_folds'] = float(por_fold[metrica].std(ddof=0))
        linha['mcc_desvio_seeds'] = float(grupo.groupby('seed').mcc.mean().std(ddof=0))
        linhas.append(linha)
    return pd.DataFrame(linhas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saida', type=Path, default=Path('resultados/gradient_boosting'))
    args = parser.parse_args()
    d0, digest = carregar_d0(Path('dados/processados/base_d0.csv'),
                            Path('dados/processados/manifesto.txt'))
    folds = obter_folds(d0)
    mlp = json.loads(Path('modelos/hiperparametros.json').read_text())
    if mlp['folds'] != 4 or mlp.get('embargo_dias') != 7:
        raise ValueError('Refaca a selecao da MLP com quatro folds e embargo de sete dias.')
    mlp_params = {k: mlp[k] for k in ('hidden_layer_sizes', 'alpha',
                                     'learning_rate_init', 'max_iter')}
    mlp_params['hidden_layer_sizes'] = tuple(mlp_params['hidden_layer_sizes'])
    log = json.loads(Path('resultados/regressao_logistica/configuracao_escolhida.json').read_text())
    log_params = {k: log[k] for k in ('taxa_aprendizado', 'class_weight', 'l2',
                                     'max_iter', 'tolerancia')}
    inicio = time.perf_counter()
    busca = []
    with threadpool_limits(limits=1):
        for i, params in enumerate(GRADE):
            linhas = avaliar(d0, folds, 'Gradient Boosting', params, 42, f'GB{i:02d}')
            busca.extend(linhas)
            print(f'GB{i:02d}: MCC={np.mean([r["mcc"] for r in linhas]):.4f}', flush=True)
        busca = pd.DataFrame(busca)
        nome = selecionar(busca)
        vencedor = GRADE[int(nome[2:])]
        comparacao = busca.loc[busca.config == nome].to_dict('records')
        for seed in SEEDS[1:]:
            comparacao.extend(avaliar(d0, folds, 'Gradient Boosting', vencedor, seed, nome))
        for seed in SEEDS:
            comparacao.extend(avaliar(d0, folds, 'MLP', mlp_params, seed, 'MLP_escolhida'))
        comparacao.extend(avaliar(d0, folds, 'Logistica do zero', log_params, 42, 'LR_escolhida'))
    comparacao = pd.DataFrame(comparacao)
    resumo = resumir(comparacao)
    protocolo = dict(
        experimento='E10', data_utc=datetime.now(timezone.utc).isoformat(),
        dados='somente D0', sha256_d0=digest, features=FEATURES, folds=4, embargo_dias=7,
        seed_busca=42, seeds_comparacao=list(SEEDS), threads=1,
        limiar=0.5, early_stopping=False, grade=GRADE,
        criterio='maior MCC medio; empate exato: primeira configuracao da grade',
        pr_auc='average_precision_score', desvio_folds_ddof=0,
        python=platform.python_version(), plataforma=platform.platform(),
        sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
        tempo_total_s=time.perf_counter() - inicio,
        configuracoes_baselines=dict(mlp=mlp_params, logistica=log_params))
    args.saida.mkdir(parents=True, exist_ok=True)
    for arquivo, tabela in [('busca_folds', busca), ('comparacao_folds', comparacao),
                            ('comparacao_familias', resumo)]:
        tabela.to_csv(args.saida / f'{arquivo}.csv', index=False)
    for arquivo, objeto in [('protocolo', protocolo),
                            ('configuracao_escolhida', dict(config=nome, **vencedor,
                             early_stopping=False, random_state=42))]:
        (args.saida / f'{arquivo}.json').write_text(
            json.dumps(objeto, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tabela = '\n'.join(
        f'| {r.familia} | {r.mcc_media:.4f} ± {r.mcc_desvio_folds:.4f} | '
        f'{r.f1_media:.4f} | {r.pr_auc_media:.4f} | {r.tempo_treino_s_media:.3f} |'
        for r in resumo.itertuples())
    melhor_familia = resumo.loc[resumo.mcc_media.idxmax(), 'familia']
    mlp_linhas = comparacao.loc[comparacao.familia == 'MLP']
    no_limite = int((mlp_linhas.iteracoes >= mlp_params['max_iter']).sum())
    relatorio = f'''# E10 — Gradient Boosting em D0

Hipotese: arvores impulsionadas capturam relacoes nao lineares e podem superar
MLP e regressao logistica nos mesmos quatro folds temporais de D0.

Busca: 16 configuracoes x quatro folds, seed 42, maior MCC medio.
Escolha: {nome}, `{json.dumps(vencedor)}`.
Early stopping desativado; sem padronizacao para arvores. Pesos balanceados
sao calculados pelo estimador somente no treino de cada fold.

| Familia | MCC medio ± desvio temporal | F1 | PR-AUC (AP) | Treino s/fold |
|---|---:|---:|---:|---:|
{tabela}

MLP e Gradient Boosting usam seeds 42, 1337 e 2024; logistica deterministica
usa uma execucao. As configuracoes anteriores da MLP e logistica foram
reavaliadas neste ambiente, sem nova busca. Tempo inclui padronizacao quando
aplicavel. Desvio temporal: ddof=0 sobre medias por fold entre seeds;
variabilidade entre seeds esta registrada separadamente no CSV.

Interpretacao: a tabela compara validacao usada na selecao, nao desempenho
futuro independente. O vencedor de Gradient Boosting fica selecionado para
baseline; a MLP permanece modelo principal dos experimentos de atualizacao.
MCC de treino e validacao por fold permitem examinar sobreajuste.
Maior MCC nesta comparacao: {melhor_familia}.
MLP atingiu o limite de epocas em {no_limite} de {len(mlp_linhas)} ajustes.
O CSV registra iteracoes: atingir o limite de {mlp_params['max_iter']} na MLP exige cautela
quanto a convergencia, sem alterar sua configuracao durante esta comparacao.

Correcao R1: embargo de sete dias em todos os folds. Nenhum alvo de treino
alcanca a validacao. Os periodos de validacao foram preservados e a semana
imediatamente anterior foi retirada do treino. As tres selecoes foram refeitas.
Resultados anteriores estao em resultados/historico_sem_embargo/.

D1 e D2 nao foram carregados. Artefatos: busca_folds.csv, comparacao_folds.csv,
comparacao_familias.csv, configuracao_escolhida.json e protocolo.json.
O protocolo registra data, hash, grade, ambiente, sementes e tempo total.
'''
    (args.saida / 'RELATORIO.md').write_text(relatorio, encoding='utf-8')
    print(resumo[['familia', 'mcc_media', 'mcc_desvio_folds']].to_string(index=False))
    print(f'Resultados em {args.saida}', flush=True)


if __name__ == '__main__':
    main()
