"""E12: treina todos os modelos, depois abre D2 e avalia o protocolo congelado."""
import argparse
import copy
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from protocolo_final import carregar_bloco, ler_protocolo, salvar_json, sha256
from regressao_logistica_zero import RegressaoLogisticaZero
from selecionar_finetuning import preparar_finetuning, uma_epoca

METRICAS = ['mcc', 'f1', 'precision', 'recall', 'pr_auc', 'roc_auc', 'accuracy']


def metricas(y, prob, limiar):
    y, prob = np.asarray(y), np.asarray(prob, dtype=float)
    if len(y) == 0 or y.shape != prob.shape or not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError('Probabilidades/alvos inválidos.')
    pred = (prob >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(mcc=float(matthews_corrcoef(y, pred)),
                f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                pr_auc=float(average_precision_score(y, prob)) if (y == 1).any() else np.nan,
                roc_auc=float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
                accuracy=float(accuracy_score(y, pred)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def probabilidades(artefato, X):
    escala = artefato['scaler']
    entrada = escala.transform(X) if escala is not None else X
    return artefato['estimador'].predict_proba(entrada)[:, 1]


def ajustar(modelo, X, y):
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always', ConvergenceWarning)
        modelo.fit(X, y)
    conv = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
    if hasattr(modelo, 'convergiu_'):
        conv = bool(modelo.convergiu_)
    return conv, [str(a.message) for a in avisos]


def complexidade(modelo):
    if isinstance(modelo, MLPClassifier):
        return int(sum(a.size for a in modelo.coefs_ + modelo.intercepts_)), 'pesos_e_vieses'
    if isinstance(modelo, RegressaoLogisticaZero):
        return int(modelo.pesos_.size + 1), 'pesos_e_vies'
    # Proxy documentada, sem depender de estruturas privadas do sklearn.
    return int(modelo.n_iter_ * modelo.max_leaf_nodes), 'limite_superior_folhas'


def treinar_modelos(d0, d1, p):
    features, hp = p['features'], p['hiperparametros']
    params = {k: hp[k] for k in ('hidden_layer_sizes', 'alpha', 'learning_rate_init', 'max_iter')}
    params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
    blocos = {'D0': d0, 'D1': d1, 'D0+D1': pd.concat([d0, d1], ignore_index=True)}
    modelos, registros, curvas = {}, [], []

    def registrar(nome, seed, estimador, scaler, dados, tempo, convergiu, avisos,
                  tempo_total=None, epocas=None):
        chave = f'{nome}_seed{seed}'
        qtd, tipo = complexidade(estimador)
        modelos[chave] = dict(modelo=nome, seed=seed, estimador=estimador, scaler=scaler)
        registros.append(dict(modelo=nome, seed=seed, dados_treino=dados,
                              tempo_treino_s=tempo, tempo_desde_zero_s=tempo if tempo_total is None else tempo_total,
                              convergiu=convergiu, avisos=' | '.join(avisos),
                              iteracoes=int(estimador.n_iter_) if epocas is None else epocas,
                              complexidade=qtd, tipo_complexidade=tipo))
        for epoca, perda in enumerate(getattr(estimador, 'loss_curve_', []), 1):
            curvas.append(dict(modelo=nome, seed=seed, epoca=epoca, perda=float(perda)))
        print(f'Treinado {chave} com {dados}', flush=True)

    for seed in p['seeds']:
        for nome, conjunto in [('M0', 'D0'), ('MRT', 'D0+D1'), ('MREC', 'D1')]:
            df = blocos[conjunto]
            inicio = time.perf_counter()
            escala = StandardScaler().fit(df[features])
            modelo = MLPClassifier(**params, random_state=seed, early_stopping=False)
            convergiu, avisos = ajustar(modelo, escala.transform(df[features]), df.alvo.to_numpy())
            tempo = time.perf_counter() - inicio
            registrar(nome, seed, modelo, escala, conjunto, tempo, convergiu, avisos)
            if nome == 'M0':
                m0, escala0, tempo0 = modelo, escala, tempo
        inicio = time.perf_counter()
        ft = p['finetuning']
        mft = preparar_finetuning(m0, {k: ft[k] for k in ('learning_rate_init', 'alpha')})
        Xt = escala0.transform(d1[features])
        # A copia deve preservar exatamente a funcao antes da primeira atualizacao.
        np.testing.assert_array_equal(m0.predict_proba(Xt[:8]), mft.predict_proba(Xt[:8]))
        for _ in range(ft['epocas']):
            uma_epoca(mft, Xt, d1.alvo.to_numpy())
        tempo = time.perf_counter() - inicio
        registrar('MFT', seed, mft, copy.deepcopy(escala0), 'D0 -> D1', tempo,
                  None, ['Epocas fixas; convergencia nao e criterio de parada do FT.'],
                  tempo_total=tempo0+tempo, epocas=ft['epocas'])
    gb = p['comparadores']['gradient_boosting']
    params_gb = {k: gb[k] for k in ('learning_rate', 'max_iter', 'max_leaf_nodes',
                                   'l2_regularization', 'class_weight')}
    for seed in p['comparadores']['seeds_gb']:
        inicio = time.perf_counter()
        modelo = HistGradientBoostingClassifier(**params_gb, random_state=seed, early_stopping=False)
        _, avisos = ajustar(modelo, d0[features], d0.alvo.to_numpy())
        registrar('GB', seed, modelo, None, 'D0', time.perf_counter()-inicio, None, avisos)
    inicio = time.perf_counter()
    lr = p['comparadores']['regressao_logistica_zero']
    modelo = RegressaoLogisticaZero(**{k: lr[k] for k in
        ('taxa_aprendizado', 'max_iter', 'l2', 'tolerancia', 'class_weight')})
    escala = StandardScaler().fit(d0[features])
    convergiu, avisos = ajustar(modelo, escala.transform(d0[features]), d0.alvo.to_numpy())
    registrar('LR', -1, modelo, escala, 'D0', time.perf_counter()-inicio, convergiu, avisos)
    return modelos, pd.DataFrame(registros), pd.DataFrame(curvas)


def avaliar_modelos(modelos, d2, p):
    linhas, previsoes = [], []
    for artefato in modelos.values():
        inicio = time.perf_counter()
        prob = probabilidades(artefato, d2[p['features']])
        tempo = time.perf_counter() - inicio
        nome, seed = artefato['modelo'], artefato['seed']
        linhas.append(dict(modelo=nome, seed=seed, tempo_inferencia_s=tempo,
                           **metricas(d2.alvo.to_numpy(), prob, p['limiar'])))
        pred = d2[['date', 'title', 'artist', 'alvo']].copy()
        pred.insert(0, 'row_id', np.arange(len(d2)))
        pred['modelo'], pred['seed'] = nome, seed
        pred['probabilidade'] = prob
        pred['predicao'] = (prob >= p['limiar']).astype(int)
        previsoes.append(pred)
    return pd.DataFrame(linhas), pd.concat(previsoes, ignore_index=True)


def resumir(metricas_seeds):
    linhas = []
    for nome, grupo in metricas_seeds.groupby('modelo'):
        linha = dict(modelo=nome, execucoes=len(grupo))
        for metrica in METRICAS:
            linha[metrica+'_media'] = grupo[metrica].mean()
            linha[metrica+'_desvio'] = grupo[metrica].std(ddof=1)
        linhas.append(linha)
    resumo = pd.DataFrame(linhas)
    base = float(resumo.loc[resumo.modelo == 'M0', 'mcc_media'].iloc[0])
    resumo['delta_mcc_vs_m0'] = resumo.mcc_media - base
    resumo['delta_mcc_relativo'] = resumo.delta_mcc_vs_m0 / abs(base) if base != 0 else np.nan
    return resumo


def executar(caminho_protocolo, dados, saida):
    p = ler_protocolo(caminho_protocolo)
    # Todos os caminhos de dados permanecem separados; nao ha glob de CSVs.
    d0, d1 = (carregar_bloco(dados, nome, p) for nome in ('D0', 'D1'))
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=False)
    salvar_json(saida/'protocolo_congelado.json', p)
    (saida/'protocolo_congelado.sha256').write_text(sha256(saida/'protocolo_congelado.json')+'\n')
    eventos = []

    def evento(etapa):
        eventos.append(dict(etapa=etapa, utc=datetime.now(timezone.utc).isoformat()))
        salvar_json(saida/'execucao.json', dict(eventos=eventos, status=etapa))

    try:
        evento('TREINANDO')
        with threadpool_limits(limits=1):
            modelos, treino, curvas = treinar_modelos(d0, d1, p)
            treino.to_csv(saida/'treinamento.csv', index=False)
            curvas.to_csv(saida/'curvas_treino.csv', index=False)
            joblib.dump(modelos, saida/'modelos.joblib', compress=3)
            # Metricas de desenvolvimento para contextualizar generalizacao.
            desenvolvimento = []
            for a in modelos.values():
                nome = a['modelo']
                conjuntos = [('D0', d0)] if nome in ('M0', 'GB', 'LR') else [('D1', d1)]
                if nome == 'M0':
                    conjuntos.append(('D1', d1))
                if nome == 'MRT':
                    conjuntos = [('D0+D1', pd.concat([d0, d1], ignore_index=True))]
                for conjunto, df in conjuntos:
                    desenvolvimento.append(dict(modelo=nome, seed=a['seed'], conjunto=conjunto,
                        **metricas(df.alvo.to_numpy(), probabilidades(a, df[p['features']]), p['limiar'])))
            pd.DataFrame(desenvolvimento).to_csv(saida/'metricas_desenvolvimento.csv', index=False)
            evento('TODOS_OS_MODELOS_TREINADOS')
            # Somente aqui os bytes/conteudo de D2 sao acessados pela primeira vez.
            d2 = carregar_bloco(dados, 'D2', p)
            evento('D2_CARREGADO')
            tabela, pred = avaliar_modelos(modelos, d2, p)
        tabela.to_csv(saida/'metricas_seeds.csv', index=False)
        pred.to_csv(saida/'previsoes.csv', index=False)
        resumo = resumir(tabela)
        resumo.to_csv(saida/'resumo.csv', index=False)
        tabela[['modelo', 'seed', 'tn', 'fp', 'fn', 'tp']].to_csv(saida/'matrizes_confusao.csv', index=False)
        treino.merge(tabela[['modelo','seed','tempo_inferencia_s']], on=['modelo','seed']).to_csv(saida/'tempos.csv', index=False)
        linhas = '\n'.join(f'| {r.modelo} | {r.mcc_media:.4f} | {r.mcc_desvio:.4f} | {r.delta_mcc_vs_m0:+.4f} |'
                           for r in resumo.itertuples())
        (saida/'RELATORIO.md').write_text(
            '# E12 — Avaliacao final\n\n| Modelo | MCC | Desvio entre seeds | Diferenca vs M0 |\n'
            '|---|---:|---:|---:|\n'+linhas+'\n\n'
            'Desvio amostral ddof=1; LR deterministica tem uma execucao (seed=-1) e desvio indefinido. '
            'PR-AUC e average precision; ROC-AUC indefinida se houver uma unica classe. '
            'Tempos de MFT separam atualizacao e custo total desde M0. '
            'Complexidade de GB e um limite superior de folhas, nao contagem de pesos. '
            'A comparacao central usa M0/MFT/MRT/MREC; GB e LR sao baselines historicos. '
            'Os desvios entre sementes nao sao intervalos de confianca para a populacao.\n', encoding='utf-8')
        evento('CONCLUIDO')
        arquivos = [f for f in saida.iterdir() if f.is_file()]
        salvar_json(saida/'manifesto_resultados.json', {f.name: sha256(f) for f in arquivos})
        return resumo
    except Exception:
        evento('FALHOU')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocolo', type=Path, default=Path('protocolo/protocolo_final.congelado.json'))
    parser.add_argument('--dados', type=Path, default=Path('dados/processados'))
    parser.add_argument('--saida', type=Path, default=Path('resultados/final'))
    args = parser.parse_args()
    print(executar(args.protocolo, args.dados, args.saida).to_string(index=False))


if __name__ == '__main__':
    main()
