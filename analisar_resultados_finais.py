"""E13-E15: drift, explicabilidade e erros, somente apos E12 concluido."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from analisar_dados_drift import calcular_drift
from experimento_final import metricas
from protocolo_final import carregar_bloco, ler_protocolo, salvar_json, sha256


def conferir_execucao(pasta):
    pasta = Path(pasta)
    manifest = json.loads((pasta/'manifesto_resultados.json').read_text(encoding='utf-8'))
    obrigatorios = {'execucao.json', 'modelos.joblib', 'previsoes.csv', 'metricas_seeds.csv',
                    'metricas_desenvolvimento.csv', 'protocolo_congelado.json', 'protocolo_congelado.sha256'}
    if not obrigatorios.issubset(manifest):
        raise ValueError('Manifesto de resultados incompleto.')
    for nome, esperado in manifest.items():
        caminho = (pasta/nome).resolve()
        if not caminho.is_relative_to(pasta.resolve()) or sha256(caminho) != esperado:
            raise ValueError(f'Artefato alterado: {nome}')
    execucao = json.loads((pasta/'execucao.json').read_text(encoding='utf-8'))
    etapas = [e['etapa'] for e in execucao['eventos']]
    if execucao['status'] != 'CONCLUIDO' or etapas.index('TODOS_OS_MODELOS_TREINADOS') >= etapas.index('D2_CARREGADO'):
        raise ValueError('Execucao final incompleta ou fora de ordem.')
    return ler_protocolo(pasta/'protocolo_congelado.json')


def cortes_erros(d2):
    cortes = dict(
        faixa_rank=pd.cut(d2['rank'], [0, 10, 40, np.inf], labels=['1-10','11-40','41-100']),
        tempo_parada=pd.cut(d2['semanas_na_parada'], [0,4,12,np.inf], labels=['1-4','5-12','13+']),
        trimestre=d2.date.dt.to_period('Q').astype(str))
    for coluna in ('estreia', 'reentrada', 'radio_presente', 'streaming_presente', 'digital_presente', 'album_presente'):
        cortes[coluna] = d2[coluna].astype(str)
    return pd.DataFrame(cortes)


def analisar_erros(predicoes, d2, p):
    grupos, exemplos = [], []
    cortes = cortes_erros(d2)
    for (nome, seed), parte in predicoes.groupby(['modelo','seed'], sort=True):
        parte = parte.sort_values('row_id').reset_index(drop=True)
        if len(parte) != len(d2) or not np.array_equal(parte.row_id, np.arange(len(d2))):
            raise ValueError('Predicoes nao cobrem D2 exatamente uma vez por modelo/seed.')
        if not np.array_equal(parte.alvo, d2.alvo):
            raise ValueError('Alvos das predicoes nao correspondem a D2.')
        if not np.array_equal(parte.predicao, (parte.probabilidade >= p['limiar']).astype(int)):
            raise ValueError('Predicao nao corresponde ao limiar congelado.')
        for coluna in cortes:
            for categoria, indices in cortes.groupby(coluna, observed=True).groups.items():
                sub = parte.loc[indices]
                grupos.append(dict(modelo=nome, seed=seed, corte=coluna, grupo=str(categoria),
                                   n=len(sub), positivos=int(sub.alvo.sum()),
                                   **metricas(sub.alvo.to_numpy(), sub.probabilidade.to_numpy(), p['limiar'])))
        for tipo, verdadeiro, previsto, crescente in [('FP',0,1,False),('FN',1,0,True)]:
            sub = parte[(parte.alvo == verdadeiro) & (parte.predicao == previsto)]
            sub = sub.sort_values(['probabilidade', 'row_id'], ascending=[crescente,True]).head(p['analises']['erros_por_tipo'])
            sub = sub.merge(d2[p['features']].reset_index(names='row_id'), on='row_id', validate='one_to_one')
            exemplos.append(sub.assign(tipo_erro=tipo))
    return pd.DataFrame(grupos), pd.concat(exemplos, ignore_index=True)


def importancias(modelos, d2, p):
    linhas = []
    X, y = d2[p['features']], d2.alvo.to_numpy()
    cfg = p['explicabilidade']
    def scoring(modelo, X, y):
        return matthews_corrcoef(y, (modelo.predict_proba(X)[:, 1] >= p['limiar']).astype(int))
    for a in modelos.values():
        if a['modelo'] not in cfg['modelos']:
            continue
        pipeline = Pipeline([('escala', a['scaler']), ('modelo', a['estimador'])])
        resultado = permutation_importance(pipeline, X, y, scoring=scoring,
                    n_repeats=cfg['n_repeats'], random_state=cfg['random_state'], n_jobs=1)
        for i, atributo in enumerate(p['features']):
            for repeticao, valor in enumerate(resultado.importances[i], 1):
                linhas.append(dict(modelo=a['modelo'], seed=a['seed'], atributo=atributo,
                                   repeticao=repeticao, queda_mcc=float(valor)))
        print(f'Importancias: {a["modelo"]} seed {a["seed"]}', flush=True)
    return pd.DataFrame(linhas)


def salvar_figura(fig, caminho):
    fig.text(.01, .01, 'Fonte: experimento final congelado — análise descritiva posterior ao teste.', fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(caminho, dpi=140, bbox_inches='tight')
    plt.close(fig)


def executar(final, dados, saida):
    final, saida = Path(final), Path(saida)
    p = conferir_execucao(final)  # Antes de abrir qualquer bloco de dados.
    blocos = {nome: carregar_bloco(dados, nome, p) for nome in ('D0','D1','D2')}
    saida.mkdir(parents=True, exist_ok=False)
    # Este bundle foi produzido localmente pelo experimento e seu hash foi validado.
    modelos = joblib.load(final/'modelos.joblib')
    predicoes = pd.read_csv(final/'previsoes.csv', parse_dates=['date'])
    chaves = {(a['modelo'],a['seed']) for a in modelos.values()}
    if set(predicoes.groupby(['modelo','seed']).groups) != chaves:
        raise ValueError('Modelos e predicoes nao correspondem.')
    drift_tabelas, bins_tabelas = [], []
    for ref, recente in [('D0','D1'), ('D0','D2'), ('D1','D2')]:
        # Reutiliza exatamente E11: os bins sao aprendidos no primeiro periodo.
        tabela, bins = calcular_drift({'D0':blocos[ref], 'D1':blocos[recente]})
        tabela = tabela.rename(columns={'media_d0':'media_referencia','media_d1':'media_recente',
                                        'wasserstein_por_iqr_d0':'wasserstein_por_iqr_referencia'})
        bins = bins.rename(columns={'proporcao_d0':'proporcao_referencia','proporcao_d1':'proporcao_recente'})
        drift_tabelas.append(tabela.assign(referencia=ref, recente=recente))
        bins_tabelas.append(bins.assign(referencia=ref, recente=recente))
    drift = pd.concat(drift_tabelas, ignore_index=True)
    drift.to_csv(saida/'drift.csv', index=False)
    pd.concat(bins_tabelas, ignore_index=True).to_csv(saida/'intervalos_psi.csv', index=False)
    alvo = pd.DataFrame([dict(conjunto=nome, linhas=len(df), proporcao_alvo=float(df.alvo.mean()))
                         for nome, df in blocos.items()])
    alvo.to_csv(saida/'alvo_periodos.csv', index=False)
    erros, exemplos = analisar_erros(predicoes, blocos['D2'], p)
    erros.to_csv(saida/'erros_grupos.csv', index=False)
    exemplos.to_csv(saida/'erros_exemplos.csv', index=False)
    with threadpool_limits(limits=1):
        imp = importancias(modelos, blocos['D2'], p)
    imp.to_csv(saida/'importancias_repeticoes.csv', index=False)
    por_seed = imp.groupby(['modelo','seed','atributo']).queda_mcc.agg(['mean','std']).reset_index()
    por_seed.to_csv(saida/'importancias_seeds.csv', index=False)
    resumo = por_seed.groupby(['modelo','atributo'])['mean'].agg(['mean','std']).reset_index()
    resumo.columns = ['modelo','atributo','importancia_media','desvio_seeds']
    resumo.to_csv(saida/'importancias_resumo.csv', index=False)
    metricas_finais = pd.read_csv(final/'metricas_seeds.csv')
    dev = pd.read_csv(final/'metricas_desenvolvimento.csv')
    trajetoria = pd.concat([dev[dev.modelo=='M0'], metricas_finais[metricas_finais.modelo=='M0'].assign(conjunto='D2')], ignore_index=True)
    trajetoria.to_csv(saida/'m0_por_periodo.csv', index=False)

    fig, axes = plt.subplots(2,2,figsize=(13,8))
    for ax, nome in zip(axes.flat, p['explicabilidade']['modelos']):
        top = resumo[resumo.modelo==nome].nlargest(10,'importancia_media').sort_values('importancia_media')
        ax.barh(top.atributo, top.importancia_media, xerr=top.desvio_seeds.fillna(0), color='#2166ac')
        ax.axvline(0, color='gray', linewidth=.8)
        ax.set(title=nome, xlabel='Queda de MCC após permutação (média ± desvio entre seeds)')
    salvar_figura(fig, saida/'importancias.png')
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax, (ref, rec) in zip(axes, [('D0','D1'),('D0','D2'),('D1','D2')]):
        sub=drift[(drift.referencia==ref)&(drift.recente==rec)&(drift.tipo=='numerico')].nlargest(6,'psi').sort_values('psi')
        ax.barh(sub.atributo,sub.psi,color='#d6604d')
        ax.set(title=f'{ref} → {rec}', xlabel='PSI')
    salvar_figura(fig, saida/'drift.png')
    fig, axes=plt.subplots(2,2,figsize=(9,7))
    for ax,nome in zip(axes.flat,p['explicabilidade']['modelos']):
        grupo=metricas_finais[metricas_finais.modelo==nome]
        matriz=grupo[['tn','fp','fn','tp']].mean().to_numpy().reshape(2,2)
        ax.imshow(matriz,cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{matriz[i,j]:.1f}',ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Previsto',ylabel='Verdadeiro',title=f'{nome} — contagem média entre seeds')
    salvar_figura(fig, saida/'matrizes_confusao.png')
    melhor = metricas_finais[metricas_finais.modelo.isin(['MFT','MRT','MREC'])].groupby('modelo').mcc.mean().idxmax()
    texto = f'''# E13–E15 — Análises após avaliação final

Melhor estratégia atualizada por MCC médio observado em D2: {melhor}.
Essa identificação é descritiva: todos os modelos foram avaliados antes e
nenhum é reajustado ou reavaliado com parâmetros escolhidos pelo resultado.

Drift usa as mesmas medidas de E11. Intervalos de PSI são definidos no período
de referência de cada comparação, incluindo D1 na comparação D1→D2. Alteração
do alvo não prova concept drift. Não são usados p-valores iid.

Importâncias foram calculadas para as quatro estratégias, em todas as seeds,
com {p['explicabilidade']['n_repeats']} permutações e limiar {p['limiar']}.
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
'''
    (saida/'RELATORIO.md').write_text(texto,encoding='utf-8')
    salvar_json(saida/'manifesto_analises.json',dict(
        protocolo_sha256=sha256(final/'protocolo_congelado.json'),
        resultados_sha256=sha256(final/'manifesto_resultados.json'),
        arquivos={f.name:sha256(f) for f in saida.iterdir() if f.is_file()}))
    return resumo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final',type=Path,default=Path('resultados/final'))
    parser.add_argument('--dados',type=Path,default=Path('dados/processados'))
    parser.add_argument('--saida',type=Path,default=Path('resultados/analises_finais'))
    args=parser.parse_args()
    executar(args.final,args.dados,args.saida)
    print(f'Análises salvas em {args.saida}')


if __name__ == '__main__':
    main()
