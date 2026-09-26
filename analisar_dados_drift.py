"""AED e E11: drift descritivo D0 -> D1. Nunca abre D2.

Gera tabelas, figuras e relatorio, compartilhados pelo notebook.
KS e distancias sao descritivos: registros semanais/musicas nao sao iid;
nao se reportam p-valores que pressuponham observacoes independentes.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from construir_base import ALVO, FEATURES, INICIO, CORTE_D1, CORTE_D2, SEMANA

CATEGORICAS = ['estreia', 'reentrada', 'radio_presente', 'streaming_presente',
               'digital_presente', 'album_presente', 'mes', 'semana_ano']
NUMERICAS = [c for c in FEATURES if c not in CATEGORICAS]
CORES = {'D0': '#2166ac', 'D1': '#d6604d'}


def carregar_desenvolvimento(pasta=Path('dados/processados')):
    manifesto = (pasta/'manifesto.txt').read_text()
    dados, hashes = {}, {}
    for nome, inicio, fim in [('d0', INICIO, CORTE_D1-2*SEMANA),
                               ('d1', CORTE_D1, CORTE_D2-2*SEMANA)]:
        caminho = pasta/f'base_{nome}.csv'
        digest = hashlib.sha256(caminho.read_bytes()).hexdigest()
        esperado = next(l for l in manifesto.splitlines() if l.startswith(nome+' ')).split('sha256=')[1]
        if digest != esperado:
            raise ValueError(f'Hash inesperado em {nome}')
        df = pd.read_csv(caminho, parse_dates=['date'])
        if not df.date.between(inicio, fim).all() or not df.date.is_monotonic_increasing:
            raise ValueError(f'Periodo invalido em {nome}')
        if not np.isfinite(df[FEATURES].to_numpy()).all() or set(df[ALVO].unique()) != {0, 1}:
            raise ValueError(f'Atributos ou alvo invalidos em {nome}')
        dados[nome.upper()], hashes[nome.upper()] = df, digest
    return dados, hashes


def psi_referencia(referencia, recente, n_bins=10):
    """Quantis exclusivamente de D0; caudas infinitas e pseudocontagem 0.5.

    Variavel constante em D0 recebe tres intervalos, permitindo detectar
    massa nova fora da constante. Duplicatas de quantis sao removidas.
    """
    ref, rec = np.asarray(referencia, dtype=float), np.asarray(recente, dtype=float)
    internos = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    if np.ptp(ref) == 0:
        internos = np.array([np.nextafter(ref[0], -np.inf), np.nextafter(ref[0], np.inf)])
    bordas = np.r_[-np.inf, internos, np.inf]
    a, _ = np.histogram(ref, bordas)
    b, _ = np.histogram(rec, bordas)
    p = (a + 0.5) / (a.sum() + 0.5 * len(a))
    q = (b + 0.5) / (b.sum() + 0.5 * len(b))
    valor = float(np.sum((q-p) * np.log(q/p)))
    return valor, bordas, p, q


def divergencia_categorica(ref, rec):
    categorias = sorted(set(ref) | set(rec))
    a = pd.Series(ref).value_counts().reindex(categorias, fill_value=0).to_numpy(dtype=float)
    b = pd.Series(rec).value_counts().reindex(categorias, fill_value=0).to_numpy(dtype=float)
    p, q = a/a.sum(), b/b.sum()
    # scipy retorna distancia: elevar ao quadrado da divergencia em bits [0,1].
    return float(jensenshannon(p, q, base=2)**2), categorias, p, q


def calcular_drift(dados):
    d0, d1 = dados['D0'], dados['D1']
    linhas, bins = [], []
    for coluna in NUMERICAS:
        a, b = d0[coluna].to_numpy(), d1[coluna].to_numpy()
        psi, bordas, p, q = psi_referencia(a, b)
        iqr = float(np.quantile(a, .75) - np.quantile(a, .25))
        distancia = float(wasserstein_distance(a, b))
        linhas.append(dict(atributo=coluna, tipo='numerico',
                           ks=float(ks_2samp(a, b).statistic), psi=psi,
                           wasserstein=distancia,
                           wasserstein_por_iqr_d0=distancia/iqr if iqr > 0 else np.nan,
                           js_bits=np.nan, media_d0=float(a.mean()), media_d1=float(b.mean())))
        bins.extend(dict(atributo=coluna, inferior=str(bordas[i]), superior=str(bordas[i+1]),
                         proporcao_d0=float(p[i]), proporcao_d1=float(q[i])) for i in range(len(p)))
    for coluna in CATEGORICAS + [ALVO]:
        js, _, _, _ = divergencia_categorica(d0[coluna], d1[coluna])
        linhas.append(dict(atributo=coluna, tipo='alvo' if coluna == ALVO else 'categorico',
                           ks=np.nan, psi=np.nan, wasserstein=np.nan,
                           wasserstein_por_iqr_d0=np.nan, js_bits=js,
                           media_d0=float(d0[coluna].mean()), media_d1=float(d1[coluna].mean())))
    return pd.DataFrame(linhas), pd.DataFrame(bins)


def salvar_figura(fig, caminho):
    fig.text(.01, .005, 'Fonte: Billboard/Kaggle v175 — somente D0 e D1.', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(caminho, dpi=140, bbox_inches='tight')
    plt.close(fig)


def gerar_figuras(dados, drift, saida):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figuras = saida/'figuras'
    figuras.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    for nome, df in dados.items():
        serie = df.groupby('date')[ALVO].mean()
        ax.plot(serie.index, serie.rolling(13, min_periods=1).mean(), color=CORES[nome], label=nome)
    ax.set(title='Proporção de músicas que melhoram na semana seguinte',
           ylabel='Proporção — média móvel de 13 semanas', xlabel='Semana')
    ax.legend()
    salvar_figura(fig, figuras/'alvo_temporal.png')

    colunas = ['rank', 'peak_pos', 'semanas_na_parada', 'variacao_1s', 'hiato_semanas', 'distancia_pico']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for coluna, ax in zip(colunas, axes.flat):
        for nome, df in dados.items():
            valores = np.sort(df[coluna].to_numpy())
            ax.plot(valores, np.arange(1, len(valores)+1)/len(valores), label=nome, color=CORES[nome])
        ax.set(title=coluna, xlabel='Valor observado', ylabel='Fração acumulada')
        ax.legend()
    fig.suptitle('Distribuições completas (CDF): D0 × D1', y=1.02)
    salvar_figura(fig, figuras/'distribuicoes.png')

    cols = ['radio_presente', 'streaming_presente', 'digital_presente', 'album_presente', 'estreia', 'reentrada']
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(cols))
    for i, (nome, df) in enumerate(dados.items()):
        ax.bar(x+(i-.5)*.35, df[cols].mean(), width=.35, label=nome, color=CORES[nome])
    ax.set_xticks(x, ['Rádio', 'Streaming', 'Digital', 'Álbum', 'Estreia*', 'Reentrada*'])
    ax.set(title='Cobertura e indicadores de passagem', ylabel='Proporção de linhas')
    ax.legend()
    salvar_figura(fig, figuras/'cobertura.png')

    numericas = drift[drift.tipo == 'numerico'].sort_values('psi', ascending=False)
    categoricas = drift[drift.tipo == 'categorico'].sort_values('js_bits', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(numericas.atributo[::-1], numericas.psi[::-1], color='#2166ac')
    axes[0].set(title='Mudança numérica', xlabel='PSI — intervalos definidos em D0')
    axes[1].barh(categoricas.atributo[::-1], categoricas.js_bits[::-1], color='#d6604d')
    axes[1].set(title='Mudança categórica', xlabel='Divergência Jensen–Shannon (bits)')
    salvar_figura(fig, figuras/'drift.png')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (nome, df) in zip(axes, dados.items()):
        corr = df[FEATURES].corr(method='spearman')
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7)
        ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=7)
        ax.set_title(f'{nome}: correlação de Spearman')
        fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    salvar_figura(fig, figuras/'correlacoes.png')

    # Datas do protocolo, sem abrir dados futuros.
    fig, ax = plt.subplots(figsize=(11, 3))
    for i, (nome, df) in enumerate(dados.items()):
        ax.plot([df.date.min(), df.date.max()], [i, i], linewidth=12,
                solid_capstyle='butt', color=CORES[nome])
    ax.axvline(CORTE_D1-SEMANA, color='gray', linestyle='--', label='Embargo antes de D1')
    ax.axvline(CORTE_D2-SEMANA, color='black', linestyle=':', label='Embargo antes de D2')
    ax.set_yticks([0, 1], ['D0 — desenvolvimento', 'D1 — atualização'])
    ax.set(ylim=(-.7, 1.8), title='Períodos observados; D2 permanece reservado', xlabel='Data')
    ax.legend(loc='upper left', fontsize=8)
    salvar_figura(fig, figuras/'linha_tempo.png')


def executar(saida=Path('resultados/aed_drift')):
    dados, hashes = carregar_desenvolvimento()
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for nome, df in dados.items():
        linhas.append(dict(conjunto=nome, linhas=len(df), semanas=df.date.nunique(),
                           inicio=str(df.date.min().date()), fim=str(df.date.max().date()),
                           proporcao_alvo=float(df[ALVO].mean()),
                           ausentes=int(df[FEATURES].isna().sum().sum()),
                           duplicatas=int(df.duplicated(['date', 'title', 'artist']).sum())))
        df[FEATURES + [ALVO]].describe().T.to_csv(saida/f'descritivas_{nome.lower()}.csv')
        df[FEATURES].corr(method='spearman').to_csv(saida/f'correlacoes_{nome.lower()}.csv')
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(saida/'resumo_base.csv', index=False)
    semanal = pd.concat([df.groupby('date')[ALVO].agg(['mean', 'size']).assign(conjunto=nome)
                         for nome, df in dados.items()])
    semanal.to_csv(saida/'alvo_semanal.csv')
    indicadores = pd.DataFrame({nome: df[CATEGORICAS[:6]].mean() for nome, df in dados.items()})
    indicadores.to_csv(saida/'indicadores.csv')
    drift, bins = calcular_drift(dados)
    drift.to_csv(saida/'drift_d0_d1.csv', index=False)
    bins.to_csv(saida/'intervalos_psi.csv', index=False)
    gerar_figuras(dados, drift, saida)
    top = drift[drift.tipo == 'numerico'].nlargest(5, 'psi')
    topcat = drift[drift.tipo == 'categorico'].nlargest(3, 'js_bits')
    prop0, prop1 = [float(dados[n][ALVO].mean()) for n in ['D0', 'D1']]
    texto = f'''# AED e E11 — Mudança de distribuição D0 → D1

Pergunta: os dados recentes diferem dos históricos? Dados: somente D0/D1,
verificados pelos hashes do manifesto. D2 não foi carregado.

D0: {len(dados['D0']):,} linhas; D1: {len(dados['D1']):,} linhas.
Proporção positiva: {prop0:.4f} → {prop1:.4f}, variação de {(prop1-prop0)*100:.2f} pontos percentuais.

## Mudanças numéricas (ordenadas por PSI)

| Atributo | PSI | KS | Wasserstein / IQR de D0 |
|---|---:|---:|---:|
'''
    texto += '\n'.join(f'| {r.atributo} | {r.psi:.4f} | {r.ks:.4f} | {r.wasserstein_por_iqr_d0:.4f} |'
                       for r in top.itertuples())
    texto += '\n\nMaiores divergências categóricas: ' + ', '.join(
        f'{r.atributo} ({r.js_bits:.4f} bits)' for r in topcat.itertuples()) + '.\n'
    texto += '''
## Método e interpretação

KS mede a maior diferença entre distribuições acumuladas. PSI usa quantis
de D0 (até dez intervalos), caudas infinitas e pseudocontagem 0,5 por intervalo.
Wasserstein está nas unidades do atributo; sua razão pelo IQR de D0 é deixada
vazia quando o IQR é zero. Jensen–Shannon é a divergência em bits, não a distância.
PSI e JS não devem ser comparados na mesma escala. Não usamos limiares automáticos
para declarar drift nem p-valores iid: músicas se repetem e semanas são dependentes.

As diferenças medem mudança de distribuição dos atributos e do alvo. Não provam
mudança da relação condicional entre atributos e alvo (concept drift), nem causalidade.
Mudanças de cobertura dos rankings auxiliares também podem refletir a coleta.
Variáveis de calendário refletem sazonalidade e composição dos períodos.
Correlação entre posições e indicadores pode ser induzida pelas regras de construção.

## Ausências, censuras e passagens

Os atributos processados não têm nulos. Ausência no Hot 100 anterior usa posição
101; ranking secundário ausente usa 51; álbum ausente usa 201, acompanhado de
indicadores de presença. Esses valores são convenções, não posições observadas.
`estreia` indica início de uma passagem; inclui retorno após hiato superior a 28 dias.
`reentrada` identifica os inícios de passagem de música já observada. Portanto,
esses indicadores não representam necessariamente primeira estreia da carreira.

## Decisão e próximos passos

Há mudança mensurável entre D0 e D1, a ser relacionada ao desempenho de M0.
Manter a avaliação temporal e as quatro estratégias previstas; nenhuma variável
foi descartada com base nesta exploração. A avaliação futura exige protocolo
congelado. Artefatos quantitativos e seis figuras acompanham este relatório.
'''
    (saida/'RELATORIO.md').write_text(texto, encoding='utf-8')
    (saida/'protocolo.json').write_text(json.dumps(dict(
        experimento='E11', data_utc=datetime.now(timezone.utc).isoformat(),
        hashes=hashes, dados=['D0', 'D1'], n_bins_psi=10, pseudocontagem=.5,
        referencia_bins='D0', js='divergencia em bits', p_valores=False,
        motivo='dependencia temporal e repeticao de musicas'), indent=2)+'\n', encoding='utf-8')
    print(resumo.to_string(index=False))
    print(top[['atributo', 'psi', 'ks']].to_string(index=False))
    return resumo, drift


if __name__ == '__main__':
    executar()
