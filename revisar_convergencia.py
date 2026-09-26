"""E06-R1: compara teto antigo (100) e revisado em todos os folds/seeds."""
import json
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ajustar_mlp import folds_temporais
from construir_base import ALVO, FEATURES
from comparar_regressao_logistica import metricas


def main():
    hp = json.loads(Path('modelos/hiperparametros.json').read_text())
    d0 = pd.read_csv('dados/processados/base_d0.csv', parse_dates=['date'])
    saida = Path('resultados/convergencia')
    saida.mkdir(parents=True, exist_ok=True)
    linhas, curvas = [], []
    for fold, (treino, val, _) in enumerate(folds_temporais(d0.date, hp['folds']), 1):
        escala = StandardScaler().fit(d0.loc[treino, FEATURES])
        Xt, Xv = escala.transform(d0.loc[treino, FEATURES]), escala.transform(d0.loc[val, FEATURES])
        yt, yv = d0.loc[treino, ALVO].to_numpy(), d0.loc[val, ALVO].to_numpy()
        for seed in (42, 1337, 2024):
            for teto in sorted({100, hp['max_iter']}):
                m = MLPClassifier(hidden_layer_sizes=tuple(hp['hidden_layer_sizes']),
                                  alpha=hp['alpha'], learning_rate_init=hp['learning_rate_init'],
                                  max_iter=teto, random_state=seed, early_stopping=False)
                inicio = time.perf_counter()
                with warnings.catch_warnings(record=True) as avisos:
                    warnings.simplefilter('always', ConvergenceWarning)
                    m.fit(Xt, yt)
                convergiu = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
                linhas.append(dict(fold=fold, seed=seed, teto=teto, epocas=m.n_iter_,
                                   convergiu=convergiu, tempo_s=time.perf_counter()-inicio,
                                   mcc_treino=metricas(yt, m.predict_proba(Xt)[:, 1])['mcc'],
                                   **metricas(yv, m.predict_proba(Xv)[:, 1])))
                curvas.extend(dict(fold=fold, seed=seed, teto=teto, epoca=e, perda=float(perda))
                              for e, perda in enumerate(m.loss_curve_, 1))
                print(f'fold={fold} seed={seed} teto={teto}: epocas={m.n_iter_}, convergiu={convergiu}', flush=True)
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(saida/'diagnostico.csv', index=False)
    pd.DataFrame(curvas).to_csv(saida/'curvas_perda.csv', index=False)
    revisado = tabela[tabela.teto == hp['max_iter']]
    ok = bool(revisado.convergiu.all())
    resumo = dict(dados='D0', embargo_dias=7, hp=hp, seeds=[42, 1337, 2024],
                  convergencia_em_todos=ok, maior_numero_epocas=int(revisado.epocas.max()),
                  criterio='ausencia de ConvergenceWarning; parada pela perda de treino',
                  observacao='Convergencia numerica nao garante generalizacao nem otimo global.')
    (saida/'resumo.json').write_text(json.dumps(resumo, indent=2)+'\n', encoding='utf-8')
    texto = '# E06-R1 — Diagnostico de convergencia\n\n'
    texto += 'Mesma configuracao escolhida em D0, com embargo, quatro folds e tres seeds.\n\n'
    for teto, grupo in tabela.groupby('teto'):
        texto += (f'- Teto {teto}: {int(grupo.convergiu.sum())}/12 ajustes convergiram; '
                  f'MCC medio {grupo.mcc.mean():.4f}; MCC treino {grupo.mcc_treino.mean():.4f}; '
                  f'maximo observado {grupo.epocas.max()} epocas.\n')
    texto += ('\nA diferenca treino/validacao deve ser interpretada como sinal potencial de '
              'sobreajuste. Aumentar o teto permite estabilizar a perda; nao implica melhorar MCC. '
              'Nenhuma decisao usa D1 ou D2.\n')
    (saida/'RELATORIO.md').write_text(texto, encoding='utf-8')
    if not ok:
        raise RuntimeError('Ainda ha ajustes sem convergencia; revisar antes do protocolo final.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
