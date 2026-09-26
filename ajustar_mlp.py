"""
Ajuste de hiperparametros da MLP, restrito a D0 (secao 6.9 do enunciado).

Grava os melhores hiperparametros em modelos/hiperparametros.json, que os
experimentos temporais (M0, MFT, MRT, MREC) leem depois.

Uso:
    python ajustar_mlp.py
    python ajustar_mlp.py --folds 5 --max-iter 150

Regras que este script garante:
  - Apenas D0 e lido. D1 e D2 nem sao abertos.
  - A validacao respeita a ordem cronologica: cada fold treina no passado e
    valida no futuro imediato, nunca o contrario.
  - O corte entre treino e validacao cai sempre em fronteira de semana, para
    que linhas da mesma semana nao aparecam dos dois lados.
  - Um embargo de sete dias exige que o horizonte do alvo de cada linha de
    treino termine antes da primeira semana de validacao.
  - O StandardScaler e ajustado dentro de cada fold, so com o treino daquele
    fold. Ajustar antes do loop vazaria estatisticas da validacao.
"""

import argparse
import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

SEED = 42

# Grade de busca. Mantida pequena de proposito: 12 combinacoes x n folds ja
# demora alguns minutos em CPU. Ampliar so se o resultado ficar na borda.
GRADE = {
    "hidden_layer_sizes": [(32,), (64, 32)],
    "alpha": [1e-4, 1e-3, 1e-2],          # regularizacao L2
    "learning_rate_init": [1e-3, 3e-3],
}

FEATURES = [
    "rank", "rank_lag1", "rank_lag2", "variacao_1s",
    "peak_pos", "distancia_pico",
    "semanas_na_parada", "hiato_semanas", "estreia", "reentrada",
    "radio_rank", "radio_presente",
    "streaming_rank", "streaming_presente",
    "digital_rank", "digital_presente",
    "album_rank", "album_presente",
    "mes", "semana_ano",
]
ALVO = "alvo"


def folds_temporais(datas: pd.Series, n_folds: int):
    """Validacao em blocos cronologicos crescentes.

    Fold 1 treina no bloco 1 e valida no bloco 2; fold 2 treina nos blocos 1-2
    e valida no bloco 3; e assim por diante. E o mesmo principio do
    TimeSeriesSplit, mas cortando em fronteira de semana em vez de por indice
    de linha, porque cada semana tem ~100 linhas e um corte por indice partiria
    uma semana ao meio. O ultimo horizonte de treino deve terminar antes da
    validacao; remove-se a semana imediatamente anterior (embargo de 7 dias).
    """
    if n_folds < 1 or datas.isna().any():
        raise ValueError('Numero de folds ou datas invalidos.')
    semanas = np.array(sorted(datas.unique()))
    if len(semanas) < n_folds + 1:
        raise ValueError('Semanas insuficientes para os folds solicitados.')
    blocos = np.array_split(semanas, n_folds + 1)

    for i in range(n_folds):
        semanas_treino = np.concatenate(blocos[: i + 1])
        semanas_val = blocos[i + 1]
        treino = datas.isin(semanas_treino).to_numpy()
        # O alvo em t utiliza t+7 dias. Exigir t+7 < inicio_validacao:
        # a ultima semana anterior a validacao nao pode entrar no treino.
        treino = treino & (datas + pd.Timedelta(days=7) < pd.Timestamp(semanas_val[0])).to_numpy()
        val = datas.isin(semanas_val).to_numpy()
        if not treino.any():
            raise ValueError('Treino vazio depois do embargo temporal de sete dias.')
        yield treino, val, pd.Timestamp(semanas_val[0])


def avaliar(params: dict, X: pd.DataFrame, y: np.ndarray,
            datas: pd.Series, n_folds: int, max_iter: int) -> dict:
    """Roda a validacao cronologica para uma combinacao e devolve as metricas."""
    mccs, tempos, iteracoes, convergencias = [], [], [], []

    for treino, val, _ in folds_temporais(datas, n_folds):
        escala = StandardScaler().fit(X[treino])
        modelo = MLPClassifier(
            **params,
            max_iter=max_iter,
            random_state=SEED,
            early_stopping=False,
        )
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always', ConvergenceWarning)
            modelo.fit(escala.transform(X[treino]), y[treino])
        convergencias.append(not any(issubclass(a.category, ConvergenceWarning) for a in avisos))
        iteracoes.append(int(modelo.n_iter_))
        tempos.append(time.perf_counter() - t0)
        mccs.append(matthews_corrcoef(y[val], modelo.predict(escala.transform(X[val]))))

    return {
        "mcc_medio": float(np.mean(mccs)),
        "mcc_desvio": float(np.std(mccs)),
        "mcc_por_fold": [round(m, 4) for m in mccs],
        "tempo_medio_s": round(float(np.mean(tempos)), 1),
        "iteracoes_por_fold": iteracoes,
        "convergiu_por_fold": convergencias,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="dados/processados/base_d0.csv", type=Path)
    p.add_argument("--saida", default="modelos/hiperparametros.json", type=Path)
    p.add_argument("--folds", default=4, type=int)
    p.add_argument("--max-iter", default=800, type=int)
    args = p.parse_args()

    d0 = pd.read_csv(args.base, parse_dates=["date"])
    X, y, datas = d0[FEATURES], d0[ALVO].to_numpy(), d0["date"]

    print(f"D0: {len(d0)} linhas, {datas.nunique()} semanas, "
          f"{y.mean():.3f} de positivos")
    print(f"Validacao em {args.folds} blocos cronologicos. Cortes:")
    for _, _, inicio_val in folds_temporais(datas, args.folds):
        print(f"  valida a partir de {inicio_val.date()}")

    # Linha de base trivial: responder sempre a classe majoritaria.
    # MCC = 0 por definicao, porque o modelo nao distingue as classes.
    print(f"\nLinha de base trivial: acuracia {1 - y.mean():.3f}, MCC 0.000")

    combinacoes = [
        dict(zip(GRADE.keys(), valores)) for valores in product(*GRADE.values())
    ]
    print(f"\nTestando {len(combinacoes)} combinacoes "
          f"({len(combinacoes) * args.folds} treinos no total)\n")

    resultados = []
    for i, params in enumerate(combinacoes, 1):
        metricas = avaliar(params, X, y, datas, args.folds, args.max_iter)
        resultados.append({"params": params, **metricas})
        print(
            f"[{i:>2}/{len(combinacoes)}] "
            f"camadas={str(params['hidden_layer_sizes']):<10} "
            f"alpha={params['alpha']:<7} lr={params['learning_rate_init']:<6} "
            f"MCC={metricas['mcc_medio']:.4f} +/- {metricas['mcc_desvio']:.4f} "
            f"({metricas['tempo_medio_s']}s/fold)"
        )

    resultados.sort(key=lambda r: r["mcc_medio"], reverse=True)
    melhor = resultados[0]

    print("\n" + "=" * 72)
    print("MELHOR COMBINACAO")
    print("=" * 72)
    for chave, valor in melhor["params"].items():
        print(f"  {chave}: {valor}")
    print(f"  MCC medio: {melhor['mcc_medio']:.4f} "
          f"(desvio {melhor['mcc_desvio']:.4f})")
    print(f"  por fold: {melhor['mcc_por_fold']}")

    if melhor["mcc_desvio"] > abs(melhor["mcc_medio"]) / 2:
        print("\n  ATENCAO: desvio alto entre folds. O desempenho varia muito")
        print("  conforme o periodo de validacao, o que ja e indicio de")
        print("  instabilidade temporal. Registrar no diario.")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    melhor_serializavel = {
        **melhor["params"],
        "hidden_layer_sizes": list(melhor["params"]["hidden_layer_sizes"]),
        "max_iter": args.max_iter,
        "mcc_validacao_d0": melhor["mcc_medio"],
        "mcc_desvio_d0": melhor["mcc_desvio"],
        "folds": args.folds,
        "seed_busca": SEED,
        "embargo_dias": 7,
        "convergiu_por_fold": melhor["convergiu_por_fold"],
        "iteracoes_por_fold": melhor["iteracoes_por_fold"],
    }
    args.saida.write_text(json.dumps(melhor_serializavel, indent=2) + "\n")

    todos = args.saida.parent / "busca_completa.json"
    todos.write_text(json.dumps(
        [{**r, "params": {**r["params"],
                          "hidden_layer_sizes": list(r["params"]["hidden_layer_sizes"])}}
         for r in resultados], indent=2) + "\n")

    print(f"\nGravado em {args.saida} e {todos}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
