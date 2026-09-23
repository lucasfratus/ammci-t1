"""Seleciona e compara regressao logistica propria e de biblioteca em D0.

O script abre exclusivamente ``base_d0.csv``. A configuracao da implementacao
didatica e escolhida por MCC medio em quatro folds temporais. Em seguida, a
configuracao vencedora e comparada com ``sklearn.linear_model.LogisticRegression``
nos mesmos folds e com o mesmo pre-processamento.

Uso:
    python comparar_regressao_logistica.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ajustar_mlp import folds_temporais
from construir_base import ALVO, FEATURES
from regressao_logistica_zero import RegressaoLogisticaZero


CONFIGURACOES = [
    {
        "nome": f"lr_{taxa:g}_{'balanceado' if peso else 'sem_balanceamento'}",
        "taxa_aprendizado": taxa,
        "class_weight": peso,
    }
    for peso in (None, "balanced")
    for taxa in (0.2, 0.5, 1.0)
]


def metricas(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def preparar_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    treino: np.ndarray,
    validacao: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta a escala somente no passado de cada fold."""
    escala = StandardScaler().fit(X.loc[treino])
    X_treino = escala.transform(X.loc[treino])
    X_validacao = escala.transform(X.loc[validacao])
    return X_treino, y[treino], X_validacao, y[validacao]


def buscar_configuracao(
    X: pd.DataFrame,
    y: np.ndarray,
    datas: pd.Series,
    n_folds: int,
    max_iter: int,
    tolerancia: float,
) -> tuple[pd.DataFrame, dict]:
    linhas = []
    folds = list(folds_temporais(datas, n_folds))

    for configuracao in CONFIGURACOES:
        print(f"\n{configuracao['nome']}")
        for numero, (treino, validacao, inicio_validacao) in enumerate(folds, 1):
            X_t, y_t, X_v, y_v = preparar_fold(X, y, treino, validacao)
            modelo = RegressaoLogisticaZero(
                taxa_aprendizado=configuracao["taxa_aprendizado"],
                max_iter=max_iter,
                l2=0.0,
                tolerancia=tolerancia,
                class_weight=configuracao["class_weight"],
            )
            inicio = time.perf_counter()
            modelo.fit(X_t, y_t)
            tempo = time.perf_counter() - inicio
            resultado = metricas(y_v, modelo.predict_proba(X_v)[:, 1])
            linhas.append(
                {
                    "config": configuracao["nome"],
                    "fold": numero,
                    "inicio_validacao": inicio_validacao.date().isoformat(),
                    "taxa_aprendizado": configuracao["taxa_aprendizado"],
                    "class_weight": configuracao["class_weight"] or "nenhum",
                    "l2": 0.0,
                    "iteracoes": modelo.n_iter_,
                    "convergiu": modelo.convergiu_,
                    "custo_final": modelo.custo_final_,
                    "tempo_treino_s": tempo,
                    **resultado,
                }
            )
            print(
                f"  fold {numero}: MCC={resultado['mcc']:.4f} "
                f"iter={modelo.n_iter_:>4} tempo={tempo:.2f}s "
                f"{'convergiu' if modelo.convergiu_ else 'limite'}"
            )

    busca = pd.DataFrame(linhas)
    ranking = (
        busca.groupby("config", as_index=False)
        .agg(
            mcc_medio=("mcc", "mean"),
            mcc_desvio=("mcc", "std"),
            tempo_medio_s=("tempo_treino_s", "mean"),
            iteracoes_medias=("iteracoes", "mean"),
            todos_convergiram=("convergiu", "all"),
        )
        .reset_index(drop=True)
    )
    # MCC e discreto porque deriva das classes previstas. Diferencas menores
    # que a quarta casa sao tratadas como empate numerico entre otimizadores
    # que convergiram para a mesma solucao; nesse caso vence o mais rapido.
    ranking["mcc_criterio"] = ranking["mcc_medio"].round(4)
    elegiveis = ranking[ranking["todos_convergiram"]].copy()
    if elegiveis.empty:
        raise RuntimeError(
            "nenhuma configuracao convergiu em todos os folds; "
            "aumente max_iter ou ajuste a grade"
        )
    elegiveis = elegiveis.sort_values(
        ["mcc_criterio", "tempo_medio_s", "config"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    nome_vencedor = elegiveis.iloc[0]["config"]
    vencedor = next(c for c in CONFIGURACOES if c["nome"] == nome_vencedor)
    vencedor = {
        **vencedor,
        "l2": 0.0,
        "max_iter": max_iter,
        "tolerancia": tolerancia,
        "criterio_selecao": (
            "maior MCC medio arredondado a 4 casas; "
            "desempate por menor tempo medio; exige convergencia em todos os folds"
        ),
        "mcc_medio": float(elegiveis.iloc[0]["mcc_medio"]),
        "mcc_desvio": float(elegiveis.iloc[0]["mcc_desvio"]),
    }
    return busca, vencedor


def comparar_com_biblioteca(
    X: pd.DataFrame,
    y: np.ndarray,
    datas: pd.Series,
    n_folds: int,
    configuracao: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    linhas = []
    concordancias = []

    for numero, (treino, validacao, inicio_validacao) in enumerate(
        folds_temporais(datas, n_folds), 1
    ):
        X_t, y_t, X_v, y_v = preparar_fold(X, y, treino, validacao)

        proprio = RegressaoLogisticaZero(
            taxa_aprendizado=configuracao["taxa_aprendizado"],
            max_iter=configuracao["max_iter"],
            l2=configuracao["l2"],
            tolerancia=configuracao["tolerancia"],
            class_weight=configuracao["class_weight"],
        )
        inicio = time.perf_counter()
        proprio.fit(X_t, y_t)
        tempo_proprio = time.perf_counter() - inicio
        prob_propria = proprio.predict_proba(X_v)[:, 1]

        # C infinito desliga a regularizacao da implementacao da biblioteca,
        # deixando a mesma funcao objetivo usada na configuracao propria.
        biblioteca = LogisticRegression(
            C=np.inf,
            class_weight=configuracao["class_weight"],
            solver="lbfgs",
            max_iter=configuracao["max_iter"],
            tol=configuracao["tolerancia"],
        )
        inicio = time.perf_counter()
        biblioteca.fit(X_t, y_t)
        tempo_biblioteca = time.perf_counter() - inicio
        prob_biblioteca = biblioteca.predict_proba(X_v)[:, 1]

        base = {
            "fold": numero,
            "inicio_validacao": inicio_validacao.date().isoformat(),
        }
        linhas.append(
            {
                **base,
                "implementacao": "NumPy (do zero)",
                "iteracoes": proprio.n_iter_,
                "convergiu": proprio.convergiu_,
                "tempo_treino_s": tempo_proprio,
                **metricas(y_v, prob_propria),
            }
        )
        linhas.append(
            {
                **base,
                "implementacao": "scikit-learn",
                "iteracoes": int(biblioteca.n_iter_[0]),
                "convergiu": bool(biblioteca.n_iter_[0] < configuracao["max_iter"]),
                "tempo_treino_s": tempo_biblioteca,
                **metricas(y_v, prob_biblioteca),
            }
        )

        pred_propria = prob_propria >= 0.5
        pred_biblioteca = prob_biblioteca >= 0.5
        concordancias.append(
            {
                **base,
                "correlacao_probabilidades": float(
                    np.corrcoef(prob_propria, prob_biblioteca)[0, 1]
                ),
                "diferenca_absoluta_media": float(
                    np.mean(np.abs(prob_propria - prob_biblioteca))
                ),
                "concordancia_classes": float(
                    np.mean(pred_propria == pred_biblioteca)
                ),
            }
        )

    return pd.DataFrame(linhas), pd.DataFrame(concordancias)


def resumir(
    d0: pd.DataFrame,
    busca: pd.DataFrame,
    vencedor: dict,
    comparacao: pd.DataFrame,
    concordancia: pd.DataFrame,
    n_folds: int,
) -> dict:
    metricas_resumo = {}
    for implementacao, grupo in comparacao.groupby("implementacao"):
        metricas_resumo[implementacao] = {
            coluna: {
                "media": float(grupo[coluna].mean()),
                "desvio": float(grupo[coluna].std()),
            }
            for coluna in [
                "mcc", "f1", "precision", "recall", "accuracy",
                "roc_auc", "pr_auc", "tempo_treino_s", "iteracoes",
            ]
        }

    return {
        "protocolo": {
            "dados_lidos": "somente D0",
            "linhas": int(len(d0)),
            "semanas": int(d0["date"].nunique()),
            "folds_temporais": n_folds,
            "metrica_de_selecao": "MCC",
            "limiar_classificacao": 0.5,
            "estocastico": False,
            "observacao_seeds": (
                "Nao se aplicam: gradiente em lote e L-BFGS sao deterministicos."
            ),
        },
        "linha_base": {
            "estrategia": "sempre prever a classe majoritaria (0)",
            "accuracy_em_D0": float(1.0 - d0[ALVO].mean()),
            "mcc": 0.0,
        },
        "configuracoes_testadas": int(busca["config"].nunique()),
        "configuracao_escolhida": vencedor,
        "comparacao": metricas_resumo,
        "coerencia_entre_implementacoes": {
            "correlacao_probabilidades_media": float(
                concordancia["correlacao_probabilidades"].mean()
            ),
            "diferenca_absoluta_media": float(
                concordancia["diferenca_absoluta_media"].mean()
            ),
            "concordancia_classes_media": float(
                concordancia["concordancia_classes"].mean()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", default="dados/processados/base_d0.csv", type=Path
    )
    parser.add_argument(
        "--saida", default="resultados/regressao_logistica", type=Path
    )
    parser.add_argument("--folds", default=4, type=int)
    parser.add_argument("--max-iter", default=6_000, type=int)
    parser.add_argument("--tolerancia", default=1e-11, type=float)
    args = parser.parse_args()

    d0 = pd.read_csv(args.base, parse_dates=["date"])
    X = d0[FEATURES]
    y = d0[ALVO].to_numpy()
    datas = d0["date"]
    print(
        f"D0: {len(d0)} linhas, {datas.nunique()} semanas, "
        f"{y.mean():.3f} de positivos"
    )
    print("D1 e D2 nao sao abertos por este experimento.")

    busca, vencedor = buscar_configuracao(
        X, y, datas, args.folds, args.max_iter, args.tolerancia
    )
    print("\nConfiguracao escolhida:")
    print(json.dumps(vencedor, ensure_ascii=False, indent=2))

    comparacao, concordancia = comparar_com_biblioteca(
        X, y, datas, args.folds, vencedor
    )
    resumo = resumir(
        d0, busca, vencedor, comparacao, concordancia, args.folds
    )

    args.saida.mkdir(parents=True, exist_ok=True)
    busca.to_csv(args.saida / "busca_zero.csv", index=False)
    comparacao.to_csv(args.saida / "comparacao_folds.csv", index=False)
    concordancia.to_csv(args.saida / "concordancia.csv", index=False)
    (args.saida / "configuracao_escolhida.json").write_text(
        json.dumps(vencedor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.saida / "resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nComparacao media nos folds temporais:")
    for nome, valores in resumo["comparacao"].items():
        print(
            f"  {nome:<16} MCC={valores['mcc']['media']:.4f} +/- "
            f"{valores['mcc']['desvio']:.4f}  "
            f"PR-AUC={valores['pr_auc']['media']:.4f}  "
            f"tempo={valores['tempo_treino_s']['media']:.3f}s/fold"
        )
    coerencia = resumo["coerencia_entre_implementacoes"]
    print(
        "  concordancia: "
        f"correlacao(prob)={coerencia['correlacao_probabilidades_media']:.6f}, "
        f"classes={coerencia['concordancia_classes_media']:.2%}"
    )
    print(f"\nResultados gravados em {args.saida}/")


if __name__ == "__main__":
    main()
