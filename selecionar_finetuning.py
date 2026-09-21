"""
Fase 1 dos experimentos temporais: escolha da configuracao de fine-tuning.

Usa apenas D0 e D1. D2 NAO e aberto neste script.

O que faz:
  1. Treina M0 do zero em D0 com os hiperparametros da busca (E05), em 3 seeds.
  2. Avalia M0 em D1 (primeira metade da secao 6.11).
  3. Divide D1 em D1a (75% iniciais das semanas) e D1b (25% finais).
  4. Para cada configuracao de fine-tuning, parte de uma copia de M0, treina
     uma epoca por vez em D1a e mede o MCC em D1b depois de cada epoca.
  5. Escolhe configuracao e numero de epocas pelo MCC medio em D1b.

A fase 2 (avaliacao final) le a configuracao escolhida, refaz o fine-tuning
sobre D1 inteiro e so entao abre D2.

Saidas em modelos/:
  finetuning_escolhido.json   configuracao vencedora
  curvas_finetuning.csv       MCC em D1b e custo de treino por epoca, por seed,
                              para o grafico de overfitting (secao 11)
  m0_em_d1.csv                metricas de M0 em D1 por seed (secao 6.11)

Uso:
    python selecionar_finetuning.py
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from construir_base import ALVO, FEATURES, SEMANA

SEEDS = [42, 1337, 2024]
FRACAO_D1A = 0.75
EPOCAS_MAX = 30

# Configuracoes de fine-tuning (secao 6.12 exige pelo menos duas).
#   A: taxa de aprendizado 10x menor que a de M0. Passos pequenos, ajusta sem
#      apagar o que M0 aprendeu. E a escolha classica para fine-tuning.
#   B: mesma taxa de M0. Adapta mais rapido ao periodo novo, com risco de
#      esquecer o padrao historico (esquecimento catastrofico).
#   C: taxa baixa com regularizacao L2 forte. A L2 puxa os pesos em direcao a
#      zero, o que reduz a complexidade do modelo durante o fine-tuning e
#      limita o sobreajuste a um conjunto pequeno como D1a. Obs.: no
#      scikit-learn o termo L2 e dividido pelo tamanho do lote (200), entao
#      alpha=1e-2 nao teve efeito mensuravel (E07). alpha=1.0 torna a
#      penalidade comparavel ao gradiente dos dados (E08).
# Congelamento de camadas nao entra: a MLP tem uma camada oculta so, e o
# MLPClassifier do scikit-learn nao oferece congelamento.
CONFIGS_FT = {
    "A_lr_baixo": {"learning_rate_init": 3e-4, "alpha": 1e-4},
    "B_lr_original": {"learning_rate_init": 3e-3, "alpha": 1e-4},
    "C_lr_baixo_reg_forte": {"learning_rate_init": 3e-4, "alpha": 1.0},
}


def metricas(y: np.ndarray, prob: np.ndarray, limiar: float = 0.5) -> dict:
    """Metricas da secao 10 para classificacao desbalanceada."""
    pred = (prob >= limiar).astype(int)
    return {
        "mcc": matthews_corrcoef(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
    }


def treinar_do_zero(X: np.ndarray, y: np.ndarray, hp: dict, seed: int) -> MLPClassifier:
    """MLP com pesos aleatorios, sem nenhum peso pre-treinado (secao 9)."""
    modelo = MLPClassifier(
        hidden_layer_sizes=tuple(hp["hidden_layer_sizes"]),
        alpha=hp["alpha"],
        learning_rate_init=hp["learning_rate_init"],
        max_iter=hp["max_iter"],
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo.fit(X, y)
    return modelo


def preparar_finetuning(m0: MLPClassifier, config: dict) -> MLPClassifier:
    """Copia de M0 pronta para continuar o treino com outra configuracao.

    Os pesos (coefs_ e intercepts_) sao preservados: e isso que caracteriza o
    fine-tuning. O otimizador Adam, por outro lado, guarda estado interno
    (medias moveis dos gradientes) e a taxa de aprendizado com que foi criado.
    Removendo-o, o proximo partial_fit cria um otimizador novo ja com a taxa
    de aprendizado da configuracao de fine-tuning.
    """
    modelo = copy.deepcopy(m0)
    modelo.set_params(**config)
    if hasattr(modelo, "_optimizer"):
        del modelo._optimizer
    return modelo


def uma_epoca(modelo: MLPClassifier, X: np.ndarray, y: np.ndarray) -> None:
    """Exatamente uma passada pelos dados, preservando os pesos.

    n_iter_ nao serve para conferir isso: em versoes recentes do scikit-learn
    ele e zerado a cada chamada de partial_fit. O loss_curve_, por outro lado,
    acumula o custo de todas as epocas e so e esvaziado quando os pesos sao
    reinicializados. Se ele crescer exatamente 1, houve uma epoca e os pesos
    de M0 foram mantidos.
    """
    antes = len(modelo.loss_curve_)
    modelo.partial_fit(X, y)
    depois = len(modelo.loss_curve_)
    if depois != antes + 1:
        raise RuntimeError(
            f"loss_curve_ foi de {antes} para {depois} itens. "
            "O partial_fit reinicializou os pesos: isso nao e fine-tuning."
        )


def mudanca_relativa_pesos(atual: MLPClassifier, referencia: MLPClassifier) -> float:
    """Quanto os pesos se afastaram dos de referencia, em fracao da norma.

    Fine-tuning move os pesos pouco a partir de M0. Uma reinicializacao
    produziria valores da ordem de 1 ou mais.
    """
    diff = sum(np.sum((a - b) ** 2) for a, b in zip(atual.coefs_, referencia.coefs_))
    norma = sum(np.sum(b ** 2) for b in referencia.coefs_)
    return float(np.sqrt(diff / norma))


def dividir_d1(d1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D1a = semanas iniciais, D1b = semanas finais, com embargo de 1 semana.

    O embargo remove a ultima semana de D1a, cujo alvo foi calculado com a
    posicao da primeira semana de D1b. Mesmo principio do construir_base.py.
    """
    semanas = np.sort(d1["date"].unique())
    corte = pd.Timestamp(semanas[int(len(semanas) * FRACAO_D1A)])
    d1a = d1[(d1["date"] < corte) & (d1["date"] != corte - SEMANA)]
    d1b = d1[d1["date"] >= corte]
    return d1a, d1b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dados", default="dados/processados", type=Path)
    p.add_argument("--modelos", default="modelos", type=Path)
    args = p.parse_args()

    hp = json.loads((args.modelos / "hiperparametros.json").read_text())
    d0 = pd.read_csv(args.dados / "base_d0.csv", parse_dates=["date"])
    d1 = pd.read_csv(args.dados / "base_d1.csv", parse_dates=["date"])
    d1a, d1b = dividir_d1(d1)

    print(f"Hiperparametros de M0: camadas={hp['hidden_layer_sizes']} "
          f"alpha={hp['alpha']} lr={hp['learning_rate_init']} "
          f"max_iter={hp['max_iter']}")
    print(f"D0  {len(d0):>6} linhas")
    print(f"D1a {len(d1a):>6} linhas  {d1a['date'].min().date()} a {d1a['date'].max().date()}")
    print(f"D1b {len(d1b):>6} linhas  {d1b['date'].min().date()} a {d1b['date'].max().date()}")

    y0 = d0[ALVO].to_numpy()
    y1, y1a, y1b = d1[ALVO].to_numpy(), d1a[ALVO].to_numpy(), d1b[ALVO].to_numpy()

    linhas_m0, linhas_curva = [], []

    for seed in SEEDS:
        # Escala ajustada so em D0. O fine-tuning usa ESTA MESMA escala:
        # os pesos de M0 foram aprendidos sobre dados nessa escala.
        escala = StandardScaler().fit(d0[FEATURES])
        X0 = escala.transform(d0[FEATURES])
        X1 = escala.transform(d1[FEATURES])
        X1a = escala.transform(d1a[FEATURES])
        X1b = escala.transform(d1b[FEATURES])

        m0 = treinar_do_zero(X0, y0, hp, seed)
        convergiu = m0.n_iter_ < hp["max_iter"]
        print(f"\nseed {seed}: M0 treinado em {m0.n_iter_} epocas "
              f"({'convergiu' if convergiu else 'parou no limite'})")

        for nome, X, y in [("D0 (treino)", X0, y0), ("D1", X1, y1)]:
            linhas_m0.append({"seed": seed, "avaliado_em": nome,
                              **metricas(y, m0.predict_proba(X)[:, 1])})

        mcc_m0_d1b = matthews_corrcoef(y1b, m0.predict(X1b))

        for nome_cfg, config in CONFIGS_FT.items():
            modelo = preparar_finetuning(m0, config)

            # Checagem: antes de qualquer epoca, a copia preve igual a M0.
            if not np.allclose(modelo.predict_proba(X1b), m0.predict_proba(X1b)):
                raise RuntimeError("A copia de M0 nao preserva os pesos.")

            linhas_curva.append({"seed": seed, "config": nome_cfg, "epoca": 0,
                                 "mcc_d1b": mcc_m0_d1b, "custo_treino": np.nan})

            for epoca in range(1, EPOCAS_MAX + 1):
                uma_epoca(modelo, X1a, y1a)
                if epoca == 1:
                    print(f"  {nome_cfg:<22} pesos mudaram "
                          f"{mudanca_relativa_pesos(modelo, m0):.1%} na 1a epoca")
                linhas_curva.append({
                    "seed": seed, "config": nome_cfg, "epoca": epoca,
                    "mcc_d1b": matthews_corrcoef(y1b, modelo.predict(X1b)),
                    "custo_treino": modelo.loss_,
                })

    args.modelos.mkdir(parents=True, exist_ok=True)

    tab_m0 = pd.DataFrame(linhas_m0)
    tab_m0.to_csv(args.modelos / "m0_em_d1.csv", index=False)

    print("\n" + "=" * 72)
    print("M0: DESEMPENHO NO TREINO (D0) E NO PERIODO RECENTE (D1)")
    print("=" * 72)
    resumo_m0 = tab_m0.drop(columns="seed").groupby("avaliado_em").agg(["mean", "std"])
    for col in ["mcc", "f1", "pr_auc", "accuracy"]:
        m = resumo_m0[col]
        print(f"  {col:<9} " + "   ".join(
            f"{idx}: {m.loc[idx, 'mean']:.4f} +/- {m.loc[idx, 'std']:.4f}"
            for idx in m.index))
    print(f"  Linha de base trivial em D1: acuracia {1 - y1.mean():.3f}, MCC 0")

    curvas = pd.DataFrame(linhas_curva)
    curvas.to_csv(args.modelos / "curvas_finetuning.csv", index=False)

    media = (curvas.groupby(["config", "epoca"])["mcc_d1b"]
             .agg(["mean", "std"]).reset_index())
    base = media[media["epoca"] == 0]["mean"].iloc[0]

    print("\n" + "=" * 72)
    print("FINE-TUNING: MELHOR EPOCA POR CONFIGURACAO (MCC medio em D1b)")
    print("=" * 72)
    print(f"  M0 sem fine-tuning (epoca 0): {base:.4f}\n")

    melhores = []
    for nome_cfg in CONFIGS_FT:
        c = media[media["config"] == nome_cfg]
        linha = c.loc[c["mean"].idxmax()]
        final = c[c["epoca"] == EPOCAS_MAX].iloc[0]
        melhores.append({"config": nome_cfg, "epocas": int(linha["epoca"]),
                         "mcc_d1b": linha["mean"], "desvio": linha["std"]})
        print(f"  {nome_cfg:<22} melhor epoca {int(linha['epoca']):>2}  "
              f"MCC {linha['mean']:.4f} +/- {linha['std']:.4f}  "
              f"(ganho {linha['mean'] - base:+.4f})  "
              f"na epoca {EPOCAS_MAX}: {final['mean']:.4f}")

    vencedor = max(melhores, key=lambda r: r["mcc_d1b"])

    if vencedor["epocas"] == 0:
        print("\n  ATENCAO: nenhuma configuracao melhorou M0 em D1b. O fine-tuning")
        print("  sera registrado com 0 epocas, e MFT sera igual a M0.")
    if vencedor["epocas"] == EPOCAS_MAX:
        print(f"\n  ATENCAO: melhor epoca coincide com o limite ({EPOCAS_MAX}).")
        print("  A curva pode ainda estar subindo. Considere aumentar EPOCAS_MAX.")

    escolhido = {
        "config": vencedor["config"],
        **CONFIGS_FT[vencedor["config"]],
        "epocas": vencedor["epocas"],
        "mcc_d1b": round(vencedor["mcc_d1b"], 4),
        "ganho_vs_m0_d1b": round(vencedor["mcc_d1b"] - base, 4),
        "fracao_d1a": FRACAO_D1A,
        "seeds": SEEDS,
    }
    (args.modelos / "finetuning_escolhido.json").write_text(
        json.dumps(escolhido, indent=2) + "\n")

    print(f"\nEscolhida: {vencedor['config']} com {vencedor['epocas']} epocas")
    print(f"Gravado em {args.modelos}/finetuning_escolhido.json, "
          f"curvas_finetuning.csv e m0_em_d1.csv")


if __name__ == "__main__":
    main()
