"""

Uso:
    python diagnostico_convergencia.py
"""

import json
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ajustar_mlp import folds_temporais
from construir_base import ALVO, FEATURES

hp = json.load(open("modelos/hiperparametros.json"))
d0 = pd.read_csv("dados/processados/base_d0.csv", parse_dates=["date"])
X, y = d0[FEATURES], d0[ALVO].to_numpy()

# Ultimo fold: o que tem mais treino e valida no periodo mais recente de D0.
treino, val, inicio_val = list(folds_temporais(d0["date"], hp["folds"]))[-1]
escala = StandardScaler().fit(X[treino])
Xt, Xv = escala.transform(X[treino]), escala.transform(X[val])

print(f"Configuracao: camadas={hp['hidden_layer_sizes']} alpha={hp['alpha']} "
      f"lr={hp['learning_rate_init']}")
print(f"Treino: {treino.sum()} linhas | validacao a partir de {inicio_val.date()}: "
      f"{val.sum()} linhas\n")

for max_iter in [50, 100, 200, 400]:
    m = MLPClassifier(
        hidden_layer_sizes=tuple(hp["hidden_layer_sizes"]),
        alpha=hp["alpha"],
        learning_rate_init=hp["learning_rate_init"],
        max_iter=max_iter,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(Xt, y[treino])
    mcc = matthews_corrcoef(y[val], m.predict(Xv))
    status = "convergiu" if m.n_iter_ < max_iter else "limite"
    print(f"max_iter={max_iter:>3}  epocas={m.n_iter_:>3} ({status:<9})  "
          f"custo_final={m.loss_curve_[-1]:.4f}  MCC={mcc:.4f}")
