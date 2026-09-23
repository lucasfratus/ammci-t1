"""Regressao logistica binaria implementada apenas com NumPy.

A classe tem finalidade didatica: explicita a funcao de custo, o gradiente e a
atualizacao dos parametros sem usar uma implementacao pronta de aprendizado de
maquina. O StandardScaler continua externo, pois o enunciado permite usar
ferramentas de pre-processamento desde que sejam ajustadas sem vazamento.
"""

from __future__ import annotations

import numpy as np


class RegressaoLogisticaZero:
    """Classificador binario treinado por gradiente descendente em lote.

    Parametros
    ----------
    taxa_aprendizado:
        Tamanho do passo do gradiente descendente.
    max_iter:
        Numero maximo de atualizacoes dos parametros.
    l2:
        Intensidade da regularizacao L2 aplicada somente aos pesos.
    tolerancia:
        Criterio de parada para a variacao relativa da funcao de custo.
    class_weight:
        ``None`` ou ``"balanced"``. No segundo caso, cada classe recebe peso
        inversamente proporcional a sua frequencia no conjunto de treino.
    """

    def __init__(
        self,
        taxa_aprendizado: float = 0.1,
        max_iter: int = 2_000,
        l2: float = 0.0,
        tolerancia: float = 1e-9,
        class_weight: str | None = None,
    ) -> None:
        if taxa_aprendizado <= 0:
            raise ValueError("taxa_aprendizado deve ser positiva")
        if max_iter <= 0:
            raise ValueError("max_iter deve ser positivo")
        if l2 < 0:
            raise ValueError("l2 nao pode ser negativo")
        if tolerancia < 0:
            raise ValueError("tolerancia nao pode ser negativa")
        if class_weight not in (None, "balanced"):
            raise ValueError('class_weight deve ser None ou "balanced"')

        self.taxa_aprendizado = float(taxa_aprendizado)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.tolerancia = float(tolerancia)
        self.class_weight = class_weight

    @staticmethod
    def _validar_X(X: np.ndarray) -> np.ndarray:
        matriz = np.asarray(X, dtype=np.float64)
        if matriz.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional")
        if matriz.shape[0] == 0 or matriz.shape[1] == 0:
            raise ValueError("X nao pode ser vazio")
        if not np.isfinite(matriz).all():
            raise ValueError("X contem NaN ou infinito")
        return matriz

    @staticmethod
    def _sigmoide(z: np.ndarray) -> np.ndarray:
        """Sigmoide estavel mesmo para logits de grande magnitude."""
        z = np.asarray(z, dtype=np.float64)
        saida = np.empty_like(z)
        positivos = z >= 0
        saida[positivos] = 1.0 / (1.0 + np.exp(-z[positivos]))
        exp_z = np.exp(z[~positivos])
        saida[~positivos] = exp_z / (1.0 + exp_z)
        return saida

    def _pesos_amostras(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            self.class_weight_ = {0: 1.0, 1: 1.0}
            return np.ones(y.shape[0], dtype=np.float64)

        n = y.shape[0]
        n_zero = int((y == 0).sum())
        n_um = int((y == 1).sum())
        self.class_weight_ = {
            0: n / (2.0 * n_zero),
            1: n / (2.0 * n_um),
        }
        return np.where(y == 0, self.class_weight_[0], self.class_weight_[1])

    def _custo(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pesos_amostras: np.ndarray,
    ) -> float:
        logits = X @ self.pesos_ + self.vies_
        # logaddexp(0, z) - y*z equivale a binary cross-entropy, mas evita
        # log(0) e overflow para probabilidades muito proximas de 0 ou 1.
        entropia = np.logaddexp(0.0, logits) - y * logits
        dados = np.sum(pesos_amostras * entropia) / y.shape[0]
        regularizacao = 0.5 * self.l2 * np.dot(self.pesos_, self.pesos_)
        return float(dados + regularizacao)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegressaoLogisticaZero:
        X = self._validar_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y deve ser um vetor unidimensional")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X e y possuem numeros diferentes de amostras")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y deve conter somente as classes 0 e 1")
        if np.unique(y).size != 2:
            raise ValueError("o treino precisa conter as duas classes")
        y = y.astype(np.float64, copy=False)

        n_amostras, n_atributos = X.shape
        self.n_features_in_ = n_atributos
        self.classes_ = np.array([0, 1])
        self.pesos_ = np.zeros(n_atributos, dtype=np.float64)
        self.vies_ = 0.0
        pesos_amostras = self._pesos_amostras(y)

        self.curva_custo_ = []
        self.convergiu_ = False
        custo_anterior = self._custo(X, y, pesos_amostras)

        for iteracao in range(1, self.max_iter + 1):
            logits = X @ self.pesos_ + self.vies_
            probabilidades = self._sigmoide(logits)
            erros_ponderados = pesos_amostras * (probabilidades - y)

            gradiente_pesos = (
                X.T @ erros_ponderados / n_amostras + self.l2 * self.pesos_
            )
            gradiente_vies = float(erros_ponderados.sum() / n_amostras)

            self.pesos_ -= self.taxa_aprendizado * gradiente_pesos
            self.vies_ -= self.taxa_aprendizado * gradiente_vies

            custo = self._custo(X, y, pesos_amostras)
            if not np.isfinite(custo):
                raise FloatingPointError(
                    "o custo divergiu; reduza a taxa de aprendizado"
                )
            self.curva_custo_.append(custo)

            variacao = abs(custo_anterior - custo)
            escala = max(1.0, abs(custo_anterior))
            if variacao <= self.tolerancia * escala:
                self.convergiu_ = True
                break
            custo_anterior = custo

        self.n_iter_ = iteracao
        self.custo_final_ = custo
        # Nomes no estilo scikit-learn facilitam a comparacao, sem mudar a
        # implementacao: coef_ e intercept_ sao apenas visoes dos parametros.
        self.coef_ = self.pesos_.reshape(1, -1)
        self.intercept_ = np.array([self.vies_])
        return self

    def _verificar_ajustado(self) -> None:
        if not hasattr(self, "pesos_"):
            raise RuntimeError("o modelo ainda nao foi treinado")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._verificar_ajustado()
        X = self._validar_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X possui quantidade inesperada de atributos")
        return X @ self.pesos_ + self.vies_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob_um = self._sigmoide(self.decision_function(X))
        return np.column_stack([1.0 - prob_um, prob_um])

    def predict(self, X: np.ndarray, limiar: float = 0.5) -> np.ndarray:
        if not 0.0 < limiar < 1.0:
            raise ValueError("limiar deve estar estritamente entre 0 e 1")
        return (self.predict_proba(X)[:, 1] >= limiar).astype(int)
