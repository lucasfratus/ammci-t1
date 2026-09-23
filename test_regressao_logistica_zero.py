"""Testes da implementacao didatica de regressao logistica."""

import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from regressao_logistica_zero import RegressaoLogisticaZero


class TestRegressaoLogisticaZero(unittest.TestCase):
    def test_aprende_problema_separavel(self) -> None:
        X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        modelo = RegressaoLogisticaZero(taxa_aprendizado=0.2, max_iter=2_000)
        modelo.fit(X, y)
        np.testing.assert_array_equal(modelo.predict(X), y)
        self.assertLess(modelo.curva_custo_[-1], modelo.curva_custo_[0])

    def test_sigmoide_e_estavel(self) -> None:
        observado = RegressaoLogisticaZero._sigmoide(
            np.array([-1_000.0, 0.0, 1_000.0])
        )
        self.assertTrue(np.isfinite(observado).all())
        np.testing.assert_allclose(observado, [0.0, 0.5, 1.0], atol=1e-15)

    def test_rejeita_alvo_invalido(self) -> None:
        X = np.ones((3, 2))
        with self.assertRaisesRegex(ValueError, "classes 0 e 1"):
            RegressaoLogisticaZero().fit(X, np.array([0, 1, 2]))

    def test_l2_reduz_norma_dos_pesos(self) -> None:
        X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        sem_l2 = RegressaoLogisticaZero(
            taxa_aprendizado=0.1, max_iter=3_000, l2=0.0
        ).fit(X, y)
        com_l2 = RegressaoLogisticaZero(
            taxa_aprendizado=0.1, max_iter=3_000, l2=1.0
        ).fit(X, y)
        self.assertLess(np.linalg.norm(com_l2.coef_), np.linalg.norm(sem_l2.coef_))

    def test_coerente_com_sklearn_sem_regularizacao(self) -> None:
        gerador = np.random.default_rng(42)
        X = gerador.normal(size=(500, 4))
        logits = 1.4 * X[:, 0] - 0.8 * X[:, 1] + 0.4 * X[:, 2] - 0.2
        y = (gerador.random(500) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
        X = StandardScaler().fit_transform(X)

        proprio = RegressaoLogisticaZero(
            taxa_aprendizado=0.2,
            max_iter=10_000,
            tolerancia=1e-12,
        ).fit(X, y)
        biblioteca = LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            max_iter=10_000,
            tol=1e-12,
        ).fit(X, y)

        prob_propria = proprio.predict_proba(X)[:, 1]
        prob_biblioteca = biblioteca.predict_proba(X)[:, 1]
        np.testing.assert_allclose(prob_propria, prob_biblioteca, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
