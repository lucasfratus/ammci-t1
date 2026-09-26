"""Testes do protocolo de E10, sem dados reais nem acesso a D1/D2."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier

from avaliar_gradient_boosting import (
    FEATURES, GRADE, avaliar, carregar_d0, obter_folds, selecionar,
    resumir, validar_d0,
)
from ajustar_mlp import folds_temporais


class TestGradientBoosting(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.d0 = pd.DataFrame(rng.normal(size=(120, len(FEATURES))), columns=FEATURES)
        self.d0['date'] = np.repeat(pd.date_range('2013-01-02', periods=20, freq='7D'), 6)
        self.d0['alvo'] = np.tile([0, 1], 60)

    def test_folds_identicos_expansivos_sem_semanas_partidas(self):
        anterior = set()
        for novo, antigo in zip(obter_folds(self.d0), folds_temporais(self.d0.date, 4)):
            treino, val, _ = novo
            np.testing.assert_array_equal(treino, antigo[0])
            np.testing.assert_array_equal(val, antigo[1])
            semanas_t = set(self.d0.loc[treino, 'date'])
            semanas_v = set(self.d0.loc[val, 'date'])
            self.assertFalse(semanas_t & semanas_v)
            self.assertTrue(anterior < semanas_t)
            self.assertLess(max(semanas_t) + pd.Timedelta(days=7), min(semanas_v))
            self.assertNotIn(min(semanas_v) - pd.Timedelta(days=7), semanas_t)
            anterior = semanas_t

    def test_folds_preservam_validacao_original_e_rejeitam_treino_vazio(self):
        semanas = np.array(sorted(self.d0.date.unique()))
        blocos = np.array_split(semanas, 5)
        for i, (_, val, _) in enumerate(obter_folds(self.d0)):
            np.testing.assert_array_equal(val, self.d0.date.isin(blocos[i + 1]))
        with self.assertRaises(ValueError):
            list(folds_temporais(pd.Series(pd.date_range('2020-01-01', periods=5, freq='7D')), 4))

    def test_recusa_dados_futuros_desordenados_e_alvo_invalido(self):
        for coluna, valor in [('date', pd.Timestamp('2024-01-01')), ('alvo', 2),
                              (FEATURES[0], np.inf)]:
            invalido = self.d0.copy()
            invalido.loc[0, coluna] = valor
            with self.assertRaises(ValueError):
                validar_d0(invalido)
        with self.assertRaises(ValueError):
            validar_d0(self.d0.iloc[::-1])

    def test_hash_incorreto_interrompe_antes_de_ler_csv(self):
        with tempfile.TemporaryDirectory() as pasta:
            base, manifesto = Path(pasta) / 'd0.csv', Path(pasta) / 'manifesto.txt'
            base.write_text('dados incorretos')
            manifesto.write_text('d0 120 linhas sha256=0000\n')
            with patch('avaliar_gradient_boosting.pd.read_csv') as leitura:
                with self.assertRaises(ValueError):
                    carregar_d0(base, manifesto)
                leitura.assert_not_called()

    def test_selecao_por_media_e_desempate_estavel(self):
        busca = pd.DataFrame({'config': ['GB00', 'GB00', 'GB01', 'GB01'],
                              'mcc': [0.0, 0.8, 0.5, 0.5]})
        self.assertEqual(selecionar(busca), 'GB01')
        busca['mcc'] = 0.5
        self.assertEqual(selecionar(busca.iloc[::-1]), 'GB00')

    def test_execucao_metricas_e_early_stopping_desativado(self):
        params = {**GRADE[0], 'max_iter': 3, 'max_leaf_nodes': 3}
        with threadpool_limits(limits=1), patch(
            'avaliar_gradient_boosting.HistGradientBoostingClassifier',
            wraps=HistGradientBoostingClassifier,
        ) as construtor:
            linhas = avaliar(self.d0, obter_folds(self.d0), 'Gradient Boosting',
                             params, 42, 'teste')
        self.assertEqual(construtor.call_count, 4)
        for chamada in construtor.call_args_list:
            self.assertIs(chamada.kwargs['early_stopping'], False)
        resultado = pd.DataFrame(linhas)
        self.assertEqual(len(resultado), 4)
        self.assertTrue((resultado.iteracoes == 3).all())
        self.assertTrue(np.isfinite(resultado.select_dtypes('number')).all().all())
        resumo = resumir(resultado)
        self.assertAlmostEqual(resumo.iloc[0].mcc_media, resultado.mcc.mean())
        self.assertAlmostEqual(resumo.iloc[0].mcc_desvio_folds, resultado.mcc.std(ddof=0))


if __name__ == '__main__':
    unittest.main()
