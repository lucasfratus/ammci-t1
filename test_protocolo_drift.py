"""Casos de fronteira temporal e verificacao das medidas de drift."""
import unittest
import numpy as np
import pandas as pd

from ajustar_mlp import folds_temporais
from analisar_dados_drift import psi_referencia, divergencia_categorica
from selecionar_finetuning import dividir_d1, configuracoes_ft


class TestProtocoloDrift(unittest.TestCase):
    def test_horizonte_nao_alcanca_validacao_com_datas_irregulares(self):
        datas = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-08', '2020-01-15',
                                         '2020-01-18', '2020-01-22', '2020-01-29']))
        for treino, val, inicio in folds_temporais(datas, 1):
            self.assertTrue((datas[treino] + pd.Timedelta(days=7) < inicio).all())
            self.assertTrue((datas[val] >= inicio).all())

    def test_d1_preserva_embargo(self):
        df = pd.DataFrame({'date': np.repeat(pd.date_range('2021-03-24', periods=20, freq='7D'), 2)})
        a, b = dividir_d1(df)
        self.assertLess(a.date.max()+pd.Timedelta(days=7), b.date.min())
        self.assertEqual(len(df)-len(a)-len(b), 2)

    def test_ft_acompanha_hiperparametros_revisados(self):
        configs = configuracoes_ft({'learning_rate_init': .001, 'alpha': .01})
        self.assertEqual(configs['B_lr_original'], dict(learning_rate_init=.001, alpha=.01))
        self.assertAlmostEqual(configs['A_lr_baixo']['learning_rate_init'], .0001)
        self.assertEqual(configs['C_lr_baixo_reg_forte']['alpha'], 1.0)

    def test_psi_identidade_e_caudas_definidas_apenas_na_referencia(self):
        ref = np.arange(100)
        zero, bordas, _, _ = psi_referencia(ref, ref)
        mudanca, outras, _, q = psi_referencia(ref, np.arange(100)+1000)
        self.assertAlmostEqual(zero, 0)
        self.assertGreater(mudanca, 0)
        np.testing.assert_array_equal(bordas, outras)
        self.assertAlmostEqual(q.sum(), 1)
        self.assertTrue(np.isneginf(bordas[0]) and np.isposinf(bordas[-1]))

    def test_psi_constante_detecta_novos_valores(self):
        valor, _, _, _ = psi_referencia(np.zeros(100), np.ones(100))
        self.assertTrue(np.isfinite(valor))
        self.assertGreater(valor, 0)

    def test_js_divergencia_em_bits(self):
        igual, _, _, _ = divergencia_categorica([0, 1], [0, 1])
        disjunto, _, _, _ = divergencia_categorica([0, 0], [1, 1])
        self.assertAlmostEqual(igual, 0)
        self.assertAlmostEqual(disjunto, 1)


if __name__ == '__main__':
    unittest.main()
