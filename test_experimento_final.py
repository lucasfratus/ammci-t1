"""Integracao completa em dados sinteticos. Nao abre os conjuntos reais."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import nbformat
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import experimento_final as final
from analisar_resultados_finais import executar as analisar, conferir_execucao, analisar_erros
from protocolo_final import (RAIZ, carregar_bloco, congelar, corrigir_congelado,
                             ler_protocolo, salvar_json, sha256, validar)


class TestExperimentoFinal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporario = tempfile.TemporaryDirectory(prefix='ammci-final-teste-')
        cls.raiz = Path(cls.temporario.name)
        cls.dados = cls.raiz/'dados'
        cls.dados.mkdir()
        cls.p = json.loads((RAIZ/'protocolo/protocolo_final.rascunho.json').read_text(encoding='utf-8'))
        p = cls.p
        p['hashes_fontes'] = {nome:sha256(RAIZ/nome) for nome in p['hashes_fontes']}
        p['hiperparametros'].update(hidden_layer_sizes=[3], max_iter=6)
        p['finetuning'].update(epocas=2, hiperparametros_m0=copy.deepcopy(p['hiperparametros']))
        p['comparadores']['gradient_boosting'].update(max_iter=3, max_leaf_nodes=3)
        p['comparadores']['regressao_logistica_zero'].update(max_iter=30, tolerancia=1e-3)
        p['explicabilidade']['n_repeats'] = 2
        p['analises']['erros_por_tipo'] = 3
        rng = np.random.default_rng(123)
        for nome, inicio, semanas in [('D0','2013-01-02',24),('D1','2014-01-01',12),('D2','2015-01-07',12)]:
            n=semanas*4
            df=pd.DataFrame(rng.normal(size=(n,len(p['features']))), columns=p['features'])
            for coluna in ('rank','rank_lag1','rank_lag2','peak_pos'):
                df[coluna]=rng.integers(1,101,size=n)
            for coluna in ('estreia','reentrada','radio_presente','streaming_presente','digital_presente','album_presente'):
                df[coluna]=rng.integers(0,2,size=n)
            df['semanas_na_parada']=rng.integers(1,30,size=n)
            df['date']=np.repeat(pd.date_range(inicio,periods=semanas,freq='7D'),4)
            df['title']=[f'Musica sintetica {i%4}' for i in range(n)]
            df['artist']='Artista sintetico'
            df['alvo']=np.tile([0,1],n//2)
            df=df[['date','title','artist']+p['features']+['alvo']]
            caminho=cls.dados/f'base_{nome.lower()}.csv'
            df.to_csv(caminho,index=False,lineterminator='\n')
            p['hashes_dados'][nome]=sha256(caminho)
            p['linhas_dados'][nome]=n
            p['periodos'][nome]=[str(df.date.min().date()),str(df.date.max().date())]
        cls.rascunho=cls.raiz/'rascunho.json'
        salvar_json(cls.rascunho,p)
        cls.rascunho.with_suffix('.sha256').write_text(sha256(cls.rascunho))
        cls.congelado=cls.raiz/'congelado.json'
        congelar(cls.rascunho,cls.congelado,'TESTE AUTOMATIZADO: somente dados sinteticos')
        cls.saida=cls.raiz/'final'
        cls.ordem=[]
        real_loader=final.carregar_bloco
        real_train=final.treinar_modelos
        def carregar(pasta,nome,p):
            cls.ordem.append('abrir_'+nome)
            if nome=='D2':
                assert 'treino_concluido' in cls.ordem
                assert (cls.saida/'modelos.joblib').exists()
            return real_loader(pasta,nome,p)
        def treinar(*args):
            resultado=real_train(*args)
            cls.ordem.append('treino_concluido')
            return resultado
        with patch.object(final,'carregar_bloco',side_effect=carregar), patch.object(final,'treinar_modelos',side_effect=treinar):
            cls.resumo=final.executar(cls.congelado,cls.dados,cls.saida)

    @classmethod
    def tearDownClass(cls):
        cls.temporario.cleanup()

    def test_todos_treinos_antes_de_d2(self):
        self.assertEqual(self.ordem,['abrir_D0','abrir_D1','treino_concluido','abrir_D2'])
        modelos=joblib.load(self.saida/'modelos.joblib')
        self.assertEqual(len(modelos),16)  # 4 MLP x 3 seeds + 3 GB + 1 LR.
        self.assertEqual(conferir_execucao(self.saida)['status'],'CONGELADO')

    def test_rascunho_bloqueia_antes_de_abrir_dados(self):
        with patch.object(final,'carregar_bloco') as loader:
            with self.assertRaisesRegex(ValueError,'congelado'):
                final.executar(self.rascunho,self.dados,self.raiz/'proibido')
            loader.assert_not_called()
        self.assertFalse((self.raiz/'proibido').exists())

    def test_hash_protocolo_alterado_recusado(self):
        caminho=self.raiz/'alterado.json'
        caminho.write_bytes(self.congelado.read_bytes()+b' ')
        caminho.with_suffix('.sha256').write_text(sha256(self.congelado))
        with self.assertRaisesRegex(ValueError,'Hash'):
            ler_protocolo(caminho)

    def test_hash_textual_independe_de_crlf_lf(self):
        lf = self.raiz/'eol_lf.md'
        crlf = self.raiz/'eol_crlf.md'
        lf.write_bytes(b'linha 1\nlinha 2\n')
        crlf.write_bytes(b'linha 1\r\nlinha 2\r\n')
        self.assertEqual(sha256(lf), sha256(crlf))
        crlf.write_bytes(b'linha 1\r\nlinha alterada\r\n')
        self.assertNotEqual(sha256(lf), sha256(crlf))

    def test_fontes_alteradas_recusadas(self):
        caminho=self.raiz/'fonte_alterada.json'
        p=json.loads(self.congelado.read_text(encoding='utf-8'))
        p['hashes_fontes']['experimento_final.py']='0'*64
        salvar_json(caminho,p)
        caminho.with_suffix('.sha256').write_text(sha256(caminho))
        with self.assertRaisesRegex(ValueError,'diverge'):
            ler_protocolo(caminho)

    def test_correcao_de_hash_recusa_mudanca_experimental(self):
        rascunho = self.raiz/'rascunho_alterado.json'
        p = copy.deepcopy(self.p)
        p['limiar'] = 0.4
        salvar_json(rascunho, p)
        rascunho.with_suffix('.sha256').write_text(sha256(rascunho))
        destino = self.raiz/'congelado_para_correcao.json'
        shutil.copyfile(self.congelado, destino)
        shutil.copyfile(self.congelado.with_suffix('.sha256'), destino.with_suffix('.sha256'))
        with self.assertRaisesRegex(ValueError, 'decisão experimental: limiar'):
            corrigir_congelado(rascunho, destino, 'teste')

    def test_hash_dados_bloqueia_antes_de_parsear(self):
        p=copy.deepcopy(self.p)
        p['hashes_dados']['D2']='0'*64
        with patch('protocolo_final.pd.read_csv') as leitura:
            with self.assertRaisesRegex(ValueError,'Hash'):
                carregar_bloco(self.dados,'D2',p)
            leitura.assert_not_called()

    def test_scalers_respeitam_conjuntos_e_ft_preserva_m0(self):
        modelos=joblib.load(self.saida/'modelos.joblib')
        d0=carregar_bloco(self.dados,'D0',self.p)
        d1=carregar_bloco(self.dados,'D1',self.p)
        for seed in self.p['seeds']:
            m0,ft,rt,rec=[modelos[f'{nome}_seed{seed}'] for nome in ['M0','MFT','MRT','MREC']]
            np.testing.assert_allclose(m0['scaler'].mean_,d0[self.p['features']].mean())
            np.testing.assert_array_equal(m0['scaler'].mean_,ft['scaler'].mean_)
            np.testing.assert_allclose(rec['scaler'].mean_,d1[self.p['features']].mean())
            np.testing.assert_allclose(rt['scaler'].mean_,pd.concat([d0,d1])[self.p['features']].mean())
            self.assertEqual(len(ft['estimador'].loss_curve_)-len(m0['estimador'].loss_curve_),2)
            self.assertFalse(any(np.shares_memory(a,b) for a,b in zip(m0['estimador'].coefs_,ft['estimador'].coefs_)))
            self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(m0['estimador'].coefs_,ft['estimador'].coefs_)))

    def test_previsoes_metricas_e_agregacao(self):
        pred=pd.read_csv(self.saida/'previsoes.csv')
        met=pd.read_csv(self.saida/'metricas_seeds.csv')
        self.assertEqual(len(pred),16*self.p['linhas_dados']['D2'])
        for r in met.itertuples():
            grupo=pred[(pred.modelo==r.modelo)&(pred.seed==r.seed)]
            self.assertEqual(r.tn+r.fp+r.fn+r.tp,len(grupo))
            m=final.metricas(grupo.alvo.to_numpy(),grupo.probabilidade.to_numpy(),self.p['limiar'])
            self.assertAlmostEqual(r.mcc,m['mcc'])
        self.assertTrue(pd.isna(self.resumo.loc[self.resumo.modelo=='LR','mcc_desvio'].iloc[0]))
        self.assertEqual(self.resumo.loc[self.resumo.modelo=='M0','delta_mcc_vs_m0'].iloc[0],0)

    def test_metricas_grupo_de_classe_unica(self):
        m=final.metricas(np.zeros(4),np.array([.1,.2,.3,.9]),.5)
        self.assertTrue(np.isnan(m['roc_auc']))
        self.assertTrue(np.isnan(m['pr_auc']))
        self.assertEqual((m['tn'],m['fp'],m['fn'],m['tp']),(3,1,0,0))

    def test_recusa_sobrescrita(self):
        with self.assertRaises(FileExistsError):
            final.executar(self.congelado,self.dados,self.saida)
        with self.assertRaises(FileExistsError):
            congelar(self.rascunho,self.congelado,'teste')

    def test_analises_completas_com_dados_sinteticos(self):
        saida=self.raiz/'analises'
        resumo=analisar(self.saida,self.dados,saida)
        self.assertEqual(len(resumo),4*len(self.p['features']))
        self.assertTrue(np.isfinite(resumo.importancia_media).all())
        drift=pd.read_csv(saida/'drift.csv')
        self.assertEqual(set(zip(drift.referencia,drift.recente)),{('D0','D1'),('D0','D2'),('D1','D2')})
        exemplos=pd.read_csv(saida/'erros_exemplos.csv')
        self.assertTrue((exemplos.alvo!=exemplos.predicao).all())
        self.assertTrue((exemplos.groupby(['modelo','seed','tipo_erro']).size()<=3).all())
        for nome in ['drift.png','importancias.png','matrizes_confusao.png','RELATORIO.md']:
            self.assertTrue((saida/nome).stat().st_size>0)
        # Exercita tambem o caminho com resultados dos notebooks, sem kernel externo.
        for nome in ['03_experimento_final_drift.ipynb', '04_explicabilidade_erros.ipynb']:
            nb = nbformat.read(RAIZ/'notebooks'/nome, as_version=4)
            nbformat.validate(nb)
            contexto = {}
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    fonte = cell.source.replace("Path('resultados/final')", repr(self.saida.as_posix()))
                    fonte = fonte.replace("Path('resultados/analises_finais')", repr(saida.as_posix()))
                    # Preserva Path nos caminhos substituidos.
                    fonte = fonte.replace('final = '+repr(self.saida.as_posix()), 'final = Path('+repr(self.saida.as_posix())+')')
                    fonte = fonte.replace('analises = '+repr(saida.as_posix()), 'analises = Path('+repr(saida.as_posix())+')')
                    exec(compile(fonte, nome, 'exec'), contexto)
                    contexto['display'] = lambda *args: None
            self.assertTrue(contexto['pronto'] and contexto['analises_prontas'])

    def test_falha_no_treino_nao_abre_d2(self):
        saida = self.raiz/'falha_treino'
        with patch.object(final, 'treinar_modelos', side_effect=RuntimeError('falha simulada')), patch.object(final, 'carregar_bloco', wraps=carregar_bloco) as loader:
            with self.assertRaisesRegex(RuntimeError, 'falha simulada'):
                final.executar(self.congelado, self.dados, saida)
        self.assertEqual([c.args[1] for c in loader.call_args_list], ['D0', 'D1'])
        self.assertEqual(json.loads((saida/'execucao.json').read_text())['status'], 'FALHOU')

    def test_artefato_alterado_bloqueia_analise(self):
        copia = self.raiz/'final_alterado'
        shutil.copytree(self.saida, copia)
        with (copia/'previsoes.csv').open('a') as arquivo:
            arquivo.write('\n')
        with patch('analisar_resultados_finais.carregar_bloco') as loader:
            with self.assertRaisesRegex(ValueError, 'Artefato alterado'):
                analisar(copia, self.dados, self.raiz/'analise_bloqueada')
            loader.assert_not_called()

    def test_erros_rejeitam_predicoes_desalinhadas(self):
        d2=carregar_bloco(self.dados,'D2',self.p)
        pred=pd.read_csv(self.saida/'previsoes.csv')
        pred.loc[0,'row_id']=9999
        with self.assertRaisesRegex(ValueError,'exatamente'):
            analisar_erros(pred,d2,self.p)

    def test_protocolo_rejeita_embargo_ausente(self):
        p=copy.deepcopy(self.p)
        p['periodos']['D1'][0]=str((pd.Timestamp(p['periodos']['D0'][1])+pd.Timedelta(days=7)).date())
        with self.assertRaisesRegex(ValueError,'embargo'):
            validar(p)

    def test_entrega_versionada_tem_integridade(self):
        protocolo = ler_protocolo(RAIZ/'protocolo/protocolo_final.congelado.json')
        self.assertEqual(protocolo['status'], 'CONGELADO')
        self.assertEqual(conferir_execucao(RAIZ/'resultados/final')['versao'], 3)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
