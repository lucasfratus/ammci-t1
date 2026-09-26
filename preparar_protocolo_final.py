"""Produz rascunho revisavel do protocolo, sem abrir nem avaliar D2."""
import json
from pathlib import Path

from construir_base import FEATURES
from protocolo_final import PACOTES, POLITICA_HASH, salvar_json, salvar_texto, sha256
from importlib.metadata import version


def ler(caminho):
    return json.loads(Path(caminho).read_text(encoding='utf-8'))


def main():
    hp = ler('modelos/hiperparametros.json')
    ft = ler('modelos/finetuning_escolhido.json')
    convergencia = ler('resultados/convergencia/resumo.json')
    if hp.get('embargo_dias') != 7 or not convergencia['convergencia_em_todos']:
        raise ValueError('Validacao temporal/convergencia ainda nao revisada.')
    if ft['hiperparametros_m0'] != hp:
        raise ValueError('Fine-tuning desatualizado em relacao a MLP.')
    gb = ler('resultados/gradient_boosting/configuracao_escolhida.json')
    lr = ler('resultados/regressao_logistica/configuracao_escolhida.json')
    hashes, linhas = {}, {}
    for linha in Path('dados/processados/manifesto.txt').read_text().splitlines():
        if 'sha256=' in linha:
            hashes[linha.split()[0].upper()] = linha.split('sha256=')[1]
            linhas[linha.split()[0].upper()] = int(linha.split()[1])
    fontes = ['ajustar_mlp.py', 'selecionar_finetuning.py', 'construir_base.py',
              'analisar_dados_drift.py', 'avaliar_gradient_boosting.py',
              'modelos/hiperparametros.json', 'modelos/finetuning_escolhido.json',
              'protocolo_final.py', 'experimento_final.py', 'analisar_resultados_finais.py',
              'preparar_protocolo_final.py', 'regressao_logistica_zero.py',
              'auditoria_billboard.py', 'comparar_regressao_logistica.py',
              'reparar_manifestos.py', 'requirements.txt']
    protocolo = dict(
        status='RASCUNHO_PARA_REVISAO_DA_EQUIPE', versao=3,
        hash_policy=POLITICA_HASH,
        dados_permitidos_antes_do_congelamento=['D0', 'D1'],
        hashes_dados=hashes,
        linhas_dados=linhas, ambiente={nome:version(nome) for nome in PACOTES},
        periodos=dict(D0=['2013-01-02','2021-03-10'],
                      D1=['2021-03-24','2023-12-06'], D2=['2023-12-20','2026-09-02']),
        features=FEATURES,
        alvo='1 se mesma musica tem rank menor em t+7 dias; 0 se igual, maior ou ausente',
        horizonte_dias=7, embargo_dias=7, seeds=[42,1337,2024],
        limiar=0.5, modelo_principal='MLPClassifier sem pesos pre-treinados',
        hiperparametros=hp, finetuning=ft,
        estrategias=dict(
            M0='treinar do zero em D0; scaler ajustado somente em D0',
            MFT='copiar M0, manter scaler de D0, reiniciar Adam e atualizar somente em D1 pelas epocas fixadas',
            MRT='mesma arquitetura e hiperparametros; novos pesos e scaler em D0+D1',
            MREC='mesma arquitetura e hiperparametros; novos pesos e scaler somente em D1'),
        comparadores=dict(gradient_boosting=gb, regressao_logistica_zero=lr,
                          uso='baselines historicos treinados em D0 e avaliados em D2; nao substituem MLP',
                          seeds_gb=[42,1337,2024], logistica_deterministica=True),
        metricas=['MCC','F1','precision','recall','average_precision','ROC_AUC','accuracy',
                  'matriz_confusao','tempo_treino_s','tempo_inferencia_s','numero_parametros'],
        metrica_principal='MCC',
        agregacao='media e desvio amostral (ddof=1) entre tres seeds; logistica uma execucao',
        variacao_vs_m0='diferenca absoluta e relativa do MCC medio; relativa indefinida se M0=0',
        ordem='concluir todos os treinos antes de carregar D2; nenhuma decisao revista pelo desempenho em D2',
        convergencia='registrar avisos e iteracoes; nao reajustar hiperparametros apos ver D2',
        drift_final='mesmas medidas de E11; PSI em D0->D2 usa bins D0; D1->D2 usa bins D1; descritivo sem p-valores iid',
        explicabilidade=dict(metodo='permutation_importance', scoring='matthews_corrcoef',
                             conjunto='D2, apenas depois de avaliar', modelos=['M0','MFT','MRT','MREC'],
                             n_repeats=10, random_state=42,
                             uso='analise descritiva; nao selecionar atributos nem reajustar modelos'),
        erros=dict(cortes=['rank: 1-10/11-40/41-100','estreia','reentrada',
                          'semanas_na_parada: 1-4/5-12/13+', 'presenca nos rankings auxiliares',
                          'trimestre de D2'], exemplos='10 falsos positivos e 10 falsos negativos mais confiantes por modelo/seed'),
        analises=dict(erros_por_tipo=10, versao_cortes='rank10_40_tempo4_12_trimestre_v1'),
        saidas=['resultados/final/previsoes.csv','resultados/final/metricas_seeds.csv',
                'resultados/final/resumo.csv','resultados/final/matrizes_confusao.csv',
                'resultados/final/tempos.csv','resultados/final/protocolo_congelado.json',
                'resultados/final/modelos.joblib','resultados/final/treinamento.csv',
                'resultados/final/curvas_treino.csv','resultados/final/metricas_desenvolvimento.csv',
                'resultados/final/execucao.json','resultados/final/manifesto_resultados.json',
                'resultados/analises_finais/'],
        correcao_software='documentar erro, preservar resultados e repetir protocolo afetado; nunca selecionar correcao pelo MCC',
        revisao_pendente='equipe revisa este rascunho; depois salvar versao congelada com hash antes de avaliar D2',
        hashes_fontes={p: sha256(p) for p in fontes})
    pasta = Path('protocolo')
    pasta.mkdir(exist_ok=True)
    destino = pasta/'protocolo_final.rascunho.json'
    salvar_json(destino, protocolo)
    salvar_texto(pasta/'protocolo_final.rascunho.sha256', sha256(destino)+'\n')
    print(f'Rascunho para revisao: {destino}. D2 nao foi aberto.')


if __name__ == '__main__':
    main()
