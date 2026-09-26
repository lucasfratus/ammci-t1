"""Validacao, congelamento explicito e leitura dos dados do experimento final."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from construir_base import FEATURES, ALVO

RAIZ = Path(__file__).resolve().parent
PACOTES = ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl')
EXTENSOES_TEXTO = {'.csv', '.json', '.md', '.py', '.sha256', '.txt'}
POLITICA_HASH = {
    'nome': 'sha256_texto_lf_v1',
    'texto': 'UTF-8 com CRLF e CR normalizados para LF antes do SHA-256',
    'binario': 'bytes exatos',
    'extensoes_texto': sorted(EXTENSOES_TEXTO),
}


def bytes_para_hash(caminho):
    """Bytes canônicos para hashes portáveis entre Windows, Linux e ZIP/Git."""
    caminho = Path(caminho)
    dados = caminho.read_bytes()
    if caminho.suffix.lower() in EXTENSOES_TEXTO:
        try:
            texto = dados.decode('utf-8')
        except UnicodeDecodeError:
            return dados
        dados = texto.replace('\r\n', '\n').replace('\r', '\n').encode('utf-8')
    return dados


def sha256(caminho):
    """SHA-256 canônico: texto independe de EOL; binários usam bytes exatos."""
    h = hashlib.sha256()
    h.update(bytes_para_hash(caminho))
    return h.hexdigest()


def sha256_bruto(caminho):
    """SHA-256 dos bytes exatos, usado somente para registrar artefatos legados."""
    h = hashlib.sha256()
    with Path(caminho).open('rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()


def salvar_json(caminho, objeto):
    conteudo = json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    Path(caminho).write_bytes(conteudo.encode('utf-8'))


def salvar_texto(caminho, texto):
    """Grava UTF-8 com LF exato, sem tradução da plataforma."""
    Path(caminho).write_bytes(str(texto).replace('\r\n', '\n').replace('\r', '\n').encode('utf-8'))


def salvar_csv(caminho, tabela):
    """Grava CSV UTF-8 com LF exato em qualquer sistema operacional."""
    caminho = Path(caminho)
    with caminho.open('w', encoding='utf-8', newline='') as arquivo:
        tabela.to_csv(arquivo, index=False, lineterminator='\n')


def validar(protocolo):
    p = protocolo
    if p.get('hash_policy') != POLITICA_HASH:
        raise ValueError('Política de hash ausente ou incompatível.')
    if p['features'] != FEATURES or p['horizonte_dias'] != 7 or p['embargo_dias'] != 7:
        raise ValueError('Atributos ou horizonte incompatíveis com o projeto.')
    if len(set(p['seeds'])) != len(p['seeds']) or len(p['seeds']) < 3:
        raise ValueError('Exigidas pelo menos três seeds distintas.')
    if any(type(s) is not int or s < 0 for s in p['seeds']):
        raise ValueError('Seeds devem ser inteiros não negativos.')
    if not 0 < p['limiar'] < 1:
        raise ValueError('Limiar inválido.')
    if set(p['estrategias']) != {'M0', 'MFT', 'MRT', 'MREC'}:
        raise ValueError('São necessárias as quatro estratégias MLP.')
    hp, ft = p['hiperparametros'], p['finetuning']
    if ft['hiperparametros_m0'] != hp:
        raise ValueError('Fine-tuning não corresponde à MLP congelada.')
    if type(ft['epocas']) is not int or ft['epocas'] < 0:
        raise ValueError('Número inválido de épocas de fine-tuning.')
    if p['comparadores']['seeds_gb'] != p['seeds']:
        raise ValueError('Seeds dos modelos estocásticos devem coincidir.')
    for nome in ('D0', 'D1', 'D2'):
        if len(p['hashes_dados'][nome]) != 64 or p['linhas_dados'][nome] < 1:
            raise ValueError(f'Metadados inválidos em {nome}.')
        inicio, fim = map(pd.Timestamp, p['periodos'][nome])
        if inicio > fim:
            raise ValueError(f'Período invertido em {nome}.')
    for anterior, seguinte in [('D0', 'D1'), ('D1', 'D2')]:
        if pd.Timestamp(p['periodos'][anterior][1]) + pd.Timedelta(days=7) >= pd.Timestamp(p['periodos'][seguinte][0]):
            raise ValueError('Fronteiras temporais sem embargo.')
    exp = p['explicabilidade']
    if p['analises']['versao_cortes'] != 'rank10_40_tempo4_12_trimestre_v1' or p['analises']['erros_por_tipo'] < 1:
        raise ValueError('Cortes de análise não implementados.')
    if exp['metodo'] != 'permutation_importance' or exp['scoring'] != 'matthews_corrcoef':
        raise ValueError('Método de explicabilidade não implementado.')
    if set(exp['modelos']) != {'M0', 'MFT', 'MRT', 'MREC'} or exp['n_repeats'] < 1:
        raise ValueError('Configuração de explicabilidade inválida.')


def conferir_fontes(p):
    for nome, digest in p['hashes_fontes'].items():
        caminho = (RAIZ / nome).resolve()
        if not caminho.is_relative_to(RAIZ) or sha256(caminho) != digest:
            raise ValueError(f'Código/configuração diverge do protocolo: {nome}')
    for pacote, esperado in p['ambiente'].items():
        if version(pacote) != esperado:
            raise ValueError(f'Versão divergente: {pacote}; esperado {esperado}.')


def ler_protocolo(caminho, exigir_congelado=True):
    caminho = Path(caminho)
    if sha256(caminho) != caminho.with_suffix('.sha256').read_text().strip():
        raise ValueError('Hash do protocolo não confere.')
    p = json.loads(caminho.read_text(encoding='utf-8'))
    if exigir_congelado and p.get('status') != 'CONGELADO':
        raise ValueError('Protocolo ainda não congelado. D2 permanece reservado.')
    validar(p)
    conferir_fontes(p)
    return p


def congelar(rascunho, destino, responsavel):
    p = ler_protocolo(rascunho, exigir_congelado=False)
    if p['status'] != 'RASCUNHO_PARA_REVISAO_DA_EQUIPE' or not responsavel.strip():
        raise ValueError('Informe o responsável pela revisão de um rascunho válido.')
    destino = Path(destino)
    if destino.exists() or destino.with_suffix('.sha256').exists():
        raise FileExistsError('Não é permitido sobrescrever um protocolo congelado.')
    p.update(status='CONGELADO', responsavel_revisao=responsavel.strip(),
             congelado_em_utc=datetime.now(timezone.utc).isoformat(),
             sha256_rascunho=sha256(rascunho))
    p.pop('revisao_pendente', None)
    destino.parent.mkdir(parents=True, exist_ok=True)
    salvar_json(destino, p)
    salvar_texto(destino.with_suffix('.sha256'), sha256(destino)+'\n')
    return p


def corrigir_congelado(rascunho, destino, responsavel):
    """Substitui somente metadados de integridade de um protocolo já avaliado.

    A operação recusa qualquer mudança em decisões experimentais. Ela existe
    para migrar a versão 2, cujo hash dependia de CRLF/LF, para a política
    canônica da versão 3. O protocolo anterior permanece recuperável no Git e
    seus hashes são registrados no novo documento.
    """
    p = ler_protocolo(rascunho, exigir_congelado=False)
    destino = Path(destino)
    if not destino.exists() or not responsavel.strip():
        raise ValueError('Protocolo anterior e responsável são obrigatórios.')
    anterior_bytes = destino.read_bytes()
    anterior = json.loads(anterior_bytes.decode('utf-8'))
    ignorados = {
        'status', 'versao', 'revisao_pendente', 'hash_policy', 'hashes_fontes',
        'responsavel_revisao', 'congelado_em_utc', 'sha256_rascunho',
        'correcao_reprodutibilidade',
    }
    for chave in sorted((set(p) | set(anterior)) - ignorados):
        if p.get(chave) != anterior.get(chave):
            raise ValueError(f'Correção alteraria decisão experimental: {chave}')
    hash_declarado = destino.with_suffix('.sha256').read_text(encoding='utf-8').strip()
    p.update(
        status='CONGELADO',
        responsavel_revisao=responsavel.strip(),
        congelado_em_utc=datetime.now(timezone.utc).isoformat(),
        sha256_rascunho=sha256(rascunho),
        correcao_reprodutibilidade={
            'tipo': 'migração de hashes dependentes de CRLF/LF para texto canônico LF',
            'protocolo_anterior_versao': anterior.get('versao'),
            'protocolo_anterior_sha256_declarado': hash_declarado,
            'protocolo_anterior_sha256_bytes_no_checkout': hashlib.sha256(anterior_bytes).hexdigest(),
            'decisoes_experimentais_alteradas': False,
            'd2_usado_para_novas_decisoes': False,
            'observacao': 'D2 já havia sido avaliado; a nova execução é apenas verificação de reprodutibilidade.',
        },
    )
    p.pop('revisao_pendente', None)
    salvar_json(destino, p)
    salvar_texto(destino.with_suffix('.sha256'), sha256(destino)+'\n')
    return p


def carregar_bloco(pasta, nome, p):
    """Primeiro confere bytes; depois esquema, datas e alvo. Não descobre arquivos."""
    caminho = Path(pasta) / f'base_{nome.lower()}.csv'
    if sha256(caminho) != p['hashes_dados'][nome]:
        raise ValueError(f'Hash divergente em {nome}.')
    df = pd.read_csv(caminho, parse_dates=['date'])
    esperadas = ['date', 'title', 'artist'] + p['features'] + [ALVO]
    if list(df.columns) != esperadas or len(df) != p['linhas_dados'][nome]:
        raise ValueError(f'Esquema ou quantidade de linhas inválido em {nome}.')
    inicio, fim = map(pd.Timestamp, p['periodos'][nome])
    if (df.date.isna().any() or not df.date.is_monotonic_increasing
            or df.date.min() != inicio or df.date.max() != fim
            or not df.date.between(inicio, fim).all()):
        raise ValueError(f'Datas inválidas em {nome}.')
    if df[['title', 'artist']].isna().any().any() or df.duplicated(['date', 'title', 'artist']).any():
        raise ValueError(f'Identidades ausentes/duplicadas em {nome}.')
    if not np.isfinite(df[p['features']].to_numpy(dtype=float)).all():
        raise ValueError(f'Atributos não finitos em {nome}.')
    if not df[ALVO].isin([0, 1]).all():
        raise ValueError(f'Alvo inválido em {nome}.')
    if nome != 'D2' and df[ALVO].nunique() != 2:
        raise ValueError('Treino precisa das duas classes.')
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rascunho', type=Path, default=Path('protocolo/protocolo_final.rascunho.json'))
    parser.add_argument('--saida', type=Path, default=Path('protocolo/protocolo_final.congelado.json'))
    parser.add_argument('--responsavel', required=True)
    parser.add_argument('--confirmar-revisao', action='store_true', required=True,
                        help='Confirma explicitamente a revisão da equipe antes de congelar.')
    parser.add_argument('--corrigir-existente', action='store_true',
                        help='Migra um protocolo v2 sem permitir mudanças experimentais.')
    args = parser.parse_args()
    operacao = corrigir_congelado if args.corrigir_existente else congelar
    operacao(args.rascunho, args.saida, args.responsavel)
    print(f'Protocolo congelado: {args.saida}. Nenhum conjunto de dados foi aberto.')


if __name__ == '__main__':
    main()
