"""Migra os artefatos finais da política legada CRLF para hashes canônicos.

Não treina modelos, não recalcula métricas e não altera previsões. Antes de
gravar os novos manifestos, exige que cada artefato ainda corresponda ao hash
legado por bytes exatos, por CRLF canônico legado ou pela política nova.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from protocolo_final import ler_protocolo, salvar_json, salvar_texto, sha256


def sha256_crlf(caminho):
    caminho = Path(caminho)
    dados = caminho.read_bytes()
    try:
        texto = dados.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    except UnicodeDecodeError:
        return hashlib.sha256(dados).hexdigest()
    return hashlib.sha256(texto.replace('\n', '\r\n').encode('utf-8')).hexdigest()


def conferir_legado(pasta, manifesto, aninhado=False):
    pasta = Path(pasta)
    m = json.loads(Path(manifesto).read_text(encoding='utf-8'))
    arquivos = m['arquivos'] if aninhado else m
    for nome, esperado in arquivos.items():
        caminho = pasta/nome
        if not caminho.is_file() or esperado not in {sha256(caminho), sha256_crlf(caminho)}:
            raise ValueError(f'Artefato legado realmente divergente: {caminho}')


def executar(protocolo, final, analises):
    protocolo, final, analises = map(Path, (protocolo, final, analises))
    ler_protocolo(protocolo)
    conferir_legado(final, final/'manifesto_resultados.json')
    conferir_legado(analises, analises/'manifesto_analises.json', aninhado=True)

    shutil.copyfile(protocolo, final/'protocolo_congelado.json')
    shutil.copyfile(protocolo.with_suffix('.sha256'), final/'protocolo_congelado.sha256')
    salvar_texto(final/'CORRECAO_REPRODUTIBILIDADE.md',
        '# Correção de reprodutibilidade\n\n'
        'Os resultados numéricos não foram alterados. O protocolo e os manifestos '
        'foram migrados para SHA-256 com normalização canônica de terminações de '
        'linha. A repetição completa deve ser executada em um diretório novo e '
        'comparada com `resumo.csv`.\n')
    salvar_json(final/'manifesto_resultados.json', {
        f.name: sha256(f) for f in sorted(final.iterdir())
        if f.is_file() and f.name != 'manifesto_resultados.json'
    })

    salvar_json(analises/'manifesto_analises.json', {
        'protocolo_sha256': sha256(final/'protocolo_congelado.json'),
        'resultados_sha256': sha256(final/'manifesto_resultados.json'),
        'arquivos': {
            f.name: sha256(f) for f in sorted(analises.iterdir())
            if f.is_file() and f.name != 'manifesto_analises.json'
        },
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocolo', type=Path, default=Path('protocolo/protocolo_final.congelado.json'))
    p.add_argument('--final', type=Path, default=Path('resultados/final'))
    p.add_argument('--analises', type=Path, default=Path('resultados/analises_finais'))
    args = p.parse_args()
    executar(args.protocolo, args.final, args.analises)
    print('Protocolos e manifestos migrados; previsões e métricas preservadas.')


if __name__ == '__main__':
    main()
