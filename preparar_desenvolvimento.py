"""Reconstroi somente D0/D1 e exige os hashes existentes, sem gerar D2."""
from pathlib import Path
import hashlib

from construir_base import (
    carregar, preparar_hot100, juntar_secundaria, juntar_album,
    INICIO, CORTE_D1, CORTE_D2, SEMANA, LIMIAR_HIATO_DIAS,
    SECUNDARIAS, IDENTIFICACAO, FEATURES, ALVO,
)


def main():
    pasta = Path('dados')
    hot = carregar(pasta, 'hot100')
    hot = hot[hot.date < CORTE_D2]
    base = preparar_hot100(hot, LIMIAR_HIATO_DIAS)
    base = base[(base.date >= INICIO) & (base.date < CORTE_D2 - SEMANA)]
    base = base[base.date != CORTE_D1 - SEMANA]
    for nome in SECUNDARIAS:
        sec = carregar(pasta, nome)
        base = juntar_secundaria(base, sec[sec.date < CORTE_D2], nome)
    album = carregar(pasta, 'billboard200')
    base = juntar_album(base, album[album.date < CORTE_D2])
    base['mes'] = base.date.dt.month
    base['semana_ano'] = base.date.dt.isocalendar().week.astype(int)
    base = base[IDENTIFICACAO + FEATURES + [ALVO]].sort_values(['date', 'rank'])
    manifesto = (pasta/'processados/manifesto.txt').read_text()
    for nome, bloco in [('d0', base[base.date < CORTE_D1]), ('d1', base[base.date >= CORTE_D1])]:
        conteudo = bloco.to_csv(index=False, lineterminator='\n').encode('utf-8')
        digest = hashlib.sha256(conteudo).hexdigest()
        esperado = next(l for l in manifesto.splitlines() if l.startswith(nome+' ')).split('sha256=')[1]
        if digest != esperado:
            raise ValueError(f'Hash divergente em {nome}; manifesto e dados preservados.')
        caminho = pasta/'processados'/f'base_{nome}.csv'
        if caminho.exists() and hashlib.sha256(caminho.read_bytes()).hexdigest() != esperado:
            raise ValueError(f'Arquivo existente divergente: {caminho}')
        caminho.write_bytes(conteudo)
        print(f'{nome}: {len(bloco)} linhas, SHA-256 conferido', flush=True)


if __name__ == '__main__':
    main()
