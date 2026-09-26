"""
Constroi a base final do Trabalho 1 de AMMCI a partir dos CSVs do Billboard.

Saida: tres arquivos (D0, D1, D2) em dados/processados/, mais um manifesto com
hashes SHA-256 para reprodutibilidade.

Uso:
    python construir_base.py
    python construir_base.py --limiar-hiato 14      # experimento alternativo
    python construir_base.py --limiar-hiato 56

Regras que este script garante:
  - Toda coluna usa apenas informacao disponivel ate a semana da linha.
  - Passagens e lags sao calculados sobre o historico completo desde 1958,
    antes do recorte, para que uma musica que ja estava na parada em 2012
    nao apareca como estreia em janeiro de 2013.
  - Nenhuma estatistica de D2 e impressa. D2 so e aberto na avaliacao final.
  - Nenhum transformador (normalizacao, imputacao por media etc.) e ajustado
    aqui. Isso acontece depois, no pipeline de cada modelo, so com dados de treino.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from auditoria_billboard import carregar

# ---------------------------------------------------------------------------
# Constantes do protocolo experimental. Mudar qualquer uma exige registro no
# diario de experimentos.
# ---------------------------------------------------------------------------
INICIO = pd.Timestamp("2013-01-01")
CORTE_D1 = pd.Timestamp("2021-03-24")   # primeira semana de D1
CORTE_D2 = pd.Timestamp("2023-12-20")   # primeira semana de D2
SEMANA = pd.Timedelta(days=7)

LIMIAR_HIATO_DIAS = 28      # ausencia maior que isso abre uma passagem nova
CENSURA_HOT100 = 101        # "nao estava no Hot 100 na semana anterior"
CENSURA_SECUNDARIA = 51     # "fora do top 50" em radio, streaming e digital
CENSURA_ALBUM = 201         # "artista sem album no Billboard 200"

CHAVE = ["key_titulo", "key_artista"]
SECUNDARIAS = ["radio", "streaming", "digital"]

FEATURES = [
    "rank", "rank_lag1", "rank_lag2", "variacao_1s",
    "peak_pos", "distancia_pico",
    "semanas_na_parada", "hiato_semanas", "estreia", "reentrada",
    "radio_rank", "radio_presente",
    "streaming_rank", "streaming_presente",
    "digital_rank", "digital_presente",
    "album_rank", "album_presente",
    "mes", "semana_ano",
]
IDENTIFICACAO = ["date", "title", "artist"]   # mantidas so para analise de erros
ALVO = "alvo"


def preparar_hot100(df: pd.DataFrame, limiar_dias: int) -> pd.DataFrame:
    """Passagens, lags, contagens e alvo, sobre o historico completo."""
    df = df.dropna(subset=["rank"])
    df = df.drop_duplicates(subset=["date"] + CHAVE, keep="first")
    df = df.sort_values(CHAVE + ["date"]).reset_index(drop=True)

    # Passagem: cada vez que a musica fica fora mais que o limiar, conta como
    # uma nova passagem pela parada. Ausencias curtas sao oscilacao de borda.
    gap = df.groupby(CHAVE, sort=False)["date"].diff().dt.days
    df["nova_passagem"] = gap.isna() | (gap > limiar_dias)
    df["passagem"] = df.groupby(CHAVE, sort=False)["nova_passagem"].cumsum()

    # Semanas ausentes dentro da mesma passagem (gap de 14 dias = 1 semana fora).
    df["hiato_semanas"] = (
        (gap.where(~df["nova_passagem"]) / 7 - 1).fillna(0).astype(int)
    )

    grupo = df.groupby(CHAVE + ["passagem"], sort=False)
    df["semanas_na_parada"] = grupo.cumcount() + 1
    df["rank_lag1"] = grupo["rank"].shift(1).fillna(CENSURA_HOT100)
    df["rank_lag2"] = grupo["rank"].shift(2).fillna(CENSURA_HOT100)
    df["estreia"] = df["nova_passagem"].astype(int)
    df["reentrada"] = (df["nova_passagem"] & (df["passagem"] > 1)).astype(int)

    # Positivo = subiu em relacao a semana anterior.
    df["variacao_1s"] = df["rank_lag1"] - df["rank"]
    df["distancia_pico"] = df["rank"] - df["peak_pos"]

    # Alvo: posicao da mesma musica exatamente 7 dias depois.
    # Ausencia na semana seguinte (saiu da parada) vira 0.
    proxima = df[["date"] + CHAVE + ["rank"]].copy()
    proxima["date"] = proxima["date"] - SEMANA
    proxima = proxima.rename(columns={"rank": "rank_proxima"})
    df = df.merge(proxima, on=["date"] + CHAVE, how="left")
    df[ALVO] = (df["rank_proxima"] < df["rank"]).astype(int)

    # rank_proxima e informacao do futuro. Sai aqui para nunca virar atributo.
    return df.drop(columns=["rank_proxima", "nova_passagem", "passagem", "last_week"])


def juntar_secundaria(base: pd.DataFrame, sec: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Posicao na parada secundaria na mesma semana, com censura e indicador."""
    s = sec.drop_duplicates(subset=["date"] + CHAVE)[["date"] + CHAVE + ["rank"]]
    s = s.rename(columns={"rank": f"{nome}_rank"})
    base = base.merge(s, on=["date"] + CHAVE, how="left")
    base[f"{nome}_presente"] = base[f"{nome}_rank"].notna().astype(int)
    base[f"{nome}_rank"] = base[f"{nome}_rank"].fillna(CENSURA_SECUNDARIA)
    return base


def juntar_album(base: pd.DataFrame, album: pd.DataFrame) -> pd.DataFrame:
    """Melhor posicao de album do artista na mesma semana.

    Limitacao conhecida: casa pelo artista normalizado. Uma musica creditada a
    'Artista A & Artista B' nao casa com um album so do Artista A.
    """
    a = (
        album.groupby(["date", "key_artista"])["rank"].min()
        .rename("album_rank").reset_index()
    )
    base = base.merge(a, on=["date", "key_artista"], how="left")
    base["album_presente"] = base["album_rank"].notna().astype(int)
    base["album_rank"] = base["album_rank"].fillna(CENSURA_ALBUM)
    return base


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./dados", type=Path)
    p.add_argument("--saida", default="./dados/processados", type=Path)
    p.add_argument("--limiar-hiato", default=LIMIAR_HIATO_DIAS, type=int)
    args = p.parse_args()

    print("Lendo arquivos...")
    hot100 = carregar(args.data_dir, "hot100")
    secundarias = {n: carregar(args.data_dir, n) for n in SECUNDARIAS}
    album = carregar(args.data_dir, "billboard200")

    print(f"Construindo passagens (limiar {args.limiar_hiato} dias), lags e alvo...")
    base = preparar_hot100(hot100, args.limiar_hiato)

    # Recorte depois das features: o historico anterior a 2013 ja foi usado
    # para lags e passagens e agora pode sair.
    base = base[base["date"] >= INICIO]

    # A ultima semana do dataset nao tem semana seguinte observavel.
    base = base[base["date"] < base["date"].max()]

    # Embargo: o alvo da ultima semana de cada bloco foi calculado com a
    # posicao da primeira semana do bloco seguinte. Remover essas semanas
    # impede que informacao de D1 entre em D0 e, principalmente, que
    # informacao de D2 entre em D1.
    embargo = base["date"].isin([CORTE_D1 - SEMANA, CORTE_D2 - SEMANA])
    print(f"Embargo: {embargo.sum()} linhas removidas nas fronteiras dos blocos")
    base = base[~embargo]

    for nome, sec in secundarias.items():
        base = juntar_secundaria(base, sec, nome)
    base = juntar_album(base, album)

    base["mes"] = base["date"].dt.month
    base["semana_ano"] = base["date"].dt.isocalendar().week.astype(int)

    base = base[IDENTIFICACAO + FEATURES + [ALVO]].sort_values(["date", "rank"])

    nulos = base[FEATURES].isna().sum()
    if nulos.any():
        print("\nATENCAO: atributos com valores ausentes:")
        print(nulos[nulos > 0])

    blocos = {
        "d0": base[base["date"] < CORTE_D1],
        "d1": base[(base["date"] >= CORTE_D1) & (base["date"] < CORTE_D2)],
        "d2": base[base["date"] >= CORTE_D2],
    }

    args.saida.mkdir(parents=True, exist_ok=True)
    manifesto = [
        f"limiar_hiato_dias={args.limiar_hiato}",
        f"inicio={INICIO.date()} corte_d1={CORTE_D1.date()} corte_d2={CORTE_D2.date()}",
        f"censuras: hot100={CENSURA_HOT100} secundaria={CENSURA_SECUNDARIA} album={CENSURA_ALBUM}",
    ]
    print(f"\n{len(FEATURES)} atributos explicativos: {', '.join(FEATURES)}\n")

    for nome, bloco in blocos.items():
        caminho = args.saida / f"base_{nome}.csv"
        bloco.to_csv(caminho, index=False, lineterminator='\n')
        manifesto.append(f"{nome} {len(bloco)} linhas sha256={sha256(caminho)}")
        print(
            f"{nome.upper()}  {bloco['date'].min().date()} a {bloco['date'].max().date()}"
            f"  {len(bloco):>6} linhas"
        )

    (args.saida / "manifesto.txt").write_text("\n".join(manifesto) + "\n")

    # Balanceamento so de D0 e D1. Olhar a taxa de alvo de D2 ja e usar D2
    # para decidir alguma coisa.
    print("\nProporcao de alvo = 1 (subiu na semana seguinte):")
    for nome in ["d0", "d1"]:
        print(f"  {nome.upper()}: {blocos[nome][ALVO].mean():.3f}")

    print("\nPor ano (D0 e D1):")
    treino = pd.concat([blocos["d0"], blocos["d1"]])
    print(treino.groupby(treino["date"].dt.year)[ALVO].agg(["mean", "size"]).round(3))

    print(f"\nArquivos em {args.saida}/, hashes em manifesto.txt")


if __name__ == "__main__":
    main()
