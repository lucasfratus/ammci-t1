"""
Auditoria inicial do dataset Billboard (ludmin/billboard) para o trabalho de AMMCI.

Responde tres perguntas que travam o registro da proposta no Classroom:
  1. Qual o periodo coberto por cada parada e quantas linhas existem?
  2. Qual a cobertura do join entre hot100 e as paradas de radio/streaming/digital?
  3. Onde caem os cortes D0/D1/D2 (60/20/20) no recorte temporal escolhido?

Uso:
    python auditoria_billboard.py
    python auditoria_billboard.py --data-dir ./dados --inicio 2013-01-01
"""

import argparse
import unicodedata
from pathlib import Path

import pandas as pd

ARQUIVOS = ["hot100", "radio", "streaming", "digital", "billboard200"]
SECUNDARIAS = ["radio", "streaming", "digital"]

# A descricao do Kaggle nao bate com o schema real dos arquivos. Em vez de fixar
# nomes, resolvemos cada papel semantico contra uma lista de apelidos possiveis.
# Ordem importa: o primeiro apelido encontrado no cabecalho vence.
ALIASES = {
    "date": ["date", "chart_date", "chart_week", "week", "weekid", "week_id",
             "issue_date", "chartdate"],
    "title": ["title", "song", "song_name", "name", "track", "track_name"],
    "artist": ["artist", "artists", "performer", "artist_name", "primary_artist"],
    "rank": ["rank", "position", "current_week", "this_week", "chart_position",
             "current_position", "pos"],
    "last_week": ["last_week", "last_week_position", "previous_week", "lw",
                  "last_position", "prev_pos", "previous_position"],
    "peak_pos": ["peak_pos", "peak_position", "peak", "peak_rank", "best_position"],
    "weeks_on_chart": ["weeks_on_chart", "weeks_on_board", "weeks", "wks",
                       "weeks_on", "total_weeks"],
}

OBRIGATORIAS = ["date", "title", "artist", "rank"]


def resolver_colunas(colunas: list[str], nome_arquivo: str) -> dict[str, str]:
    """Mapeia papel semantico -> nome real da coluna no arquivo."""
    lookup = {c.strip().lower().replace(" ", "_").replace("-", "_"): c for c in colunas}
    mapa = {}
    for papel, apelidos in ALIASES.items():
        for apelido in apelidos:
            if apelido in lookup:
                mapa[papel] = lookup[apelido]
                break

    faltando = [p for p in OBRIGATORIAS if p not in mapa]
    if faltando:
        raise ValueError(
            f"\n{nome_arquivo}.csv: nao consegui identificar {faltando}."
            f"\nColunas reais do arquivo: {list(colunas)}"
            f"\nAdicione o nome correto na lista ALIASES no topo do script."
        )
    return mapa


def normalizar(serie: pd.Series) -> pd.Series:
    """Padroniza titulo/artista para o join.

    Sem isso o match falha por diferencas bobas: 'Beyonce' vs 'Beyoncé',
    'Feat.' vs 'Featuring', maiusculas, espaco duplo, pontuacao.
    """
    s = serie.fillna("").astype(str).str.lower()
    s = s.apply(
        lambda x: unicodedata.normalize("NFKD", x)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    s = s.str.replace(r"\bfeaturing\b|\bfeat\.?\b|\bft\.?\b", "&", regex=True)
    s = s.str.replace(r"[^a-z0-9&]+", " ", regex=True)
    return s.str.strip()


def carregar(data_dir: Path, nome: str) -> pd.DataFrame:
    caminho = data_dir / f"{nome}.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} nao encontrado. Baixe o dataset com:\n"
            f"  kaggle datasets download ludmin/billboard -p {data_dir} --unzip"
        )

    # Le so o cabecalho para descobrir o schema antes de carregar 37 MB.
    cabecalho = pd.read_csv(caminho, nrows=0)
    mapa = resolver_colunas(list(cabecalho.columns), nome)

    df = pd.read_csv(caminho)
    df = df.rename(columns={real: papel for papel, real in mapa.items()})

    # errors="coerce" transforma data invalida em NaT em vez de estourar excecao.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalidas = df["date"].isna().sum()
    if invalidas:
        print(f"  aviso: {nome}.csv tem {invalidas} datas invalidas, descartadas")
        df = df.dropna(subset=["date"])

    for papel in ["rank", "last_week", "peak_pos", "weeks_on_chart"]:
        if papel in df.columns:
            df[papel] = pd.to_numeric(df[papel], errors="coerce")
        else:
            df[papel] = pd.NA

    df["key_titulo"] = normalizar(df["title"])
    df["key_artista"] = normalizar(df["artist"])
    df.attrs["schema_real"] = mapa
    return df


def bloco_1_inventario(dados: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 72)
    print("0. SCHEMA REAL ENCONTRADO NOS ARQUIVOS")
    print("=" * 72)
    for nome, df in dados.items():
        mapa = df.attrs.get("schema_real", {})
        ausentes = [p for p in ALIASES if p not in mapa]
        print(f"{nome:<14} {mapa}")
        if ausentes:
            print(f"{'':<14} papeis nao encontrados: {ausentes}")

    print("\n" + "=" * 72)
    print("1. INVENTARIO POR PARADA")
    print("=" * 72)

    linhas = []
    for nome, df in dados.items():
        por_semana = df.groupby("date").size()
        linhas.append(
            {
                "arquivo": nome,
                "inicio": df["date"].min().date(),
                "fim": df["date"].max().date(),
                "linhas": len(df),
                "semanas": df["date"].nunique(),
                "posicoes_por_semana": int(por_semana.median()),
                "nulos_last_week": df["last_week"].isna().sum(),
                "duplicatas": df.duplicated(
                    subset=["date", "key_titulo", "key_artista"]
                ).sum(),
            }
        )
    print(pd.DataFrame(linhas).to_string(index=False))
    print(
        "\nLeia 'posicoes_por_semana': se streaming lista 50 e o hot100 lista 100,"
        "\nmetade das linhas do hot100 nunca tera par no streaming. Isso e teto"
        "\nestrutural de cobertura, nao sujeira de dados."
    )


def bloco_2_cobertura(dados: dict[str, pd.DataFrame], inicio: str) -> None:
    print("\n" + "=" * 72)
    print(f"2. COBERTURA DO JOIN COM hot100 (a partir de {inicio})")
    print("=" * 72)

    base = dados["hot100"]
    base = base[base["date"] >= inicio]
    chaves = ["date", "key_titulo", "key_artista"]

    linhas = []
    for nome in SECUNDARIAS:
        outro = dados[nome][chaves].drop_duplicates()
        outro = outro[outro["date"] >= inicio]
        casado = base.merge(outro, on=chaves, how="inner")
        linhas.append(
            {
                "parada": nome,
                "linhas_hot100": len(base),
                "com_par": len(casado),
                "cobertura_%": round(100 * len(casado) / max(len(base), 1), 1),
            }
        )
    print(pd.DataFrame(linhas).to_string(index=False))

    # Cobertura simultanea: linha do hot100 presente nas tres paradas ao mesmo tempo.
    juncao = base[chaves].copy()
    for nome in SECUNDARIAS:
        outro = dados[nome][chaves].drop_duplicates()
        outro[f"tem_{nome}"] = True
        juncao = juncao.merge(outro, on=chaves, how="left")
    tem_todas = juncao[[f"tem_{n}" for n in SECUNDARIAS]].fillna(False).all(axis=1)
    print(
        f"\nLinhas do hot100 com par nas TRES paradas: {tem_todas.sum()} "
        f"({round(100 * tem_todas.mean(), 1)}%)"
    )
    print(
        "\nInterpretacao:"
        "\n  cada parada secundaria deve ser usada separadamente, sempre acompanhada"
        "\n  por seu indicador de presenca. A cobertura simultanea das tres nao e um"
        "\n  requisito e nao se deve filtrar apenas os casos completos. Estar fora do"
        "\n  top 50 e informacao potencialmente preditiva, nao um ausente aleatorio."
    )


def bloco_3_cortes(dados: dict[str, pd.DataFrame], inicio: str) -> None:
    print("\n" + "=" * 72)
    print(f"3. CORTES TEMPORAIS D0/D1/D2 (60/20/20, a partir de {inicio})")
    print("=" * 72)

    base = dados["hot100"]
    base = base[base["date"] >= inicio].sort_values("date")
    semanas = base["date"].drop_duplicates().sort_values().reset_index(drop=True)

    corte_0 = semanas.iloc[int(len(semanas) * 0.60)]
    corte_1 = semanas.iloc[int(len(semanas) * 0.80)]

    d0 = base[base["date"] < corte_0]
    d1 = base[(base["date"] >= corte_0) & (base["date"] < corte_1)]
    d2 = base[base["date"] >= corte_1]

    for nome, parte in [("D0 historico", d0), ("D1 recente", d1), ("D2 futuro", d2)]:
        print(
            f"{nome:<14} {parte['date'].min().date()} a {parte['date'].max().date()}  "
            f"{len(parte):>7} linhas  {parte['date'].nunique():>4} semanas"
        )

    print(
        f"\nCorte D0/D1: {corte_0.date()}"
        f"\nCorte D1/D2: {corte_1.date()}"
        "\n\nAnote essas duas datas. Elas viram constante no codigo e entram na secao"
        "\nde Materiais e Metodos do short paper. D2 fica congelado ate a avaliacao final."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".", type=Path)
    p.add_argument(
        "--inicio",
        default="2013-01-01",
        help="Recorte temporal inicial. Ajuste depois de ver o bloco 1.",
    )
    args = p.parse_args()

    dados = {nome: carregar(args.data_dir, nome) for nome in ARQUIVOS}

    bloco_1_inventario(dados)
    bloco_2_cobertura(dados, args.inicio)
    bloco_3_cortes(dados, args.inicio)

    print("\n" + "=" * 72)
    print("Proximo passo: com esses numeros em maos, registre a proposta no Classroom.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
