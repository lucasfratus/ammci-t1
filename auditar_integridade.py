"""Auditoria de integridade anterior aos experimentos finais.

Esta auditoria usa somente o intervalo de modelagem anterior a D2
([2013-01-01, 2023-12-20)) para procurar vazamento em ``peak_pos`` e
colisoes introduzidas pela normalizacao de titulo/artista. O historico anterior
a 2013 e lido apenas para reconstruir o estado acumulado das musicas.

Saidas:
    resultados/auditoria_integridade/resumo.json
    resultados/auditoria_integridade/duplicatas_normalizadas.csv
    resultados/auditoria_integridade/variantes_titulo_normalizadas.csv
    resultados/auditoria_integridade/variantes_artista_normalizadas.csv

Uso:
    python auditar_integridade.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from auditoria_billboard import ARQUIVOS, carregar
from baixar_dados import ARQUIVOS_V175, sha256


INICIO_MODELAGEM = pd.Timestamp("2013-01-01")
CORTE_D2 = pd.Timestamp("2023-12-20")
LIMIAR_HIATO_DIAS = 28

def conferir_fontes(data_dir: Path) -> dict[str, dict]:
    resultado = {}
    for arquivo, esperado in ARQUIVOS_V175.items():
        caminho = data_dir / arquivo
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo fonte ausente: {caminho}")
        observado = sha256(caminho)
        resultado[arquivo] = {
            "bytes": caminho.stat().st_size,
            "sha256": observado,
            "confere_com_v175": observado == esperado,
        }
        if observado != esperado:
            raise ValueError(
                f"Hash inesperado para {arquivo}. "
                "A auditoria exige exatamente a versao 175 do Kaggle."
            )
    return resultado


def auditar_peak_pos(hot100: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Verifica se peak_pos evolui apenas com informacao disponivel em t.

    Dentro de uma passagem, o pico da linha atual deve ser o minimo entre o
    pico da linha anterior, o rank atual e last_week. ``last_week`` e incluido
    porque o dataset possui algumas lacunas de calendario, mas essa posicao ja
    e conhecida na data da linha. A primeira linha de uma passagem e aceita
    como estado historico informado pela propria Billboard.
    """
    limite = hot100[hot100["date"] < CORTE_D2].copy()
    limite = limite.dropna(subset=["rank", "peak_pos"])
    limite = limite.sort_values(
        ["key_titulo", "key_artista", "date", "rank"]
    )
    limite = limite.drop_duplicates(
        ["date", "key_titulo", "key_artista"], keep="first"
    )

    chaves = ["key_titulo", "key_artista"]
    gap = limite.groupby(chaves, sort=False)["date"].diff().dt.days
    limite["passagem"] = (
        gap.isna() | (gap > LIMIAR_HIATO_DIAS)
    ).groupby([limite["key_titulo"], limite["key_artista"]], sort=False).cumsum()

    grupo = limite.groupby(chaves + ["passagem"], sort=False)
    limite["ordem_na_passagem"] = grupo.cumcount()
    limite["peak_anterior"] = grupo["peak_pos"].shift()
    limite["melhor_posicao_conhecida"] = limite[
        ["peak_anterior", "rank", "last_week"]
    ].min(axis=1, skipna=True)

    continuacao = limite["ordem_na_passagem"] > 0
    limite["antecipa_pico_nao_observado"] = (
        continuacao
        & (limite["peak_pos"] < limite["melhor_posicao_conhecida"])
    )
    limite["esquece_pico_anterior"] = (
        continuacao
        & (limite["peak_pos"] > limite["melhor_posicao_conhecida"])
    )

    recorte = limite[limite["date"] >= INICIO_MODELAGEM].copy()
    problemas = recorte[
        recorte["antecipa_pico_nao_observado"]
        | recorte["esquece_pico_anterior"]
    ].copy()

    primeiras = recorte["ordem_na_passagem"] == 0
    resumo = {
        "linhas_auditadas": int(len(recorte)),
        "continuacoes_de_passagem": int((~primeiras).sum()),
        "novas_passagens": int(primeiras.sum()),
        "antecipa_pico_nao_observado": int(
            recorte["antecipa_pico_nao_observado"].sum()
        ),
        "esquece_pico_anterior": int(recorte["esquece_pico_anterior"].sum()),
        "conclusao": (
            "peak_pos e causal no recorte auditado"
            if problemas.empty
            else "peak_pos exige investigacao antes da modelagem"
        ),
    }
    return resumo, problemas


def variantes_normalizadas(
    dados: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    duplicatas = []
    variantes_titulo = []
    artistas = []
    por_arquivo = {}

    for nome, bruto in dados.items():
        df = bruto[
            (bruto["date"] >= INICIO_MODELAGEM) & (bruto["date"] < CORTE_D2)
        ].copy()
        chave_semana = ["date", "key_titulo", "key_artista"]
        mascara_dup = df.duplicated(chave_semana, keep=False)
        dup = df.loc[
            mascara_dup,
            ["date", "title", "artist", "rank", "key_titulo", "key_artista"],
        ].copy()
        dup.insert(0, "arquivo", nome)
        duplicatas.append(dup)

        identidades = df[
            ["key_titulo", "key_artista", "title", "artist"]
        ].drop_duplicates()
        contagem = identidades.groupby(["key_titulo", "key_artista"]).size()
        chaves_colisao = contagem[contagem > 1].rename("variantes").reset_index()
        variacao = identidades.merge(
            chaves_colisao, on=["key_titulo", "key_artista"], how="inner"
        )
        variacao.insert(0, "arquivo", nome)
        variantes_titulo.append(variacao)

        artistas_arquivo = df[["key_artista", "artist"]].drop_duplicates().copy()
        artistas_arquivo.insert(0, "arquivo", nome)
        artistas.append(artistas_arquivo)

        por_arquivo[nome] = {
            "linhas_no_recorte": int(len(df)),
            "linhas_em_chaves_semanais_duplicadas": int(mascara_dup.sum()),
            "chaves_titulo_artista_com_multiplas_grafias": int(
                len(chaves_colisao)
            ),
        }

    tab_duplicatas = pd.concat(duplicatas, ignore_index=True)
    tab_titulos = pd.concat(variantes_titulo, ignore_index=True)
    tab_artistas = pd.concat(artistas, ignore_index=True).drop_duplicates(
        ["key_artista", "artist"]
    )
    contagem_artista = tab_artistas.groupby("key_artista")["artist"].nunique()
    chaves_artista = (
        contagem_artista[contagem_artista > 1]
        .rename("variantes")
        .reset_index()
    )
    tab_artistas = tab_artistas.merge(chaves_artista, on="key_artista", how="inner")

    resumo = {
        "por_arquivo": por_arquivo,
        "chaves_de_artista_com_multiplas_grafias": int(len(chaves_artista)),
        "observacao": (
            "As duplicatas do billboard200 sao os albuns '+', '-' e '=' de "
            "Ed Sheeran. Elas nao alteram album_rank, pois a feature usa "
            "intencionalmente o melhor rank do artista na semana."
        ),
    }
    return tab_duplicatas, tab_titulos, tab_artistas, resumo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dados", type=Path)
    parser.add_argument(
        "--saida", default="resultados/auditoria_integridade", type=Path
    )
    args = parser.parse_args()

    print("Conferindo hashes da versao 175...")
    fontes = conferir_fontes(args.data_dir)
    print("  cinco arquivos conferem com a versao congelada")

    print("Carregando fontes e auditando apenas dados anteriores a D2...")
    dados = {nome: carregar(args.data_dir, nome) for nome in ARQUIVOS}
    peak, problemas_peak = auditar_peak_pos(dados["hot100"])
    duplicatas, variantes_titulo, variantes_artista, normalizacao = (
        variantes_normalizadas(dados)
    )

    args.saida.mkdir(parents=True, exist_ok=True)
    colunas_peak = [
        "date", "title", "artist", "rank", "last_week", "peak_pos",
        "peak_anterior", "melhor_posicao_conhecida",
        "antecipa_pico_nao_observado", "esquece_pico_anterior",
    ]
    problemas_peak.reindex(columns=colunas_peak).to_csv(
        args.saida / "problemas_peak_pos.csv", index=False
    )
    duplicatas.to_csv(args.saida / "duplicatas_normalizadas.csv", index=False)
    variantes_titulo.to_csv(
        args.saida / "variantes_titulo_normalizadas.csv", index=False
    )
    variantes_artista.to_csv(
        args.saida / "variantes_artista_normalizadas.csv", index=False
    )

    resumo = {
        "dataset": {
            "kaggle": "ludmin/billboard",
            "versao": 175,
            "inicio_auditoria": str(INICIO_MODELAGEM.date()),
            "fim_exclusivo_auditoria": str(CORTE_D2.date()),
            "d2_inspecionado": False,
        },
        "fontes": fontes,
        "peak_pos": peak,
        "normalizacao": normalizacao,
    }
    (args.saida / "resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(resumo["peak_pos"], ensure_ascii=False, indent=2))
    print("\nColisoes por arquivo:")
    for nome, item in normalizacao["por_arquivo"].items():
        print(
            f"  {nome:<12} "
            f"duplicadas={item['linhas_em_chaves_semanais_duplicadas']:<3} "
            "chaves com variantes="
            f"{item['chaves_titulo_artista_com_multiplas_grafias']}"
        )
    print(f"\nResultados gravados em {args.saida}/")


if __name__ == "__main__":
    main()
