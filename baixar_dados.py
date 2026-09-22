"""Baixa e confere a versao 175 do dataset Billboard no Kaggle.

O dataset e publico e nao exige credenciais. Nenhum arquivo existente e
sobrescrito: arquivos ja presentes precisam ter o hash esperado, caso
contrario a execucao para com erro.

Uso:
    python baixar_dados.py
    python baixar_dados.py --data-dir /caminho/para/dados
"""

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from zipfile import BadZipFile, ZipFile


DATASET = "ludmin/billboard"
VERSAO = 175
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    f"{DATASET}?datasetVersionNumber={VERSAO}"
)
ZIP_SHA256 = "833d4bb30936c20a6be52ca5512ccb87e3c2e38f69d48e17848b809d2f508744"

ARQUIVOS_V175 = {
    "billboard200.csv": "935e07ee3344865a04dca15a4fb60536f6bb16f89b3cf58284988dd7a3c2018c",
    "digital.csv": "6a2ce94ed2a8e80f5be357bfd8953006dbb604ae58e4d9e61c75976700099560",
    "hot100.csv": "6fc2df245ae37eac5281635e448d9faf0c8106e44acefb42bd1de1c31f9ea6a4",
    "radio.csv": "705aff2774eca122578aea60e21c1285287ff74af8201f540d9d88bf3b1741e5",
    "streaming.csv": "8edee61e7721c8c0e217d44d42427bb6e1015e26a9f61bae7c3cd4420f298be1",
}


def sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def conferir_existentes(data_dir: Path) -> set[str]:
    """Confere os arquivos presentes e devolve os nomes ainda ausentes."""
    ausentes = set()
    for nome, esperado in ARQUIVOS_V175.items():
        caminho = data_dir / nome
        if not caminho.exists():
            ausentes.add(nome)
            continue
        observado = sha256(caminho)
        if observado != esperado:
            raise ValueError(
                f"{caminho} ja existe, mas nao pertence a versao {VERSAO}. "
                "Mova ou remova esse arquivo conscientemente antes de tentar novamente."
            )
        print(f"OK existente: {nome}")
    return ausentes


def baixar_zip(destino: Path) -> None:
    requisicao = urllib.request.Request(
        URL,
        headers={"User-Agent": "ammci-t1-reprodutibilidade/1.0"},
    )
    print(f"Baixando {DATASET}, versao {VERSAO}...")
    with urllib.request.urlopen(requisicao, timeout=300) as resposta:
        with destino.open("wb") as arquivo:
            shutil.copyfileobj(resposta, arquivo, length=1 << 20)
    if sha256(destino) != ZIP_SHA256:
        raise ValueError("O ZIP baixado nao possui o hash esperado da versao 175.")


def extrair(arquivo_zip: Path, data_dir: Path, ausentes: set[str]) -> None:
    try:
        with ZipFile(arquivo_zip) as zipado:
            nomes = set(zipado.namelist())
            faltando_no_zip = ausentes - nomes
            if faltando_no_zip:
                raise ValueError(
                    f"Arquivos ausentes no ZIP: {sorted(faltando_no_zip)}"
                )
            if zipado.testzip() is not None:
                raise ValueError("O ZIP falhou na verificacao de integridade.")

            for nome in sorted(ausentes):
                destino = data_dir / nome
                temporario = data_dir / f".{nome}.part"
                with zipado.open(nome) as origem, temporario.open("wb") as saida:
                    shutil.copyfileobj(origem, saida, length=1 << 20)
                if sha256(temporario) != ARQUIVOS_V175[nome]:
                    temporario.unlink(missing_ok=True)
                    raise ValueError(f"Hash inesperado depois de extrair {nome}.")
                os.replace(temporario, destino)
                print(f"Extraido e conferido: {nome}")
    except BadZipFile as erro:
        raise ValueError("O download nao e um ZIP valido.") from erro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dados", type=Path)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    ausentes = conferir_existentes(args.data_dir)
    if not ausentes:
        print(f"Dataset versao {VERSAO} ja esta completo e integro.")
        return

    with tempfile.TemporaryDirectory(prefix="ammci-billboard-") as pasta:
        arquivo_zip = Path(pasta) / f"billboard-v{VERSAO}.zip"
        baixar_zip(arquivo_zip)
        extrair(arquivo_zip, args.data_dir, ausentes)

    restantes = conferir_existentes(args.data_dir)
    if restantes:
        raise RuntimeError(f"Download incompleto: {sorted(restantes)}")
    print(f"Dataset versao {VERSAO} pronto em {args.data_dir}/")


if __name__ == "__main__":
    main()
