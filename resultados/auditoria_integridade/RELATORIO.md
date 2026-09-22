# Auditoria de integridade da base

## Escopo

A auditoria foi executada sobre a versão 175 do dataset
`ludmin/billboard`, publicada no Kaggle em 16/09/2026. Os hashes dos cinco
arquivos brutos estão registrados em `dados/MANIFESTO_FONTE.txt`.

As verificações de conteúdo ficaram restritas ao período de 01/01/2013 até
19/12/2023, isto é, antes do início de D2. O histórico anterior a 2013 foi
usado somente para reconstruir passagens e estados acumulados das músicas.
Nenhuma distribuição, alvo ou métrica de D2 foi examinada.

## Validação de `peak_pos`

Segundo a documentação do dataset, `peak_pos` representa a melhor posição
alcançada **até a data da linha**. Para verificar essa propriedade, cada música
foi dividida em passagens usando o mesmo limiar de 28 dias da construção da
base. Em toda continuação de passagem foi testada a recorrência:

```text
peak_pos(t) = min(peak_pos(t-1), rank(t), last_week(t))
```

`last_week` foi incluído porque há lacunas pontuais no calendário bruto, mas
essa posição já é conhecida na data `t`.

Resultados:

- 57.200 linhas auditadas no recorte D0+D1 bruto;
- 50.855 continuações de passagem verificadas;
- zero casos em que `peak_pos` antecipou uma posição ainda não observada;
- zero casos em que `peak_pos` esqueceu um pico anterior dentro da passagem.

A primeira linha de cada passagem pode carregar um pico histórico de uma
aparição anterior. Isso não constitui vazamento: o valor já era conhecido na
data da linha. Portanto, `peak_pos` e `distancia_pico` foram mantidos como
atributos causais.

## Normalização de títulos e artistas

Não foram encontradas chaves semanais duplicadas após a normalização em
`hot100`, `radio`, `streaming` ou `digital` no período auditado.

No `billboard200`, 30 linhas pertencem a chaves duplicadas. Elas correspondem
aos álbuns `+`, `-` e `=` de Ed Sheeran, cujos títulos viram uma string vazia
quando a pontuação é removida. Isso não altera o atributo `album_rank`: a
construção da base ignora o título do álbum e usa intencionalmente a melhor
posição de qualquer álbum do artista em cada semana.

Foram encontradas 91 chaves normalizadas de artista com duas grafias brutas.
As variantes inspecionadas são diferenças esperadas, como `Feat.` versus
`Featuring`, pontuação, capitalização e acentos. A lista completa foi mantida
em `variantes_artista_normalizadas.csv` para rastreabilidade.

## Decisão

A versão 175 foi congelada e reproduz exatamente os hashes das bases
processadas registradas anteriormente. Não foi encontrada evidência de data
leakage em `peak_pos`, nem colisão de normalização que altere as linhas do
Hot 100 ou das três paradas secundárias no intervalo D0+D1.

Limitação: a auditoria verifica a coerência interna do snapshot disponibilizado
pelo Kaggle, mas não permite provar que a fonte nunca realizou correções
retroativas antes da publicação da versão 175.
