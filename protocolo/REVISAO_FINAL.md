# Revisão, congelamento e correção de reprodutibilidade

A avaliação original foi executada em 26/09/2026 com protocolo versão 2. O
registro original foi produzido em uma sessão de assistência por IA e não
substitui a revisão humana da equipe.

Antes da avaliação, foram conferidos os hashes do código e das configurações,
as versões das dependências, os esquemas e períodos de D0/D1 e o embargo de
sete dias. A suíte disponível naquela execução tinha 31 testes, incluindo
integração sintética.

Foram mantidas as decisões anteriores: 20 atributos, MLP com 32 neurônios,
alpha 0,01, taxa 0,003, teto de 800 épocas; fine-tuning por uma época em D1,
preservando pesos e scaler de M0 e reiniciando Adam. MRT usa D0+D1 e MREC usa
D1, com novos pesos e scalers próprios. As sementes são 42, 1337 e 2024,
o limiar é 0,5 e MCC é a métrica principal. GB e regressão logística são
comparadores históricos treinados em D0. Nenhuma configuração foi escolhida
com base em D2.

O protocolo versão 2 declarava SHA-256
`ad6e1d1fd71aa741a9454fdad1b2f48dc7a1111e4759630e7f02033a35d0b3a6`.

Após o congelamento, os CSVs foram materializados pelo construtor existente.
Os três hashes coincidiram com os previamente especificados. A construção
dos atributos precede o treinamento; a leitura de D2 pelo avaliador ocorre
somente depois de todos os treinamentos e da gravação dos modelos.

As análises posteriores são descritivas e não autorizam ajustar parâmetros
ou selecionar atributos usando D2. Eventos, modelos, previsões e métricas da
execução ficam em `resultados/final/`, com manifesto de integridade.

## Correção R2 — hashes portáveis

Após obter o projeto pelo Git em Linux, verificou-se que a versão 2 dependia
dos bytes de terminação de linha usados no Windows. O Git normalizou CRLF para
LF, invalidando o protocolo e os manifestos apesar de o conteúdo lógico ser o
mesmo. Cinco fontes também haviam sido registradas com terminações misturadas.

Em 26/09/2026, a equipe solicitou a correção de reprodutibilidade. A versão 3:

- normaliza CRLF e CR para LF antes de calcular hashes de arquivos textuais;
- continua usando bytes exatos para binários;
- grava JSON, CSV e Markdown com LF determinístico;
- inclui `.gitattributes` e testes de portabilidade;
- recusa a migração se qualquer decisão experimental for diferente da versão 2.

Não foram alterados atributos, dados, modelos, hiperparâmetros, seeds, limiar,
fine-tuning ou critérios de análise. O protocolo registra os hashes da versão
anterior e declara explicitamente que D2 não foi usado para novas decisões.

Protocolo versão 3: `protocolo_final.congelado.json`.
SHA-256 canônico: `069305da99cfcc7824dfba1f3402f2c6f61275144d842f7f5e31ce60697526fe`.

Depois da migração, os 16 modelos e todas as análises foram reproduzidos em
diretórios novos. Previsões, métricas, matrizes, curvas, drift, importâncias e
erros coincidiram exatamente com os artefatos anteriores; somente os tempos de
execução variaram. A suíte ampliada passou com 34 testes.
