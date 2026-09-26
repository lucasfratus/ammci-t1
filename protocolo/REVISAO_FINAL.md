# Revisão e congelamento para avaliação real

Em 26/09/2026, o usuário autorizou explicitamente nesta conversa a revisão,
o congelamento e a execução da avaliação real. A revisão técnica foi realizada
pelo Codex; o registro não representa uma revisão humana adicional da equipe.

Antes da avaliação, foram conferidos os hashes do código e das configurações,
as versões das dependências, os esquemas e períodos de D0/D1 e o embargo de
sete dias. A suíte de 31 testes já havia passado, incluindo integração sintética.

Foram mantidas as decisões anteriores: 20 atributos, MLP com 32 neurônios,
alpha 0,01, taxa 0,003, teto de 800 épocas; fine-tuning por uma época em D1,
preservando pesos e scaler de M0 e reiniciando Adam. MRT usa D0+D1 e MREC usa
D1, com novos pesos e scalers próprios. As sementes são 42, 1337 e 2024,
o limiar é 0,5 e MCC é a métrica principal. GB e regressão logística são
comparadores históricos treinados em D0. Nenhuma configuração foi escolhida
com base em D2.

Protocolo congelado: `protocolo_final.congelado.json`.
SHA-256: `ad6e1d1fd71aa741a9454fdad1b2f48dc7a1111e4759630e7f02033a35d0b3a6`.

Após o congelamento, os CSVs foram materializados pelo construtor existente.
Os três hashes coincidiram com os previamente especificados. A construção
dos atributos precede o treinamento; a leitura de D2 pelo avaliador ocorre
somente depois de todos os treinamentos e da gravação dos modelos.

As análises posteriores são descritivas e não autorizam ajustar parâmetros
ou selecionar atributos usando D2. Eventos, modelos, previsões e métricas da
execução ficam em `resultados/final/`, com manifesto de integridade.
