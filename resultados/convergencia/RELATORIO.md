# E06-R1 — Diagnostico de convergencia

Mesma configuracao escolhida em D0, com embargo, quatro folds e tres seeds.

- Teto 100: 7/12 ajustes convergiram; MCC medio 0.3143; MCC treino 0.4259; maximo observado 100 epocas.
- Teto 800: 12/12 ajustes convergiram; MCC medio 0.3134; MCC treino 0.4281; maximo observado 171 epocas.

A diferenca treino/validacao deve ser interpretada como sinal potencial de sobreajuste. Aumentar o teto permite estabilizar a perda; nao implica melhorar MCC. Nenhuma decisao usa D1 ou D2.
