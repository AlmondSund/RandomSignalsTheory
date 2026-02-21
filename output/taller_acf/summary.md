# Taller ACF - Resumen Computacional
Criterio teórico principal: la ACF es válida para WSS real solo si `A0 - 2*A1 >= 0`.
| Caso | A0 | A1 | sigma [s] | A0-2A1 | min S(f) cerrada | min S(f) FFT | negativos cerrada | min eig Toeplitz | ¿válida teóricamente? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Caso 1 Valido | 2.800 | 1.000 | 0.350 | 0.800 | 1.917911e-07 | -3.613753e-16 | 0 | -7.874479e-14 | SI |
| Caso 2 Umbral | 2.000 | 1.000 | 0.650 | 0.000 | 0.000000e+00 | -1.021647e-14 | 0 | -7.157493e-14 | SI |
| Caso 3 Invalido | 1.200 | 1.000 | 0.900 | -0.800 | -7.505049e-02 | -7.477940e-02 | 946 | -1.314595e-09 | NO |

## Figuras
- `output/taller_acf/caso_1_valido.png`
- `output/taller_acf/caso_2_umbral.png`
- `output/taller_acf/caso_3_invalido.png`
