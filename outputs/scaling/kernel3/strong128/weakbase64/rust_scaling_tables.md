# Tabele merenja (mean / std / Tukey outlier-i)

*Ubrzanje i efikasnost od srednjih vremena (`t_seq_sec` / `t_par_sec`).*

## Jako skaliranje

| workers | n_runs | t_serial_sec | t_serial_std_sec | t_serial_outliers | t_seq_sec | t_seq_std_sec | t_seq_outliers | t_par_sec | t_par_std_sec | t_par_outliers | speedup | efficiency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 30 | 0.000004 | 0.000006 | 3 | 0.012560 | 0.001702 | 3 | 0.010613 | 0.000386 | 0 | 1.183485 | 1.183485 |
| 2 | 30 | 0.000004 | 0.000006 | 3 | 0.012560 | 0.001702 | 3 | 0.005319 | 0.000187 | 2 | 2.361587 | 1.180793 |
| 4 | 30 | 0.000004 | 0.000006 | 3 | 0.012560 | 0.001702 | 3 | 0.003825 | 0.001058 | 0 | 3.284122 | 0.821031 |

## Slabo skaliranje (paralelno vreme)

| workers | H | W | n_runs | t_par_sec | t_par_std_sec | t_par_outliers |
|---|---|---|---|---|---|---|
| 1 | 64 | 64 | 30 | 0.002967 | 0.000263 | 1 |
| 2 | 88 | 88 | 30 | 0.003055 | 0.000478 | 3 |
| 4 | 128 | 128 | 30 | 0.005385 | 0.001129 | 0 |
