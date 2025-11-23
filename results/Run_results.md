Status=OPTIMAL, obj_value=26.0

Saved detailed results to: results/nl4opt_gpt5_lp_gurobi_full.csv

=== Summary (LP + Gurobi, numeric gt>0 & OPTIMAL) ===
n_total          = 231
n_numeric_eval   = 223
n_ok             = 181
acc              = 0.8116591928251121
mean_abs_err     = 212.3757262169818
mean_rel_err     = 0.07274102972411392
mean_latency_sec = 3.5494091190030224
mean_calls       = 1.0089686098654709


------------
**selfcheck-goroubi

Saved detailed results to: results/nl4opt_gpt5_lp_gurobi_selfcheck_full.csv

=== Summary (LP + Gurobi + Self-Check, numeric gt>0 & OPTIMAL) ===
n_total          = 231
n_numeric_eval   = 220
n_ok             = 170
acc              = 0.7727272727272727
mean_abs_err     = 2003.0667432828402
mean_rel_err     = 0.25651114565662203
mean_latency_sec = 6.01289162310687
mean_calls       = 2.0
--------
=== Summary (LP + Gurobi + Self-Check PATCH, numeric gt>0 & OPTIMAL) ===
n_total          = 231
n_numeric_eval   = 216
n_ok             = 155
acc              = 0.7175925925925926
mean_abs_err     = 12711.939671285507
mean_rel_err     = 0.2950805295188549
mean_latency_sec = 7.0051955106081785
mean_calls       = 2.0