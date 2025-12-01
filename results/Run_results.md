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
-----
Saved detailed results to: results/nl4opt_gpt5_lp_gurobi_semantic_verifier_full.csv

=== Summary (LP + Gurobi + Semantic + Verifier, numeric gt>0 & OPTIMAL) ===
n_total          = 231
n_numeric_eval   = 222
n_ok             = 180
acc              = 0.8108108108108109
mean_abs_err     = 191.35204330204328
mean_rel_err     = 0.10430116180653355
mean_latency_sec = 7.034229675928752
mean_calls       = 2.018018018018018

-----------
python src/experiment/analyze_livemathbench_cot_vs_sc.py
Loading: results/livemathbench_all_gpt5_cot.csv
  Columns: ['idx', 'split', 'question', 'gold_answer', 'raw_response', 'pred_str', 'correct', 'numeric_match', 'string_match', 'pred_norm', 'gold_norm', 'latency_sec', 'error']
  Shape:   (140, 13)
Loading: results/livemathbench_all_gpt5_cot_selfconsistency.csv
  Columns: ['idx', 'split', 'question', 'gold_answer', 'n_samples', 'sample_raw_texts_json', 'sample_answers_raw_json', 'sample_answers_norm_json', 'sample_correct_flags_json', 'sample_numeric_match_json', 'sample_string_match_json', 'sample_latencies_json', 'sample_errors_json', 'any_sample_correct', 'sc_winning_norm', 'sc_vote_count', 'sc_tie', 'sc_correct', 'sc_numeric_match', 'sc_string_match', 'mean_latency_sec']
  Shape:   (140, 21)
Merged rows: 140
[WARN] 140 rows have different split_cot vs split_sc

=== (A) Global Summary ===
Total rows: 140
  CoT correct: 65  (acc = 0.464)
  SC  correct: 78  (acc = 0.557)

  Fixed      (CoT wrong -> SC correct): 32
  Regressed  (CoT correct -> SC wrong): 19
  Both wrong (CoT wrong & SC wrong):    43

Global summary table:
                             metric    value  n_correct  count
                       CoT accuracy 0.464286       65.0    NaN
                    CoT+SC accuracy 0.557143       78.0    NaN
    Fixed (CoT wrong -> SC correct) 0.228571        NaN   32.0
Regressed (CoT correct -> SC wrong) 0.135714        NaN   19.0

=== Failure Group Counts ===
fixed:      32
regressed:  19
both_wrong: 43


======== Fixed by Self-Consistency (n=32) ========

--------------------------------------------------
idx: 1
split: CNMO
Question:
  Let an infinite geometric sequence $\{a_n\}$ have a common ratio $q$ that satisfies $0 < |q| < 1$. If the sum of all terms in $\{a_n\}$ equals the sum of the squares of all terms, then what is the range of values for $a_2$?
Gold answer: $[-\frac{1}{4}, 0) \cup (0, 2)$

CoT prediction (normalized): [-1/4,2)
CoT correct: False
SC winning answer norm: 246
SC correct: True

Raw CoT response (first 200 chars):
  Let the geometric sequence be \(\{a_n\}\) with first term \(a_1\) and common ratio \(q\), where \(0<|q|<1\).  Then: \[ a_n = a_1 q^{n-1}. \]  **1. Sum of all terms**  The sum of the infinite geometric

--------------------------------------------------
idx: 3
split: CNMO
Question:
  In a triangular pyramid $P-ABC$, if $PA \perp$ base $ABC$, and the lengths of edges $AB$, $BP$, $BC$, $CP$ are 1, 2, 3, 4 respectively, what is the volume of this triangular pyramid?
Gold answer: $\frac{3}{4}$

CoT prediction (normalized): 9/4
CoT correct: False
SC winning answer norm: 23
SC correct: True

Raw CoT response (first 200 chars):
  Let the base triangle be \( \triangle ABC \) and apex \(P\) such that \(PA \perp \text{plane }ABC\).  Given: \[ AB = 1,\quad BC = 3,\quad BP = 2,\quad CP = 4. \]  We want the volume: \[ V = \frac{1}{3

--------------------------------------------------
idx: 4
split: CNMO
Question:
  For a non-uniform die, the probabilities of rolling $1, 2, 3, 4, 5, 6$ points form an arithmetic sequence. Rolling the die twice independently, let the points be $a, b$ respectively. If the probability of the event $a+b=7$ occurring is $\frac{1}{7}$, then what is the probability of the event $a=b$ occurring?
Gold answer: $\frac{4}{21}$

CoT prediction (normalized): 4/21
CoT correct: False
SC winning answer norm: 7
SC correct: True

Raw CoT response (first 200 chars):
  Let the probabilities of rolling \(1,2,3,4,5,6\) be \[ p_1, p_2, p_3, p_4, p_5, p_6 \] in an arithmetic sequence. Then there exist real numbers \(a\) and \(d\) such that \[ p_k = a + (k-1)d,\quad k=1,


======== Regressed due to Self-Consistency (n=19) ========

--------------------------------------------------
idx: 12
split: CNMO
Question:
  What is the minimum value of $y=\frac{4^x+1}{2^x+1}$?
Gold answer: $2\sqrt{2}-2$

CoT prediction (normalized): 2\sqrt{2}-2
CoT correct: True
SC winning answer norm: (0,\frac{1}{2})
SC correct: False

Raw CoT response (first 200 chars):
  We want the minimum of \[ y = \frac{4^x + 1}{2^x + 1}. \]  Let \(t = 2^x\). Since \(2^x > 0\) for all real \(x\), we have \(t > 0\). Then \[ 4^x = (2^x)^2 = t^2, \] so \[ y = \frac{t^2 + 1}{t + 1}, \q

--------------------------------------------------
idx: 18
split: AMC
Question:
  What is the value of $9901 \times 101 - 99 \times 10101$?
Gold answer: $2$

CoT prediction (normalized): 2
CoT correct: True
SC winning answer norm: \frac{1}{2}+\frac{1}{2}\ln2
SC correct: False

Raw CoT response (first 200 chars):
  Compute each product:  1) \(9901 \times 101\)  \[ 9901 \times 101 = 9901 \times (100 + 1) = 9901 \times 100 + 9901 = 990100 + 9901 = 1000001 \]  2) \(99 \times 10101\)  \[ 10101 \times 99 = 10101 \tim

--------------------------------------------------
idx: 19
split: AMC
Question:
  A model used to estimate the time it will take to hike to the top of the mountain on a trail is of the form $T = aL + bG$, where $a$ and $b$ are constants, $T$ is the time in minutes, $L$ is the length of the trail in miles, and $G$ is the altitude gain in feet. The model estimates that it will take $69$ minutes to hike to the top if a trail is $1.5$ miles long and ascends $800$ feet, as well as if a trail is $1.2$ miles long and ascends $1100$ feet. How many minutes does the model estimates it will take to hike to the top if the trail is $4.2$ miles long and ascends $4000$ feet?
Gold answer: $246$

CoT prediction (normalized): 246
CoT correct: True
SC winning answer norm: 338551
SC correct: False

Raw CoT response (first 200 chars):
  We are given the model \[ T = aL + bG \] where \(T\) is time in minutes, \(L\) is trail length in miles, and \(G\) is altitude gain in feet.  We know: 1. For a 1.5-mile trail with 800 ft gain, \(T = 6


======== Both CoT and SC wrong (n=43) ========

--------------------------------------------------
idx: 11
split: CNMO
Question:
  Given real numbers $a, b$, if the solution set of the quadratic inequality $x^2 + b < ax$ is $(1,5)$, then what is the solution set of the quadratic inequality $x^2 - a > bx$?
Gold answer: $(-\infty,-1)\cup(6,+\infty)$

CoT prediction (normalized): (-\infty,-1)\cup(6,\infty)
CoT correct: False
SC winning answer norm: 9
SC correct: False

Raw CoT response (first 200 chars):
  We are given that the solution set of \[ x^2 + b < ax \] is exactly the open interval \((1,5)\).  Rewrite the inequality: \[ x^2 - ax + b < 0. \] This is a quadratic in \(x\). Its solution set is the 

--------------------------------------------------
idx: 31
split: AMC
Question:
  The numbers, in order, of each row and the numbers, in order, of each column of a $5 \times 5$ array of integers form an arithmetic progression of length $5$. The numbers in positions $(5, 5)$, $(2,4)$, $(4,3)$, and $(3, 1)$ are $0$, $48$, $16$, and $12$, respectively. What number is in position $(1, 2)$?
Gold answer: $29$

CoT prediction (normalized): 4
CoT correct: False
SC winning answer norm: 0
SC correct: False

Raw CoT response (first 200 chars):
  Let the entry in row \(i\), column \(j\) be \(a_{i,j}\).  Each row is an arithmetic progression of length 5, so for each row \(i\), \[ a_{i,j} = r_i + (j-1)d_i \] for some row start \(r_i\) and row di

--------------------------------------------------
idx: 36
split: AMC
Question:
  Points $P$ and $Q$ are chosen uniformly and independently at random on sides $\overline{AB}$ and $\overline{AC}$, respectively, of equilateral triangle $\triangle ABC$. Which of the following intervals contains the probability that the area of $\triangle APQ$ is less than half the area of $\triangle ABC$?
Gold answer: $\left(\frac{3}{4}, \frac{7}{8}\right]$

CoT prediction (normalized): \frac{1+\ln2}{2}
CoT correct: False
SC winning answer norm: \frac{1}{2}\log_2\left(\frac{9}{7}\right)
SC correct: False

Raw CoT response (first 200 chars):
  Let the side length of equilateral triangle \(ABC\) be 1. Then its area is \[ [ABC] = \frac{\sqrt{3}}{4}. \]  Let \(P\) be on \(\overline{AB}\) and \(Q\) on \(\overline{AC}\). Define \[ AP = x,\quad A

=== (B) Per-category analysis by 'split' ===
split  n  n_cot_ok  n_sc_ok  acc_cot   acc_sc  delta_acc_sc_minus_cot
  AMC 46        28       30 0.608696 0.652174                0.043478
 CCEE 44        24       25 0.545455 0.568182                0.022727
 CNMO 18         6       16 0.333333 0.888889                0.555556
WLPMC 11         2        2 0.181818 0.181818                0.000000
 hard 21         5        5 0.238095 0.238095                0.000000

=== (C) Error-type clustering ===
error_cluster  n  n_cot_ok  n_sc_ok  acc_cot   acc_sc  delta_acc_sc_minus_cot
      algebra 14         5        7 0.357143 0.500000                0.142857
     geometry 34        12       18 0.352941 0.529412                0.176471
number_theory  8         6        6 0.750000 0.750000                0.000000
        other 84        42       47 0.500000 0.559524                0.059524

=== (D) Heatmap saved to: results/livemathbench_cot_sc_heatmap.png ===
   
