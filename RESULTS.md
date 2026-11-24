# Results

This page gives the commands that turn the predictions of `scripts/run.py` into the paper's metrics,
comparisons and figures.

Every interval is a 95% confidence interval from 1,000 bootstrap draws that resample patients (or
studies, where patients are unknown) with replacement (`eval/analysis/metrics.py`). Within a cohort,
every model uses the same draws, so differences between models are paired.

## Paper results

The table gives the paper's main results, from linear probes on each model's frozen study embeddings.
Classification tasks are pooled across the private test set, MIMIC-IV-Echo and CardioBench, and
severity-grading tasks across the private test set and MIMIC-IV-Echo. Values are means over tasks with 95% confidence
intervals from 1,000 paired bootstrap draws.

| Model | Mean AUROC, 68 classification tasks | Mean QWK, 31 severity-grading tasks |
|---|---|---|
| **EchoAlign** | **0.890 (0.881 to 0.897)** | **0.530 (0.511 to 0.544)** |
| EchoPrime | 0.880 (0.872 to 0.887) | 0.505 (0.486 to 0.520) |
| EchoCLIP | 0.860 (0.852 to 0.867) | 0.477 (0.453 to 0.492) |
| PanEcho | 0.856 (0.847 to 0.863) | 0.457 (0.436 to 0.471) |
| EchoJEPA | 0.853 (0.846 to 0.859) | 0.441 (0.419 to 0.456) |
| EchoFM | 0.822 (0.812 to 0.829) | 0.389 (0.364 to 0.408) |

EchoAlign's mean is higher than that of each of the five models on both metrics (P < 0.01).

## Cohorts

Each command below runs on one cohort. Set these variables first, from `configs/cohorts.yaml`:

| Cohort | `C` | `PROBE` (probe labels) | `ZS` (zero-shot labels) | `SEED` (seed key) | Resampled unit |
|---|---|---|---|---|---|
| Private | `private` | `data/labels/private_probe.csv` | `data/labels/private_probe.csv` | `private` | study |
| MIMIC-IV-Echo | `mimic_iv_echo` | `data/labels/mimic_iv_echo_probe.csv` | `data/labels/mimic_iv_echo.csv` | `mimic/patient_cluster/absolute` | patient |
| CardioBench | `cardiobench` | `data/labels/cardiobench_probe.csv` | `data/labels/cardiobench.csv` | `cardiobench/cluster/absolute` | study (patient for HMC-QU) |

For example, for MIMIC-IV-Echo:

```bash
C=mimic_iv_echo
PROBE=data/labels/mimic_iv_echo_probe.csv
ZS=data/labels/mimic_iv_echo.csv
SEED=mimic/patient_cluster/absolute
R=results/$C
```

The private cohort is not distributed, so its commands run only where that data is available.

## Steps

To reproduce the results, follow these steps from the repository root:

1. Run the evaluation. It writes the embeddings, zero-shot scores, linear-probe predictions and
   segmentation runs:

   ```bash
   python scripts/run.py
   ```

2. Compute the AUROC (area under the ROC curve) of each task and model, for the linear probes and
   for zero-shot:

   ```bash
   python scripts/evaluate.py --labels $PROBE --seed-key $SEED --missing drop \
       --scores $R/probe/*_test_scores.csv --out $R/probe/auroc.csv
   python scripts/evaluate.py --labels $ZS --seed-key $SEED --missing drop \
       --scores $R/zeroshot/scores_*.csv --out $R/zeroshot/auroc.csv
   ```

   `--missing drop` leaves out TEST studies that a model does not score and counts them in the
   `n_missing` column. EchoPrime gives no zero-shot score for a study when none of its 50 retrieved
   reports has a label for the finding.

   The linear probes also write the regression and severity-grade metrics:
   `<model>_regression.csv` (RMSE), `<model>_ordinal.csv` (quadratic weighted kappa) and
   `<model>_ordinal_boundaries.csv` (AUROC at each grade boundary), in `$R/probe/`.

3. Compare the linear probes of EchoAlign with each baseline, on the TEST studies every model
   scores. `compare_inputs.py prepare` joins the score files and writes the paired labels and the
   bootstrap universe; `compare_paired.py` computes the task means and the paired differences. The
   loop skips a task type the cohort does not have (CardioBench has no severity grades):

   ```bash
   MODELS="echoalign echoprime panecho echoclip echojepa echofm"
   PAIRS="--pair echoalign:echoprime --pair echoalign:panecho --pair echoalign:echoclip
          --pair echoalign:echojepa --pair echoalign:echofm"
   for KIND in test_scores regression_predictions ordinal_predictions; do
       [ -e $R/probe/echoalign_$KIND.csv ] || continue
       OUT=$R/compare_$KIND
       python scripts/compare_inputs.py prepare --labels $PROBE --reference echoalign \
           $(for m in $MODELS; do echo --method $m=$R/probe/${m}_$KIND.csv; done) --out $OUT
       python scripts/compare_paired.py --labels $OUT/paired_labels.csv --universe $OUT/universe.csv \
           --seed-key $SEED --save-draws \
           $(for m in $MODELS; do echo --method $m=$OUT/scores_$m.csv; done) $PAIRS --out $OUT/paired
   done
   ```

4. Compare zero-shot scores. `scripts/run.py` already ran `prepare` in `$R/zeroshot/`. EchoJEPA and
   EchoFM have no zero-shot method:

   ```bash
   python scripts/compare_paired.py --labels $R/zeroshot/paired_labels.csv --universe $R/zeroshot/universe.csv \
       --seed-key $SEED --save-draws \
       --method echoalign=$R/zeroshot/scores_echoalign.csv --method echoprime=$R/zeroshot/scores_echoprime.csv \
       --method panecho=$R/zeroshot/scores_panecho.csv --method echoclip=$R/zeroshot/scores_echoclip.csv \
       --pair echoalign:echoprime --pair echoalign:panecho --pair echoalign:echoclip \
       --out $R/zeroshot/paired
   ```

5. After steps 2 to 4 have run for every cohort, collect the comparisons into the tables the
   figures read (`per_task.csv`, `paired_per_task.csv` and `pooled.csv`):

   ```bash
   python scripts/collect_results.py --root results --cohorts mimic_iv_echo cardiobench \
       --pair echoalign:echoprime --pair echoalign:panecho --pair echoalign:echoclip \
       --pair echoalign:echojepa --pair echoalign:echofm --out results/collected
   ```

6. Write the segmentation tables (test Dice per model, head and dataset, with patient-bootstrap
   intervals) to `results/segmentation/tables/`:

   ```bash
   python scripts/segment.py --steps report
   ```

## Further analyses

These analyses run on the private cohort and MIMIC-IV-Echo. Run each script with `--help` for all
of its options.

- **Subgroups:** AUROC within patient subgroups, from a covariate table with one row per study and
  the columns that `configs/subgroups.yaml` names for the cohort:

  ```bash
  python scripts/subgroups.py --cohort $C --labels $R/compare_test_scores/paired_labels.csv \
      --universe $R/compare_test_scores/universe.csv --seed-key $SEED --covariates <covariate table> \
      --method echoalign=$R/compare_test_scores/scores_echoalign.csv \
      --method echoprime=$R/compare_test_scores/scores_echoprime.csv --out $R/subgroups
  ```

- **Label efficiency:** linear probes on 1% to 100% of the TRAIN labels, then the mean AUROC per
  fraction. EchoPrime uses its section features:

  ```bash
  python scripts/label_efficiency.py --method echoalign --embeddings embeddings/$C/echoalign.npz \
      --labels $PROBE --out $R/label_efficiency
  python scripts/label_efficiency.py --method echoprime --echoprime-sections embeddings/$C/echoprime_sections.npz \
      --cohort $C --labels $PROBE --missing drop --out $R/label_efficiency
  python scripts/summarize_label_efficiency.py --labels $R/compare_test_scores/paired_labels.csv \
      --universe $R/compare_test_scores/universe.csv --seed-key $SEED \
      --method echoalign=$R/label_efficiency/echoalign_label_efficiency_scores.csv \
      --method echoprime=$R/label_efficiency/echoprime_sections_all_sections_label_efficiency_scores.csv \
      --out $R/label_efficiency/summary
  ```

- **ROC curves:** from the probe predictions, with the findings and styles in
  `configs/figures_roc.yaml`:

  ```bash
  python scripts/plot_roc.py --cohort $C --predictions $R/probe --labels $PROBE --seed-key $SEED \
      --out $R/figures
  ```

- **Output comparison:** the assigned structure output against the global output on the zero-shot
  tasks, with each output using prompts selected for itself on the private VAL split. The script
  stops if selection does not reproduce `configs/echoalign_prompts.yaml`:

  ```bash
  python scripts/compare_output_prompts.py --cohort $C --weights weights/EchoAlign.pt \
      --private-embeddings embeddings/private/echoalign.npz --embeddings embeddings/$C/echoalign.npz \
      --labels $R/zeroshot/paired_labels.csv --universe $R/zeroshot/universe.csv \
      --auroc $R/zeroshot/auroc.csv --seed-key $SEED --out $R/output_prompts
  ```
