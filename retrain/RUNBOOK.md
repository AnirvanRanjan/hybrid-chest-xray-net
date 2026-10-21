# Retraining runbook — MS-Hybrid CXR paper

## 0. One-time cache (CPU session, no GPU quota used)

Upload the four `.py` files as a private Kaggle dataset (`retrain-code`). In a new notebook, add **nih-chest-xrays/data** and `retrain-code` as inputs, set Accelerator to None, and run:

```
!python $(find /kaggle/input -name 01_build_cache.py | head -1)
```

The script finds the NIH folder by itself and prints progress every 2,000 images. Then **Save Version → Save & Run All**. The output is four files: `nih384.zip` (~10 GB, all 112,120 images at 384px) plus `Data_Entry_2017.csv`, `train_val_list.txt` and `test_list.txt`. It is a single zip on purpose, because Kaggle saves notebook output file by file and 112k loose files make the commit crawl or fail.

## 1. Training notebook setup

New notebook → Add Input: the output of step 0, plus `retrain-code` → Accelerator: **GPU T4 x1** → **Internet: On** (needed for the ImageNet weights). Copy the scripts in with `!cp $(dirname $(find /kaggle/input -name 02_train.py | head -1))/*.py .`

`--data_dir` defaults to `auto`, which finds the cache under `/kaggle/input`. Pass it explicitly only if the script says it found zero or several candidates.

**Always start with a 2-minute smoke test** so a typo doesn't burn a session:

```
!python 02_train.py --run_name smoke --debug_n 256 --epochs 1
```

## 2. Experiments (in this order)

| # | Command flags | Purpose |
|---|---|---|
| A | `--model hybrid --sampler floor --stage 224 --run_name hyb_floor` | Main model, fixed sampler |
| B | `--model hybrid --sampler none --stage 224 --run_name hyb_none` | Sampler ablation |
| C | `--model densenet --loss bce --sampler none --stage 224 --run_name dn_bce` | CheXNet-style baseline |
| D | `--model densenet --loss asl --sampler floor --stage 224 --run_name dn_asl` | Same recipe, CNN only |
| E | `--model hybrid --sampler <winner> --stage 320 --tta --init /kaggle/working/<winner>/best.pt --run_name hyb_320` | Final model |

Pick the winner of A vs B by **best validation AUROC** in `history.csv`, never by test results. Every run finishes by evaluating the test set once and writing its files, so A–D also become the baseline and ablation tables in the paper.

Watch the first epoch's time in the log; the 224 hybrid runs are the slow ones. If a session is going to exceed ~11 hours, the script stops itself after an epoch. Then: **Save Version**, attach that version's output as an input to a new session, and rerun the same command plus `--resume /kaggle/input/<prev-output>/<run_name>/last.pt`.

Kaggle gives about 30 GPU-hours per week, so plan A–E across roughly two weeks.

## 3. What to send back after each run

**Never commit a second time just to collect files.** Every Save & Run All starts a fresh session with an empty `/kaggle/working` and reruns the whole notebook, so the training would start over. When a run completes, `02_train.py` itself writes `/kaggle/working/<run_name>_results.zip`. Download that single file from the version's **Output** tab. The `.pt` checkpoints stay in the output, where later notebooks attach them for `--init` / `--resume`. If the time guard stopped the run, there is no zip yet; resume as above, and the session that finishes writes it.

The zip contains the whole `<run_name>/` folder except the `.pt` files:

- `history.csv` — per-epoch train loss, val AUROC, val F1
- `split_info.json`, `split.csv`, `split_class_counts.csv` — exact split and per-split class counts, for the paper's dataset section
- `args.json` — the flags the run used (04_gradcam.py reads it)
- `pred_val.csv`, `pred_test.csv` — per-image probabilities + age, sex, patient ID
- `thresholds_val.npy`, `test_per_class.csv`, `test_summary.json`

Keep `best.pt` from the final model (run E) safe; Grad-CAM figures are regenerated from it.

## 4. Evaluation and paper outputs (on the Mac, no GPU)

Put each run's returned folder under `retrain/results/<run_name>/`. **Keep the run names from §2** (`hyb_floor`, `hyb_none`, `dn_bce`, `dn_asl`, `hyb_320`), because the paper's value names use them. Then, from `retrain/`:

```
# single-model results (tags A, B, E are what the manuscript reads)
python 03_evaluate.py run --run_dir results/hyb_floor --tag A --paper_dir ../paper/generated
python 03_evaluate.py run --run_dir results/hyb_none  --tag B --paper_dir ../paper/generated
python 03_evaluate.py run --run_dir results/hyb_320   --tag E --paper_dir ../paper/generated
python 03_evaluate.py run --run_dir results/hyb_320   --tag E --paper_dir ../paper/generated --split val

# paired comparisons (baselines: reference = run E, the final hybrid)
python 03_evaluate.py compare --run_dirs results/hyb_320 results/hyb_none results/dn_bce results/dn_asl \
    --ref hyb_320 --tag baselines --paper_dir ../paper/generated --out eval/baselines \
    --labels "MS-Hybrid, 320 + TTA (E)" "MS-Hybrid, 224 (B)" "DenseNet-121, BCE (C)" "DenseNet-121, ASL (D)"
python 03_evaluate.py compare --run_dirs results/dn_bce results/dn_asl \
    --ref dn_bce --tag loss --paper_dir ../paper/generated --out eval/loss \
    --labels "DenseNet-121, BCE (C)" "DenseNet-121, ASL (D)"
python 03_evaluate.py compare --run_dirs results/hyb_floor results/hyb_none \
    --ref hyb_floor --tag sampler --paper_dir ../paper/generated \
    --labels "With sampler (A)" "Without sampler (B)"

# Grad-CAM (needs the nih384 cache images: run on Kaggle CPU with the cache attached,
# or copy only the images listed in gradcam_examples.csv after a first run)
python 04_gradcam.py --run_dir results/hyb_320 --ckpt results/hyb_320/best.pt \
    --data_dir <folder containing nih384/> --out ../paper/generated/E

# Additional experiment: tuned hybrid (run T)
python 03_evaluate.py run --run_dir results/hyb_tuned_320 --tag T --paper_dir ../paper/generated
python 03_evaluate.py run --run_dir results/hyb_tuned_320 --tag T --paper_dir ../paper/generated --split val
python 03_evaluate.py run --run_dir results/hyb_tuned --tag Tp --paper_dir ../paper/generated
python 03_evaluate.py compare --run_dirs results/hyb_tuned_320 results/hyb_tuned results/hyb_320 results/dn_bce results/dn_asl \
    --ref hyb_tuned_320 --tag tuned --paper_dir ../paper/generated --out eval/tuned \
    --labels "MS-Hybrid tuned, 320 + TTA (T)" "MS-Hybrid tuned, 224" "MS-Hybrid, 320 + TTA (E)" "DenseNet-121, BCE (C)" "DenseNet-121, ASL (D)"
python 03_evaluate.py compare --run_dirs results/hyb_tuned results/dn_bce results/dn_asl results/hyb_none \
    --ref hyb_tuned --tag tuned224 --paper_dir ../paper/generated --out eval/tuned224 \
    --labels "MS-Hybrid tuned, 224" "DenseNet-121, BCE (C)" "DenseNet-121, ASL (D)" "MS-Hybrid, 224 (B)"

# Post-hoc check of the 60+ cardiomegaly sensitivity gap by view (AP vs PA); descriptive only
python 05_view_check.py --run_dir results/hyb_320 --tag E --out ../paper/generated/E
```

**Decided 21 Oct 2026 on validation AUROC: B won (0.8092 vs 0.8088 for A).** E is fine-tuned from `hyb_none/best.pt` with `--sampler none`, The baselines comparison uses the final model E (`hyb_320`) as reference. Age groups are <40, 40–59 and 60+ (merged because only 31 test patients are 80+).

What each `run` writes to `eval/<run_name>/` (prefixed `test_` or `val_`), and copies to `paper/generated/<tag>/`:

- `metrics_ci.csv`: per-class and macro AUROC, plus F1/TPR/FPR at the validation thresholds, with patient-level bootstrap 95% CIs (2,000 resamples); also macro F1 at a fixed 0.5 threshold
- `results.tex`: every number as `\resdef{E.test.<name>}{value}`; the manuscript prints them with `\res{...}`
- `auroc_table.tex`, `calibration_table.tex`, `subgroup_table.tex`, `dataset_table.tex`: LaTeX tables
- `roc.png`, `pr.png`, `reliability.png`, `subgroup_macro_auroc.png`, `cooccurrence.png`, `history.png`: figures
- `calibration.csv`, `subgroups.csv`, `subgroup_gaps.csv`, `summary.json`: full detail for writing the text

`compare` reuses the same patient resamples for every model (a paired bootstrap) and reports AUROC differences against `--ref`, with CIs and p-values. The p-values are unadjusted, so correct them (e.g. Holm) when reporting per-class differences. A full `run` takes about 2 minutes on the M1.

## 5. Building the paper

From `paper/`: `pdflatex sn-article && bibtex sn-article && pdflatex sn-article && pdflatex sn-article`. Until `generated/` is filled, every result prints as a red `[name]` placeholder and every generated table or figure as a red box, so a missing number can't go unnoticed. The red `\TBD{...}` notes mark text that has to be written after looking at the results. Search the .tex for `FIXME` to find the remaining decisions.
