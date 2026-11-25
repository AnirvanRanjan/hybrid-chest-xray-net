"""
12_seeds.py  —  paper 2: seed variability of the DenseNet-121 loss grid (CPU).

DenseNet-121 x {BCE, wBCE, focal, ASL} x training seeds {42, 1, 2}; all on the same seed-42 patient split
(checked here again). For every run, on the NIH test split (14 classes, macro average):
  * macro AUROC;
  * Brier, classwise adaptive ECE (10 equal-frequency bins), calibration slope and intercept, mean
    predicted / prevalence ratio, for raw probabilities and after each recalibration method of
    06_recalibrate (temperature, per-class Platt, isotonic; all fitted on that run's NIH validation set).
Outputs (eval/p2_seeds/):
  seed_runs.csv     one row per run x method
  seed_summary.csv  per loss x method: mean, SD and min-max across the three seeds
  seed_diffs.csv    within-seed AUROC difference of each loss vs BCE (does the loss ranking hold per seed?)
  seed_chexpert.csv (with --ext) per run, CheXpert sample: NIH 7-class AUROC, CheXpert AUROC and drop, ECE and
                    intercept raw / after NIH-fitted Platt, underdiagnosis at F1 and 80%-sensitivity thresholds,
                    and the under-40 and PA underdiagnosis gaps (sensitivity-0.8 thresholds as in the paper, and F1;
                    unadjusted bootstrap p), read from
                    a 09_external.py run over the 12 DenseNet-121 models (eval/p2_external_seeds)

  python 12_seeds.py --results results --out eval/p2_seeds --ext eval/p2_external_seeds
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("ev", "03_evaluate.py")
rc = _load("rc", "06_recalibrate.py")

SEED42 = {"bce": "dn_bce", "wbce": "p2_dn_wbce", "focal": "p2_dn_focal", "asl": "dn_asl"}
LOSSES = list(SEED42)
SEEDS = [42, 1, 2]
METHODS = ["none", "temperature", "platt", "isotonic"]
METRICS = ["auroc", "brier", "ece_q10", "cal_slope", "cal_intercept", "overpred"]


def run_name(loss, seed):
    return SEED42[loss] if seed == 42 else f"p2_dn_{loss}_s{seed}"


def metrics(y, p):
    rows = [ev.calibration_row(y[:, k].astype(float), p[:, k]) for k in range(y.shape[1])]
    out = {m: float(np.mean([r[m] for r in rows])) for m in ["brier", "ece_q10", "cal_slope", "cal_intercept"]}
    out["overpred"] = float(np.mean([r["mean_pred"] / r["prevalence"] for r in rows]))
    out["auroc"] = float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1])]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="eval/p2_seeds")
    ap.add_argument("--ext", default=None, help="09_external.py output over the 12 DenseNet-121 runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ref = pd.read_csv(os.path.join(a.results, "dn_bce", "split.csv")).sort_values("Image Index").reset_index(drop=True)
    rows = []
    for loss in LOSSES:
        for seed in SEEDS:
            run = run_name(loss, seed)
            d = os.path.join(a.results, run)
            sp = pd.read_csv(os.path.join(d, "split.csv")).sort_values("Image Index").reset_index(drop=True)
            assert ref.equals(sp[ref.columns]), f"{run}: split differs"
            _, y_val, p_val, _ = ev.load_run(d, "val")
            df, y, p, _ = ev.load_run(d, "test")
            for m in METHODS:
                pm = rc.fit_apply(m, y_val, p_val, p)[0]
                rows.append({"loss": loss, "seed": seed, "run": run, "method": m, **metrics(y, pm)})
            print(f"{run} done", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "seed_runs.csv"), index=False)

    g = runs.groupby(["loss", "method"])[METRICS]
    summ = pd.concat({"mean": g.mean(), "sd": g.std(ddof=1), "min": g.min(), "max": g.max()}, axis=1)
    summ.columns = [f"{m}_{s}" for s, m in summ.columns]
    summ = summ[[f"{m}_{s}" for m in METRICS for s in ["mean", "sd", "min", "max"]]].reset_index()
    summ.to_csv(os.path.join(a.out, "seed_summary.csv"), index=False)

    raw = runs[runs.method == "none"].pivot(index="seed", columns="loss", values="auroc")
    diffs = pd.DataFrame({f"{l}-bce": raw[l] - raw["bce"] for l in LOSSES if l != "bce"})
    diffs.to_csv(os.path.join(a.out, "seed_diffs.csv"))
    if a.ext:
        chexpert(a)
    print(f"Wrote {a.out}/")


def chexpert(a):
    ex = _load("ex", "09_external.py")
    ov = pd.read_csv(os.path.join(a.ext, "external_overall.csv"))
    sub = pd.read_csv(os.path.join(a.ext, "external_subgroups.csv"))
    ov, sub = ov[ov.set == "sample"].set_index(["run", "method"]), sub.set_index(["run", "attribute", "group"])
    rows = []
    for loss in LOSSES:
        for seed in SEEDS:
            run = run_name(loss, seed)
            _, y, p, _ = ev.load_run(os.path.join(a.results, run), "test")
            nih7 = float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in ex.NIH_IDX]))
            g = lambda m, c: float(ov.loc[(run, m), c])
            r = {"loss": loss, "seed": seed, "run": run, "nih7_auroc": nih7, "auroc": g("none", "auroc"),
                 "drop": g("none", "auroc") - nih7, "ece_none": g("none", "ece"), "ece_platt": g("platt", "ece"),
                 "citl_none": g("none", "cal_intercept"), "citl_platt": g("platt", "cal_intercept"),
                 "ud_f1": g("thresholds_f1", "underdiag"), "ud_sens80": g("thresholds_sens80", "underdiag")}
            for name, key in [("lt40", ("age", "<40")), ("PA", ("view", "PA"))]:
                q = sub.loc[(run, *key)]
                # sensitivity-0.8 thresholds, as for the 12-model grid in the paper; F1 kept for reference
                r[f"{name}_udgap"], r[f"{name}_udgap_p"] = float(q.underdiag_sens80_gap), float(q.underdiag_sens80_gap_p)
                r[f"{name}_udgap_f1"], r[f"{name}_udgap_f1_p"] = float(q.underdiag_f1_gap), float(q.underdiag_f1_gap_p)
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "seed_chexpert.csv"), index=False)


if __name__ == "__main__":
    main()
