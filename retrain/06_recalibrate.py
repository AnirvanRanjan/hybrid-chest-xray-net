"""
06_recalibrate.py  —  paper 2: does post-hoc recalibration fix the calibration of each model?

For every run, three recalibration maps are FITTED ON VALIDATION predictions and APPLIED TO
TEST predictions (the test labels are only used to score the result):
  temperature  one scale a for all classes:      p' = sigmoid(a * logit p)       (T = 1/a)
  platt        per class:                         p' = sigmoid(b0 + b1 * logit p)
  isotonic     per class, monotone step function (sklearn IsotonicRegression)
plus "none" (the raw probabilities). Temperature and Platt are strictly monotone, so AUROC is
unchanged; isotonic can merge scores into ties and lower AUROC slightly, which is reported.

Reported on test, macro-averaged over the 14 classes: AUROC, Brier, ECE (10 equal-count bins),
calibration slope and intercept, mean predicted / prevalence (ratio of macro means, as in 03_evaluate.py). Brier and ECE get patient-level
bootstrap 95% CIs (2,000 resamples of patients, as in 03_evaluate.py) and a paired bootstrap
difference against "none". In the bootstrap the ECE bin edges stay fixed at the full-sample
equal-count edges; only the weights change.

Per-class thresholds are not re-searched here (they matter for F1, not for calibration).

  python 06_recalibrate.py --run_dirs results/dn_bce results/dn_asl results/hyb_320 \\
      results/hyb_none results/hyb_floor results/hyb_tuned_320 --out eval/recal
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

_spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

METHODS = ["none", "temperature", "platt", "isotonic"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_apply(method, y_val, p_val, p_test):
    """Fit on validation (images x classes), return recalibrated test probabilities and the fitted parameters."""
    lv, lt = ev._logit(p_val), ev._logit(p_test)
    if method == "none":
        return p_test.copy(), {}
    if method == "temperature":
        a = ev.logistic_fit(lv.reshape(-1, 1), y_val.ravel().astype(float))[0]
        return sigmoid(a * lt), {"T": 1 / a}
    out, params = np.empty_like(p_test), {}
    for k, c in enumerate(ev.CLASSES):
        if method == "platt":
            b0, b1 = ev.logistic_fit(np.column_stack([np.ones(len(lv)), lv[:, k]]), y_val[:, k].astype(float))
            out[:, k] = sigmoid(b0 + b1 * lt[:, k])
            params[c] = (b0, b1)
        else:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val[:, k], y_val[:, k])
            out[:, k] = iso.predict(p_test[:, k])
    return out, params


class WeightedCal:
    """Brier and fixed-bin ECE for one class under bootstrap weights (all ones = point estimate)."""
    def __init__(self, y, p, n_bins=10):
        self.y, self.p = y.astype(float), p
        self.idx = ev.quantile_bins(p, n_bins)
        self.nb = self.idx.max() + 1

    def __call__(self, w):
        tot = w.sum()
        nw = np.bincount(self.idx, w, minlength=self.nb)
        yw = np.bincount(self.idx, w * self.y, minlength=self.nb)
        pw = np.bincount(self.idx, w * self.p, minlength=self.nb)
        m = nw > 0
        ece = float(np.abs(yw[m] - pw[m]).sum() / tot)
        brier = float((w * (self.p - self.y) ** 2).sum() / tot)
        return brier, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="eval/recal")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    summary, perclass, fitted = [], [], []
    for rd in a.run_dirs:
        run = os.path.basename(os.path.normpath(rd))
        _, y_val, p_val, _ = ev.load_run(rd, "val")
        df, y, p_raw, _ = ev.load_run(rd, "test")
        preds = {}
        for m in METHODS:
            preds[m], params = fit_apply(m, y_val, p_val, p_raw)
            if m == "temperature":
                fitted.append({"run": run, "T": params["T"]})

        # point estimates, per class
        for m in METHODS:
            for k, c in enumerate(ev.CLASSES):
                row = ev.calibration_row(y[:, k], preds[m][:, k])
                row.update(run=run, method=m, cls=c,
                           auroc=ev.FastAUC(y[:, k], preds[m][:, k])(np.ones(len(y))))
                perclass.append(row)

        # patient bootstrap of macro Brier / ECE, paired across methods (same resamples)
        cal = {m: [WeightedCal(y[:, k], preds[m][:, k]) for k in range(len(ev.CLASSES))] for m in METHODS}
        boots = {m: [] for m in METHODS}
        for w in ev.boot_weights(df["Patient ID"].values, a.n_boot, a.seed):
            for m in METHODS:
                r = np.array([f(w) for f in cal[m]])
                boots[m].append(r.mean(axis=0))          # (macro brier, macro ece)
        pc = pd.DataFrame(perclass)
        for m in METHODS:
            sel = pc[(pc.run == run) & (pc.method == m)]
            b = np.array(boots[m]); b0 = np.array(boots["none"])
            row = {"run": run, "method": m,
                   "auroc": sel.auroc.mean(), "brier": sel.brier.mean(), "ece_q10": sel.ece_q10.mean(),
                   "cal_slope": sel.cal_slope.mean(), "cal_intercept": sel.cal_intercept.mean(),
                   "overprediction": sel.mean_pred.mean() / sel.prevalence.mean()}   # as in 03_evaluate.py compare
            row["brier_lo"], row["brier_hi"] = ev.ci(b[:, 0])
            row["ece_lo"], row["ece_hi"] = ev.ci(b[:, 1])
            if m != "none":
                d = b[:, 1] - b0[:, 1]
                row["ece_diff_vs_none"] = row["ece_q10"] - summary_none(summary, run)
                row["ece_diff_lo"], row["ece_diff_hi"] = ev.ci(d)
                row["ece_diff_p"] = ev.p_two_sided(d)
            summary.append(row)

    s = pd.DataFrame(summary)
    pd.DataFrame(perclass).to_csv(os.path.join(a.out, "recal_perclass.csv"), index=False)
    s.to_csv(os.path.join(a.out, "recal_summary.csv"), index=False)
    pd.DataFrame(fitted).to_csv(os.path.join(a.out, "temperatures.csv"), index=False)
    cols = ["run", "method", "auroc", "brier", "ece_q10", "ece_lo", "ece_hi", "cal_slope", "cal_intercept", "overprediction"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s[cols].round(4).to_string(index=False))
        print(pd.DataFrame(fitted).round(3).to_string(index=False))
    print(f"Wrote {a.out}/")


def summary_none(summary, run):
    return next(r["ece_q10"] for r in summary if r["run"] == run and r["method"] == "none")


if __name__ == "__main__":
    main()
