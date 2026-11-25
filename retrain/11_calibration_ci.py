"""
11_calibration_ci.py  —  paper 2: bootstrap CIs for calibration slope and intercept, and reliability-diagram bins (CPU).

06_recalibrate.py and 09_external.py already give patient-bootstrap CIs for Brier and ECE but only point
estimates for the calibration slope and intercept. This script adds those CIs, with the same protocol:
  * NIH test (14 classes) and the CheXpert sample (7 shared classes, uncertain labels masked per class);
  * raw probabilities ("none") and per-class Platt fitted on NIH validation (06_recalibrate.fit_apply);
  * per class: slope b in y ~ sigmoid(a + b logit p); intercept (calibration in the large) a in
    y ~ sigmoid(a + logit p); macro = mean over classes;
  * 2,000 patient-level bootstrap resamples (03_evaluate.boot_weights, seed 0), percentile 95% CIs.
It also writes the reliability-diagram bins (10 equal-count bins per class: mean predicted, observed
frequency, count) for both sets and methods.

  python 11_calibration_ci.py --results results --pred_dir results/_dl_ext/ext_preds \\
      --chex results/_dl_chexpert --out eval/p2_calci
"""
import argparse
import importlib.util
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker; oversubscription made it ~50x slower
from multiprocessing import Pool

import numpy as np
import pandas as pd


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("ev", "03_evaluate.py")
rc = _load("rc", "06_recalibrate.py")
ex = _load("ex", "09_external.py")
METHODS = ["none", "platt"]


def wfit(X, y, w, beta, offset=0.0, iters=30):
    """Weighted Newton-Raphson for P(y=1) = sigmoid(X @ beta + offset), warm-started; step length capped at 5."""
    beta = beta.astype(float).copy()
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta + offset, -35, 35)))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(len(beta)), X.T @ (w * (y - mu)))
        n = np.abs(step).max()
        if n > 5:
            step *= 5 / n
        beta += step
        if n < 1e-7:
            break
    return beta


class SlopeIntercept:
    """Per-class slope and calibration-in-the-large under bootstrap weights (ones = point estimate)."""
    def __init__(self, y, p, masks):
        self.cls = []
        for k, m in enumerate(masks):
            yk, lp = y[m, k].astype(float), ev._logit(p[m, k])
            X = np.column_stack([np.ones(len(lp)), lp])
            one = np.ones(len(lp))
            b0 = wfit(X, yk, one, np.zeros(2))
            a0 = np.array([ex.citl(yk, p[m, k])])
            self.cls.append((m, yk, lp, X, b0, a0))

    def __call__(self, w):
        s, a = [], []
        for m, yk, lp, X, b0, a0 in self.cls:
            wk = w[m]
            s.append(wfit(X, yk, wk, b0)[1])
            a.append(wfit(np.ones((len(lp), 1)), yk, wk, a0, offset=lp)[0])
        return np.array(s), np.array(a)


def reliability(y, p, masks, classes):
    rows = []
    for k, (m, c) in enumerate(zip(masks, classes)):
        yk, pk = y[m, k].astype(float), p[m, k]
        idx = ev.quantile_bins(pk, 10)
        for b in np.unique(idx):
            sel = idx == b
            rows.append({"cls": c, "bin": int(b), "mean_pred": pk[sel].mean(), "frac_pos": yk[sel].mean(), "n": int(sel.sum())})
    return rows


def job(task):
    run, sname, a = task
    _, y_val, p_val, _ = ev.load_run(os.path.join(a.results, run), "val")
    if sname == "nih":
        df, y, p14, _ = ev.load_run(os.path.join(a.results, run), "test")
        pid, cols, classes = df["Patient ID"].values, slice(None), list(ev.CLASSES)
    else:
        d = ex.load_labels(a.chex)["sample"]
        pr = pd.read_csv(os.path.join(a.pred_dir, f"{run}.csv.gz")).set_index("key")
        d = d[d.key.isin(pr.index)].reset_index(drop=True)
        y = np.stack(d.y.values)
        p14 = pr.loc[d.key, [f"p_{c}" for c in ev.CLASSES]].values.astype(float)
        pid, cols, classes = d.pid.values, ex.NIH_IDX, list(ex.SHARED)
    masks = [y[:, k] >= 0 for k in range(len(classes))]
    out, rel = [], []
    for m in METHODS:
        pm = rc.fit_apply(m, y_val, p_val, p14)[0][:, cols]
        si = SlopeIntercept(y, pm, masks)
        s0, a0 = si(np.ones(len(y)))
        bs, ba = [], []
        for w in ev.boot_weights(pid, a.n_boot, a.seed):
            s, i = si(w)
            bs.append(s.mean()); ba.append(i.mean())
        row = {"run": run, "set": sname, "method": m, "cal_slope": s0.mean(), "cal_intercept": a0.mean()}
        row["cal_slope_lo"], row["cal_slope_hi"] = ev.ci(bs)
        row["cal_intercept_lo"], row["cal_intercept_hi"] = ev.ci(ba)
        out.append(row)
        rel += [{"run": run, "set": sname, "method": m, **r} for r in reliability(y, pm, masks, classes)]
    print(f"{run}/{sname} done", flush=True)
    return out, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pred_dir", default="results/_dl_ext/ext_preds")
    ap.add_argument("--chex", default="results/_dl_chexpert")
    ap.add_argument("--out", default="eval/p2_calci")
    ap.add_argument("--runs", nargs="*", default=ex.RUNS)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [(r, s, a) for r in a.runs for s in ("nih", "chexpert")]
    with Pool(a.workers) as pool:
        res = pool.map(job, tasks)
    pd.DataFrame([r for o, _ in res for r in o]).to_csv(os.path.join(a.out, "slope_intercept_ci.csv"), index=False)
    pd.DataFrame([r for _, rel in res for r in rel]).to_csv(os.path.join(a.out, "reliability_bins.csv"), index=False)
    print(f"Wrote {a.out}/")


if __name__ == "__main__":
    main()
