"""
13_review_checks.py  —  paper 2: CPU-only checks requested in review, on saved predictions (no training).

Nothing is tuned on a test set. Every map below is fitted on NIH VALIDATION predictions, except the
"oracle" prior shift on CheXpert, which uses CheXpert label prevalence on purpose and is reported only as
an upper bound on what a prior correction could achieve.

  (1) recal_ablation.csv   NIH test, 12 runs: recalibration maps that separate "scale vs offset" from
                           "one parameter set vs one per class":
                             temperature        1 scale for all classes (as in the paper)
                             temp_perclass      1 scale per class, no intercept           (14 params)
                             shift_perclass     1 intercept per class, slope fixed at 1   (14 params)
                             platt_global       1 intercept + 1 slope for all classes     (2 params)
                             platt              intercept + slope per class (as in the paper, 28 params)
  (2) prevalence.csv       per class: NIH train, validation and test prevalence, CheXpert sample prevalence.
  (3) prior_shift.csv      CheXpert sample, 12 runs, after NIH-fitted per-class Platt: no correction, oracle
                           prior shift (logit of CheXpert prevalence minus logit of NIH-validation prevalence,
                           added per class), and the label-free EM estimate of the CheXpert prior
                           (Saerens et al. 2002, per class; checked on synthetic prior-shift data, where it
                           recovers the true prior). ECE, calibration intercept, and TPR / FPR /
                           underdiagnosis at the F1 thresholds carried in probability space.
  (4) age_view.csv         CheXpert sample, 12 runs: underdiagnosis at the sensitivity-0.8 thresholds for
                           age <40 vs 40+ within AP and within PA scans (point estimates).
  (5) thresholds7.csv      NIH test restricted to the 7 classes shared with CheXpert (like-for-like with the
                           CheXpert sample), and the CheXpert sample, at the NIH-validation thresholds (F1 and
                           sensitivity 0.8, p2_thresholds/thresholds_fitted.csv): macro TPR, FPR, underdiagnosis,
                           and the share of scans without any of the 7 findings that get >= 1 positive call.
  (6) ece_bins.csv         NIH test, 12 runs: macro ECE with 10, 20 and 50 equal-count bins (raw, temperature,
                           per-class Platt), to check that the conclusions do not depend on the bin count.

  python 13_review_checks.py --results results --pred_dir results/_dl_ext/ext_preds --chex results/_dl_chexpert --out eval/p2_review
"""
import argparse
import importlib.util
import os

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
sig = rc.sigmoid
logit = ev._logit


def fit_map(method, lv, y):
    """Return f(logits) -> probabilities, fitted on validation logits lv (n x K) and labels y."""
    K = lv.shape[1]
    if method == "temperature":
        a = ev.logistic_fit(lv.reshape(-1, 1), y.ravel().astype(float))[0]
        return lambda l: sig(a * l)
    if method == "platt_global":
        b0, b1 = ev.logistic_fit(np.column_stack([np.ones(lv.size), lv.ravel()]), y.ravel().astype(float))
        return lambda l: sig(b0 + b1 * l)
    par = []
    for k in range(K):
        yk = y[:, k].astype(float)
        if method == "temp_perclass":
            par.append((0.0, ev.logistic_fit(lv[:, [k]], yk)[0]))
        elif method == "shift_perclass":
            par.append((ev.logistic_fit(np.ones((len(yk), 1)), yk, offset=lv[:, k])[0], 1.0))
        elif method == "platt":
            par.append(tuple(ev.logistic_fit(np.column_stack([np.ones(len(yk)), lv[:, k]]), yk)))
    b0, b1 = np.array(par).T
    return lambda l: sig(b0 + b1 * l)


def macro_cal(y, p, mask=None):
    rows = []
    for k in range(y.shape[1]):
        m = (y[:, k] >= 0) if mask is None else mask & (y[:, k] >= 0)
        r = ev.calibration_row(y[m, k].astype(float), p[m, k])
        r["cal_intercept"] = ex.citl(y[m, k].astype(float), p[m, k])   # ev's Newton fit diverges under large shift
        rows.append(r)
    return {c: float(np.mean([r[c] for r in rows])) for c in ["ece_q10", "cal_slope", "cal_intercept", "brier"]}


def flag_rate(y, p, th):
    """Share of scans with no positive and no uncertain label (all 0) that get >= 1 positive call."""
    neg = (y == 0).all(axis=1)
    return float((p[neg] >= th).any(axis=1).mean())


def em_prior(p, prior_src, iters=200):
    """Saerens et al. (2002), one binary class: estimate the target prior from calibrated probabilities."""
    q = prior_src
    for _ in range(iters):
        w1, w0 = q / prior_src, (1 - q) / (1 - prior_src)
        post = w1 * p / (w1 * p + w0 * (1 - p))
        q_new = post.mean()
        if abs(q_new - q) < 1e-8:
            break
        q = q_new
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pred_dir", default="results/_dl_ext/ext_preds")
    ap.add_argument("--chex", default="results/_dl_chexpert")
    ap.add_argument("--out", default="eval/p2_review")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # (1) recalibration ablation on NIH test
    rows = []
    for run in ex.RUNS:
        d = os.path.join(a.results, run)
        _, yv, pv_, _ = ev.load_run(d, "val")
        _, yt, pt, _ = ev.load_run(d, "test")
        for m in ["none", "temperature", "temp_perclass", "shift_perclass", "platt_global", "platt"]:
            p = pt if m == "none" else fit_map(m, logit(pv_), yv)(logit(pt))
            rows.append({"run": run, "method": m, **macro_cal(yt, p)})
    rec = pd.DataFrame(rows)
    rec.to_csv(os.path.join(a.out, "recal_ablation.csv"), index=False)
    print(rec.pivot(index="run", columns="method", values="ece_q10").round(3).to_string())

    # (2) prevalence by split (NIH: dn_bce split; CheXpert: sample set)
    cc = pd.read_csv(os.path.join(a.results, "dn_bce", "split_class_counts.csv")).set_index("class")
    sets = ex.load_labels(a.chex)
    prev = {}
    for s in ["train", "val"]:
        prev[f"nih_{s}"] = cc.loc[ev.CLASSES, s] / cc.loc["images", s]
    _, yt, _, _ = ev.load_run(os.path.join(a.results, "dn_bce"), "test")
    prev["nih_test"] = pd.Series(yt.mean(0), index=ev.CLASSES)
    pr0 = pd.read_csv(os.path.join(a.pred_dir, "dn_bce.csv.gz")).set_index("key")
    sm = sets["sample"][sets["sample"].key.isin(pr0.index)].reset_index(drop=True)
    ys = np.stack(sm.y.values)
    chex_prev = np.array([(ys[:, j] == 1).sum() / (ys[:, j] >= 0).sum() for j in range(ys.shape[1])])
    prev["chexpert_sample"] = pd.Series(chex_prev, index=list(ex.SHARED))
    pv = pd.DataFrame(prev)
    pv.to_csv(os.path.join(a.out, "prevalence.csv"))
    print(pv.round(4).to_string())

    # (3) prior shift on CheXpert, and (4) age x view
    rows, av = [], []
    for run in ex.RUNS:
        pr = pd.read_csv(os.path.join(a.pred_dir, f"{run}.csv.gz")).set_index("key")
        d = sets["sample"][sets["sample"].key.isin(pr.index)].reset_index(drop=True)
        y = np.stack(d.y.values)
        p14 = pr.loc[d.key, [f"p_{c}" for c in ev.CLASSES]].values.astype(float)
        _, yv, pv_, th_f1 = ev.load_run(os.path.join(a.results, run), "val")
        platt = fit_map("platt", logit(pv_), yv)
        cal = platt(logit(p14))[:, ex.NIH_IDX]
        thr = platt(logit(np.tile(th_f1, (1, 1))))[0, ex.NIH_IDX]       # F1 thresholds in Platt-probability space
        src = yv[:, ex.NIH_IDX].mean(0)
        variants = {"none": cal}
        variants["oracle_prior"] = sig(logit(cal) + (logit(chex_prev) - logit(src)))
        q_em = np.array([em_prior(cal[:, j], src[j]) for j in range(cal.shape[1])])
        variants["em_prior"] = sig(logit(cal) + (logit(q_em) - logit(src)))
        for name, p in variants.items():
            dec = ex.Decisions(y, p, thr, np.ones(len(y), bool), range(y.shape[1]))(np.ones(len(y)))
            rows.append({"run": run, "variant": name, **macro_cal(y, p), **dec,
                         **({"em_prior_mean_abs_err": float(np.abs(q_em - chex_prev).mean())} if name == "em_prior" else {})})
        th_s = ex.sens_thresholds(yv, pv_)[ex.NIH_IDX]
        b = p14[:, ex.NIH_IDX] >= th_s
        ud_mask = (y == 1).any(axis=1); nop = ~b.any(axis=1)
        young = (d.age == "<40").values; known = d.age.notna().values
        for view in ["AP", "PA"]:
            vm = (d.view == view).values & ud_mask & known
            u_y, u_o = nop[vm & young].mean(), nop[vm & ~young].mean()
            av.append({"run": run, "view": view, "n_pos_lt40": int((vm & young).sum()), "n_pos_40plus": int((vm & ~young).sum()),
                       "ud_lt40": u_y, "ud_40plus": u_o, "gap": u_y - u_o})
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(a.out, "prior_shift.csv"), index=False)
    av = pd.DataFrame(av)
    av.to_csv(os.path.join(a.out, "age_view.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(ps.groupby("variant")[["ece_q10", "cal_intercept", "cal_slope", "tpr", "fpr", "underdiag"]].agg(["min", "max"]).round(3).T)
        print("EM prior mean abs error vs CheXpert prevalence:", ps.em_prior_mean_abs_err.dropna().round(3).tolist())
        print(av.round(3).to_string(index=False))
    # (5) NIH 7-class and CheXpert thresholds, like for like
    thf = pd.read_csv(os.path.join(a.results, "..", "eval", "p2_thresholds", "thresholds_fitted.csv")).set_index(["run", "threshold"])
    rows = []
    for run in ex.RUNS:
        _, yt, pt, _ = ev.load_run(os.path.join(a.results, run), "test")
        pr = pd.read_csv(os.path.join(a.pred_dir, f"{run}.csv.gz")).set_index("key")
        d = sets["sample"][sets["sample"].key.isin(pr.index)].reset_index(drop=True)
        yc = np.stack(d.y.values)
        pc = pr.loc[d.key, [f"p_{c}" for c in ev.CLASSES]].values.astype(float)[:, ex.NIH_IDX]
        for t in ["f1", "sens80"]:
            th = thf.loc[(run, t), ev.CLASSES].values.astype(float)[ex.NIH_IDX]
            for ds, y_, p_ in [("nih7", yt[:, ex.NIH_IDX], pt[:, ex.NIH_IDX]), ("chexpert", yc, pc)]:
                dec = ex.Decisions(y_, p_, th, np.ones(len(y_), bool), range(y_.shape[1]))(np.ones(len(y_)))
                rows.append({"run": run, "threshold": t, "set": ds, **dec, "flag_rate_neg": flag_rate(y_, p_, th)})
    t7 = pd.DataFrame(rows)
    t7.to_csv(os.path.join(a.out, "thresholds7.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(t7.groupby(["threshold", "set"])[["tpr", "fpr", "underdiag", "flag_rate_neg"]].agg(["min", "max"]).round(3))

    # (6) ECE with 10 / 20 / 50 equal-count bins on NIH test
    rows = []
    for run in ex.RUNS:
        d = os.path.join(a.results, run)
        _, yv, pv_, _ = ev.load_run(d, "val")
        _, yt, pt, _ = ev.load_run(d, "test")
        for m in ["none", "temperature", "platt"]:
            p = pt if m == "none" else fit_map(m, logit(pv_), yv)(logit(pt))
            rows.append({"run": run, "method": m, **{f"ece_q{nb}": float(np.mean([ev.ece_quantile(yt[:, k].astype(float), p[:, k], nb)
                                                                                    for k in range(yt.shape[1])])) for nb in (10, 20, 50)}})
    eb = pd.DataFrame(rows)
    eb.to_csv(os.path.join(a.out, "ece_bins.csv"), index=False)
    print(eb.round(3).to_string(index=False))
    print(f"Wrote {a.out}/")


if __name__ == "__main__":
    main()
