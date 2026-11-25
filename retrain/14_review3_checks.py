"""
14_review3_checks.py  —  paper 2, third review round: CPU-only checks on saved predictions (no training).
Every threshold and map is fitted on NIH VALIDATION predictions; nothing is tuned on a test set. The
exact-F1 thresholds (3) are a sensitivity check reported next to, not instead of, the pre-specified
0.01-grid thresholds.

  (1) uncertain_rescue.csv  CheXpert sample, 12 runs, F1 and sensitivity-0.8 thresholds: underdiagnosis when
                            a positive call on a class whose label is uncertain on that scan counts as flagging
                            the scan (as in the paper) vs when it does not ("confirmed-only"); share of positive
                            scans with >= 1 uncertain label overall and for <40 / 40+ and PA / AP; the <40 and
                            PA underdiagnosis gaps (crude and finding-count standardised, as in 10_paper2_extras
                            casemix) under the confirmed-only rule, with patient-bootstrap 95% CIs.
  (2) perclass_fnr.csv      CheXpert sample, 12 runs, sensitivity-0.8 thresholds: per-class false-negative rate
                            (uncertain labels masked) for <40 vs 40+ and PA vs AP.
  (3) f1_exact.csv          F1-optimal thresholds searched over the exact validation scores instead of the 0.01
                            grid; macro TPR and underdiagnosis on NIH test (7 shared classes) and CheXpert.
  (5) asl_margin.csv        NIH test, ASL runs: share of negative-label predictions below the ASL margin m = 0.05
                            (where the margin-shifted negative loss and its gradient are zero) and their 1st
                            percentile and median.
  (4) nih_intercepts.csv    NIH test after the validation-fitted per-class Platt maps: calibration intercept over
                            14 and over the 7 shared classes, before and after an oracle validation-to-test prior
                            shift (logit of test prevalence minus logit of validation prevalence, per class).

  python 14_review3_checks.py --results results --pred_dir results/_dl_ext/ext_preds --chex results/_dl_chexpert --out eval/p2_review3
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
ex = _load("ex", "09_external.py")
rv = _load("rv", "13_review_checks.py")
sig, logit = rv.sig, rv.logit


def f1_exact_thresholds(y, p):
    """Per class: the threshold (a validation score) that maximises F1 for the rule p >= t."""
    out = np.zeros(p.shape[1])
    for k in range(p.shape[1]):
        o = np.argsort(-p[:, k], kind="stable")
        s, yk = p[o, k], y[o, k].astype(float)
        tp = np.cumsum(yk); fp = np.cumsum(1 - yk)
        last = np.r_[s[1:] != s[:-1], True]                 # only cut between distinct scores
        f1 = 2 * tp / (tp + fp + yk.sum())
        f1 = np.where(last, f1, -1)
        out[k] = s[int(np.argmax(f1))]
    return out


def std_gap(missed, keep, stratum, ing, rest, w):
    ref = np.array([(keep & (ing | rest) & (stratum == s)).sum() for s in (1, 2, 3)], float)
    ref /= ref.sum()
    r = {}
    for name, m in (("in", ing & keep), ("rest", rest & keep)):
        rs = []
        for s in (1, 2, 3):
            ms = m & (stratum == s)
            rs.append((w[ms] * missed[ms]).sum() / w[ms].sum() if w[ms].sum() > 0 else np.nan)
        r[name + "_std"] = float(np.nansum(np.array(rs) * ref))
        r[name + "_crude"] = (w[m] * missed[m]).sum() / w[m].sum()
    return r["in_crude"] - r["rest_crude"], r["in_std"] - r["rest_std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pred_dir", default="results/_dl_ext/ext_preds")
    ap.add_argument("--chex", default="results/_dl_chexpert")
    ap.add_argument("--out", default="eval/p2_review3")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d0 = ex.load_labels(a.chex)["sample"]
    thf = pd.read_csv(os.path.join(a.results, "..", "eval", "p2_thresholds", "thresholds_fitted.csv")).set_index(["run", "threshold"])

    resc, fnr, f1x, icp, asl = [], [], [], [], []
    for run in ex.RUNS:
        pr = pd.read_csv(os.path.join(a.pred_dir, f"{run}.csv.gz")).set_index("key")
        d = d0[d0.key.isin(pr.index)].reset_index(drop=True)
        y = np.stack(d.y.values)
        p = pr.loc[d.key, [f"p_{c}" for c in ev.CLASSES]].values.astype(float)[:, ex.NIH_IDX]
        _, yv, pv, _ = ev.load_run(os.path.join(a.results, run), "val")
        _, yt, pt, _ = ev.load_run(os.path.join(a.results, run), "test")
        pid = d.pid.values
        keep = (y == 1).any(axis=1)
        unc = (y == -1)
        stratum = np.minimum((y == 1).sum(axis=1), 3)
        young, known_a = (d.age == "<40").values, d.age.notna().values
        pa, known_v = (d.view == "PA").values, d.view.isin(["AP", "PA"]).values
        groups = {"lt40": (known_a & young, known_a & ~young), "PA": (known_v & pa, known_v & ~pa)}
        th_all = {"f1": thf.loc[(run, "f1"), ev.CLASSES].values.astype(float)[ex.NIH_IDX],
                  "sens80": ex.sens_thresholds(yv, pv)[ex.NIH_IDX]}

        # (1) uncertain labels rescuing a scan
        for t, th in th_all.items():
            call = p >= th
            missed_std = ~call.any(axis=1)
            missed_conf = ~(call & ~unc).any(axis=1)
            row = {"run": run, "threshold": t,
                   "ud_paper": missed_std[keep].mean(), "ud_confirmed_only": missed_conf[keep].mean(),
                   "share_pos_with_uncertain": unc[keep].any(axis=1).mean()}
            for g, (ing, rest) in groups.items():
                row[f"share_unc_{g}_in"] = unc[keep & ing].any(axis=1).mean()
                row[f"share_unc_{g}_rest"] = unc[keep & rest].any(axis=1).mean()
                for rule, miss in (("paper", missed_std), ("conf", missed_conf)):
                    cg, sg = std_gap(miss, keep, stratum, ing, rest, np.ones(len(d)))
                    row[f"{g}_{rule}_crude"], row[f"{g}_{rule}_std"] = cg, sg
                if t == "sens80":
                    bs = np.array([std_gap(missed_conf, keep, stratum, ing, rest, w)
                                   for w in ev.boot_weights(pid, a.n_boot, a.seed)])
                    row[f"{g}_conf_crude_lo"], row[f"{g}_conf_crude_hi"] = np.percentile(bs[:, 0], [2.5, 97.5])
                    row[f"{g}_conf_std_lo"], row[f"{g}_conf_std_hi"] = np.percentile(bs[:, 1], [2.5, 97.5])
            resc.append(row)

        # (2) per-class false-negative rate by age and view, sensitivity-0.8 thresholds
        call = p >= th_all["sens80"]
        for j, c in enumerate(ex.SHARED):
            pos = y[:, j] == 1
            r = {"run": run, "cls": c}
            for g, (ing, rest) in groups.items():
                for nm, m in (("in", ing), ("rest", rest)):
                    mm = pos & m
                    r[f"{g}_{nm}_n"] = int(mm.sum())
                    r[f"{g}_{nm}_fnr"] = (~call[mm, j]).mean() if mm.sum() else np.nan
            fnr.append(r)

        # (3) exact-score F1 thresholds
        th_x = f1_exact_thresholds(yv, pv)[ex.NIH_IDX]
        for ds, y_, p_ in (("nih7", yt[:, ex.NIH_IDX], pt[:, ex.NIH_IDX]), ("chexpert", y, p)):
            for name, th in (("grid", th_all["f1"]), ("exact", th_x)):
                dec = ex.Decisions(y_, p_, th, np.ones(len(y_), bool), range(y_.shape[1]))(np.ones(len(y_)))
                f1x.append({"run": run, "set": ds, "search": name, **dec})

        # (4) NIH test intercept after Platt, 14 and 7 classes, with and without an oracle val->test prior shift
        cal = rv.fit_map("platt", logit(pv), yv)(logit(pt))
        pi_v, pi_t = yv.mean(0), yt.mean(0)
        shifted = sig(logit(cal) + (logit(pi_t) - logit(pi_v)))
        r = {"run": run}
        for nm, q in (("platt", cal), ("platt_oracle", shifted)):
            for cs, idx in (("14", list(range(14))), ("7", list(ex.NIH_IDX))):
                r[f"citl_{nm}_{cs}"] = rv.macro_cal(yt[:, idx], q[:, idx])["cal_intercept"]
        icp.append(r)
        if run in ("dn_asl", "p2_swin_asl", "p2_hyb_asl"):
            neg = pt[yt == 0]
            asl.append({"run": run, "share_below_m": float((neg < 0.05).mean()),
                        "q01": float(np.quantile(neg, 0.01)), "median": float(np.median(neg))})
        print(run, "done", flush=True)

    out = {"uncertain_rescue": resc, "perclass_fnr": fnr, "f1_exact": f1x, "nih_intercepts": icp, "asl_margin": asl}
    for k, v in out.items():
        pd.DataFrame(v).to_csv(os.path.join(a.out, f"{k}.csv"), index=False)
    r = pd.DataFrame(resc)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(r.groupby("threshold")[["ud_paper", "ud_confirmed_only", "share_pos_with_uncertain",
                                     "lt40_paper_crude", "lt40_conf_crude", "lt40_conf_std", "PA_paper_crude", "PA_conf_crude", "PA_conf_std",
                                     "share_unc_lt40_in", "share_unc_lt40_rest", "share_unc_PA_in", "share_unc_PA_rest"]].agg(["min", "max"]).round(3).T)
        s = r[r.threshold == "sens80"]
        print("sens80 confirmed-only std gap CI lower > 0:  lt40", int((s.lt40_conf_std_lo > 0).sum()), " PA", int((s.PA_conf_std_lo > 0).sum()))
        f = pd.DataFrame(fnr).groupby("cls").mean(numeric_only=True)
        print(f.round(3).to_string())
        print(pd.DataFrame(f1x).groupby(["set", "search"])[["tpr", "underdiag"]].agg(["min", "max"]).round(3))
        print(pd.DataFrame(icp).round(3).to_string(index=False))
        print(pd.DataFrame(asl).round(4).to_string(index=False))
    print(f"Wrote {a.out}/")


if __name__ == "__main__":
    main()
