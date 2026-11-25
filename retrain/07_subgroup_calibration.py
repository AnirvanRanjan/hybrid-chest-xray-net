"""
07_subgroup_calibration.py  —  paper 2: is a model that is calibrated overall also calibrated
within age and sex groups, before and after recalibration?

Methods (all fitted on VALIDATION predictions, scored on TEST):
  none          raw probabilities
  platt_global  per-class Platt map fitted on the whole validation split
  platt_group   per-class Platt map fitted separately on each group's validation images
                (falls back to the global map for a class with < MIN_VAL_POS positive validation
                images in that group; the number of fallbacks is reported)

Per subgroup (sex F/M; age <40, 40-59, 60+), macro-averaged over the classes eligible for that
attribute (>= 10 positive test patients in every group, as in 03_evaluate.py):
  ece      expected calibration error, 10 equal-count bins within the subgroup
  citl     calibration-in-the-large as a probability difference: mean predicted - observed rate
           (> 0 over-predicts, < 0 under-predicts)
  gaps     group minus all other groups of the attribute, for ece and citl
All with patient-level bootstrap 95% CIs (2,000 resamples of patients, bin edges fixed).

  python 07_subgroup_calibration.py --run_dirs results/dn_bce results/dn_asl results/hyb_tuned_320 --out eval/subcal
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
METHODS = ["none", "platt_global", "platt_group"]
MIN_VAL_POS = 20


def platt(lv, yv):
    return ev.logistic_fit(np.column_stack([np.ones(len(lv)), lv]), yv.astype(float))


def recalibrate(dfv, yv, pv, dft, pt):
    """Returns {method: test probabilities} and the number of group-map fallbacks."""
    lv, lt = ev._logit(pv), ev._logit(pt)
    glob = [platt(lv[:, k], yv[:, k]) for k in range(len(ev.CLASSES))]
    out = {"none": pt.copy(), "platt_global": np.empty_like(pt)}
    for k, (b0, b1) in enumerate(glob):
        out["platt_global"][:, k] = rc.sigmoid(b0 + b1 * lt[:, k])
    # group maps: one prediction matrix per attribute (each test image gets its own group's map)
    group_out, fallbacks = {}, 0
    for attr in ["sex", "age"]:
        g_v = [(name, m) for a, name, m, _ in ev.subgroup_masks(dfv) if a == attr]
        g_t = {name: m for a, name, m, _ in ev.subgroup_masks(dft) if a == attr}
        arr = out["platt_global"].copy()   # images with unknown group keep the global map
        for name, mv in g_v:
            mt = g_t[name]
            for k in range(len(ev.CLASSES)):
                if yv[mv, k].sum() >= MIN_VAL_POS:
                    b0, b1 = platt(lv[mv, k], yv[mv, k])
                    arr[mt, k] = rc.sigmoid(b0 + b1 * lt[mt, k])
                else:
                    fallbacks += 1
        group_out[attr] = arr
    return out, group_out, fallbacks


class SubCal:
    """Weighted ECE (bins fixed from the subgroup's own quantiles) and mean(p) - mean(y)."""
    def __init__(self, y, p, n_bins=10):
        self.y, self.p = y.astype(float), p
        self.idx = ev.quantile_bins(p, n_bins)
        self.nb = self.idx.max() + 1

    def __call__(self, w):
        tot = w.sum()
        if tot == 0:
            return np.nan, np.nan
        yw = np.bincount(self.idx, w * self.y, minlength=self.nb)
        pw = np.bincount(self.idx, w * self.p, minlength=self.nb)
        return float(np.abs(yw - pw).sum() / tot), float((pw.sum() - yw.sum()) / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="eval/subcal")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for rd in a.run_dirs:
        run = os.path.basename(os.path.normpath(rd))
        dfv, yv, pv, _ = ev.load_run(rd, "val")
        dft, yt, pt, _ = ev.load_run(rd, "test")
        flat, grouped, fb = recalibrate(dfv, yv, pv, dft, pt)
        groups = ev.subgroup_masks(dft)
        elig = ev.eligible_classes(dft, yt, groups)
        print(f"{run}: group-map fallbacks to global = {fb}; eligible classes "
              + ", ".join(f"{k}={len(v)}" for k, v in elig.items()))

        # one SubCal per (method, group or rest, class); masks select the images
        cells = {}
        for m in METHODS:
            for attr, name, gm, rest in groups:
                p = grouped[attr] if m == "platt_group" else flat[m]
                for k in elig[attr]:
                    cells[(m, attr, name, "g", k)] = (gm, SubCal(yt[gm, k], p[gm, k]))
                    cells[(m, attr, name, "r", k)] = (rest, SubCal(yt[rest, k], p[rest, k]))

        def macro(w):
            res = {}
            for (m, attr, name, side, k), (mask, f) in cells.items():
                res.setdefault((m, attr, name, side), []).append(f(w[mask]))
            return {key: np.nanmean(np.array(v), axis=0) for key, v in res.items()}   # (ece, citl)

        point = macro(np.ones(len(dft)))
        boots = [macro(w) for w in ev.boot_weights(dft["Patient ID"].values, a.n_boot, a.seed)]
        for m in METHODS:
            for attr, name, gm, _ in groups:
                g, r = point[(m, attr, name, "g")], point[(m, attr, name, "r")]
                bg = np.array([b[(m, attr, name, "g")] for b in boots])
                br = np.array([b[(m, attr, name, "r")] for b in boots])
                row = {"run": run, "method": m, "attr": attr, "group": name, "n_images": int(gm.sum()),
                       "n_classes": len(elig[attr]), "ece": g[0], "citl": g[1],
                       "gap_ece": g[0] - r[0], "gap_citl": g[1] - r[1]}
                row["ece_lo"], row["ece_hi"] = ev.ci(bg[:, 0])
                row["citl_lo"], row["citl_hi"] = ev.ci(bg[:, 1])
                row["gap_ece_lo"], row["gap_ece_hi"] = ev.ci(bg[:, 0] - br[:, 0])
                row["gap_citl_lo"], row["gap_citl_hi"] = ev.ci(bg[:, 1] - br[:, 1])
                rows.append(row)

    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(a.out, "subgroup_calibration.csv"), index=False)
    cols = ["run", "method", "attr", "group", "ece", "ece_lo", "ece_hi", "citl", "gap_citl", "gap_citl_lo", "gap_citl_hi"]
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(s[cols].round(4).to_string(index=False))
    print(f"Wrote {a.out}/")


if __name__ == "__main__":
    main()
