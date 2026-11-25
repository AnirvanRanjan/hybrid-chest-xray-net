"""
08_thresholds.py  —  paper 2: do thresholds chosen on NIH validation transfer across subgroups of the NIH test split?

Thresholds (per class, fitted on VALIDATION only, applied unchanged to TEST):
  f1      F1-optimal threshold of 02_train.py (thresholds_val.npy)
  sens80  largest threshold with validation sensitivity >= 0.8 (exact; 09_external.sens_thresholds)
Per subgroup (sex F/M; age <40, 40-59, 60+), macro over the classes with >= 10 positive test patients
in every group of the attribute (as in 03_evaluate.py):
  tpr, fpr     sensitivity and false-positive rate at the threshold
  underdiag    among images with >= 1 positive finding, the share where no class reaches its threshold,
               i.e. the image would be read as "No Finding" (Seyyed-Kalantari et al. 2021), over all 14 classes
Gaps = group minus the other known groups of the attribute; patient-level bootstrap 95% CIs
(2,000 resamples, paired) and bootstrap p-values. Also the overall values for every run.

  python 08_thresholds.py --run_dirs results/dn_bce results/dn_asl ... --out eval/p2_thresholds
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
METS = ["tpr", "fpr", "underdiag"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="eval/p2_thresholds")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    overall, sub, ths = [], [], []
    for rd in a.run_dirs:
        run = os.path.basename(os.path.normpath(rd))
        _, y_val, p_val, th_f1 = ev.load_run(rd, "val")
        df, y, p, _ = ev.load_run(rd, "test")
        th = {"f1": th_f1, "sens80": ex.sens_thresholds(y_val, p_val)}
        for t, v in th.items():
            ths.append({"run": run, "threshold": t, **dict(zip(ev.CLASSES, v))})
        pid = df["Patient ID"].values
        groups = ev.subgroup_masks(df)
        elig = ev.eligible_classes(df, y, groups)
        allk = list(range(len(ev.CLASSES)))
        one = np.ones(len(df))
        objs = {("all", "all", t, "in"): ex.Decisions(y, p, th[t], np.ones(len(df), bool), allk) for t in th}
        for attr, name, m, rest in groups:
            for t in th:
                objs[(attr, name, t, "in")] = ex.Decisions(y, p, th[t], m, elig[attr])
                objs[(attr, name, t, "rest")] = ex.Decisions(y, p, th[t], rest, elig[attr])
        pt = {k: o(one) for k, o in objs.items()}
        bs = {k: [] for k in objs}
        for w in ev.boot_weights(pid, a.n_boot, a.seed):
            for k, o in objs.items():
                bs[k].append(o(w))
        for t in th:
            k = ("all", "all", t, "in")
            row = {"run": run, "threshold": t}
            for met in METS:
                row[met] = pt[k][met]
                row[f"{met}_lo"], row[f"{met}_hi"] = ev.ci([b[met] for b in bs[k]])
            overall.append(row)
        for attr, name, m, _ in groups:
            for t in th:
                ki, kr = (attr, name, t, "in"), (attr, name, t, "rest")
                row = {"run": run, "threshold": t, "attribute": attr, "group": name,
                       "n_images": int(m.sum()), "n_patients": len(np.unique(pid[m])),
                       "classes": ";".join(ev.CLASSES[c] for c in elig[attr])}
                for met in METS:
                    row[met] = pt[ki][met]
                    row[f"{met}_lo"], row[f"{met}_hi"] = ev.ci([b[met] for b in bs[ki]])
                    gap = [bi[met] - br[met] for bi, br in zip(bs[ki], bs[kr])]
                    row[f"{met}_gap"] = pt[ki][met] - pt[kr][met]
                    row[f"{met}_gap_lo"], row[f"{met}_gap_hi"] = ev.ci(gap)
                    row[f"{met}_gap_p"] = ev.p_two_sided(gap)
                sub.append(row)
        o = pd.DataFrame(overall)
        print(o[o.run == run][["run", "threshold"] + METS].round(3).to_string(index=False, header=run == overall[0]["run"]), flush=True)
    pd.DataFrame(overall).to_csv(os.path.join(a.out, "thresholds_overall.csv"), index=False)
    pd.DataFrame(sub).to_csv(os.path.join(a.out, "thresholds_subgroups.csv"), index=False)
    pd.DataFrame(ths).to_csv(os.path.join(a.out, "thresholds_fitted.csv"), index=False)
    print(f"Wrote {a.out}/")


if __name__ == "__main__":
    main()
