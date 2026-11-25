"""
09_external.py  —  paper 2: external validation of the frozen NIH grid models on CheXpert (CPU).

Reads the CheXpert predictions written by 09_infer_external.py (Kaggle GPU) and evaluates them with
the protocol locked in paper2/PLAN.md §4 before any CheXpert result was seen:
  * 7 findings shared with NIH (Effusion = CheXpert "Pleural Effusion"); frontal images only.
  * Nothing is fitted on CheXpert. Recalibration maps (none / temperature / per-class Platt / isotonic,
    06_recalibrate.fit_apply) are fitted on each run's NIH VALIDATION predictions (all 14 classes, as
    in 06) and applied unchanged. Thresholds: NIH-validation F1-optimal (thresholds_val.npy) and a
    fixed-sensitivity point (largest threshold with NIH-validation sensitivity >= 0.8, exact,
    rule p >= t). Temperature and Platt are monotone per class, so thresholds are applied to the raw
    probabilities.
Sets:
  valid   CheXpert validation set, radiologist labels (valid.csv; no uncertain labels).
  sample  frontal train images of a seeded 25% patient sample (chexpert-probe), CheXbert impression
          labels: 1 positive, 0 / blank negative, -1 (uncertain) masked for that class only.
Per run, set and method: macro (over the 7 classes) AUROC, ECE (10 equal-count bins), Brier,
calibration intercept (in the large) and slope, mean predicted / prevalence; TPR and FPR at both
thresholds; underdiagnosis rate = share of images with >= 1 positive shared finding for which no
shared finding reaches its threshold (Seyyed-Kalantari et al. 2021, restricted to the 7 classes;
images with no positive but an uncertain label are excluded).
Patient-level bootstrap 95% CIs (resamples of patients, ECE bin edges fixed), and for each
recalibration method the paired difference in ECE versus none.
Subgroups (sample set only): sex, age (<40, 40-59, 60+), race (White, Asian, Black; others/unknown
excluded from that attribute), view (AP, PA). Gaps = group minus the other known groups of the
attribute. Macro metrics over the classes with >= 10 positive patients in every group of the attribute.

  python 09_external.py --pred_dir results/_dl_ext/ext_preds --results results --chex results/_dl_chexpert --out eval/p2_external
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

SHARED = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
          "Edema": "Edema", "Effusion": "Pleural Effusion", "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax"}
NIH_IDX = [ev.CLASSES.index(c) for c in SHARED]
RUNS = ["dn_bce", "p2_dn_wbce", "p2_dn_focal", "dn_asl", "p2_swin_bce", "p2_swin_wbce", "p2_swin_focal", "p2_swin_asl",
        "hyb_tuned", "p2_hyb_wbce", "p2_hyb_focal", "p2_hyb_asl"]
RACES = ["White", "Asian", "Black"]
SENS_TARGET = 0.8


def sens_thresholds(y_val, p_val, target=SENS_TARGET):
    """Per class: the largest threshold t with NIH-validation sensitivity (p >= t) >= target, i.e. a
    validation positive's score (exact, no grid: a 0.01 grid is too coarse for BCE models, whose rare-class
    probabilities are mostly < 0.05)."""
    out = np.zeros(p_val.shape[1])
    for k in range(p_val.shape[1]):
        pos = np.sort(p_val[y_val[:, k] == 1, k])
        out[k] = pos[int(np.floor((1 - target) * len(pos) + 1e-9))]
        assert (pos >= out[k]).mean() >= target
    return out


def load_labels(chex):
    """Frontal CheXpert images with labels (7 shared classes; -1 = uncertain), patient, age, sex, race, view."""
    meta = pd.read_csv(os.path.join(chex, "chexpert_meta.csv"))
    meta["key"] = meta.path_to_image.str.replace(".jpg", "", regex=False)
    meta = meta[meta.frontal_lateral == "Frontal"]
    lab = pd.read_csv(os.path.join(chex, "chexpert_labels_impression.csv"))
    lab["key"] = lab.path_to_image.str.replace(".jpg", "", regex=False)
    val = pd.read_csv(os.path.join(chex, "valid.csv"))
    val["key"] = val.Path.str.replace("CheXpert-v1.0-small/", "", regex=False).str.replace(".jpg", "", regex=False)
    val = val[val["Frontal/Lateral"] == "Frontal"]
    sets = {}
    for name, src in [("valid", val), ("sample", lab)]:
        d = meta.merge(src[["key"] + list(SHARED.values())], on="key", how="inner")
        d = d[d.split == ("valid" if name == "valid" else "train")]
        y = d[list(SHARED.values())].fillna(0).values.astype(float)       # blank = negative
        assert set(np.unique(y)) <= {-1.0, 0.0, 1.0}
        if name == "valid":
            assert (y >= 0).all() and len(d) == len(val), "valid.csv: unexpected uncertain labels or unmatched rows"
        age = pd.to_numeric(d.age, errors="coerce").where(lambda s: s <= 100)
        sets[name] = pd.DataFrame({
            "key": d.key.values, "pid": d.deid_patient_id.values, "y": list(y),
            "age": ev.age_group(age.reset_index(drop=True)).values,
            "sex": d.sex.map({"Female": "F", "Male": "M"}).values,
            "race": d.race.where(d.race.isin(RACES)).values,
            "view": d.ap_pa.where(d.ap_pa.isin(["AP", "PA"])).values})
    return sets


class Metrics:
    """Macro metrics over classes for one prediction matrix, under bootstrap weights w (ones = point estimate).
    Uncertain labels (-1) are masked per class; ECE bins are fixed from the (sub)set's own point sample."""
    def __init__(self, y, p, mask, classes):
        self.cls = []
        for k in classes:
            m = mask & (y[:, k] >= 0)
            yk, pk = y[m, k], p[m, k]
            idx = ev.quantile_bins(pk, 10)
            self.cls.append((m, yk, pk, ev.FastAUC(yk, pk), idx, idx.max() + 1))

    def __call__(self, w):
        auc, ece, brier = [], [], []
        for m, yk, pk, fa, idx, nb in self.cls:
            wk = w[m]
            tot = wk.sum()
            auc.append(fa(wk))
            yw = np.bincount(idx, wk * yk, minlength=nb)
            pw = np.bincount(idx, wk * pk, minlength=nb)
            ece.append(np.abs(yw - pw).sum() / tot if tot > 0 else np.nan)
            brier.append((wk * (pk - yk) ** 2).sum() / tot if tot > 0 else np.nan)
        return {"auroc": np.nanmean(auc), "ece": np.nanmean(ece), "brier": np.nanmean(brier)}


class Decisions:
    """TPR / FPR at fixed per-class thresholds (macro over classes) and the underdiagnosis rate."""
    def __init__(self, y, p, th, mask, classes):
        self.y, self.b, self.mask, self.classes = y, (p >= th).astype(float), mask, classes
        self.ud_mask = mask & (y == 1).any(axis=1)   # >= 1 positive (images with only uncertain labels excluded)
        self.nopred = ~self.b.astype(bool).any(axis=1)

    def __call__(self, w):
        tpr, fpr = [], []
        for k in self.classes:
            m = self.mask & (self.y[:, k] >= 0)
            t, f, _ = ev.rates(self.y[m, k], self.b[m, k], w[m])
            tpr.append(t); fpr.append(f)
        wu = w[self.ud_mask]
        ud = (wu * self.nopred[self.ud_mask]).sum() / wu.sum() if wu.sum() > 0 else np.nan
        return {"tpr": np.nanmean(tpr), "fpr": np.nanmean(fpr), "underdiag": ud}


def point_rows(y, p, mask):
    rows = []
    for k in range(len(SHARED)):
        m = mask & (y[:, k] >= 0)
        r = ev.calibration_row(y[m, k], p[m, k])
        r["cal_intercept"] = citl(y[m, k], p[m, k])   # ev's Newton fit diverges under large prevalence shift
        r.update(cls=list(SHARED)[k], n=int(m.sum()), n_pos=int(y[m, k].sum()),
                 auroc=ev.FastAUC(y[m, k], p[m, k])(np.ones(m.sum())))
        rows.append(r)
    return rows


def citl(y, p):
    """Calibration-in-the-large a in y ~ sigmoid(a + logit p), by bisection on the score equation
    sum(y) = sum(sigmoid(a + logit p)), which is monotone in a (always converges)."""
    lp, target = ev._logit(p), float(y.sum())
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + lp)))).sum() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def eligible(pid, y, masks):
    return [k for k in range(len(SHARED))
            if all(len(np.unique(pid[m & (y[:, k] == 1)])) >= ev.MIN_POS_PATIENTS for m in masks)]


def summarise(boots, point):
    out = {}
    for k, v in point.items():
        out[k] = v
        out[f"{k}_lo"], out[f"{k}_hi"] = ev.ci([b[k] for b in boots])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--chex", default="results/_dl_chexpert")
    ap.add_argument("--out", default="eval/p2_external")
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = load_labels(a.chex)
    for s, d in sets.items():
        y = np.stack(d.y.values)
        print(f"{s}: {len(d)} images, {d.pid.nunique()} patients; positives "
              + ", ".join(f"{c} {int((y[:, j] == 1).sum())}" for j, c in enumerate(SHARED))
              + f"; uncertain masked {int((y == -1).sum())}", flush=True)

    overall, perclass, sub, fitted = [], [], [], []
    for run in a.runs:
        f = os.path.join(a.pred_dir, f"{run}.csv.gz")
        if not os.path.exists(f):
            print(f"{run}: no CheXpert predictions, skipped"); continue
        pr = pd.read_csv(f).set_index("key")
        _, y_val, p_val, th_f1 = ev.load_run(os.path.join(a.results, run), "val")
        th_s = sens_thresholds(y_val, p_val)
        fitted.append({"run": run, **{f"th_f1_{c}": th_f1[i] for c, i in zip(SHARED, NIH_IDX)},
                       **{f"th_sens_{c}": th_s[i] for c, i in zip(SHARED, NIH_IDX)}})
        for sname, d in sets.items():
            missing = ~d.key.isin(pr.index)   # sample: labels exist for all train images, only the sample was cached
            if sname == "valid":
                assert not missing.any(), f"{run}: {missing.sum()} CheXpert validation images have no prediction"
            d = d[~missing].reset_index(drop=True)
            y = np.stack(d.y.values)
            p14 = pr.loc[d.key, [f"p_{c}" for c in ev.CLASSES]].values.astype(float)
            preds = {m: rc.fit_apply(m, y_val, p_val, p14)[0][:, NIH_IDX] for m in rc.METHODS}
            raw = preds["none"]
            pid = d.pid.values
            allm = np.ones(len(d), bool)
            cls_all = list(range(len(SHARED)))
            # overall, with bootstrap
            mets = {m: Metrics(y, preds[m], allm, cls_all) for m in rc.METHODS}
            dec = {t: Decisions(y, raw, th[NIH_IDX], allm, cls_all) for t, th in (("f1", th_f1), ("sens80", th_s))}
            one = np.ones(len(d))
            pt = {m: mets[m](one) for m in rc.METHODS}
            pd_ = {t: dec[t](one) for t in dec}
            bm = {m: [] for m in rc.METHODS}; bd = {t: [] for t in dec}
            for w in ev.boot_weights(pid, a.n_boot, a.seed):
                for m in rc.METHODS:
                    bm[m].append(mets[m](w))
                for t in dec:
                    bd[t].append(dec[t](w))
            for m in rc.METHODS:
                row = {"run": run, "set": sname, "method": m, "n_images": len(d), "n_patients": len(np.unique(pid)),
                       **summarise(bm[m], pt[m])}
                pcr = pd.DataFrame(point_rows(y, preds[m], allm))
                row.update(cal_intercept=pcr.cal_intercept.mean(), cal_slope=pcr.cal_slope.mean(),
                           overprediction=pcr.mean_pred.mean() / pcr.prevalence.mean())
                if m != "none":
                    dif = [b["ece"] - b0["ece"] for b, b0 in zip(bm[m], bm["none"])]
                    row["ece_diff_vs_none"] = pt[m]["ece"] - pt["none"]["ece"]
                    row["ece_diff_lo"], row["ece_diff_hi"] = ev.ci(dif)
                    row["ece_diff_p"] = ev.p_two_sided(dif)
                overall.append(row)
                for r in pcr.to_dict("records"):
                    perclass.append({"run": run, "set": sname, "method": m, **r})
            for t in dec:
                overall.append({"run": run, "set": sname, "method": f"thresholds_{t}", "n_images": len(d),
                                "n_patients": len(np.unique(pid)), **summarise(bd[t], pd_[t])})
            print(f"{run}/{sname}: AUROC {pt['none']['auroc']:.3f}  ECE none {pt['none']['ece']:.3f} "
                  f"temp {pt['temperature']['ece']:.3f} platt {pt['platt']['ece']:.3f}  "
                  f"TPR@F1 {pd_['f1']['tpr']:.3f} underdiag {pd_['f1']['underdiag']:.3f}", flush=True)

            if sname != "sample":
                continue
            # subgroups: point values, CIs, and gaps (group minus the other known groups), paired bootstrap
            for attr, groups in (("sex", ["F", "M"]), ("age", [b[2] for b in ev.AGE_BINS]), ("race", RACES), ("view", ["AP", "PA"])):
                g = d[attr].values
                masks = {v: (g == v) for v in groups}
                known = np.isin(g, groups)
                elig = eligible(pid, y, list(masks.values()))
                if not elig:
                    continue
                objs = {}
                for v, m in masks.items():
                    objs[(v, "in")] = (Metrics(y, preds["none"], m, elig), Metrics(y, preds["platt"], m, elig),
                                       Decisions(y, raw, th_f1[NIH_IDX], m, elig), Decisions(y, raw, th_s[NIH_IDX], m, elig))
                    objs[(v, "rest")] = (Metrics(y, preds["none"], known & ~m, elig), Metrics(y, preds["platt"], known & ~m, elig),
                                         Decisions(y, raw, th_f1[NIH_IDX], known & ~m, elig), Decisions(y, raw, th_s[NIH_IDX], known & ~m, elig))

                def flat(o, w):
                    mn, mp, d1, d2 = o
                    a_, b_, c_, e_ = mn(w), mp(w), d1(w), d2(w)
                    return {"auroc": a_["auroc"], "ece_none": a_["ece"], "ece_platt": b_["ece"],
                            "tpr_f1": c_["tpr"], "fpr_f1": c_["fpr"], "underdiag_f1": c_["underdiag"],
                            "tpr_sens80": e_["tpr"], "fpr_sens80": e_["fpr"], "underdiag_sens80": e_["underdiag"]}
                ptv = {k: flat(o, one) for k, o in objs.items()}
                bs = {k: [] for k in objs}
                for w in ev.boot_weights(pid, a.n_boot, a.seed + 1):
                    for k, o in objs.items():
                        bs[k].append(flat(o, w))
                for v in groups:
                    row = {"run": run, "attribute": attr, "group": v, "n_images": int(masks[v].sum()),
                           "n_patients": len(np.unique(pid[masks[v]])), "classes": ";".join(list(SHARED)[k] for k in elig)}
                    for met, val in ptv[(v, "in")].items():
                        row[met] = val
                        row[f"{met}_lo"], row[f"{met}_hi"] = ev.ci([b[met] for b in bs[(v, "in")]])
                        gap = [bi[met] - br[met] for bi, br in zip(bs[(v, "in")], bs[(v, "rest")])]
                        row[f"{met}_gap"] = val - ptv[(v, "rest")][met]
                        row[f"{met}_gap_lo"], row[f"{met}_gap_hi"] = ev.ci(gap)
                        row[f"{met}_gap_p"] = ev.p_two_sided(gap)
                    sub.append(row)
            print(f"  {run}: subgroups done", flush=True)

    pd.DataFrame(overall).to_csv(os.path.join(a.out, "external_overall.csv"), index=False)
    pd.DataFrame(perclass).to_csv(os.path.join(a.out, "external_perclass.csv"), index=False)
    pd.DataFrame(sub).to_csv(os.path.join(a.out, "external_subgroups.csv"), index=False)
    pd.DataFrame(fitted).to_csv(os.path.join(a.out, "nih_val_thresholds.csv"), index=False)
    print(f"Wrote {a.out}/")


if __name__ == "__main__":
    main()
