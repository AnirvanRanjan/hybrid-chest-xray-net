"""
03_evaluate.py  —  statistics for the paper, from the prediction files 02_train.py writes.
CPU only; runs on the Mac. Reads pred_val.csv / pred_test.csv / thresholds_val.npy and
never retrains or re-tunes anything. Thresholds always come from VALIDATION.

  run      one model: per-class + macro AUROC with patient-level bootstrap 95% CIs,
           F1 / TPR / FPR at the validation thresholds, calibration (Brier, ECE with
           quantile bins, calibration slope and intercept, reliability curves), and
           age / sex subgroups with bootstrap CIs on the gaps.
  compare  several models on the same images: paired patient-level bootstrap of the
           AUROC differences against a reference run (baseline table, sampler ablation).

Bootstrap: 2,000 resamples of PATIENTS (not images) with replacement; an image's weight is
the number of times its patient was drawn. Percentile 95% CIs. DeLong is not used because
it treats images as independent, and NIH patients contribute many images each.

Paper outputs (--paper_dir ../paper/generated --tag E): tables, figures and a results.tex of
named values (\resdef{E.test.macro_auroc}{0.812}) that the manuscript reads through \res{...}.
Nothing in the paper is typed by hand; a missing value shows as a red placeholder.

Examples
  python 03_evaluate.py run --run_dir results/hyb_320 --tag E --paper_dir ../paper/generated
  python 03_evaluate.py run --run_dir results/hyb_320 --split val      # validation, labelled as such
  python 03_evaluate.py compare --run_dirs results/hyb_320 results/hyb_floor results/hyb_none \\
        results/dn_bce results/dn_asl --ref hyb_320 --out eval/compare --tag baselines --paper_dir ../paper/generated
"""
import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
# 60-79 and 80+ were merged into 60+ because the official test list has only 31 patients aged 80+
# (decided on group sizes alone, 21 Oct 2026; with four bins only 1 of 14 classes met MIN_POS_PATIENTS).
AGE_BINS = [(0, 40, "<40"), (40, 60, "40-59"), (60, 101, "60+")]
MIN_POS_PATIENTS = 10   # per class, in every group of an attribute, for subgroup per-class results and gaps
KEY = {"<40": "lt40", "40-59": "40to59", "60+": "60plus", "F": "F", "M": "M"}  # LaTeX-safe names
SERIES, REF_GREY, INK = "#2a78d6", "#9a9890", "#0b0b0b"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_run(run_dir, split):
    df = pd.read_csv(os.path.join(run_dir, f"pred_{split}.csv"))
    df = df.sort_values("Image Index").reset_index(drop=True)
    y = df[[f"y_{c}" for c in CLASSES]].values.astype(int)
    p = df[[f"p_{c}" for c in CLASSES]].values.astype(float)
    th = np.load(os.path.join(run_dir, "thresholds_val.npy"))
    return df, y, p, th


def age_group(age):
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    for lo, hi, name in AGE_BINS:
        out[(age >= lo) & (age < hi)] = name
    return out


# ----------------------------------------------------------------------------
# Weighted metrics. w = bootstrap multiplicity per image (all ones = the plain estimate);
# a subgroup is w * mask, so the same code serves the full set and every subgroup.
# ----------------------------------------------------------------------------
class FastAUC:
    """Weighted Mann-Whitney AUROC (ties count 1/2). Scores are ranked once; each
    bootstrap replicate is then three bincounts. Equals sklearn's roc_auc_score at w = 1."""
    def __init__(self, y, p):
        self.y = y.astype(float)
        _, self.inv = np.unique(p, return_inverse=True)
        self.g = self.inv.max() + 1

    def __call__(self, w):
        pos = np.bincount(self.inv, w * self.y, minlength=self.g)
        neg = np.bincount(self.inv, w * (1 - self.y), minlength=self.g)
        den = pos.sum() * neg.sum()
        if den == 0:
            return np.nan
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / den)


def rates(y, b, w):
    """TPR, FPR, F1 for binary predictions b, weighted."""
    tp = (w * y * b).sum(); fn = (w * y * (1 - b)).sum()
    fp = (w * (1 - y) * b).sum(); tn = (w * (1 - y) * (1 - b)).sum()
    tpr = tp / (tp + fn) if tp + fn > 0 else np.nan
    fpr = fp / (fp + tn) if fp + tn > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else np.nan
    return tpr, fpr, f1


def boot_weights(patient_ids, n_boot, seed):
    codes, uniq = pd.factorize(patient_ids)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        yield counts[codes].astype(float)


def ci(samples):
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan, np.nan
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def p_two_sided(diffs):
    """Bootstrap two-sided p-value for H0: difference = 0, floored at 1/n."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    return float(max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1.0 / len(d)))


# ----------------------------------------------------------------------------
# Calibration (point estimates)
# ----------------------------------------------------------------------------
def quantile_bins(p, n_bins):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)


def ece_quantile(y, p, n_bins=10):
    idx = quantile_bins(p, n_bins)
    e = 0.0
    for b in np.unique(idx):
        m = idx == b
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logistic_fit(X, y, offset=0.0, iters=50):
    """Newton-Raphson for P(y=1) = sigmoid(X @ beta + offset)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta + offset)))
        H = (X * (mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(len(beta)), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def calibration_row(y, p):
    lp = _logit(p)
    ones = np.ones((len(y), 1))
    slope = logistic_fit(np.column_stack([ones, lp]), y)[1]   # y ~ a + b * logit(p)
    citl = logistic_fit(ones, y, offset=lp)[0]                # y ~ a + offset(logit p): calibration-in-the-large
    return {"brier": float(np.mean((p - y) ** 2)), "ece_q10": ece_quantile(y, p),
            "cal_slope": float(slope), "cal_intercept": float(citl),
            "mean_pred": float(p.mean()), "prevalence": float(y.mean())}


def plot_reliability(y, p, path, title, n_bins=10):
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for k, ax in enumerate(axes.flat):
        if k >= len(CLASSES):
            ax.axis("off")
            continue
        idx = quantile_bins(p[:, k], n_bins)
        mp = np.array([p[idx == b, k].mean() for b in np.unique(idx)])
        obs = np.array([y[idx == b, k].mean() for b in np.unique(idx)])
        lim = max(mp.max(), obs.max()) * 1.08
        ax.plot([0, lim], [0, lim], color=REF_GREY, lw=1, ls="--", zorder=1)
        ax.plot(mp, obs, color=SERIES, lw=2, marker="o", ms=4.5, zorder=2,
                markeredgecolor="white", markeredgewidth=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_title(CLASSES[k].replace("_", " "), fontsize=10, color=INK)
        ax.tick_params(labelsize=7, colors="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    fig.supxlabel("Mean predicted probability (decile bin)", fontsize=10)
    fig.supylabel("Observed frequency", fontsize=10)
    fig.suptitle(f"{title}\nAxes scaled per panel; dashed line = perfect calibration", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Named values for the manuscript
# ----------------------------------------------------------------------------
def f3(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def fp(p):
    return "n/a" if not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def fsgn(x):
    """Signed 3-decimal number with a true minus sign for LaTeX text mode."""
    return "n/a" if not np.isfinite(x) else f"{x:+.3f}".replace("-", "$-$")


def f2(x):
    """2-decimal number with a true minus sign for LaTeX text mode."""
    return "n/a" if not np.isfinite(x) else f"{x:.2f}".replace("-", "$-$")


def fp_text(p):
    """p-value as a complete LaTeX phrase for running text: '$p = 0.019$' or '$p < 0.001$'."""
    return "n/a" if not np.isfinite(p) else ("$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$")


def fint(n):
    return f"{int(n):,}".replace(",", "{,}")


def pretty(c):
    return "Macro average" if c == "macro" else c.replace("_", " ")


class Results:
    """Collects name -> formatted string; written as \\resdef{name}{value} lines."""
    def __init__(self, prefix):
        self.prefix, self.vals = prefix, {}

    def __setitem__(self, name, value):
        self.vals[f"{self.prefix}.{name}"] = value

    def write(self, path, source):
        lines = [f"% generated by 03_evaluate.py from {source} -- do not edit by hand"]
        lines += [f"\\resdef{{{k}}}{{{v}}}" for k, v in self.vals.items()]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


def write_lines(path, lines, source):
    with open(path, "w") as f:
        f.write("\n".join([f"% generated by 03_evaluate.py from {source} -- do not edit by hand"] + lines) + "\n")


def copy_to_paper(out, files, paper_dir, tag):
    if not paper_dir:
        return
    dst = os.path.join(paper_dir, tag)
    os.makedirs(dst, exist_ok=True)
    for f in files:
        if os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(out, f), os.path.join(dst, f))
    print(f"Copied {len(files)} paper files to {dst}/")


# ----------------------------------------------------------------------------
# `run`: one model
# ----------------------------------------------------------------------------
def subgroup_masks(df):
    sex = df["sex"].astype(str)
    ag = age_group(pd.to_numeric(df["age_clean"], errors="coerce"))
    groups = [("sex", s, (sex == s).values, (sex.isin(["M", "F"]) & (sex != s)).values) for s in ["F", "M"]]
    known_age = ag.notna().values
    for _, _, name in AGE_BINS:
        m = (ag == name).fillna(False).values.astype(bool)
        groups.append(("age", name, m, known_age & ~m))   # gap = group minus all other known ages
    return groups


def eligible_classes(df, y, groups):
    """Per attribute: classes with >= MIN_POS_PATIENTS positive patients in EVERY group of that attribute.
    Smaller cells give degenerate bootstrap intervals (one positive patient -> a 'tight' CI around 0 or 1)."""
    pid = df["Patient ID"].values
    out = {}
    for attr in dict.fromkeys(g[0] for g in groups):
        masks = [m for a_, _, m, _ in groups if a_ == attr]
        out[attr] = [k for k in range(len(CLASSES))
                     if all(len(np.unique(pid[m & (y[:, k] == 1)])) >= MIN_POS_PATIENTS for m in masks)]
    return out


def eval_one(y, p, th, w, aucs, groups, elig):
    """All bootstrapped quantities for one weight vector -> flat dict."""
    out = {}
    b = (p >= th[None, :]).astype(float)
    per = [aucs[k](w) for k in range(len(CLASSES))]
    for k, c in enumerate(CLASSES):
        out[("all", "all", c, "auroc")] = per[k]
        tpr, fpr, f1 = rates(y[:, k], b[:, k], w)
        out[("all", "all", c, "tpr")], out[("all", "all", c, "fpr")], out[("all", "all", c, "f1")] = tpr, fpr, f1
    out[("all", "all", "macro", "auroc")] = np.nanmean(per)
    out[("all", "all", "macro", "f1")] = np.nanmean([out[("all", "all", c, "f1")] for c in CLASSES])
    b05 = (p >= 0.5).astype(float)   # fixed-threshold reference for the threshold-search claim
    out[("all", "all", "macro", "f1_at_0.5")] = np.nanmean([rates(y[:, k], b05[:, k], w)[2] for k in range(len(CLASSES))])
    for attr, name, m, comp in groups:
        for tag, mask in (("", m), ("~", comp)):
            wg = w * mask
            per = [aucs[k](wg) for k in range(len(CLASSES))]
            for k, c in enumerate(CLASSES):
                tpr, fpr, _ = rates(y[:, k], b[:, k], wg)
                out[(attr, tag + name, c, "auroc")] = per[k]
                out[(attr, tag + name, c, "tpr")], out[(attr, tag + name, c, "fpr")] = tpr, fpr
            sel = [per[k] for k in elig[attr]]   # same classes for every group of the attribute
            with np.errstate(all="ignore"):
                out[(attr, tag + name, "macro", "auroc")] = np.nanmean(sel) if np.isfinite(sel).any() else np.nan
    return out


def cmd_run(a):
    df, y, p, th = load_run(a.run_dir, a.split)
    run = os.path.basename(os.path.normpath(a.run_dir))
    out = a.out or os.path.join("eval", run)
    os.makedirs(out, exist_ok=True)
    pre = f"{a.split}_"
    aucs = [FastAUC(y[:, k], p[:, k]) for k in range(len(CLASSES))]
    groups = subgroup_masks(df)
    elig = eligible_classes(df, y, groups)
    for attr, ks in elig.items():
        print(f"{attr}: {len(ks)}/14 classes have >= {MIN_POS_PATIENTS} positive patients in every group; "
              f"excluded from subgroup results: {[CLASSES[k] for k in range(14) if k not in ks]}")

    point = eval_one(y, p, th, np.ones(len(df)), aucs, groups, elig)
    boots = [eval_one(y, p, th, w, aucs, groups, elig)
             for w in boot_weights(df["Patient ID"].values, a.n_boot, a.seed)]
    keys = list(point)
    B = {k: np.array([bt[k] for bt in boots]) for k in keys}

    # --- overall metrics with CIs
    rows = []
    for k in keys:
        if k[0] == "all":
            lo, hi = ci(B[k])
            rows.append({"class": k[2], "metric": k[3], "value": point[k], "ci_lo": lo, "ci_hi": hi})
    overall = pd.DataFrame(rows)
    npos = {c: int(y[:, i].sum()) for i, c in enumerate(CLASSES)}
    overall["n_pos"] = overall["class"].map(npos)
    overall["threshold_val"] = overall["class"].map(dict(zip(CLASSES, th)))
    overall.to_csv(os.path.join(out, pre + "metrics_ci.csv"), index=False)

    # --- subgroups and gaps
    sg_rows, gap_rows = [], []
    for attr, name, m, _ in groups:
        for c in CLASSES + ["macro"]:
            ci_k = CLASSES.index(c) if c != "macro" else None
            for met in (["auroc"] if c == "macro" else ["auroc", "tpr", "fpr"]):
                kg, kc = (attr, name, c, met), (attr, "~" + name, c, met)
                lo, hi = ci(B[kg])
                sg_rows.append({"attribute": attr, "group": name, "class": c, "metric": met,
                                "value": point[kg], "ci_lo": lo, "ci_hi": hi,
                                "n_images": int(m.sum()), "n_patients": int(df.loc[m, "Patient ID"].nunique()),
                                "n_pos": int(y[m, ci_k].sum()) if ci_k is not None else np.nan,
                                "n_pos_patients": int(df.loc[m & (y[:, ci_k] == 1), "Patient ID"].nunique()) if ci_k is not None else np.nan,
                                "eligible": c == "macro" or ci_k in elig[attr]})
                if attr == "sex" and name == "M":
                    continue   # F - M already covers it
                if c != "macro" and ci_k not in elig[attr]:
                    continue   # too few positive patients for a meaningful gap
                d = B[kg] - B[kc]
                lo, hi = ci(d)
                vs = "M" if attr == "sex" else "other ages"
                gap_rows.append({"attribute": attr, "group": name, "vs": vs, "class": c, "metric": met,
                                 "gap": point[kg] - point[kc], "ci_lo": lo, "ci_hi": hi, "p_boot": p_two_sided(d)})
    sg = pd.DataFrame(sg_rows)
    sg.to_csv(os.path.join(out, pre + "subgroups.csv"), index=False)
    pd.DataFrame(gap_rows).to_csv(os.path.join(out, pre + "subgroup_gaps.csv"), index=False)
    plot_subgroups(sg, point[("all", "all", "macro", "auroc")],
                   os.path.join(out, pre + "subgroup_macro_auroc.png"), f"{run} ({a.split})")

    # --- calibration
    cal = pd.DataFrame([{"class": c, **calibration_row(y[:, k], p[:, k])} for k, c in enumerate(CLASSES)])
    cal = pd.concat([cal, pd.DataFrame([{"class": "macro", **cal.drop(columns="class").mean().to_dict()}])])
    cal.to_csv(os.path.join(out, pre + "calibration.csv"), index=False)
    plot_reliability(y, p, os.path.join(out, pre + "reliability.png"), f"Reliability, {run} ({a.split})")

    src = f"{run}/pred_{a.split}.csv"
    tag = a.tag or run
    R = Results(f"{tag}.{a.split}")
    write_latex_auroc(overall, os.path.join(out, pre + "auroc_table.tex"), src)
    write_latex_calibration(cal, os.path.join(out, pre + "calibration_table.tex"), src)
    write_latex_subgroups(sg, pd.DataFrame(gap_rows), os.path.join(out, pre + "subgroup_table.tex"), src)
    plot_roc_pr(y, p, os.path.join(out, pre + "roc.png"), os.path.join(out, pre + "pr.png"), f"{run} ({a.split})")
    plot_cooccurrence(y, (p >= th[None, :]).astype(int), os.path.join(out, pre + "cooccurrence.png"))

    # --- named values
    R["n_images"], R["n_patients"] = fint(len(df)), fint(df["Patient ID"].nunique())
    o = overall.set_index(["class", "metric"])
    for c in CLASSES + ["macro"]:
        key = "macro" if c == "macro" else c
        for met in ["auroc", "f1"] + ([] if c == "macro" else ["tpr", "fpr"]):
            R[f"{met}.{key}"] = f3(o.loc[(c, met), "value"])
            R[f"{met}_lo.{key}"], R[f"{met}_hi.{key}"] = f3(o.loc[(c, met), "ci_lo"]), f3(o.loc[(c, met), "ci_hi"])
        if c != "macro":
            R[f"threshold.{c}"], R[f"n_pos.{c}"] = f"{th[CLASSES.index(c)]:.2f}", fint(npos[c])
    R["macro_auroc"], R["macro_f1"] = R.vals[f"{R.prefix}.auroc.macro"], R.vals[f"{R.prefix}.f1.macro"]
    m05 = o.loc[("macro", "f1_at_0.5")]
    R["macro_f1_at_05"], R["macro_f1_at_05_lo"], R["macro_f1_at_05_hi"] = f3(m05["value"]), f3(m05["ci_lo"]), f3(m05["ci_hi"])
    ranked = overall[(overall["metric"] == "auroc") & (overall["class"] != "macro")].sort_values("value")
    for name, row in [("worst", ranked.iloc[0]), ("second_worst", ranked.iloc[1]),
                      ("second_best", ranked.iloc[-2]), ("best", ranked.iloc[-1])]:
        R[f"{name}_class"], R[f"{name}_auroc"] = pretty(row["class"]), f3(row["value"])
    R["threshold_min"], R["threshold_max"] = f"{th.min():.2f}", f"{th.max():.2f}"
    ratio = (cal["mean_pred"] / cal["prevalence"])[cal["class"] != "macro"]
    R["overprediction_min"], R["overprediction_max"] = f"{ratio.min():.1f}", f"{ratio.max():.1f}"
    R["n_classes_overpredicted"] = str(int((ratio > 1).sum()))
    bpred = (p >= th[None, :]).astype(int)
    R["n_multilabel_true"], R["n_multilabel_pred"] = fint((y.sum(1) >= 2).sum()), fint((bpred.sum(1) >= 2).sum())
    for name, m in (("true", y), ("pred", bpred)):
        co = (m.T @ m).astype(float); np.fill_diagonal(co, 0); iu = np.triu_indices(len(CLASSES), 1)
        i = int(np.argmax(co[iu]))
        R[f"top_pair_{name}"] = f"{pretty(CLASSES[iu[0][i]])} and {pretty(CLASSES[iu[1][i]])}"
        R[f"top_pair_{name}_n"] = fint(co[iu][i])
    for _, r in cal.iterrows():
        for m in ["brier", "ece_q10", "cal_slope", "cal_intercept"]:
            R[f"{m}.{r['class']}"] = f3(r[m])
    for r in sg[(sg["class"] == "macro")].itertuples():
        R[f"sub.{r.attribute}.{KEY[r.group]}.auroc"] = f3(r.value)
        R[f"sub.{r.attribute}.{KEY[r.group]}.auroc_lo"], R[f"sub.{r.attribute}.{KEY[r.group]}.auroc_hi"] = f3(r.ci_lo), f3(r.ci_hi)
        R[f"sub.{r.attribute}.{KEY[r.group]}.n_images"] = fint(r.n_images)
    for attr, ks in elig.items():
        R[f"sub.{attr}.excluded"] = ", ".join(pretty(CLASSES[k]) for k in range(14) if k not in ks) or "none"
        R[f"sub.{attr}.n_classes"] = str(len(ks))
    gaps = pd.DataFrame(gap_rows)
    for _, r in gaps[(gaps["class"] != "macro") & (gaps["metric"] == "tpr")].iterrows():
        k = f"gap.{r['attribute']}.{KEY[r['group']]}.tpr.{r['class']}"
        R[k], R[k + "_lo"], R[k + "_hi"], R[k + "_p"] = fsgn(r["gap"]), fsgn(r["ci_lo"]), fsgn(r["ci_hi"]), fp_text(r["p_boot"])
    for _, r in gaps[(gaps["class"] == "macro") & (gaps["metric"] == "auroc")].iterrows():
        k = f"gap.{r['attribute']}.{KEY[r['group']]}"
        R[k], R[k + "_lo"], R[k + "_hi"], R[k + "_p"] = fsgn(r["gap"]), fsgn(r["ci_lo"]), fsgn(r["ci_hi"]), fp_text(r["p_boot"])
    extra_run_values(a.run_dir, tag, out)
    R.write(os.path.join(out, pre + "results.tex"), src)

    mac = overall[(overall["class"] == "macro") & (overall["metric"] == "auroc")].iloc[0]
    summary = {"run": run, "split": a.split, "n_images": len(df), "n_patients": int(df["Patient ID"].nunique()),
               "n_boot": a.n_boot, "seed": a.seed,
               "macro_auroc": mac["value"], "macro_auroc_ci": [mac["ci_lo"], mac["ci_hi"]],
               "macro_f1_at_val_thresholds": point[("all", "all", "macro", "f1")],
               "macro_brier": float(cal.iloc[-1]["brier"]), "macro_ece_q10": float(cal.iloc[-1]["ece_q10"]),
               "note": ("thresholds were chosen on this split; F1/TPR/FPR are optimistic"
                        if a.split == "val" else "thresholds chosen on validation, applied to test")}
    with open(os.path.join(out, pre + "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}/")
    copy_to_paper(out, [pre + f for f in ["results.tex", "auroc_table.tex", "calibration_table.tex",
                                          "subgroup_table.tex", "reliability.png", "subgroup_macro_auroc.png",
                                          "roc.png", "pr.png", "cooccurrence.png"]]
                  + ["run_results.tex", "dataset_table.tex", "history.png"], a.paper_dir, tag)


def extra_run_values(run_dir, tag, out):
    """Split sizes, class counts and training history: written once per run (not per split)."""
    R = Results(tag)
    p = os.path.join(run_dir, "split_info.json")
    if os.path.exists(p):
        for k, v in json.load(open(p)).items():
            R[f"split.{k}"] = fint(v)
        si = json.load(open(p))
        for sp in ("train", "val", "test"):
            R[f"split.{sp}_images_per_patient"] = f"{si[f'{sp}_images'] / si[f'{sp}_patients']:.1f}"
    p = os.path.join(run_dir, "split_class_counts.csv")
    if os.path.exists(p):
        cnt = pd.read_csv(p, index_col="class")
        for split in cnt.columns:
            for c in cnt.index:
                key = c.replace(" ", "_")
                R[f"count.{split}.{key}"] = fint(cnt.loc[c, split])
                if c != "images":
                    R[f"prev.{split}.{key}"] = f"{100 * cnt.loc[c, split] / cnt.loc['images', split]:.2f}"
        write_latex_dataset(cnt, os.path.join(out, "dataset_table.tex"), f"{os.path.basename(run_dir)}/split_class_counts.csv")
    p = os.path.join(run_dir, "history.csv")
    if os.path.exists(p):
        h = pd.read_csv(p)
        best = h.loc[h["val_auroc"].idxmax()]
        R["hist.epochs_run"], R["hist.best_epoch"] = str(len(h)), str(int(best["epoch"]))
        R["hist.best_val_auroc"], R["hist.first_val_auroc"] = f3(best["val_auroc"]), f3(h["val_auroc"].iloc[0])
        R["hist.first_loss"], R["hist.last_loss"] = f"{h['train_loss'].iloc[0]:.4f}", f"{h['train_loss'].iloc[-1]:.4f}"
        plot_history(h, os.path.join(out, "history.png"))
    R.write(os.path.join(out, "run_results.tex"), os.path.basename(run_dir))


def plot_subgroups(sg, overall_auc, path, title):
    d = sg[(sg["class"] == "macro") & (sg["metric"] == "auroc")].reset_index(drop=True)
    labels = [f"{'Sex' if r.attribute == 'sex' else 'Age'} {r.group}  (n={r.n_images:,})" for r in d.itertuples()]
    ypos = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(d) + 1.4))
    ax.axvline(overall_auc, color=REF_GREY, lw=1, ls="--", zorder=1)
    ax.errorbar(d["value"], ypos, xerr=[d["value"] - d["ci_lo"], d["ci_hi"] - d["value"]],
                fmt="o", color=SERIES, ms=8, elinewidth=2, capsize=0, zorder=2,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Macro AUROC (95% patient-level bootstrap CI)", fontsize=9)
    ax.set_title(f"{title}\ndashed line = overall macro AUROC", fontsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def write_latex_auroc(overall, path, src):
    o = overall.set_index(["class", "metric"])
    lines = [r"\begin{tabular}{@{}lrcccc@{}}", r"\toprule",
             r"Class & $n_{\mathrm{pos}}$ & AUROC (95\% CI) & F1 & Sensitivity & Threshold \\", r"\midrule"]
    for c in CLASSES + ["macro"]:
        if c == "macro":
            lines.append(r"\midrule")
        a = o.loc[(c, "auroc")]
        npos = "" if c == "macro" else fint(a["n_pos"])
        tpr = "" if c == "macro" else f3(o.loc[(c, "tpr"), "value"])
        thr = "" if c == "macro" else f"{a['threshold_val']:.2f}"
        lines.append(f"{pretty(c)} & {npos} & {f3(a['value'])} ({f3(a['ci_lo'])}--{f3(a['ci_hi'])})"
                     f" & {f3(o.loc[(c, 'f1'), 'value'])} & {tpr} & {thr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_lines(path, lines, src)


def write_latex_calibration(cal, path, src):
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Class & Prevalence & Mean predicted & Brier & ECE & Slope / intercept \\", r"\midrule"]
    for r in cal.itertuples():
        if r[1] == "macro":
            lines.append(r"\midrule")
        lines.append(f"{pretty(r[1])} & {f3(r.prevalence)} & {f3(r.mean_pred)} & {f3(r.brier)} & {f3(r.ece_q10)}"
                     f" & {f2(r.cal_slope)} / {f2(r.cal_intercept)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_lines(path, lines, src)


def write_latex_subgroups(sg, gaps, path, src):
    m = sg[(sg["class"] == "macro") & (sg["metric"] == "auroc")]
    g = gaps[(gaps["class"] == "macro") & (gaps["metric"] == "auroc")].set_index(["attribute", "group"])
    lines = [r"\begin{tabular}{@{}llrccc@{}}", r"\toprule",
             r"Attribute & Group & Images & Macro AUROC (95\% CI) & Gap (95\% CI) & $p$ \\", r"\midrule"]
    for r in m.itertuples():
        if (r.attribute, r.group) in g.index:
            x = g.loc[(r.attribute, r.group)]
            gap = f"{fsgn(x['gap'])} ({fsgn(x['ci_lo'])} to {fsgn(x['ci_hi'])})"
            pv = fp(x["p_boot"]).replace("<", "$<$")
        else:
            gap, pv = "reference", ""
        grp = r.group.replace("<", "$<$").replace("-", "--")
        lines.append(f"{'Sex' if r.attribute == 'sex' else 'Age'} & {grp} & {fint(r.n_images)} & "
                     f"{f3(r.value)} ({f3(r.ci_lo)}--{f3(r.ci_hi)}) & {gap} & {pv} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_lines(path, lines, src)


def write_latex_dataset(cnt, path, src):
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Class & All (\%) & Train & Validation & Test \\", r"\midrule"]
    for c in CLASSES + ["No Finding", "images"]:
        if c in ("No Finding", "images"):
            lines.append(r"\midrule")
        name = "Images" if c == "images" else pretty(c)
        allc = fint(cnt.loc[c, "all"])
        if c != "images":
            allc += f" ({100 * cnt.loc[c, 'all'] / cnt.loc['images', 'all']:.2f})"
        lines.append(f"{name} & {allc} & {fint(cnt.loc[c, 'train'])} & {fint(cnt.loc[c, 'val'])} & {fint(cnt.loc[c, 'test'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_lines(path, lines, src)


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, colors="#52514e")
    ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)


def plot_roc_pr(y, p, roc_path, pr_path, title):
    for kind, path in (("roc", roc_path), ("pr", pr_path)):
        fig, axes = plt.subplots(4, 4, figsize=(11, 11))
        for k, ax in enumerate(axes.flat):
            if k >= len(CLASSES):
                ax.axis("off")
                continue
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y[:, k], p[:, k])
                ax.plot([0, 1], [0, 1], color=REF_GREY, lw=1, ls="--")
                ax.plot(fpr, tpr, color=SERIES, lw=2)
                stat = f"AUROC {FastAUC(y[:, k], p[:, k])(np.ones(len(y))):.3f}"
            else:
                prec, rec, _ = precision_recall_curve(y[:, k], p[:, k])
                ax.axhline(y[:, k].mean(), color=REF_GREY, lw=1, ls="--")
                ax.plot(rec, prec, color=SERIES, lw=2)
                stat = f"AP {average_precision_score(y[:, k], p[:, k]):.3f}"
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.set_title(f"{pretty(CLASSES[k])}\n{stat}", fontsize=9, color=INK)
            _style(ax)
        fig.supxlabel("False positive rate" if kind == "roc" else "Recall", fontsize=10)
        fig.supylabel("True positive rate" if kind == "roc" else "Precision", fontsize=10)
        ref = "dashed line = chance" if kind == "roc" else "dashed line = prevalence (chance AP)"
        fig.suptitle(f"{'ROC' if kind == 'roc' else 'Precision-recall'} curves, {title}; {ref}", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def plot_cooccurrence(y, b, path):
    names = [pretty(c) for c in CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    for ax, mat, ttl in ((axes[0], y, "Ground-truth labels"), (axes[1], b, "Predictions at validation thresholds")):
        co = (mat.T @ mat).astype(float)
        np.fill_diagonal(co, np.nan)
        im = ax.imshow(co, cmap="Blues")
        vmax = np.nanmax(co)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if i != j:
                    ax.text(j, i, f"{int(co[i, j])}", ha="center", va="center", fontsize=5.5,
                            color="white" if co[i, j] > 0.6 * vmax else INK)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{ttl} (test set, image counts)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_history(h, path):
    best = h.loc[h["val_auroc"].idxmax(), "epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, lab in ((axes[0], "train_loss", "Training loss"), (axes[1], "val_auroc", "Validation macro AUROC")):
        ax.axvline(best, color=REF_GREY, lw=1, ls="--")
        ax.plot(h["epoch"], h[col], color=SERIES, lw=2, marker="o", ms=4, markeredgecolor="white", markeredgewidth=1)
        ax.set_xlabel("Epoch", fontsize=9); ax.set_title(lab, fontsize=10)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _style(ax)
    fig.suptitle("dashed line = epoch with the best validation AUROC (checkpoint used)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# `compare`: paired bootstrap across runs on identical images
# ----------------------------------------------------------------------------
def cmd_compare(a):
    runs = {}
    for d in a.run_dirs:
        df, y, p, _ = load_run(d, a.split)
        runs[os.path.basename(os.path.normpath(d))] = (df, y, p)
    names = list(runs)
    ref = a.ref or names[0]
    assert ref in runs, f"--ref {ref} not among {names}"
    df0, y0, _ = runs[ref]
    for n, (df, y, _) in runs.items():
        assert df["Image Index"].equals(df0["Image Index"]), f"{n}: different images from {ref}"
        assert (y == y0).all(), f"{n}: labels differ from {ref}"

    aucs = {n: [FastAUC(y0[:, k], p[:, k]) for k in range(len(CLASSES))] for n, (_, _, p) in runs.items()}
    point = {n: np.array([f(np.ones(len(df0))) for f in aucs[n]]) for n in names}
    boots = {n: [] for n in names}
    for w in boot_weights(df0["Patient ID"].values, a.n_boot, a.seed):   # same w for every model = paired
        for n in names:
            boots[n].append([f(w) for f in aucs[n]])
    boots = {n: np.array(v) for n, v in boots.items()}

    rows = []
    for n in names:
        for k, c in enumerate(CLASSES + ["macro"]):
            if c == "macro":
                v, s, s_ref, v_ref = np.nanmean(point[n]), np.nanmean(boots[n], 1), np.nanmean(boots[ref], 1), np.nanmean(point[ref])
            else:
                v, s, s_ref, v_ref = point[n][k], boots[n][:, k], boots[ref][:, k], point[ref][k]
            lo, hi = ci(s)
            row = {"run": n, "class": c, "auroc": v, "ci_lo": lo, "ci_hi": hi}
            if n != ref:
                dlo, dhi = ci(s - s_ref)
                row.update({"diff_vs_ref": v - v_ref, "diff_ci_lo": dlo, "diff_ci_hi": dhi,
                            "p_boot": p_two_sided(s - s_ref)})
            rows.append(row)
    out = a.out or os.path.join("eval", "compare")
    os.makedirs(out, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, f"{a.split}_compare_vs_{ref}.csv"), index=False)
    labels = dict(zip(names, a.labels)) if a.labels else {n: n.replace("_", r"\_") for n in names}
    tag = a.tag or "compare"
    R = Results(f"{tag}.{a.split}")
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Model & Macro AUROC (95\% CI) & $\Delta$ vs.\ reference (95\% CI) & $p$ \\", r"\midrule"]
    for r in res[res["class"] == "macro"].itertuples():
        R[f"{r.run}.macro_auroc"], R[f"{r.run}.macro_auroc_lo"], R[f"{r.run}.macro_auroc_hi"] = f3(r.auroc), f3(r.ci_lo), f3(r.ci_hi)
        if r.run == ref:
            diff, pv = "reference", ""
        else:
            diff, pv = f"{fsgn(r.diff_vs_ref)} ({fsgn(r.diff_ci_lo)} to {fsgn(r.diff_ci_hi)})", fp(r.p_boot).replace("<", "$<$")
            R[f"{r.run}.diff"], R[f"{r.run}.diff_lo"], R[f"{r.run}.diff_hi"] = fsgn(r.diff_vs_ref), fsgn(r.diff_ci_lo), fsgn(r.diff_ci_hi)
            R[f"{r.run}.diff_p"] = fp_text(r.p_boot)
            pc = res[(res.run == r.run) & (res["class"] != "macro")]
            R[f"{r.run}.n_classes_higher"] = str(int((pc.diff_ci_lo > 0).sum()))   # CI entirely above the reference
            R[f"{r.run}.n_classes_lower"] = str(int((pc.diff_ci_hi < 0).sum()))
            for _, q in pc.iterrows():
                R[f"{r.run}.diff.{q['class']}"] = fsgn(q["diff_vs_ref"])
        lines.append(f"{labels[r.run]} & {f3(r.auroc)} ({f3(r.ci_lo)}--{f3(r.ci_hi)}) & {diff} & {pv} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    src = f"pred_{a.split}.csv of " + ", ".join(names)
    write_lines(os.path.join(out, f"{a.split}_compare_table.tex"), lines, src)
    # per-class AUROC, one column per model
    # calibration of each model (point estimates; macro = mean over classes)
    lines_c = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
               r"Model & Mean predicted / prevalence & Brier & ECE & Slope & Intercept \\", r"\midrule"]
    for n in names:
        _, yy, pp = runs[n]
        cal = pd.DataFrame([calibration_row(yy[:, k], pp[:, k]) for k in range(len(CLASSES))]).mean()
        for m in ["brier", "ece_q10", "cal_slope", "cal_intercept", "mean_pred", "prevalence"]:
            R[f"{n}.{m}"] = f3(cal[m])
        R[f"{n}.overprediction"] = f"{cal['mean_pred'] / cal['prevalence']:.1f}"
        lines_c.append(f"{labels[n]} & {cal['mean_pred']:.3f} / {cal['prevalence']:.3f} & {f3(cal['brier'])} & "
                       f"{f3(cal['ece_q10'])} & {f2(cal['cal_slope'])} & {f2(cal['cal_intercept'])} \\\\")
    lines_c += [r"\bottomrule", r"\end{tabular}"]
    w = res.pivot(index="class", columns="run", values="auroc").loc[CLASSES + ["macro"], names]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(names) + "@{}}", r"\toprule",
             "Class & " + " & ".join(labels[n] for n in names) + r" \\", r"\midrule"]
    for c, row in w.iterrows():
        if c == "macro":
            lines.append(r"\midrule")
        best = row.max()
        cells = [(r"\textbf{%s}" % f3(v)) if v == best else f3(v) for v in row]
        lines.append(f"{pretty(c)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_lines(os.path.join(out, f"{a.split}_compare_perclass_table.tex"), lines, src)
    write_lines(os.path.join(out, f"{a.split}_compare_calibration_table.tex"), lines_c, src)
    R.write(os.path.join(out, f"{a.split}_results.tex"), src)
    copy_to_paper(out, [f"{a.split}_{f}" for f in ["compare_table.tex", "compare_perclass_table.tex",
                                                   "compare_calibration_table.tex", "results.tex"]],
                  a.paper_dir, tag)
    with pd.option_context("display.width", 200):
        print(res[res["class"] == "macro"].round(4).to_string(index=False))
    print(f"Reference: {ref}. p_boot is unadjusted; correct for multiple comparisons when reporting per class.")
    print(f"Wrote {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--run_dir", required=True)
    c = sub.add_parser("compare"); c.add_argument("--run_dirs", nargs="+", required=True)
    c.add_argument("--ref", default=None, help="run name (folder name) the others are compared against")
    c.add_argument("--labels", nargs="+", default=None, help="LaTeX row labels, one per --run_dirs entry")
    for s in (r, c):
        s.add_argument("--split", choices=["test", "val"], default="test")
        s.add_argument("--out", default=None)
        s.add_argument("--n_boot", type=int, default=2000)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--tag", default=None, help="name prefix for paper values, e.g. E or baselines")
        s.add_argument("--paper_dir", default=None, help="also copy tables/figures/values to <paper_dir>/<tag>/")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    main()
