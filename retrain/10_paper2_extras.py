"""
10_paper2_extras.py  —  paper 2: the three analyses that complete 07-09 (CPU).

1. nih7     NIH test values restricted to the 7 classes shared with CheXpert, so the external drop is
            like-for-like: macro AUROC and ECE (raw, NIH-validation Platt), patient bootstrap CIs, and the
            CheXpert-minus-NIH difference (independent bootstraps of the two test sets).
2. casemix  Is the higher underdiagnosis of <40 and PA patients on CheXpert (sens-0.8 thresholds) explained
            by case mix? Underdiagnosis by number of positive shared findings (1, 2, 3+), and the group gap
            directly standardised to the overall finding-count distribution, with patient bootstrap CIs.
3. holm     Holm-adjusted p-values for the subgroup gaps of 08 and 09. Family = one run x one threshold
            (08) or one run (09) x one metric, over all groups of all attributes reported for it.

  python 10_paper2_extras.py --out eval/p2_extras
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


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.maximum.accumulate((len(q) - np.arange(len(q))) * q[order])
    res = np.empty(len(q))
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


# For a binary attribute, group-vs-rest for one level is the same test as for the other level, so only one
# level enters the Holm family (holm_dup = False); the other level gets the same adjusted p-value.
HOLM_DUP = {("sex", "M"), ("view", "AP")}


def add_holm(df, family_cols):
    df["holm_dup"] = [(a, g) in HOLM_DUP for a, g in zip(df.attribute, df.group)]
    for c in [c for c in df.columns if c.endswith("_gap_p")]:
        df[c + "_holm"] = np.nan
        for _, idx in df.groupby(family_cols).groups.items():
            keep = [i for i in idx if not df.at[i, "holm_dup"]]
            df.loc[keep, c + "_holm"] = holm(df.loc[keep, c].values)
            for i in idx:
                if df.at[i, "holm_dup"]:
                    partner = [j for j in keep if df.at[j, "attribute"] == df.at[i, "attribute"]]
                    assert len(partner) == 1 and np.isclose(df.at[partner[0], c], df.at[i, c], equal_nan=True)
                    df.at[i, c + "_holm"] = df.at[partner[0], c + "_holm"]
    return df


def nih7(runs, results, n_boot, seed, ext_overall):
    rows = []
    o = pd.read_csv(ext_overall)
    for run in runs:
        _, yv, pv, _ = ev.load_run(os.path.join(results, run), "val")
        df, y, p, _ = ev.load_run(os.path.join(results, run), "test")
        pid = df["Patient ID"].values
        allm = np.ones(len(df), bool)
        y7 = y[:, ex.NIH_IDX]
        mets = {m: ex.Metrics(y7, rc.fit_apply(m, yv, pv, p)[0][:, ex.NIH_IDX], allm, range(7)) for m in ["none", "platt"]}
        one = np.ones(len(df))
        pt = {m: mets[m](one) for m in mets}
        bs = {m: [] for m in mets}
        for w in ev.boot_weights(pid, n_boot, seed):
            for m in mets:
                bs[m].append(mets[m](w))
        row = {"run": run}
        for m in mets:
            for k in ["auroc", "ece"]:
                key = f"nih_{k}_{m}" if k == "ece" else "nih_auroc"
                if k == "auroc" and m != "none":
                    continue
                row[key] = pt[m][k]
                row[key + "_lo"], row[key + "_hi"] = ev.ci([b[k] for b in bs[m]])
        # CheXpert sample values from 09 (same metric definitions); difference CI from independent bootstraps
        # approximated by combining the two percentile half-widths in quadrature
        for s in ["sample", "valid"]:
            e = o[(o.run == run) & (o.set == s)].set_index("method")
            row[f"chex_{s}_auroc"] = e.loc["none", "auroc"]
            row[f"chex_{s}_auroc_lo"], row[f"chex_{s}_auroc_hi"] = e.loc["none", "auroc_lo"], e.loc["none", "auroc_hi"]
            row[f"chex_{s}_ece_none"] = e.loc["none", "ece"]
            row[f"chex_{s}_ece_platt"] = e.loc["platt", "ece"]
            d = e.loc["none", "auroc"] - row["nih_auroc"]
            hw = np.sqrt(((e.loc["none", "auroc_hi"] - e.loc["none", "auroc_lo"]) / 2) ** 2
                         + ((row["nih_auroc_hi"] - row["nih_auroc_lo"]) / 2) ** 2)
            row[f"drop_{s}"], row[f"drop_{s}_lo"], row[f"drop_{s}_hi"] = d, d - hw, d + hw
        rows.append(row)
        print(f"nih7 {run}: NIH AUROC {row['nih_auroc']:.3f} -> CheXpert sample {row['chex_sample_auroc']:.3f}; "
              f"ECE platt NIH {row['nih_ece_platt']:.3f} -> {row['chex_sample_ece_platt']:.3f}", flush=True)
    return pd.DataFrame(rows)


def casemix(runs, results, pred_dir, chex, n_boot, seed):
    d0 = ex.load_labels(chex)["sample"]
    rows = []
    for run in runs:
        pr = pd.read_csv(os.path.join(pred_dir, f"{run}.csv.gz")).set_index("key")
        _, yv, pv, _ = ev.load_run(os.path.join(results, run), "val")
        th = ex.sens_thresholds(yv, pv)[ex.NIH_IDX]
        d = d0[d0.key.isin(pr.index)].reset_index(drop=True)
        y = np.stack(d.y.values)
        p = pr.loc[d.key, [f"p_{c}" for c in ev.CLASSES]].values[:, ex.NIH_IDX]
        npos = (y == 1).sum(axis=1)
        keep = npos >= 1
        stratum = np.minimum(npos, 3)                         # 1, 2, 3+
        missed = ~(p >= th).any(axis=1)
        pid = d.pid.values
        for attr, grp in (("age", "<40"), ("view", "PA")):
            g = d[attr].values
            known = pd.notna(g) & keep
            ing = known & (g == grp)
            rest = known & (g != grp)
            strata = [1, 2, 3]
            ref = np.array([(known & (stratum == s)).sum() for s in strata], float)
            ref /= ref.sum()                                  # standard: overall finding-count distribution

            def stats(w):
                out = {}
                for name, m in (("in", ing), ("rest", rest)):
                    r = []
                    for s in strata:
                        ms = m & (stratum == s)
                        ww = w[ms]
                        r.append((ww * missed[ms]).sum() / ww.sum() if ww.sum() > 0 else np.nan)
                    out[name] = r
                    wm = w[m]
                    out[name + "_crude"] = (wm * missed[m]).sum() / wm.sum()
                    out[name + "_std"] = float(np.nansum(np.array(r) * ref))
                return out

            pt = stats(np.ones(len(d)))
            bs = [stats(w) for w in ev.boot_weights(pid, n_boot, seed)]
            row = {"run": run, "attribute": attr, "group": grp,
                   "share_1_in": float((ing & (stratum == 1)).sum() / ing.sum()),
                   "share_1_rest": float((rest & (stratum == 1)).sum() / rest.sum())}
            for kind in ["crude", "std"]:
                gap = pt[f"in_{kind}"] - pt[f"rest_{kind}"]
                row[f"gap_{kind}"] = gap
                row[f"gap_{kind}_lo"], row[f"gap_{kind}_hi"] = ev.ci([b[f"in_{kind}"] - b[f"rest_{kind}"] for b in bs])
                row[f"gap_{kind}_p"] = ev.p_two_sided([b[f"in_{kind}"] - b[f"rest_{kind}"] for b in bs])
            for i, s in enumerate(strata):
                row[f"ud_in_{s}"], row[f"ud_rest_{s}"] = pt["in"][i], pt["rest"][i]
                gs = [b["in"][i] - b["rest"][i] for b in bs]
                row[f"gap_{s}"] = pt["in"][i] - pt["rest"][i]
                row[f"gap_{s}_lo"], row[f"gap_{s}_hi"] = ev.ci(gs)
            rows.append(row)
            print(f"casemix {run} {grp}: crude gap {row['gap_crude']:+.3f}, standardised {row['gap_std']:+.3f} "
                  f"({row['gap_std_lo']:+.3f} to {row['gap_std_hi']:+.3f}); one-finding share {row['share_1_in']:.2f} vs {row['share_1_rest']:.2f}",
                  flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pred_dir", default="results/_dl_ext/ext_preds")
    ap.add_argument("--chex", default="results/_dl_chexpert")
    ap.add_argument("--eval", default="eval")
    ap.add_argument("--out", default="eval/p2_extras")
    ap.add_argument("--runs", nargs="*", default=ex.RUNS)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", nargs="*", default=["holm", "nih7", "casemix"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if "holm" in a.parts:
        t = add_holm(pd.read_csv(os.path.join(a.eval, "p2_thresholds/thresholds_subgroups.csv")), ["run", "threshold"])
        t.to_csv(os.path.join(a.out, "thresholds_subgroups_holm.csv"), index=False)
        e = add_holm(pd.read_csv(os.path.join(a.eval, "p2_external/external_subgroups.csv")), ["run"])
        e.to_csv(os.path.join(a.out, "external_subgroups_holm.csv"), index=False)
        print("holm: NIH underdiag gaps p<0.05 raw", int((t.underdiag_gap_p < 0.05).sum()),
              "-> Holm", int((t.underdiag_gap_p_holm < 0.05).sum()), "of", len(t), flush=True)
    if "nih7" in a.parts:
        nih7(a.runs, a.results, a.n_boot, a.seed, os.path.join(a.eval, "p2_external/external_overall.csv")).to_csv(
            os.path.join(a.out, "nih7_vs_chexpert.csv"), index=False)
    if "casemix" in a.parts:
        casemix(a.runs, a.results, a.pred_dir, a.chex, a.n_boot, a.seed).to_csv(os.path.join(a.out, "casemix.csv"), index=False)
    print(f"Wrote {a.out}/")


if __name__ == "__main__":
    main()
