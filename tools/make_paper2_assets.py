"""
make_assets.py  —  paper 2 (MIDL): every figure, table and number in the paper, built from the evaluation CSVs.

Nothing in midl/paper.tex is typed by hand: numbers come from midl/generated/values.tex as \\pv{key},
tables from midl/generated/*.tex, figures from midl/figures/*.pdf.

  python3 make_assets.py            (from paper2/; reads ../retrain/eval/)
"""
import os
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "..", "retrain", "eval")
OUT = os.path.join(HERE, "midl")
GEN, FIG = os.path.join(OUT, "generated"), os.path.join(OUT, "figures")

ARCH = {"densenet": "DenseNet-121", "swin": "Swin-B", "hybrid": "Hybrid"}
LOSSES = ["bce", "wbce", "focal", "asl"]
LOSS_NAME = {"bce": "BCE", "wbce": "wBCE", "focal": "Focal", "asl": "ASL"}
RUN = {("densenet", "bce"): "dn_bce", ("densenet", "wbce"): "p2_dn_wbce", ("densenet", "focal"): "p2_dn_focal",
       ("densenet", "asl"): "dn_asl", ("swin", "bce"): "p2_swin_bce", ("swin", "wbce"): "p2_swin_wbce",
       ("swin", "focal"): "p2_swin_focal", ("swin", "asl"): "p2_swin_asl", ("hybrid", "bce"): "hyb_tuned",
       ("hybrid", "wbce"): "p2_hyb_wbce", ("hybrid", "focal"): "p2_hyb_focal", ("hybrid", "asl"): "p2_hyb_asl"}
KEY = {v: f"{a[:3]}{l}" for (a, l), v in RUN.items()}          # LaTeX-safe: denbce, swiasl, hybfocal ...
# validated categorical palette (dataviz validate_palette.js, light: all checks pass; two slots < 3:1 contrast,
# so identity is never colour-alone: marker shape + legend + direct labels)
COL = {"bce": "#2a78d6", "wbce": "#eb6834", "focal": "#1baf7a", "asl": "#eda100"}
MARK = {"bce": "o", "wbce": "s", "focal": "^", "asl": "D"}
AMARK = {"densenet": "o", "swin": "s", "hybrid": "^"}
LABEL_POS = {"Effusion": (0.97, 0.62, "center"), "Atelectasis": (0.92, 1.45, "right"), "Pneumothorax": (0.9, 0.62, "right"),
             "Consolidation": (0.88, 1.25, "right"), "Cardiomegaly": (1.12, 0.85, "left"), "Edema": (0.9, 0.75, "right"),
             "Pneumonia": (1.12, 0.8, "left")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2})

V = {}


def unsign0(t):
    """'+0.000' / '-0.000' -> '0.000' (a rounded zero has no sign)."""
    return t[1:] if t[:1] in "+-" and float(t) == 0 else t


def pv(key, val, fmt="{:.3f}"):
    V[key] = unsign0(fmt.format(val)) if isinstance(val, (float, np.floating, int, np.integer)) else str(val)


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load():
    rec = pd.concat([pd.read_csv(os.path.join(EV, d, "recal_summary.csv")) for d in ["p2_recal_dn", "p2_recal_swin", "p2_recal_hyb"]])
    recpc = pd.concat([pd.read_csv(os.path.join(EV, d, "recal_perclass.csv")) for d in ["p2_recal_dn", "p2_recal_swin", "p2_recal_hyb"]])
    temp = pd.concat([pd.read_csv(os.path.join(EV, d, "temperatures.csv")) for d in ["p2_recal_dn", "p2_recal_swin", "p2_recal_hyb"]])
    cmp_ = []
    for f, ref in [("p2_compare/test_compare_vs_dn_bce.csv", "dn_bce"), ("p2_compare_swin/test_compare_vs_p2_swin_bce.csv", "p2_swin_bce"),
                   ("p2_compare_hyb/test_compare_vs_hyb_tuned.csv", "hyb_tuned")]:
        c = pd.read_csv(os.path.join(EV, f))
        c = c[c["class"] == "macro"].assign(ref=ref)
        cmp_.append(c)
    cmp_ = pd.concat(cmp_)
    ext = pd.read_csv(os.path.join(EV, "p2_external/external_overall.csv"))
    extpc = pd.read_csv(os.path.join(EV, "p2_external/external_perclass.csv"))
    n7 = pd.read_csv(os.path.join(EV, "p2_extras/nih7_vs_chexpert.csv"))
    th = pd.read_csv(os.path.join(EV, "p2_thresholds/thresholds_overall.csv"))
    thsub = pd.read_csv(os.path.join(EV, "p2_extras/thresholds_subgroups_holm.csv"))
    exsub = pd.read_csv(os.path.join(EV, "p2_extras/external_subgroups_holm.csv"))
    cm = pd.read_csv(os.path.join(EV, "p2_extras/casemix.csv"))
    sc = pd.read_csv(os.path.join(EV, "p2_subcal/subgroup_calibration.csv"))
    return rec, recpc, temp, cmp_, ext, extpc, n7, th, thsub, exsub, cm, sc


def load_calci():
    si = pd.read_csv(os.path.join(EV, "p2_calci/slope_intercept_ci.csv"))
    rb = pd.read_csv(os.path.join(EV, "p2_calci/reliability_bins.csv"))
    return si, rb


def calib_rows(rec, ext, si):
    """One row per run, set (nih / chexpert) and method (none / platt): Brier, ECE, slope, intercept, each with its CI."""
    rows = []
    for run in RUN.values():
        for m in ["none", "platt"]:
            r = rec[(rec.run == run) & (rec.method == m)].iloc[0]
            e = ext[(ext.run == run) & (ext.set == "sample") & (ext.method == m)].iloc[0]
            for s, src, ece in [("nih", r, "ece_q10"), ("chexpert", e, "ece")]:
                q = si[(si.run == run) & (si.set == s) & (si.method == m)].iloc[0]
                elo, ehi = ("ece_lo", "ece_hi")
                rows.append({"run": run, "set": s, "method": m,
                             "brier": src.brier, "brier_lo": src.brier_lo, "brier_hi": src.brier_hi,
                             "ece": src[ece], "ece_lo": src[elo], "ece_hi": src[ehi],
                             "slope": q.cal_slope, "slope_lo": q.cal_slope_lo, "slope_hi": q.cal_slope_hi,
                             "citl": q.cal_intercept, "citl_lo": q.cal_intercept_lo, "citl_hi": q.cal_intercept_hi})
    return pd.DataFrame(rows)


def values(rec, temp, cmp_, ext, extpc, n7, th, thsub, exsub, cm, sc):
    r = rec.set_index(["run", "method"])
    for run, k in KEY.items():
        for m in ["none", "temperature", "platt", "isotonic"]:
            pv(f"{k}.ece.{m}", r.loc[(run, m), "ece_q10"])
            pv(f"{k}.citl.{m}", r.loc[(run, m), "cal_intercept"], "{:+.2f}")
        pv(f"{k}.auroc", r.loc[(run, "none"), "auroc"])
        pv(f"{k}.auroc.iso", r.loc[(run, "isotonic"), "auroc"])
        pv(f"{k}.T", temp.set_index("run").loc[run, "T"], "{:.2f}")
        pv(f"{k}.overpred", r.loc[(run, "none"), "overprediction"], "{:.1f}")
        c = cmp_[cmp_.run == run].iloc[0]
        pv(f"{k}.auroc.lo", c.ci_lo); pv(f"{k}.auroc.hi", c.ci_hi)
        if pd.notna(c.diff_vs_ref):
            pv(f"{k}.dauc", c.diff_vs_ref, "{:+.3f}"); pv(f"{k}.dauc.lo", c.diff_ci_lo, "{:+.3f}")
            pv(f"{k}.dauc.hi", c.diff_ci_hi, "{:+.3f}"); pv(f"{k}.dauc.p", fp(c.p_boot))
            pv(f"{k}.dauc.peq", "$p<0.001$" if c.p_boot < 0.001 else f"$p={c.p_boot:.3f}$")
        n = n7.set_index("run").loc[run]
        pv(f"{k}.nih7.auroc", n.nih_auroc); pv(f"{k}.nih7.ece.none", n.nih_ece_none); pv(f"{k}.nih7.ece.platt", n.nih_ece_platt)
        for s in ["sample", "valid"]:
            e = ext[(ext.run == run) & (ext.set == s)].set_index("method")
            pv(f"{k}.{s}.auroc", e.loc["none", "auroc"]); pv(f"{k}.{s}.auroc.lo", e.loc["none", "auroc_lo"]); pv(f"{k}.{s}.auroc.hi", e.loc["none", "auroc_hi"])
            for m in ["none", "temperature", "platt", "isotonic"]:
                pv(f"{k}.{s}.ece.{m}", e.loc[m, "ece"])
            pv(f"{k}.{s}.overpred.platt", e.loc["platt", "overprediction"], "{:.2f}")
            for t in ["f1", "sens80"]:
                pv(f"{k}.{s}.tpr.{t}", e.loc[f"thresholds_{t}", "tpr"]); pv(f"{k}.{s}.ud.{t}", e.loc[f"thresholds_{t}", "underdiag"])
            pv(f"{k}.{s}.drop", n[f"drop_{s}"], "{:+.3f}")
        t_ = th[th.run == run].set_index("threshold")
        for t in ["f1", "sens80"]:
            pv(f"{k}.nih.tpr.{t}", t_.loc[t, "tpr"]); pv(f"{k}.nih.ud.{t}", t_.loc[t, "underdiag"])
    # ranges over the 12 models (and by loss family) for the text
    def rng(key, sub=None, fmt="{:.3f}"):
        vals = [float(V[f"{KEY[run]}.{key}"].replace("+", "")) for (a, l), run in RUN.items() if sub is None or l in sub]
        pv(f"min.{key}" + ("" if sub is None else "." + "".join(sub)), min(vals), fmt)
        pv(f"max.{key}" + ("" if sub is None else "." + "".join(sub)), max(vals), fmt)
    for key in ["auroc", "nih7.auroc", "sample.auroc", "valid.auroc", "sample.ece.platt", "valid.ece.platt", "ece.platt",
                "sample.tpr.f1", "sample.ud.f1", "sample.tpr.sens80", "sample.ud.sens80", "nih.tpr.f1", "nih.ud.f1",
                "nih.tpr.sens80", "nih.ud.sens80", "sample.overpred.platt", "nih7.ece.platt"]:
        rng(key)
    for sub in (["bce"], ["focal"], ["wbce", "asl"]):
        for key in ["ece.none", "sample.ece.none", "ece.temperature", "nih7.ece.none"]:
            rng(key, sub)
    rng("sample.drop", None, "{:+.3f}")
    # CheXpert per-class shift (mean over 12 models, after Platt)
    pc = extpc[(extpc.set == "sample") & (extpc.method == "platt")].groupby("cls")[["mean_pred", "prevalence"]].mean()
    for c_ in pc.index:
        pv(f"shift.{c_}.pred", pc.loc[c_, "mean_pred"], "{:.3f}"); pv(f"shift.{c_}.prev", pc.loc[c_, "prevalence"], "{:.3f}")
    # subgroups
    s_ = sc[sc.method == "platt_global"]
    pv("subcal.platt.min", s_.ece.min()); pv("subcal.platt.max", s_.ece.max()); pv("subcal.platt.maxgap", s_.gap_citl.abs().max())
    s_ = sc[sc.method == "none"]; pv("subcal.none.max", s_.ece.max())
    s_ = sc[sc.method == "platt_group"]; pv("subcal.group.min", s_.ece.min()); pv("subcal.group.max", s_.ece.max())
    tu = thsub[~thsub.holm_dup]   # one contrast per binary attribute (10_paper2_extras.add_holm)
    pv("thsub.n", len(tu), "{:d}"); pv("thsub.sig", int((tu.underdiag_gap_p < 0.05).sum()), "{:d}")
    pv("thsub.sigholm", int((tu.underdiag_gap_p_holm < 0.05).sum()), "{:d}")
    for grp, tag in [("<40", "lt40"), ("60+", "60plus")]:
        g = thsub[(thsub.threshold == "sens80") & (thsub.group == grp)]
        pv(f"thsub.{tag}.sens80.min", g.underdiag_gap.min(), "{:+.3f}"); pv(f"thsub.{tag}.sens80.max", g.underdiag_gap.max(), "{:+.3f}")
        pv(f"thsub.{tag}.sens80.nholm", int((g.underdiag_gap_p_holm < 0.05).sum()), "{:d}")
    for grp, attr in [("<40", "age"), ("PA", "view")]:
        g = exsub[(exsub.group == grp)]
        tag = "lt40" if grp == "<40" else "PA"
        pv(f"ex.{tag}.udgap.min", g.underdiag_sens80_gap.min(), "{:+.3f}"); pv(f"ex.{tag}.udgap.max", g.underdiag_sens80_gap.max(), "{:+.3f}")
        pv(f"ex.{tag}.udgap.nholm", int((g.underdiag_sens80_gap_p_holm < 0.05).sum()), "{:d}")
        c = cm[cm.group == grp]
        pv(f"cm.{tag}.crude.min", c.gap_crude.min(), "{:+.3f}"); pv(f"cm.{tag}.crude.max", c.gap_crude.max(), "{:+.3f}")
        pv(f"cm.{tag}.std.min", c.gap_std.min(), "{:+.3f}"); pv(f"cm.{tag}.std.max", c.gap_std.max(), "{:+.3f}")
        pv(f"cm.{tag}.std.nexcl0", int((c.gap_std_lo > 0).sum()), "{:d}")
        pv(f"cm.{tag}.share1.in", c.share_1_in.mean(), "{:.2f}"); pv(f"cm.{tag}.share1.rest", c.share_1_rest.mean(), "{:.2f}")
    for attr in ["sex", "race"]:
        g = exsub[exsub.attribute == attr]
        pv(f"ex.{attr}.aucgap.max", g.auroc_gap.abs().max())
        pv(f"ex.{attr}.udgap.max", g.underdiag_sens80_gap.abs().max())
    import json
    sij = json.load(open(os.path.join(EV, "..", "results", "p2_swin_bce", "split_info.json")))
    for k_, v_ in sij.items():
        pv("n." + k_.replace("_", "."), int(v_), "{:,d}")
    nc = pd.read_csv(os.path.join(EV, "..", "results", "_dl_ext", "ext_preds", "nih_check.csv"))
    pv("nihcheck.maxdiff", nc.max_abs_diff.max(), "{:.4f}")
    e = ext[ext.set == "sample"].iloc[0]; pv("n.sample.images", int(e.n_images), "{:,d}"); pv("n.sample.patients", int(e.n_patients), "{:,d}")
    e = ext[ext.set == "valid"].iloc[0]; pv("n.valid.images", int(e.n_images), "{:d}"); pv("n.valid.patients", int(e.n_patients), "{:d}")


def add_ranges():
    """For every min/max pair, a range key that prints one value when both ends are equal:
    min.K / max.K -> rng.K ('a--b') and rngto.K ('a to b'); K.min / K.max -> K.rng ('a to b') and K.rngd ('a--b')."""
    for k in list(V):
        if k.startswith("min.") and "max." + k[4:] in V:
            lo, hi = V[k], V["max." + k[4:]]
            V["rng." + k[4:]] = lo if lo == hi else f"{lo}--{hi}"
            V["rngto." + k[4:]] = lo if lo == hi else f"{lo} to {hi}"
        if k.endswith(".min") and k[:-4] + ".max" in V:
            lo, hi = V[k], V[k[:-4] + ".max"]
            V[k[:-4] + ".rng"] = lo if lo == hi else f"{lo} to {hi}"
            V[k[:-4] + ".rngd"] = lo if lo == hi else f"{lo}--{hi}"


def write_values():
    add_ranges()
    with open(os.path.join(GEN, "values.tex"), "w") as f:
        f.write("% generated by paper2/make_assets.py -- do not edit\n")
        for k, v in sorted(V.items()):
            if "$" not in v:   # plain values: true minus signs (not the '--' of a range), text-mode '<'
                v = re.sub(r"(?<![-\w])-(?=[\d.])", "$-$", v).replace("<", "$<$")
            f.write(f"\\pvdef{{{k}}}{{{v}}}\n")


def values_calib(cr):
    """Ranges over models for the text: slope by loss family (NIH raw), and slope / Brier after Platt."""
    loss_of = {run: l for (a, l), run in RUN.items()}
    cr = cr.assign(loss=cr.run.map(loss_of))
    for s in ["nih", "chexpert"]:
        for m in ["none", "platt"]:
            d = cr[(cr.set == s) & (cr.method == m)]
            for col in ["slope", "brier", "citl"]:
                fmt = "{:+.2f}" if col == "citl" else ("{:.2f}" if col == "slope" else "{:.3f}")
                pv(f"cal.{s}.{m}.{col}.min", d[col].min(), fmt); pv(f"cal.{s}.{m}.{col}.max", d[col].max(), fmt)
                for fam, sub in [("bce", ["bce"]), ("focal", ["focal"]), ("wbceasl", ["wbce", "asl"]), ("wbce", ["wbce"]), ("asl", ["asl"])]:
                    f = d[d.loss.isin(sub)]
                    pv(f"cal.{s}.{m}.{col}.min.{fam}", f[col].min(), fmt); pv(f"cal.{s}.{m}.{col}.max.{fam}", f[col].max(), fmt)


def table_calib(cr, s):
    """Appendix table: every model, raw and NIH-fitted Platt, with patient-bootstrap 95% CIs."""
    f3 = lambda r, c: f"{r[c]:.3f} ({r[c + '_lo']:.3f}--{r[c + '_hi']:.3f})"
    f2 = lambda r, c, sg="": (f"{r[c]:{sg}.2f} ({r[c + '_lo']:{sg}.2f} to {r[c + '_hi']:{sg}.2f})").replace("-", "$-$").replace("$-$$-$", "--")
    L = [r"\begin{tabular}{lllcccc}", r"\toprule",
         r"Model & Loss & Recal. & Brier & ECE & Slope & Intercept \\", r"\midrule"]
    short = {"densenet": "DenseNet", "swin": "Swin-B", "hybrid": "Hybrid"}
    for a in ARCH:
        for i, l in enumerate(LOSSES):
            for j, m in enumerate(["none", "platt"]):
                r = cr[(cr.run == RUN[(a, l)]) & (cr.set == s) & (cr.method == m)].iloc[0]
                L.append(f"{short[a] if i == 0 and j == 0 else ''} & {LOSS_NAME[l] if j == 0 else ''} & {'raw' if m == 'none' else 'Platt'} & "
                         f"{f3(r, 'brier')} & {f3(r, 'ece')} & {f2(r, 'slope')} & {f2(r, 'citl', '+')} \\\\")
        L.append(r"\midrule" if a != "hybrid" else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(GEN, f"table_calib_{s}.tex"), "w").write("% generated by paper2/make_assets.py\n" + "\n".join(L) + "\n")


def fig_reliability(rb, s, m, classes, fname, arch="densenet"):
    """Per-class reliability diagrams (10 equal-count bins) for one architecture, the four losses overlaid."""
    ncol = 5 if len(classes) > 7 else 4
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.0, 1.3 * nrow + 0.35))
    axs = np.atleast_1d(axs).ravel()
    for ax, c in zip(axs, classes):
        top = 0.0
        for l in LOSSES:
            d = rb[(rb.run == RUN[(arch, l)]) & (rb.set == s) & (rb.method == m) & (rb.cls == c)].sort_values("bin")
            ax.plot(d.mean_pred, d.frac_pos, color=COL[l], marker=MARK[l], ms=2.5, lw=0.9, label=LOSS_NAME[l], zorder=3)
            top = max(top, d.mean_pred.max(), d.frac_pos.max())
        top = min(1.0, top * 1.05)
        ax.plot([0, top], [0, top], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(0, top); ax.set_ylim(0, top)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(c.replace("_", " "), fontsize=7, pad=2); ax.tick_params(labelsize=5.5, length=2, pad=1)
        grid(ax)
    for ax in axs[len(classes):]:
        ax.axis("off")
    fig.supxlabel("Mean predicted probability", fontsize=7, y=0.01); fig.supylabel("Observed frequency", fontsize=7, x=0.01)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0.02, 0.03, 1, 1 - 0.3 / (1.3 * nrow + 0.35)), h_pad=0.6, w_pad=0.4)
    fig.savefig(os.path.join(FIG, fname + ".pdf")); fig.savefig(os.path.join(FIG, fname + ".png"), dpi=200)
    plt.close(fig)


def table_main():
    L = [r"\begin{tabular}{llccccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{NIH test (14 classes)} & NIH (7 cl.) & \multicolumn{2}{c}{CheXpert (7 classes)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}",
         r"Model & Loss & AUROC (95\% CI) & $\Delta$ vs BCE ($p$) & ECE raw/temp./Platt & AUROC & AUROC & ECE raw/Platt \\", r"\midrule"]
    short = {"densenet": "DenseNet", "swin": "Swin-B", "hybrid": "Hybrid"}
    for a in ARCH:
        for i, l in enumerate(LOSSES):
            k = KEY[RUN[(a, l)]]
            d = "--" if l == "bce" else f"{V[k + '.dauc'].replace('-', '$-$')} ({V[k + '.dauc.p'].replace('<', '$<$')})"
            L.append(f"{short[a] if i == 0 else ''} & {LOSS_NAME[l]} & {V[k + '.auroc']} ({V[k + '.auroc.lo']}--{V[k + '.auroc.hi']}) & {d} & "
                     f"{V[k + '.ece.none']}/{V[k + '.ece.temperature']}/{V[k + '.ece.platt']} & "
                     f"{V[k + '.nih7.auroc']} & {V[k + '.sample.auroc']} & {V[k + '.sample.ece.none']}/{V[k + '.sample.ece.platt']} \\\\")
        L.append(r"\midrule" if a != "hybrid" else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(GEN, "table_main.tex"), "w").write("% generated by paper2/make_assets.py\n" + "\n".join(L) + "\n")


def fig_calibration(rec):
    r = rec.set_index(["run", "method"])
    meths, labels = ["none", "temperature", "platt"], ["Raw", "Temperature", "Per-class\nPlatt"]
    fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.1), sharey=True)
    for ax, a in zip(axs, ARCH):
        for l in LOSSES:
            y = [r.loc[(RUN[(a, l)], m), "ece_q10"] for m in meths]
            ax.plot(range(3), y, color=COL[l], marker=MARK[l], ms=4.5, mec="white", mew=0.6, label=LOSS_NAME[l], zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_xlim(-0.3, 2.3)
        ax.set_title(ARCH[a], color=INK); grid(ax)
    axs[0].set_ylabel("Macro ECE, NIH test")
    axs[0].set_ylim(0, 0.33)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(FIG, "calibration_grid.pdf")); fig.savefig(os.path.join(FIG, "calibration_grid.png"), dpi=200)
    plt.close(fig)


def scatter_cols(ax, xs, getter, jitter=0.13):
    for (a, l), run in RUN.items():
        off = {"densenet": -jitter, "swin": 0.0, "hybrid": jitter}[a]
        ys = [getter(run, x) for x in xs]
        ax.scatter(np.arange(len(xs)) + off, ys, color=COL[l], marker=AMARK[a], s=16, edgecolor="white", linewidth=0.5, zorder=4, clip_on=False)


def legend_two(fig, y=0.995):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[l], marker="o", ls="", ms=5, label=LOSS_NAME[l]) for l in LOSSES]
    h += [Line2D([], [], color=INK2, marker=AMARK[a], ls="", ms=5, mfc="none", label=ARCH[a]) for a in ARCH]
    fig.legend(handles=h, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, y), columnspacing=1.0, handletextpad=0.3)


def fig_transfer(rec, ext, recpc, extpc, n7):
    r = n7.set_index("run")   # NIH test over the 7 classes shared with CheXpert
    e = ext[ext.set == "sample"].set_index(["run", "method"])
    xs = [("nih", "none"), ("nih", "platt"), ("chex", "none"), ("chex", "platt")]
    get = lambda run, x: (r.loc[run, f"nih_ece_{x[1]}"] if x[0] == "nih" else e.loc[(run, x[1]), "ece"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.4), gridspec_kw={"width_ratios": [1.15, 1]})
    scatter_cols(a1, xs, get)
    a1.set_xticks(range(4)); a1.set_xticklabels(["NIH\nraw", "NIH\nPlatt", "CheXpert\nraw", "CheXpert\nPlatt"])
    a1.axvline(1.5, color=GRID, lw=0.8); a1.set_ylabel("Macro ECE"); a1.set_ylim(0, 0.35); grid(a1)
    a1.set_title("(a) Macro ECE before and after NIH-fitted Platt", loc="left")
    # (b) mean predicted vs prevalence per class after NIH Platt, averaged over the 12 models
    n = recpc[recpc.method == "platt"].groupby("cls")[["mean_pred", "prevalence"]].mean()
    c = extpc[(extpc.set == "sample") & (extpc.method == "platt")].groupby("cls")[["mean_pred", "prevalence"]].mean()
    lim = (0.004, 0.6)
    a2.plot(lim, lim, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    a2.text(0.006, 0.0085, "calibrated", color=INK2, fontsize=6.5, rotation=33, ha="left")
    for cls in c.index:
        nn = "Effusion" if cls == "Effusion" else cls
        x0, y0 = n.loc[nn, "prevalence"], n.loc[nn, "mean_pred"]
        x1, y1 = c.loc[cls, "prevalence"], c.loc[cls, "mean_pred"]
        a2.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=7), zorder=2)
        a2.scatter([x0], [y0], s=16, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=3)
        a2.scatter([x1], [y1], s=16, color=INK, zorder=3)
        dx, dy, ha = LABEL_POS.get(cls, (1.1, 0.8, "left"))
        a2.text(x1 * dx, y1 * dy, cls, fontsize=6.5, color=INK, va="center", ha=ha)
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("Prevalence"); a2.set_ylabel("Mean predicted probability")
    a2.set_title("(b) Per class, after Platt", loc="left")
    legend_two(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(FIG, "transfer.pdf")); fig.savefig(os.path.join(FIG, "transfer.png"), dpi=200)
    plt.close(fig)


def fig_thresholds(th, ext):
    t = th.set_index(["run", "threshold"])
    e = ext[ext.set == "sample"].set_index(["run", "method"])
    xs = [("nih", "f1"), ("chex", "f1"), ("nih", "sens80"), ("chex", "sens80")]
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.2))
    for ax, met, lab in [(axs[0], "tpr", "Macro sensitivity"), (axs[1], "underdiag", "Underdiagnosis rate")]:
        get = lambda run, x: (t.loc[(run, x[1]), met] if x[0] == "nih" else e.loc[(run, f"thresholds_{x[1]}"), met])
        scatter_cols(ax, xs, get)
        ax.set_xticks(range(4)); ax.set_xticklabels(["NIH", "CheXpert", "NIH", "CheXpert"])
        ax.axvline(1.5, color=GRID, lw=0.8); ax.set_ylabel(lab); ax.set_ylim(0, 1 if met == "tpr" else 0.45); grid(ax)
        for x, s in [(0.5, "F1-optimal"), (2.5, "Sensitivity 0.8")]:
            ax.text(x, -0.25, s, transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color=INK2)
    legend_two(fig)
    fig.tight_layout(rect=(0, 0.0, 1, 0.88))
    fig.savefig(os.path.join(FIG, "thresholds.pdf")); fig.savefig(os.path.join(FIG, "thresholds.png"), dpi=200)
    plt.close(fig)


def seeds():
    """DenseNet-121 x 4 losses x training seeds {42, 1, 2} (retrain/12_seeds.py): values and appendix table."""
    r = pd.read_csv(os.path.join(EV, "p2_seeds/seed_runs.csv"))
    d = pd.read_csv(os.path.join(EV, "p2_seeds/seed_diffs.csv"))
    num = lambda v, fmt: unsign0(fmt.format(v)).replace("-", "$-$")
    rng = lambda x, fmt: num(x.min(), fmt) if fmt.format(x.min()) == fmt.format(x.max()) else f"{num(x.min(), fmt)} to {num(x.max(), fmt)}"
    for l in LOSSES:
        for m in ["none", "temperature", "platt", "isotonic"]:
            x = r[(r.loss == l) & (r.method == m)]
            for col, fmt in [("auroc", "{:.3f}"), ("ece_q10", "{:.3f}"), ("cal_slope", "{:.2f}"), ("cal_intercept", "{:+.2f}"), ("overpred", "{:.1f}")]:
                pv(f"seed.{m}.{col}.{l}.min", x[col].min(), fmt); pv(f"seed.{m}.{col}.{l}.max", x[col].max(), fmt)
        if l != "bce":
            pv(f"seed.dauc.{l}.min", d[f"{l}-bce"].min(), "{:+.3f}"); pv(f"seed.dauc.{l}.max", d[f"{l}-bce"].max(), "{:+.3f}")
    pv("seed.dauc.absmax", d[[c for c in d.columns if c.endswith("-bce")]].abs().values.max())
    pv("seed.dauc.absmax.nowbce", d[["focal-bce", "asl-bce"]].abs().values.max())
    pv("seed.platt.ece.min", r[r.method == "platt"].ece_q10.min()); pv("seed.platt.ece.max", r[r.method == "platt"].ece_q10.max())
    pv("seed.n", r.seed.nunique(), "{:d}")
    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Discrimination} & \multicolumn{3}{c}{Raw} & Temperature & Platt \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
         r"Loss & AUROC & $\Delta$ vs BCE & ECE & Slope & Intercept & ECE & ECE \\", r"\midrule"]
    for l in LOSSES:
        g = lambda m, c: r[(r.loss == l) & (r.method == m)][c]
        dd = "--" if l == "bce" else rng(d[f"{l}-bce"], "{:+.3f}")
        L.append(f"{LOSS_NAME[l]} & {rng(g('none', 'auroc'), '{:.3f}')} & {dd} & {rng(g('none', 'ece_q10'), '{:.3f}')} & "
                 f"{rng(g('none', 'cal_slope'), '{:.2f}')} & {rng(g('none', 'cal_intercept'), '{:+.2f}')} & "
                 f"{rng(g('temperature', 'ece_q10'), '{:.3f}')} & {rng(g('platt', 'ece_q10'), '{:.3f}')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "table_seeds.tex"), "w").write("% generated by paper2/make_assets.py\n" + "\n".join(L) + "\n")

    # CheXpert sample, DenseNet-121 over the three seeds (12_seeds.py --ext)
    c = pd.read_csv(os.path.join(EV, "p2_seeds/seed_chexpert.csv"))
    for col, fmt in [("drop", "{:+.3f}"), ("auroc", "{:.3f}"), ("ece_none", "{:.3f}"), ("ece_platt", "{:.3f}"),
                     ("citl_platt", "{:+.2f}"), ("ud_f1", "{:.3f}"), ("ud_sens80", "{:.3f}")]:
        pv(f"seedx.{col}.min", c[col].min(), fmt); pv(f"seedx.{col}.max", c[col].max(), fmt)
    for g in ["lt40", "PA"]:
        pv(f"seedx.{g}.npos", int((c[f"{g}_udgap"] > 0).sum()), "{:d}")
        pv(f"seedx.{g}.nsig", int(((c[f"{g}_udgap"] > 0) & (c[f"{g}_udgap_p"] < 0.05)).sum()), "{:d}")
        pv(f"seedx.{g}.udgap.min", c[f"{g}_udgap"].min(), "{:+.3f}"); pv(f"seedx.{g}.udgap.max", c[f"{g}_udgap"].max(), "{:+.3f}")
    pv("seedx.n", len(c), "{:d}")
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & AUROC change & \multicolumn{2}{c}{ECE} & Intercept & \multicolumn{2}{c}{Underdiagnosis} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
         r"Loss & NIH$\to$CheXpert & Raw & Platt & Platt & F1 thr. & Sens.\ 0.8 thr. \\", r"\midrule"]
    for l in LOSSES:
        x = c[c.loss == l]
        L.append(f"{LOSS_NAME[l]} & {rng(x['drop'], '{:+.3f}')} & {rng(x['ece_none'], '{:.3f}')} & {rng(x['ece_platt'], '{:.3f}')} & "
                 f"{rng(x['citl_platt'], '{:+.2f}')} & {rng(x['ud_f1'], '{:.3f}')} & {rng(x['ud_sens80'], '{:.3f}')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "table_seeds_chexpert.tex"), "w").write("% generated by paper2/make_assets.py\n" + "\n".join(L) + "\n")


def review():
    """Review checks (retrain/13_review_checks.py, saved predictions only): recalibration-map ablation table,
    oracle and EM prior shift on CheXpert, specificity at both thresholds, age x view gaps, NIH val/test prevalence."""
    rv = os.path.join(EV, "p2_review")
    loss_of = {run: l for (a, l), run in RUN.items()}
    r = pd.read_csv(os.path.join(rv, "recal_ablation.csv")).assign(loss=lambda d: d.run.map(loss_of))
    meths = ["none", "temperature", "temp_perclass", "shift_perclass", "platt_global", "platt"]
    def rr(x):
        lo, hi = f"{x.min():.3f}", f"{x.max():.3f}"
        return lo if lo == hi else f"{lo}--{hi}"
    for l in LOSSES:
        for m in meths:
            x = r[(r.loss == l) & (r.method == m)].ece_q10
            pv(f"abl.{l}.{m}.min", x.min()); pv(f"abl.{l}.{m}.max", x.max())
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Loss & Raw & Temp.\ (1) & Class temp.\ (14) & Class shift (14) & Shared Platt (2) & Class Platt (28) \\", r"\midrule"]
    for l in LOSSES:
        L.append(f"{LOSS_NAME[l]} & " + " & ".join(rr(r[(r.loss == l) & (r.method == m)].ece_q10) for m in meths) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "table_recal_ablation.tex"), "w").write("% generated by paper2/make_assets.py\n" + "\n".join(L) + "\n")
    p = pd.read_csv(os.path.join(rv, "prior_shift.csv"))
    for v in ["none", "oracle_prior", "em_prior"]:
        x = p[p.variant == v]
        pv(f"prior.{v}.ece.min", x.ece_q10.min()); pv(f"prior.{v}.ece.max", x.ece_q10.max())
        pv(f"prior.{v}.citl.min", x.cal_intercept.min(), "{:+.2f}"); pv(f"prior.{v}.citl.max", x.cal_intercept.max(), "{:+.2f}")
    e = p.em_prior_mean_abs_err.dropna(); pv("prior.em.err.min", e.min(), "{:.2f}"); pv("prior.em.err.max", e.max(), "{:.2f}")
    o = pd.read_csv(os.path.join(EV, "p2_external/external_overall.csv"))
    t7 = pd.read_csv(os.path.join(rv, "thresholds7.csv"))
    t = t7[t7.set == "nih7"]
    for tt in ["f1", "sens80"]:
        for ds, tag in [("nih7", "nih"), ("chexpert", "chex")]:
            x = t7[(t7.set == ds) & (t7.threshold == tt)]
            pv(f"flag.{tag}.{tt}.min", x.flag_rate_neg.min(), "{:.2f}"); pv(f"flag.{tag}.{tt}.max", x.flag_rate_neg.max(), "{:.2f}")
        w = t7[t7.threshold == tt].pivot(index="run", columns="set", values=["tpr", "underdiag"])
        d_ = w["tpr"].chexpert - w["tpr"].nih7; pv(f"dtpr.{tt}.min", d_.min(), "{:+.2f}"); pv(f"dtpr.{tt}.max", d_.max(), "{:+.2f}")
        d_ = w["underdiag"].chexpert - w["underdiag"].nih7; pv(f"dud.{tt}.min", d_.min(), "{:+.3f}"); pv(f"dud.{tt}.max", d_.max(), "{:+.3f}")
        x = o[(o.set == "sample") & (o.method == f"thresholds_{tt}")]
        pv(f"spec.chex.{tt}.min", 1 - x.fpr.max(), "{:.2f}"); pv(f"spec.chex.{tt}.max", 1 - x.fpr.min(), "{:.2f}")
        x = t[t.threshold == tt]
        pv(f"spec.nih.{tt}.min", 1 - x.fpr.max(), "{:.2f}"); pv(f"spec.nih.{tt}.max", 1 - x.fpr.min(), "{:.2f}")
    a = pd.read_csv(os.path.join(rv, "age_view.csv"))
    for v in ["AP", "PA"]:
        x = a[a.view == v]
        pv(f"ageview.{v}.min", x.gap.min(), "{:+.3f}"); pv(f"ageview.{v}.max", x.gap.max(), "{:+.3f}")
        pv(f"ageview.{v}.npos", int((x.gap > 0).sum()), "{:d}")
    pr = pd.read_csv(os.path.join(rv, "prevalence.csv"), index_col=0)
    for c in ["Pneumothorax", "Cardiomegaly"]:
        pv(f"prev.val.{c}", 100 * pr.loc[c, "nih_val"], "{:.1f}\\%"); pv(f"prev.test.{c}", 100 * pr.loc[c, "nih_test"], "{:.1f}\\%")
    pv("prev.n_test_higher", int((pr.nih_test > pr.nih_val).sum()), "{:d}")
    import json
    sij = json.load(open(os.path.join(EV, "..", "results", "p2_swin_bce", "split_info.json")))
    pv("ipp.val", sij["val_images"] / sij["val_patients"], "{:.1f}"); pv("ipp.test", sij["test_images"] / sij["test_patients"], "{:.1f}")
    pv("n.total.images", sum(sij[f"{s_}_images"] for s_ in ["train", "val", "test"]), "{:,d}")
    pv("n.total.patients", sum(sij[f"{s_}_patients"] for s_ in ["train", "val", "test"]), "{:,d}")
    # wBCE positive weights n_neg / n_pos (training split, 14 findings)
    cc = pd.read_csv(os.path.join(EV, "..", "results", "dn_bce", "split_class_counts.csv")).set_index("class")
    n_tr = cc.loc["images", "train"]; w = [(n_tr - cc.loc[c, "train"]) / cc.loc[c, "train"] for c in cc.index if c not in ("images", "No Finding")]
    pv("wbce.w.min", min(w), "{:.1f}"); pv("wbce.w.max", max(w), "{:.0f}")
    # fitted temperatures by loss; largest AUROC gain over BCE
    temp = pd.concat([pd.read_csv(os.path.join(EV, d, "temperatures.csv")) for d in ["p2_recal_dn", "p2_recal_swin", "p2_recal_hyb"]])
    temp["loss"] = temp.run.map(loss_of)
    for l in LOSSES:
        x = temp[temp.loss == l]["T"]; pv(f"T.{l}.min", x.min(), "{:.2f}"); pv(f"T.{l}.max", x.max(), "{:.2f}")
    g = [float(V[f"{KEY[run]}.dauc"].replace("+", "")) for (a_, l), run in RUN.items() if l != "bce"]
    pv("dauc.maxgain", max(g), "{:+.3f}"); pv("dauc.absmax", max(abs(x) for x in g))
    # NIH subgroup contrasts surviving Holm, by threshold
    th_ = pd.read_csv(os.path.join(EV, "p2_extras/thresholds_subgroups_holm.csv"))
    for tt in ["f1", "sens80"]:
        pv(f"thsub.{tt}.nholm", int(((th_.threshold == tt) & ~th_.holm_dup & (th_.underdiag_gap_p_holm < 0.05)).sum()), "{:d}")
    pv("thsub.groups", th_[~th_.holm_dup].group.nunique(), "{:d}")
    pv("thsub.nclass.age", len(th_[th_.attribute == "age"].classes.iloc[0].split(";")), "{:d}")
    # CheXpert subgroups: gaps at F1 thresholds, and underdiagnosis within and outside the <40 / PA groups
    ex_ = pd.read_csv(os.path.join(EV, "p2_extras/external_subgroups_holm.csv"))
    pv("exsub.groups", ex_[~ex_.holm_dup].group.nunique(), "{:d}")
    for attr in ["sex", "race"]:
        pv(f"ex.{attr}.udgap.f1.max", ex_[ex_.attribute == attr].underdiag_f1_gap.abs().max())
    for grp, tag in [("<40", "lt40"), ("PA", "PA")]:
        x = ex_[ex_.group == grp]
        pv(f"ex.{tag}.ud.in.min", x.underdiag_sens80.min()); pv(f"ex.{tag}.ud.in.max", x.underdiag_sens80.max())
        rest = x.underdiag_sens80 - x.underdiag_sens80_gap
        pv(f"ex.{tag}.ud.rest.min", rest.min()); pv(f"ex.{tag}.ud.rest.max", rest.max())
    # ECE with 10 / 20 / 50 equal-count bins
    eb = pd.read_csv(os.path.join(rv, "ece_bins.csv"))
    pv("ecebins.maxdiff", max((eb.ece_q20 - eb.ece_q10).abs().max(), (eb.ece_q50 - eb.ece_q10).abs().max()))


def review3():
    """Third review round (retrain/14_review3_checks.py and existing outputs): uncertain-label check, per-class miss
    rates by age and view, exact-score F1 thresholds, NIH test intercepts; appendix tables for isotonic regression,
    NIH subgroups, prior-shift corrections and per-class miss rates."""
    r3 = os.path.join(EV, "p2_review3")
    hdr = "% generated by paper2/make_assets.py\n"
    def mm(key, x, fmt="{:.3f}"):
        pv(f"{key}.min", x.min(), fmt); pv(f"{key}.max", x.max(), fmt)
    def rr(x, fmt="{:.3f}"):
        lo, hi = unsign0(fmt.format(x.min())), unsign0(fmt.format(x.max()))
        if lo == hi:
            return lo.replace("-", "$-$")
        return f"{lo} to {hi}".replace("-", "$-$") if "+" in fmt else f"{lo}--{hi}"
    # (1) uncertain labels
    u = pd.read_csv(os.path.join(r3, "uncertain_rescue.csv"))
    s, f = u[u.threshold == "sens80"], u[u.threshold == "f1"]
    pv("unc.share", s.share_pos_with_uncertain.iloc[0], "{:.2f}")
    for g in ["lt40", "PA"]:
        pv(f"unc.share.{g}.in", s[f"share_unc_{g}_in"].iloc[0], "{:.2f}"); pv(f"unc.share.{g}.rest", s[f"share_unc_{g}_rest"].iloc[0], "{:.2f}")
        mm(f"unc.{g}.std", s[f"{g}_conf_std"], "{:+.3f}"); mm(f"unc.{g}.crude", s[f"{g}_conf_crude"], "{:+.3f}")
        pv(f"unc.{g}.nexcl0", int((s[f"{g}_conf_std_lo"] > 0).sum()), "{:d}")
    mm("unc.ud.sens80", s.ud_confirmed_only); mm("unc.ud.f1", f.ud_confirmed_only)
    # (3) exact F1 thresholds
    x = pd.read_csv(os.path.join(r3, "f1_exact.csv"))
    for ds in ["nih7", "chexpert"]:
        e = x[(x.set == ds) & (x.search == "exact")]
        mm(f"f1x.{ds}.ud", e.underdiag); mm(f"f1x.{ds}.tpr", e.tpr)
    # (4) NIH test intercepts
    ic = pd.read_csv(os.path.join(r3, "nih_intercepts.csv"))
    for c in ["platt_7", "platt_oracle_14", "platt_oracle_7"]:
        mm(f"icp.{c.replace('_', '')}", ic[f"citl_{c}"], "{:+.2f}")
    am = pd.read_csv(os.path.join(r3, "asl_margin.csv"))
    pv("asl.belowm.max", 100 * am.share_below_m.max(), "{:.2f}"); mm("asl.q01", am.q01); mm("asl.median", am["median"])
    cc_ = pd.read_csv(os.path.join(EV, "..", "results", "dn_bce", "split_class_counts.csv")).set_index("class")
    pv("hernia.val.n", int(cc_.loc["Hernia", "val"]), "{:d}")
    # Table: per-class miss rates by age and view (mean over the 12 models)
    fn_ = pd.read_csv(os.path.join(r3, "perclass_fnr.csv")).groupby("cls").mean(numeric_only=True)
    L = [r"\begin{tabular}{lrcrcrcrc}", r"\toprule",
         r" & \multicolumn{2}{c}{$<40$} & \multicolumn{2}{c}{$\geq 40$} & \multicolumn{2}{c}{PA} & \multicolumn{2}{c}{AP} \\",
         r"Finding & $n$ & FNR & $n$ & FNR & $n$ & FNR & $n$ & FNR \\", r"\midrule"]
    for c in fn_.index:
        z = fn_.loc[c]
        L.append(f"{c} & " + " & ".join(f"{int(z[f'{g}_{nm}_n']):,} & {z[f'{g}_{nm}_fnr']:.2f}" for g in ["lt40", "PA"] for nm in ["in", "rest"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "table_fnr.tex"), "w").write(hdr + "\n".join(L) + "\n")
    # Table: isotonic regression vs raw and Platt, NIH test, every model
    rec = pd.concat([pd.read_csv(os.path.join(EV, d, "recal_summary.csv")) for d in ["p2_recal_dn", "p2_recal_swin", "p2_recal_hyb"]]).set_index(["run", "method"])
    L = [r"\begin{tabular}{llccccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{AUROC} & \multicolumn{2}{c}{ECE} & \multicolumn{2}{c}{Brier} \\",
         r"Model & Loss & Raw/Platt & Isotonic & $\Delta$ & Platt & Isotonic & Platt & Isotonic \\", r"\midrule"]
    iso_d = []
    for a_, an in ARCH.items():
        for i, l in enumerate(LOSSES):
            run = RUN[(a_, l)]
            n_, pl, iso = rec.loc[(run, "none")], rec.loc[(run, "platt")], rec.loc[(run, "isotonic")]
            iso_d.append(iso.auroc - n_.auroc)
            L.append(f"{an if i == 0 else ''} & {LOSS_NAME[l]} & {n_.auroc:.3f} & {iso.auroc:.3f} & {unsign0(f'{iso.auroc - n_.auroc:+.3f}')} & "
                     f"{pl.ece_q10:.3f} & {iso.ece_q10:.3f} & {pl.brier:.4f} & {iso.brier:.4f} \\\\")
        if a_ != "hybrid":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "table_isotonic.tex"), "w").write(hdr + "\n".join(L) + "\n")
    iso_d = pd.Series(iso_d); mm("iso.dauc", iso_d, "{:+.3f}")
    # Table: NIH subgroups (ranges over the 12 models)
    sc = pd.read_csv(os.path.join(EV, "p2_subcal", "subgroup_calibration.csv"))
    tsub = pd.read_csv(os.path.join(EV, "p2_extras", "thresholds_subgroups_holm.csv"))
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{ECE} & CITL gap & \multicolumn{2}{c}{Underdiagnosis} \\",
         r"Group & Raw & Platt (all) & Platt (group) & Platt (all) & F1 & Sens.\ 0.8 \\", r"\midrule"]
    for g in ["F", "M", "<40", "40-59", "60+"]:
        e = lambda m: sc[(sc.group == g) & (sc.method == m)]
        ud = lambda t: tsub[(tsub.group == g) & (tsub.threshold == t)].underdiag
        gname = {"F": "Female", "M": "Male", "<40": "$<40$", "40-59": "40--59", "60+": r"$\geq 60$"}[g]
        L.append(f"{gname} & {rr(e('none').ece)} & {rr(e('platt_global').ece)} & {rr(e('platt_group').ece)} & "
                 f"{rr(e('platt_global').gap_citl, '{:+.3f}')} & {rr(ud('f1'))} & {rr(ud('sens80'))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "table_subgroups_nih.tex"), "w").write(hdr + "\n".join(L) + "\n")
    # Table: prior-shift corrections on CheXpert, every model
    ps = pd.read_csv(os.path.join(EV, "p2_review", "prior_shift.csv")).set_index(["run", "variant"])
    L = [r"\begin{tabular}{llccccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{ECE} & \multicolumn{3}{c}{Intercept} & EM \\",
         r"Model & Loss & Platt & + oracle & + EM & Platt & + oracle & + EM & MAE \\", r"\midrule"]
    for a_, an in ARCH.items():
        for i, l in enumerate(LOSSES):
            run = RUN[(a_, l)]
            z = {v: ps.loc[(run, v)] for v in ["none", "oracle_prior", "em_prior"]}
            L.append(f"{an if i == 0 else ''} & {LOSS_NAME[l]} & " + " & ".join(f"{z[v].ece_q10:.3f}" for v in z) + " & "
                     + " & ".join(unsign0(f"{z[v].cal_intercept:+.2f}").replace("-", "$-$") for v in z)
                     + f" & {z['em_prior'].em_prior_mean_abs_err:.2f} \\\\")
        if a_ != "hybrid":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "table_prior.tex"), "w").write(hdr + "\n".join(L) + "\n")


def main():
    os.makedirs(GEN, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    rec, recpc, temp, cmp_, ext, extpc, n7, th, thsub, exsub, cm, sc = load()
    # NIH threshold metrics (nih.tpr.*, nih.ud.*, Figure 3) over the 7 classes shared with CheXpert, so that the
    # NIH -> CheXpert comparison uses the same label set (underdiagnosis depends on the number of classes)
    t7 = pd.read_csv(os.path.join(EV, "p2_review", "thresholds7.csv"))
    th = t7[t7.set == "nih7"].drop(columns="set").reset_index(drop=True)
    values(rec, temp, cmp_, ext, extpc, n7, th, thsub, exsub, cm, sc)
    table_main()
    fig_calibration(rec); fig_transfer(rec, ext, recpc, extpc, n7); fig_thresholds(th, ext)
    si, rb = load_calci()
    cr = calib_rows(rec, ext, si)
    values_calib(cr)
    table_calib(cr, "nih"); table_calib(cr, "chexpert")
    fig_reliability(rb, "nih", "none", list(rb[rb.set == "nih"].cls.unique()), "reliability_nih_raw")
    fig_reliability(rb, "nih", "platt", list(rb[rb.set == "nih"].cls.unique()), "reliability_nih_platt")
    fig_reliability(rb, "chexpert", "platt", list(rb[rb.set == "chexpert"].cls.unique()), "reliability_chexpert_platt")
    seeds()
    review()
    review3()
    write_values()
    print(f"{len(V)} values; wrote {GEN} and {FIG}")


if __name__ == "__main__":
    main()
