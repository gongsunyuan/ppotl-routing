import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHOD_LABELS = {
    "ospf": "OSPF", "ppo_vanilla": "PPO", "ppo_naive": "PPO+NaiveTL", "dqn": "DQN",
    "grlps": "GRL-PS", "ppotl": "PPO-TL (ours)", "ppo_local": "PPO-L",
    "ppotl_freeze_only": "PPO-TL (freeze only)", "ppotl_pbrs_only": "PPO-TL (PBRS only)",
}
COLORS = {
    "ospf": "#666666", "ppo_vanilla": "#1f77b4", "ppo_naive": "#ff7f0e", "dqn": "#2ca02c",
    "grlps": "#9467bd", "ppotl": "#d62728", "ppo_local": "#8c564b",
    "ppotl_freeze_only": "#e377c2", "ppotl_pbrs_only": "#17becf",
}
METHOD_ORDER = ["ospf", "ppo_vanilla", "ppo_naive", "dqn", "grlps", "ppotl"]


def collect(mode):
    rows = []
    curves = []
    for man_path in glob.glob(os.path.join(ROOT, "logs", mode, "*", "manifest.json")):
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
        if man.get("status") not in ("done", "pretrain_done"):
            continue
        job = man["job"]
        if job.get("scenario") == "P":
            continue
        d = os.path.dirname(man_path)
        ev_path = os.path.join(d, "eval.json")
        if not os.path.exists(ev_path):
            continue
        with open(ev_path, encoding="utf-8") as f:
            ev = json.load(f)
        row = {
            "scenario": job["scenario"], "topo": job.get("target_topo") or job["topo"],
            "method": job["method"], "seed": job["seed"], "rate": job.get("rate"),
            "fail_ratio": job.get("fail_ratio", 0.0), "variant": job.get("variant"),
            "final_delay": ev["final"]["delay"], "final_loss": ev["final"]["loss"],
            "final_reward": ev["final"]["reward"],
            "zs_delay": ev["zero_shot"].get("delay"), "zs_loss": ev["zero_shot"].get("loss"),
            "converge_ep": ev["converge_episode"],
        }
        rows.append(row)
        mpath = os.path.join(d, "metrics.csv")
        if os.path.exists(mpath):
            df = pd.read_csv(mpath)
            df["topo"] = row["topo"]; df["method"] = row["method"]; df["seed"] = row["seed"]
            df["rate"] = row["rate"]; df["fail_ratio"] = row["fail_ratio"]
            curves.append(df)
        rs = ev.get("rate_sweep")
        if rs:
            for r, v in rs.items():
                rows.append({**row, "rate": float(r), "final_delay": v["delay"],
                             "final_loss": v["loss"], "zs_delay": None, "zs_loss": None,
                             "variant": (row["variant"] or "") + "_sweep"})
    summary = pd.DataFrame(rows)
    curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return summary, curves


def ci95(x):
    x = np.asarray([v for v in x if not pd.isna(v)], dtype=float)
    if len(x) < 2:
        return 0.0
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def savefig(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(curves, out_dir, scenario, topo, rate=None, name=None, methods=None):
    sub = curves[(curves.get("phase") == "target")]
    if sub.empty:
        return
    data = sub[sub["topo"] == topo]
    if rate is not None and "rate" in data and data["rate"].notna().any():
        data = data[data["rate"] == rate]
    if data.empty:
        return
    methods = methods or [m for m in METHOD_ORDER if m in set(data["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, ylabel in zip(axes, ["delay", "loss"], ["Normalized delay", "Normalized packet loss"]):
        for m in methods:
            g = data[data["method"] == m].sort_values("episode")
            if g.empty:
                continue
            piv = g.groupby("episode")[metric].mean()
            w = max(3, min(12, len(piv) // 4))
            sm = piv.rolling(w, min_periods=1).mean()
            ax.plot(sm.index, sm.values, label=METHOD_LABELS.get(m, m), color=COLORS.get(m, None))
        ax.set_xlabel("Episode"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("%s %s%s" % (scenario, topo, " (rate=%s)" % rate if rate else ""))
    savefig(fig, out_dir, name or "fig_%s_%s_convergence" % (scenario, topo))


def grouped_bars(df, out_dir, name, title, methods=None, value_cols=("final_delay", "final_loss"),
                 groupby=("topo",), suptitle=""):
    if df.empty:
        return
    methods = methods or [m for m in METHOD_ORDER if m in set(df["method"])]
    keys = sorted(set(tuple(r) for r in df[list(groupby)].dropna().values.tolist()))
    fig, axes = plt.subplots(1, len(value_cols), figsize=(5.2 * len(value_cols), 4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, value_cols):
        width = 0.8 / max(len(methods), 1)
        for gi, key in enumerate(keys):
            key_df = df
            for kcol, kval in zip(groupby, key if isinstance(key, tuple) else (key,)):
                key_df = key_df[key_df[kcol] == kval]
            xs, ys, es = [], [], []
            for mi, m in enumerate(methods):
                v = key_df[key_df["method"] == m][col]
                if v.empty:
                    continue
                xs.append(gi + (mi - len(methods) / 2) * width + width / 2)
                ys.append(v.mean()); es.append(ci95(v))
            ax.bar(xs, ys, width=width * 0.92, yerr=es, capsize=2,
                   color=[COLORS.get(m) for m in methods if not key_df[key_df["method"] == m][col].empty])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([str(k) for k in keys], fontsize=8)
        ax.set_ylabel(col.replace("final_", "").replace("_", " "))
        ax.grid(alpha=0.3, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in methods if m in set(df["method"])]
    labels = [METHOD_LABELS.get(m, m) for m in methods if m in set(df["method"])]
    axes[0].legend(handles, labels, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    savefig(fig, out_dir, name)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def significance_table(summary, out_dir, reference="ppotl"):
    rows = []
    for (scen, topo), grp in summary.groupby(["scenario", "topo"]):
        if scen == "H":
            continue
        for metric in ("final_delay", "final_loss"):
            ref = grp[grp["method"] == reference][metric].dropna().values
            if len(ref) < 2:
                continue
            fam = []
            for m, g in grp.groupby("method"):
                if m == reference:
                    continue
                alt = g[metric].dropna().values
                if len(alt) < 2:
                    continue
                t, p_t = sps.ttest_ind(ref, alt, equal_var=False)
                p_w = np.nan
                if len(alt) >= 5 and len(ref) >= 5:
                    try:
                        p_w = sps.wilcoxon(ref[:min(len(ref), len(alt))], alt[:min(len(ref), len(alt))]).pvalue
                    except ValueError:
                        p_w = np.nan
                fam.append({"scenario": scen, "topo": topo, "metric": metric, "method": m,
                            "ref_mean": ref.mean(), "alt_mean": alt.mean(),
                            "delta_pct": 100 * (alt.mean() - ref.mean()) / abs(ref.mean()) if ref.mean() != 0 else np.nan,
                            "welch_p": p_t, "wilcoxon_p": p_w})
            if fam:
                ps = [r["welch_p"] for r in fam]
                adj = holm(ps)
                for r, a in zip(fam, adj):
                    r["welch_p_holm"] = a
                rows.extend(fam)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "stats_tests.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "results", args.mode)
    os.makedirs(out_dir, exist_ok=True)
    summary, curves = collect(args.mode)
    if summary.empty:
        print("no completed runs found under logs/%s" % args.mode)
        return
    summary.to_csv(os.path.join(out_dir, "summary_all.csv"), index=False)

    agg = summary.groupby(["scenario", "topo", "method"]).agg(
        delay_mean=("final_delay", "mean"), delay_ci=("final_delay", ci95),
        loss_mean=("final_loss", "mean"), loss_ci=("final_loss", ci95),
        reward_mean=("final_reward", "mean"), converge_mean=("converge_ep", "mean"),
        n_seeds=("seed", "nunique")).reset_index()
    agg.to_csv(os.path.join(out_dir, "summary_grouped.csv"), index=False)

    s1 = summary[summary["scenario"] == "S1"]
    if not s1.empty:
        for topo in sorted(set(s1["topo"])):
            rates = sorted(set(s1[s1["topo"] == topo]["rate"].dropna()))
            rate = rates[-1] if rates else None
            if not curves.empty:
                plot_convergence(curves, out_dir, "S1", topo, rate=rate)
        grouped_bars(s1, out_dir, "fig_S1_final_bars", "S1 constant traffic",
                     groupby=("topo", "rate"), suptitle="Final performance under constant traffic")

    s2 = summary[summary["scenario"] == "S2"]
    if not s2.empty and not curves.empty:
        for topo in sorted(set(s2["topo"])):
            plot_convergence(curves, out_dir, "S2", topo, name="fig_S2_%s_convergence" % topo)
    s2s = summary[summary["variant"].fillna("").str.endswith("_sweep")] if "variant" in summary else pd.DataFrame()
    if not s2s.empty:
        for topo in sorted(set(s2s["topo"])):
            df = s2s[s2s["topo"] == topo]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, col, yl in zip(axes, ["final_delay", "final_loss"], ["Normalized delay", "Normalized packet loss"]):
                for m in METHOD_ORDER:
                    g = df[df["method"] == m]
                    if g.empty:
                        continue
                    agg = g.groupby("rate")[col].agg(["mean", ci95]).reset_index()
                    ax.errorbar(agg["rate"], agg["mean"], yerr=agg["ci95"],
                                marker="o", capsize=3, label=METHOD_LABELS.get(m, m), color=COLORS.get(m))
                ax.set_xlabel("Traffic rate (pkt/s)"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
            axes[0].legend(fontsize=8)
            fig.suptitle("S2 varying traffic %s" % topo)
            savefig(fig, out_dir, "fig_S2_%s_rate_sweep" % topo)

    s3 = summary[summary["scenario"] == "S3"]
    if not s3.empty:
        grouped_bars(s3, out_dir, "fig_S3_final_bars", "S3",
                     groupby=("topo", "fail_ratio"), suptitle="Performance under link failures")
        if not curves.empty:
            for topo in sorted(set(s3["topo"])):
                plot_convergence(curves, out_dir, "S3", topo, name="fig_S3_%s_convergence" % topo)

    s4 = summary[summary["scenario"] == "S4"]
    if not s4.empty:
        zs = s4[s4["zs_delay"].notna()][["scenario", "topo", "method", "seed", "zs_delay", "zs_loss"]].rename(
            columns={"zs_delay": "final_delay", "zs_loss": "final_loss"})
        grouped_bars(zs, out_dir, "fig_S4_zeroshot", "S4", methods=["ospf", "ppo_naive", "ppotl", "ppo_local"],
                     suptitle="Zero-shot transfer (CERNET -> target)")
        grouped_bars(s4, out_dir, "fig_S4_final", "S4", methods=["ospf", "ppo_naive", "ppotl", "ppo_local"],
                     suptitle="After adaptation (CERNET -> target)")

    a = summary[summary["scenario"] == "A"]
    if not a.empty:
        abl_methods = ["ppo_vanilla", "ppo_naive", "ppotl_freeze_only", "ppotl_pbrs_only", "ppotl"]
        grouped_bars(a, out_dir, "fig_A_final_bars", "Ablation", methods=abl_methods,
                     groupby=("topo",), suptitle="Ablation: contribution of freezing and PBRS")
        fig, ax = plt.subplots(figsize=(7, 4))
        abl = a.groupby("method")["converge_ep"].agg(["mean", ci95]).loc[[m for m in abl_methods if m in set(a["method"])]]
        ax.bar(range(len(abl)), abl["mean"], yerr=abl["ci95"], capsize=4,
               color=[COLORS[m] for m in abl.index])
        ax.set_xticks(range(len(abl)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in abl.index], fontsize=8, rotation=20)
        ax.set_ylabel("Episodes to converge"); ax.grid(alpha=0.3, axis="y")
        savefig(fig, out_dir, "fig_A_convergence_episodes")

    h = summary[summary["scenario"] == "H"]
    if not h.empty and "variant" in h:
        sweeps = [("k", "k_paths"), ("freeze", "freeze_fraction"), ("zeta", "zeta1"), ("alpha", "alpha1")]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, (prefix, _) in zip(axes, sweeps):
            df = h[h["variant"].fillna("").str.startswith(prefix)]
            if df.empty:
                continue
            df = df.assign(x=df["variant"].str.replace(prefix, "", regex=False).astype(float))
            agg = df.groupby("x").agg(mean=("final_delay", "mean"), ci=("final_delay", ci95),
                                      lmean=("final_loss", "mean"), lci=("final_loss", ci95)).reset_index()
            ax.errorbar(agg["x"], agg["mean"], yerr=agg["ci"], marker="o", capsize=3, color="#d62728", label="delay")
            ax2 = ax.twinx()
            ax2.errorbar(agg["x"], agg["lmean"], yerr=agg["lci"], marker="s", capsize=3, color="#1f77b4", label="loss")
            ax.set_xlabel(prefix); ax.set_ylabel("Normalized delay", color="#d62728")
            ax2.set_ylabel("Normalized loss", color="#1f77b4"); ax.grid(alpha=0.3)
        fig.suptitle("Sensitivity analysis (PPO-TL, %s)" % cfg_topo(h))
        savefig(fig, out_dir, "fig_H_sensitivity")

    significance_table(summary, out_dir)
    print("figures and tables written to %s" % out_dir)


def cfg_topo(h):
    return sorted(set(h["topo"]))[0] if not h.empty else ""


if __name__ == "__main__":
    main()
