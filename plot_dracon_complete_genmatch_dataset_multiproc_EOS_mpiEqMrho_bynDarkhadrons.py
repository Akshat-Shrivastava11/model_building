#!/usr/bin/env python3
"""
plot_dracon_npz_by_ndark.py

Make comparison plots for DRACON NPZ files split by the number of dark hadrons
inside each approved reco FatJet:

  <prefix>_ndark1.npz
  <prefix>_ndark2.npz
  <prefix>_ndarkgt2.npz

Expected arrays from the dataset builder:
  X, y, kinematics, masses, macro,
  containment_fraction, containment_category,
  matched_genfatjet_mass, matched_genfatjet_pt,
  visible_desc_multiplicity, file_source,
  ndark_hadrons_per_jet,
  ndark_piD0_per_jet, ndark_piDplus_per_jet, ndark_piDminus_per_jet,
  ndark_rhoD0_per_jet, ndark_rhoDplus_per_jet, ndark_rhoDminus_per_jet

Macro feature order:
  [0] Jet_mass
  [1] Jet_sdmass
  [2] Jet_sdpt
  [3] Jet_ptD
  [4] Jet_msqdr
  [5] Jet_nef
  [6] Jet_cef
  [7] Jet_tau1
  [8] Jet_tau2
  [9] Jet_tau3
  [10] Jet_tau21
  [11] Jet_tau32
"""

import os
import csv
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


BUCKETS = [
    ("ndark1", r"$N_{\mathrm{dark}}=1$"),
    ("ndark2", r"$N_{\mathrm{dark}}=2$"),
    ("ndarkgt2", r"$N_{\mathrm{dark}}>2$"),
]

PID_KEYS = [
    ("ndark_piD0_per_jet", r"$\pi_D^0$"),
    ("ndark_piDplus_per_jet", r"$\pi_D^+$"),
    ("ndark_piDminus_per_jet", r"$\pi_D^-$"),
    ("ndark_rhoD0_per_jet", r"$\rho_D^0$"),
    ("ndark_rhoDplus_per_jet", r"$\rho_D^+$"),
    ("ndark_rhoDminus_per_jet", r"$\rho_D^-$"),
]

CONTAINMENT_LABELS = {
    0: "none",
    1: "partial",
    2: "full",
}


# =============================================================================
# IO helpers
# =============================================================================
def npz_path(input_dir, prefix, bucket):
    return Path(input_dir) / f"{prefix}_{bucket}.npz"


def load_one(path):
    if not Path(path).exists():
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files}
    except Exception as exc:
        print(f"[WARN] Failed loading {path}: {exc}")
        return None


def load_datasets(input_dir, prefix):
    datasets = {}
    for bucket, label in BUCKETS:
        path = npz_path(input_dir, prefix, bucket)
        ds = load_one(path)
        if ds is None:
            continue
        datasets[bucket] = {
            "label": label,
            "path": str(path),
            "data": ds,
        }
        print(f"[INFO] Loaded {bucket}: {path}")
        print(f"       keys: {sorted(ds.keys())}")
        print(f"       entries: {n_entries(ds)}")
    return datasets


def n_entries(ds):
    if "y" in ds:
        return int(len(ds["y"]))
    if "masses" in ds:
        return int(len(ds["masses"]))
    return 0


def get_1d(ds, key, default=None):
    if key not in ds:
        if default is None:
            return np.array([], dtype=np.float32)
        return default
    arr = np.asarray(ds[key])
    return arr.reshape(-1)


def get_mass(ds):
    return get_1d(ds, "masses")


def get_kin(ds, idx):
    if "kinematics" not in ds or len(ds["kinematics"]) == 0:
        return np.array([], dtype=np.float32)
    return np.asarray(ds["kinematics"])[:, idx]


def get_macro(ds, idx):
    if "macro" not in ds or len(ds["macro"]) == 0:
        return np.array([], dtype=np.float32)
    macro = np.asarray(ds["macro"])
    if macro.ndim != 2 or macro.shape[1] <= idx:
        return np.array([], dtype=np.float32)
    return macro[:, idx]


def finite(x):
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]


def finite_positive(x):
    x = finite(x)
    return x[x > 0]


def common_range_from_datasets(datasets, getter, fallback=(0.0, 1.0), positive=False, qclip=None):
    vals = []
    for obj in datasets.values():
        ds = obj["data"]
        x = getter(ds)
        x = finite_positive(x) if positive else finite(x)
        if len(x):
            vals.append(x)
    if not vals:
        return fallback
    x = np.concatenate(vals)
    if len(x) == 0:
        return fallback
    if qclip is not None and len(x) > 20:
        lo, hi = np.nanpercentile(x, qclip)
    else:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return fallback
    pad = 0.04 * (hi - lo)
    return (lo - pad, hi + pad)


# =============================================================================
# Style helpers
# =============================================================================
def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=15, pad=8)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if grid:
        ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)


def savefig(fig, outdir, name, pdf=None):
    outpath = Path(outdir) / name
    fig.tight_layout()
    fig.savefig(outpath)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    return outpath


# =============================================================================
# Plot helpers
# =============================================================================
def overlay_hist(
    datasets,
    getter,
    outdir,
    name,
    xlabel,
    title,
    bins=80,
    hist_range=None,
    density=True,
    positive=False,
    logy=False,
    pdf=None,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    for bucket, label in BUCKETS:
        if bucket not in datasets:
            continue
        ds = datasets[bucket]["data"]
        x = getter(ds)
        x = finite_positive(x) if positive else finite(x)
        if len(x) == 0:
            continue
        ax.hist(
            x,
            bins=bins,
            range=hist_range,
            density=density,
            histtype="step",
            linewidth=2.0,
            label=f"{label}  (N={len(x)})",
        )

    if logy:
        ax.set_yscale("log")
    ylabel = "Density" if density else "Entries"
    style_ax(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=11, frameon=True)
    savefig(fig, outdir, name, pdf=pdf)


def ratio_hist_by_mass(datasets, outdir, mass_range=(1.0, 200.0), mass_bin_width=1.0, pdf=None):
    bins = np.arange(mass_range[0], mass_range[1] + mass_bin_width, mass_bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])

    counts = {}
    total = np.zeros(len(centers), dtype=np.float64)
    for bucket, _ in BUCKETS:
        if bucket not in datasets:
            continue
        mass = get_mass(datasets[bucket]["data"])
        c, _ = np.histogram(mass[np.isfinite(mass)], bins=bins)
        counts[bucket] = c.astype(np.float64)
        total += c

    fig, ax = plt.subplots(figsize=(8, 6))
    for bucket, label in BUCKETS:
        if bucket not in counts:
            continue
        frac = np.divide(counts[bucket], total, out=np.zeros_like(counts[bucket]), where=total > 0)
        ax.step(centers, frac, where="mid", linewidth=2.0, label=label)

    ax.set_ylim(0.0, 1.05)
    style_ax(
        ax,
        xlabel=r"Target dark-hadron mass $m_\pi=m_\rho$ [GeV]",
        ylabel="Fraction of approved jets",
        title=r"Fraction of jets in each $N_{\mathrm{dark}}$ bucket vs target mass",
    )
    ax.legend(fontsize=11, frameon=True)
    savefig(fig, outdir, "fraction_by_ndark_vs_mass.pdf", pdf=pdf)


def entry_count_bar(datasets, outdir, pdf=None):
    labels = []
    counts = []
    for bucket, label in BUCKETS:
        if bucket not in datasets:
            continue
        labels.append(label)
        counts.append(n_entries(datasets[bucket]["data"]))

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(np.arange(len(labels)), counts)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    style_ax(ax, ylabel="Entries", title="Dataset entries by dark-hadron multiplicity bucket")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}", ha="center", va="bottom", fontsize=12)
    savefig(fig, outdir, "entries_by_bucket.pdf", pdf=pdf)


def containment_category_bar(datasets, outdir, pdf=None):
    cats = [0, 1, 2]
    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (bucket, label) in enumerate(BUCKETS):
        if bucket not in datasets:
            continue
        ds = datasets[bucket]["data"]
        c = get_1d(ds, "containment_category").astype(int)
        vals = [np.sum(c == cat) for cat in cats]
        ax.bar(x + (i - 1) * width, vals, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([CONTAINMENT_LABELS[c] for c in cats], fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    style_ax(ax, xlabel="Containment category", ylabel="Entries", title="Containment categories in each bucket")
    ax.legend(fontsize=11, frameon=True)
    savefig(fig, outdir, "containment_category_by_bucket.pdf", pdf=pdf)


def pid_breakdown_bar(datasets, outdir, pdf=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(PID_KEYS))
    width = 0.25

    for i, (bucket, label) in enumerate(BUCKETS):
        if bucket not in datasets:
            continue
        ds = datasets[bucket]["data"]
        vals = []
        for key, _ in PID_KEYS:
            vals.append(np.sum(get_1d(ds, key)))
        ax.bar(x + (i - 1) * width, vals, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in PID_KEYS], fontsize=12)
    style_ax(ax, xlabel="Dark-hadron species", ylabel="Total count inside approved jets", title="PID composition by bucket")
    ax.legend(fontsize=11, frameon=True)
    savefig(fig, outdir, "pid_composition_by_bucket.pdf", pdf=pdf)


def hist2d_for_each_bucket(
    datasets,
    getter_x,
    getter_y,
    outdir,
    name_prefix,
    xlabel,
    ylabel,
    title_prefix,
    bins=80,
    x_range=None,
    y_range=None,
    positive_y=False,
    pdf=None,
):
    for bucket, label in BUCKETS:
        if bucket not in datasets:
            continue
        ds = datasets[bucket]["data"]
        x = getter_x(ds)
        y = getter_y(ds)
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if positive_y:
            mask &= y > 0
        x = x[mask]
        y = y[mask]

        fig, ax = plt.subplots(figsize=(8, 6))
        if len(x) == 0:
            ax.text(0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes)
        else:
            h = ax.hist2d(x, y, bins=bins, range=[x_range, y_range] if x_range and y_range else None)
            cbar = fig.colorbar(h[3], ax=ax)
            cbar.set_label("Entries", fontsize=12)
            cbar.ax.tick_params(labelsize=11)
            if x_range is not None:
                ax.set_xlim(*x_range)
            if y_range is not None:
                ax.set_ylim(*y_range)
        style_ax(ax, xlabel=xlabel, ylabel=ylabel, title=f"{title_prefix}: {label}")
        savefig(fig, outdir, f"{name_prefix}_{bucket}.pdf", pdf=pdf)


def residual_vs_mass(datasets, outdir, mass_range=(1.0, 200.0), pdf=None):
    for bucket, label in BUCKETS:
        if bucket not in datasets:
            continue
        ds = datasets[bucket]["data"]
        true_m = get_mass(ds)
        sd_m = get_macro(ds, 1)
        mask = np.isfinite(true_m) & np.isfinite(sd_m) & (sd_m > 0)
        true_m = true_m[mask]
        sd_m = sd_m[mask]
        residual = sd_m - true_m
        rel_res = np.divide(residual, true_m, out=np.full_like(residual, np.nan), where=true_m > 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        if len(true_m) == 0:
            ax.text(0.5, 0.5, "No valid SoftDrop entries", ha="center", va="center", transform=ax.transAxes)
        else:
            y_lo, y_hi = np.nanpercentile(residual, [1, 99]) if len(residual) > 20 else (np.nanmin(residual), np.nanmax(residual))
            if y_lo == y_hi:
                y_lo -= 1
                y_hi += 1
            h = ax.hist2d(true_m, residual, bins=80, range=[mass_range, (y_lo, y_hi)])
            cbar = fig.colorbar(h[3], ax=ax)
            cbar.set_label("Entries", fontsize=12)
            ax.axhline(0.0, linestyle="--", linewidth=1.2)
        style_ax(ax, xlabel=r"True $m_\pi=m_\rho$ [GeV]", ylabel=r"SoftDrop mass $-$ true mass [GeV]", title=f"SoftDrop residual: {label}")
        savefig(fig, outdir, f"sdmass_residual_vs_true_mass_{bucket}.pdf", pdf=pdf)

        fig, ax = plt.subplots(figsize=(8, 6))
        mask2 = np.isfinite(true_m) & np.isfinite(rel_res)
        if np.sum(mask2) == 0:
            ax.text(0.5, 0.5, "No valid SoftDrop entries", ha="center", va="center", transform=ax.transAxes)
        else:
            rr = rel_res[mask2]
            tm = true_m[mask2]
            y_lo, y_hi = np.nanpercentile(rr, [1, 99]) if len(rr) > 20 else (np.nanmin(rr), np.nanmax(rr))
            if y_lo == y_hi:
                y_lo -= 0.1
                y_hi += 0.1
            h = ax.hist2d(tm, rr, bins=80, range=[mass_range, (y_lo, y_hi)])
            cbar = fig.colorbar(h[3], ax=ax)
            cbar.set_label("Entries", fontsize=12)
            ax.axhline(0.0, linestyle="--", linewidth=1.2)
        style_ax(ax, xlabel=r"True $m_\pi=m_\rho$ [GeV]", ylabel=r"$(m_{SD}-m_{true})/m_{true}$", title=f"SoftDrop relative residual: {label}")
        savefig(fig, outdir, f"sdmass_relative_residual_vs_true_mass_{bucket}.pdf", pdf=pdf)


def write_summary_csv(datasets, outdir):
    path = Path(outdir) / "npz_plot_summary.csv"
    rows = []
    for bucket, label in BUCKETS:
        if bucket not in datasets:
            continue
        ds = datasets[bucket]["data"]
        mass = get_mass(ds)
        jet_pt = get_kin(ds, 0)
        jet_mass = get_kin(ds, 3)
        sd_mass = get_macro(ds, 1)
        ndark = get_1d(ds, "ndark_hadrons_per_jet")
        cf = get_1d(ds, "containment_fraction")
        rows.append({
            "bucket": bucket,
            "label": label.replace("$", ""),
            "file": datasets[bucket]["path"],
            "entries": n_entries(ds),
            "mass_min": np.nanmin(mass) if len(mass) else "",
            "mass_max": np.nanmax(mass) if len(mass) else "",
            "mass_mean": np.nanmean(mass) if len(mass) else "",
            "jet_pt_mean": np.nanmean(jet_pt) if len(jet_pt) else "",
            "jet_mass_mean": np.nanmean(jet_mass) if len(jet_mass) else "",
            "sdmass_valid_entries": int(np.sum(np.isfinite(sd_mass) & (sd_mass > 0))) if len(sd_mass) else 0,
            "sdmass_mean_valid": np.nanmean(sd_mass[np.isfinite(sd_mass) & (sd_mass > 0)]) if len(sd_mass) and np.any(np.isfinite(sd_mass) & (sd_mass > 0)) else "",
            "ndark_min": np.nanmin(ndark) if len(ndark) else "",
            "ndark_max": np.nanmax(ndark) if len(ndark) else "",
            "containment_fraction_mean": np.nanmean(cf) if len(cf) else "",
        })

    if not rows:
        return path

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Make plots for DRACON NPZ files split by nDark hadrons per approved reco FatJet.")
    parser.add_argument("--input-dir", required=True, help="Directory containing <prefix>_ndark1/2/gt2.npz")
    parser.add_argument("--prefix", required=True, help="Common output prefix, e.g. test100_fullonly")
    parser.add_argument("--outdir", default=None, help="Plot output directory. Default: <input-dir>/<prefix>_plots")
    parser.add_argument("--mass-min", type=float, default=1.0, help="Minimum mass for mass plots")
    parser.add_argument("--mass-max", type=float, default=200.0, help="Maximum mass for mass plots")
    parser.add_argument("--mass-bin-width", type=float, default=1.0, help="Mass histogram bin width")
    parser.add_argument("--pt-max", type=float, default=None, help="Optional maximum reco jet pT for overlay histogram")
    parser.add_argument("--no-combined-pdf", action="store_true", help="Do not write combined multipage PDF")
    args = parser.parse_args()

    outdir = args.outdir
    if outdir is None:
        outdir = str(Path(args.input_dir) / f"{args.prefix}_plots")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(args.input_dir, args.prefix)
    if not datasets:
        raise RuntimeError("No input NPZ files could be loaded. Check --input-dir and --prefix.")

    mass_range = (args.mass_min, args.mass_max)
    mass_bins = np.arange(args.mass_min, args.mass_max + args.mass_bin_width, args.mass_bin_width)

    combined_pdf = None
    pdf = None
    if not args.no_combined_pdf:
        combined_pdf = Path(outdir) / f"{args.prefix}_ndark_plots.pdf"
        pdf = PdfPages(combined_pdf)

    try:
        print(f"[INFO] Writing plots to: {outdir}")

        summary_csv = write_summary_csv(datasets, outdir)
        print(f"[INFO] Summary CSV: {summary_csv}")

        entry_count_bar(datasets, outdir, pdf=pdf)
        pid_breakdown_bar(datasets, outdir, pdf=pdf)
        containment_category_bar(datasets, outdir, pdf=pdf)

        overlay_hist(
            datasets,
            getter=get_mass,
            outdir=outdir,
            name="target_mass_overlay.pdf",
            xlabel=r"Target dark-hadron mass $m_\pi=m_\rho$ [GeV]",
            title="Target mass spectrum by nDark bucket",
            bins=mass_bins,
            hist_range=mass_range,
            density=False,
            pdf=pdf,
        )

        overlay_hist(
            datasets,
            getter=get_mass,
            outdir=outdir,
            name="target_mass_overlay_density.pdf",
            xlabel=r"Target dark-hadron mass $m_\pi=m_\rho$ [GeV]",
            title="Target mass density by nDark bucket",
            bins=mass_bins,
            hist_range=mass_range,
            density=True,
            pdf=pdf,
        )

        ratio_hist_by_mass(
            datasets,
            outdir=outdir,
            mass_range=mass_range,
            mass_bin_width=args.mass_bin_width,
            pdf=pdf,
        )

        jet_pt_range = common_range_from_datasets(
            datasets,
            getter=lambda ds: get_kin(ds, 0),
            fallback=(0.0, 1000.0),
            positive=True,
            qclip=(0.0, 99.5),
        )
        if args.pt_max is not None:
            jet_pt_range = (jet_pt_range[0], args.pt_max)

        overlay_hist(
            datasets,
            getter=lambda ds: get_kin(ds, 0),
            outdir=outdir,
            name="recojet_pt_overlay.pdf",
            xlabel=r"Reco FatJet $p_T$ [GeV]",
            title=r"Reco FatJet $p_T$ by nDark bucket",
            bins=80,
            hist_range=jet_pt_range,
            density=True,
            positive=True,
            pdf=pdf,
        )

        jet_mass_range = common_range_from_datasets(
            datasets,
            getter=lambda ds: get_kin(ds, 3),
            fallback=(0.0, 250.0),
            positive=True,
            qclip=(0.0, 99.5),
        )
        overlay_hist(
            datasets,
            getter=lambda ds: get_kin(ds, 3),
            outdir=outdir,
            name="recojet_mass_overlay.pdf",
            xlabel="Reco FatJet mass [GeV]",
            title="Reco FatJet mass by nDark bucket",
            bins=80,
            hist_range=jet_mass_range,
            density=True,
            positive=True,
            pdf=pdf,
        )

        sd_mass_range = common_range_from_datasets(
            datasets,
            getter=lambda ds: get_macro(ds, 1),
            fallback=(0.0, 250.0),
            positive=True,
            qclip=(0.0, 99.5),
        )
        overlay_hist(
            datasets,
            getter=lambda ds: get_macro(ds, 1),
            outdir=outdir,
            name="softdrop_mass_overlay.pdf",
            xlabel="SoftDrop mass [GeV]",
            title="SoftDrop mass by nDark bucket",
            bins=80,
            hist_range=sd_mass_range,
            density=True,
            positive=True,
            pdf=pdf,
        )

        overlay_hist(
            datasets,
            getter=lambda ds: get_1d(ds, "containment_fraction"),
            outdir=outdir,
            name="containment_fraction_overlay.pdf",
            xlabel="Containment fraction",
            title="Containment fraction by nDark bucket",
            bins=50,
            hist_range=(0.0, 1.05),
            density=True,
            pdf=pdf,
        )

        overlay_hist(
            datasets,
            getter=lambda ds: get_1d(ds, "visible_desc_multiplicity"),
            outdir=outdir,
            name="visible_desc_multiplicity_overlay.pdf",
            xlabel="Visible descendant multiplicity",
            title="Visible descendant multiplicity by nDark bucket",
            bins=np.arange(-0.5, 60.5, 1.0),
            hist_range=(-0.5, 60.5),
            density=True,
            pdf=pdf,
        )

        overlay_hist(
            datasets,
            getter=lambda ds: get_1d(ds, "ndark_hadrons_per_jet"),
            outdir=outdir,
            name="ndark_hadrons_per_jet_overlay.pdf",
            xlabel=r"Counted dark hadrons inside reco FatJet",
            title=r"Sanity check: $N_{\mathrm{dark}}$ per approved reco FatJet",
            bins=np.arange(0.5, 20.5, 1.0),
            hist_range=(0.5, 20.5),
            density=False,
            pdf=pdf,
        )

        overlay_hist(
            datasets,
            getter=lambda ds: get_1d(ds, "matched_genfatjet_mass"),
            outdir=outdir,
            name="matched_genfatjet_mass_overlay.pdf",
            xlabel="Matched GenFatJet mass [GeV]",
            title="Matched GenFatJet mass by nDark bucket",
            bins=80,
            hist_range=common_range_from_datasets(datasets, lambda ds: get_1d(ds, "matched_genfatjet_mass"), fallback=(0, 250), positive=True, qclip=(0, 99.5)),
            density=True,
            positive=True,
            pdf=pdf,
        )

        hist2d_for_each_bucket(
            datasets,
            getter_x=get_mass,
            getter_y=lambda ds: get_macro(ds, 1),
            outdir=outdir,
            name_prefix="sdmass_vs_true_mass",
            xlabel=r"True $m_\pi=m_\rho$ [GeV]",
            ylabel="SoftDrop mass [GeV]",
            title_prefix="SoftDrop mass vs true mass",
            bins=80,
            x_range=mass_range,
            y_range=sd_mass_range,
            positive_y=True,
            pdf=pdf,
        )

        hist2d_for_each_bucket(
            datasets,
            getter_x=get_mass,
            getter_y=lambda ds: get_kin(ds, 3),
            outdir=outdir,
            name_prefix="recojet_mass_vs_true_mass",
            xlabel=r"True $m_\pi=m_\rho$ [GeV]",
            ylabel="Reco FatJet mass [GeV]",
            title_prefix="Reco FatJet mass vs true mass",
            bins=80,
            x_range=mass_range,
            y_range=jet_mass_range,
            positive_y=True,
            pdf=pdf,
        )

        residual_vs_mass(datasets, outdir, mass_range=mass_range, pdf=pdf)

    finally:
        if pdf is not None:
            pdf.close()
            print(f"[INFO] Combined PDF: {combined_pdf}")

    print("[DONE] Plotting complete.")


if __name__ == "__main__":
    main()
