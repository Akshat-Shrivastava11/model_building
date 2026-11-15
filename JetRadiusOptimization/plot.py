"""Plot the jet-radius metrics, one curve per model.

Reads JetRadius.pkl (from scan.py) in each model directory matching the
pattern; /store/... patterns are fetched from EOS. Each point is the mean over
truth jets.

  * <metric>.pdf: metric vs visible-jet radius R, at one truth radius R_DH
  * summary_<score>.pdf: mean score vs the model parameter (-x) at several R
    (--dist-radii) and at the best R, and summary_<score>_quantiles.pdf, the
    median and 16-84% / 5-95% ranges of the score at R = R_DH and at the best R
  * <score>_ptbins_<model>.pdf: each score vs R in bins of the truth jet's
    visible pT, and best_radius_vs_pt_<score>.pdf, the best R per pT bin (not
    drawn for bins with fewer than --min-jets truth jets or a zero score)
  * truth_pt_dist.pdf: the truth jets' visible pT, all models overlaid
  * <quantity>_dist_<model>.pdf: distributions at several radii R (--dist-radii),
    truth_radius_dist_<model>.pdf: the truth jets' 90/95/99% containment radii,
    and truth_r<pct>_by_rdh_<model>.pdf: each of them for several truth radii
    R_DH (--dist-truth-radii)
  * truth_radius_vs_rdh.pdf: radii containing 90/95/99% of the truth jets'
    visible pT, vs R_DH; truth_jets_vs_rdh.pdf: truth jets per event and the
    fraction of their visible pT in the two leading ones, vs R_DH
  * if the scans have several truth radii, for each score:
    <metric>_2d_<model>.pdf (quality and iou), the mean over (R_DH, R) with
    the best R per R_DH marked, and best_radius_<metric>.pdf, the best-R stars
    of every model overlaid on the same axes; both also show the truth jets'
    90/95/99% radii (only the 95% one when several models are overlaid);
    best_ratio_<metric>.pdf, best R / R_DH vs R_DH for every model, and
    best_ratio_<model>.pdf, the same for every score with truth DeltaR_95,99 / R_DH
  * if the scans have the event table (scan.py event_scan): mjj_truth_vs_rdh.pdf,
    the two leading truth jets' mass vs R_DH; mjj_vs_r.pdf and mt_vs_r.pdf,
    the visible dijet mass and transverse mass vs R (median and 16-84% range,
    divided by mmed when the models have one); msd_vs_r.pdf, the soft-drop
    mass of the two leading jets (divided by mpi); <mjj|mt|msd>_dist_<model>.pdf
    at several R
  * with --displays, for models with a local events.root (first
    --display-max-events events), at --truth-radius:
    event_<i>_<model>.pdf, one event per truth-jet pT bin: the visible
    particles around the leading truth jet's axis by origin, its truth
    90/95/99% radii, and the matched visible jet at each --display-radii R
    with its own 95% radius; stacked_<model>[_pt<bin>].pdf, all truth jets
    stacked around their axes, one image per origin (this truth jet's dark
    hadrons, other dark hadrons, no dark-hadron ancestor), and their pT vs
    DeltaR stacked by origin, inclusive and per pT bin

Also prints the best R per model and truth radius.

    python3 JetRadiusOptimization/plot.py -m "models/*" -x rinv
"""

import os
import pickle

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from magiconfig import ArgumentParser, ArgumentDefaultsRawHelpFormatter
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

import radius
from common import resolve_models, set_plot_style
from darkancestry import dark_hadron_ancestry
from Histogram import radius_percentile

LABELS = {
    "quality": "quality = acceptance $\\times$ purity",
    "iou": "$p_{\\text{T}}$-weighted IoU with the truth jet",
    "acceptance": "acceptance ($p_{\\text{T}}$ fraction in complete dark hadrons)",
    "purity": "purity ($f_{\\text{full}}$)",
    "f_partial": "$p_{\\text{T}}$ fraction from partly contained dark hadrons",
    "f_nodh": "$p_{\\text{T}}$ fraction with no dark-hadron ancestor",
    "matched": "fraction of truth jets matched",
    "psi_truth": "$\\Psi_{\\text{T}}(R)$: truth visible $p_{\\text{T}}$ fraction within R",
    "jet_r95": "jet $\\Delta R_{95}$",
    "jet_girth": "jet girth $g_{\\text{jet}}$",
    "jet_ptd": "jet $D_{p_{\\text{T}}}$",
    "jet_major": "jet $\\sigma_{\\text{major}}$",
    "jet_minor": "jet $\\sigma_{\\text{minor}}$",
    "jet_nconst": "jet $n_{\\text{const}}$",
}
DIST = ("quality", "iou", "acceptance", "purity", "psi_truth",
        "jet_r95", "jet_girth", "jet_ptd", "jet_major", "jet_minor", "jet_nconst")
# line style of each truth containment radius (percent of visible pT)
TRUTH_STYLES = {90: ":", 95: "--", 99: "-."}
# particle origin relative to one truth jet (displays), drawn in this order
ORIGINS = {
    "no_dh": {"label": "no dark-hadron ancestor (ISR/UE)", "color": "0.6", "cmap": "Greys"},
    "other_dh": {"label": "other dark hadrons", "color": "tab:blue", "cmap": "Blues"},
    "truth_dh": {"label": "this truth jet's dark hadrons", "color": "tab:red", "cmap": "Reds"},
}
DISPLAY_LIM = 2.5   # half-width of the displays in eta and phi
# smaller fonts than set_plot_style for the multi-panel displays
DISPLAY_STYLE = {"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 8,
                 "xtick.labelsize": 10, "ytick.labelsize": 10}


SCORES = ("quality", "iou", "acceptance", "purity")
# color, marker and line style all change from curve to curve, so overlapping curves stay distinguishable
MARKERS = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h")
LINES = ("-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--")
STYLE_CYCLE = cycler(color=plt.cm.tab10.colors) + cycler(marker=MARKERS) + cycler(linestyle=LINES)
QUANTILES = (5, 16, 50, 84, 95)


def styles(n, cmap=None):
    """n curve styles: the cycle above, or colors from a color map with the same markers and lines."""
    colors = cmap(np.linspace(0, 0.9, n)) if cmap else [plt.cm.tab10(i % 10) for i in range(n)]
    return [{"color": c, "marker": MARKERS[i % 10], "ls": LINES[i % 10]} for i, c in enumerate(colors)]


def mean_grid(scan, metric, pt_range=(0, np.inf)):
    """Mean of a metric over truth jets in a visible-pT range: array[truth radius, radius].

    NaN values are skipped; a cell with no values is NaN.
    """
    m = scan["metrics"]
    n_r = len(scan["radii"])
    cell = np.searchsorted(scan["truth_radii"], m["truth_radius"]) * n_r + np.searchsorted(scan["radii"], m["radius"])
    values = m[metric]
    keep = (m["truth_pt"] >= pt_range[0]) & (m["truth_pt"] < pt_range[1]) & ~np.isnan(values)
    size = len(scan["truth_radii"]) * n_r
    total = np.bincount(cell[keep], weights=values[keep], minlength=size)
    count = np.bincount(cell[keep], minlength=size)
    with np.errstate(invalid="ignore"):
        return (total / count).reshape(-1, n_r)


def square_axes(ax, scan, line_color, pad=0.0):
    """Truth radius on x (it is chosen first), visible radius on y: same range, ticks and scale, with R = R_DH.

    pad: extra margin on both axes, e.g. for points spread within their column
    """
    half = [np.min(np.diff(r)) / 2 if len(r) > 1 else 0.1 for r in (scan["radii"], scan["truth_radii"])]
    lo = max(scan["radii"][0] - half[0], scan["truth_radii"][0] - half[1])
    hi = min(scan["radii"][-1] + half[0], scan["truth_radii"][-1] + half[1])
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "-", color=line_color, lw=0.8, alpha=0.5, label="$R = R_{\\text{DH}}$")
    ticks = np.arange(np.ceil(lo * 5) / 5, hi + 1e-9, 0.2)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xticks=ticks, yticks=ticks, aspect="equal",
           xlabel="truth radius $R_{\\text{DH}}$", ylabel="visible jet radius R")


def best_radii(scan, metric, pt_range=(0, np.inf), min_jets=1):
    """Best R per truth radius by the mean metric; NaN where the optimum is undefined.

    Undefined: fewer than min_jets truth jets, or a score that is zero (or NaN) at every R,
    where argmax would just return the first R.
    """
    grid = mean_grid(scan, metric, pt_range)
    m = scan["metrics"]
    first = (m["radius"] == scan["radii"][0]) & (m["truth_pt"] >= pt_range[0]) & (m["truth_pt"] < pt_range[1])
    count = np.array([np.sum(first & (m["truth_radius"] == t)) for t in scan["truth_radii"]])
    filled = np.nan_to_num(grid, nan=-1)
    best = np.array(scan["radii"])[filled.argmax(axis=1)]
    return np.where((count >= min_jets) & (filled.max(axis=1) > 0), best, np.nan), grid


def values_at(scan, metric, truth_radius, radius_value):
    """The metric of every truth jet at one (R_DH, R), NaNs dropped."""
    m = scan["metrics"]
    v = m[metric][(m["truth_radius"] == truth_radius) & (m["radius"] == radius_value)]
    return v[np.isfinite(v)]


def band(ax, x, samples, style, label, alpha=0.15):
    """Median line with the 16-84% range shaded, for a list of samples at the x values."""
    q = np.array([np.nanpercentile(s, (16, 50, 84)) if len(s) else (np.nan,) * 3 for s in samples])
    ax.plot(x, q[:, 1], label=label, **style)
    ax.fill_between(x, q[:, 0], q[:, 2], color=style["color"], alpha=alpha, lw=0)


def legend(ax):
    """Inside the axes for a few curves; below them, in columns, for many."""
    if len(ax.get_legend_handles_labels()[1]) <= 5:
        ax.legend(fontsize="small")
    else:
        ax.legend(fontsize="x-small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))


def truth_radius(scan, pct):
    """Mean radius containing pct% of the truth jets' visible pT, per truth radius."""
    m = scan["metrics"]
    first = m["radius"] == scan["radii"][0]   # one row per truth jet
    return np.array([np.nanmean(m[f"truth_r{pct}"][first & (m["truth_radius"] == t)]) for t in scan["truth_radii"]])


def truth_jet_counts(scan):
    """Per truth radius: mean truth jets per event (with visible decays) and the mean fraction
    of an event's truth visible pT in its two leading truth jets."""
    m = scan["metrics"]
    first = m["radius"] == scan["radii"][0]   # one row per truth jet
    n_events = len(np.unique(scan["events"]["event"])) if "events" in scan else int(m["event"].max()) + 1
    n_jets, lead2 = [], []
    for t in scan["truth_radii"]:
        sel = first & (m["truth_radius"] == t)
        order = np.lexsort((-m["truth_pt"][sel], m["event"][sel]))   # by event, leading jet first
        event, pt = m["event"][sel][order].astype(int), m["truth_pt"][sel][order]
        rank = np.arange(len(event)) - np.searchsorted(event, event)
        total = np.bincount(event, weights=pt)
        leading = np.bincount(event[rank < 2], weights=pt[rank < 2], minlength=len(total))
        n_jets.append(len(event) / n_events)
        lead2.append(np.mean(leading[total > 0] / total[total > 0]))
    return np.array(n_jets), np.array(lead2)


def circle(ax, center, r, label=None, **kw):
    """Circle with a line (not a box) in the legend."""
    ax.add_patch(Circle(center, r, fill=False, **kw))
    ax.plot([], [], color=kw.get("color"), ls=kw.get("ls", "-"), lw=kw.get("lw", 1), marker="none", label=label)


def truth_views(events, masks, truth):
    """Truth jets with visible descendants, [event][truth jet]: the dark-hadron axis, visible pT,
    90/95/99% radii, and every visible particle of the event around that axis with its origin."""
    gen = events.GenCandidate[:, None]   # broadcast over the truth jets
    jets = radius.truth_jets(events, masks, truth)
    in_truth = radius.truth_visible(masks, truth)
    pt = ak.values_astype(events.GenCandidate.pt, np.float64)
    particles = ak.zip({
        "deta": gen.eta - jets.eta[:, :, None],
        "dphi": gen.deltaphi(jets[:, :, None]),
        "dr": gen.deltaR(jets[:, :, None]),
        "pt": pt[:, None],
        "truth_dh": in_truth,
        "other_dh": ~in_truth & (masks != 0)[:, None],
        "no_dh": (masks == 0)[:, None],
    })
    views = ak.zip({
        "truth_jet": ak.local_index(jets.pt), "axis": jets, "truth_pt": ak.sum(pt[:, None] * in_truth, axis=-1),
        **{f"r{p}": radius_percentile(jets, p) for p in radius.SHAPE_PCTS}, "particles": particles,
    }, depth_limit=2)
    return views[ak.any(in_truth, axis=-1)]


def event_display(events, masks, truth, visible, views, i, title, name):
    """Leading truth jet of event i: truth containment (left) and matched visible jets (right)."""
    v = views[i][ak.argmax(views[i].truth_pt)]
    p = v.particles
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    for ax in axes:
        for key, o in ORIGINS.items():
            sel = p[key] & (p.dr < 1.5 * DISPLAY_LIM)
            ax.scatter(p.deta[sel], p.dphi[sel], s=3 + 4 * p.pt[sel], c=o["color"], alpha=0.6, lw=0, label=o["label"])
        ax.plot(0, 0, "k+", ms=12, mew=2)
        ax.set(xlim=(-DISPLAY_LIM, DISPLAY_LIM), ylim=(-DISPLAY_LIM, DISPLAY_LIM), aspect="equal",
               xlabel="$\\Delta\\eta$ to dark-hadron jet axis", ylabel="$\\Delta\\phi$ to dark-hadron jet axis")

    ax = axes[0]
    dark = events.DarkHadronCandidate[i][truth[i][v.truth_jet]]
    ax.plot(dark.eta - v.axis.eta, dark.deltaphi(v.axis), "kx", ms=7, mew=1.5, label=f"dark hadrons ({len(dark)})")
    circle(ax, (0, 0), args.truth_radius, color="0.3", label=f"$R_{{\\text{{DH}}}}$ = {args.truth_radius}")
    for pct, style in TRUTH_STYLES.items():
        circle(ax, (0, 0), v[f"r{pct}"], color="k", ls=style, lw=1.6,
               label=f"truth $\\Delta R_{{{pct}}}$ = {v[f'r{pct}']:.2f}")
    ax.set_title(f"truth jet: visible $p_{{\\text{{T}}}}$ = {v.truth_pt:.0f} GeV", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.8)

    ax = axes[1]
    circle(ax, (0, 0), v.r95, color="k", ls="--", lw=1.2, label=f"truth $\\Delta R_{{95}}$ = {v.r95:.2f}")
    mask = ak.to_numpy(masks[i])
    descends = radius.descendants(mask, len(events.DarkHadronCandidate[i]))
    pt = ak.to_numpy(events.GenCandidate.pt[i]).astype(np.float64)
    truth_dh = [ak.to_numpy(j) for j in truth[i]]
    for r, color in zip(args.display_radii, plt.cm.viridis(np.linspace(0, 0.85, len(args.display_radii)))):
        jets = [ak.to_numpy(j) for j in visible[r][i]]
        rows = radius.event_metrics(descends, pt, mask == 0, truth_dh, jets, 0.95)
        row = next(x for x in rows if x["truth_jet"] == v.truth_jet)
        if row["jet"] < 0:
            continue
        matched = radius.jet_records(radius.gather(events.GenCandidate[i : i + 1], visible[r][i : i + 1]))
        jet = matched[0, row["jet"]]
        center = (jet.eta - v.axis.eta, jet.deltaphi(v.axis))
        j_r95 = radius_percentile(matched, 95)[0, row["jet"]]
        circle(ax, center, r, color=color, lw=2, label=f"R = {r}: quality {row['quality']:.2f}, IoU {row['iou']:.2f}")
        circle(ax, center, j_r95, color=color, ls="--", lw=1.2, label=f"   jet $\\Delta R_{{95}}$ = {j_r95:.2f}")
        ax.plot(*center, "o", color=color, ms=5)
    ax.set_title("matched visible jets: R (solid), jet $\\Delta R_{95}$ (dashed)", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.8)
    fig.suptitle(f"{title}, event {i}, truth jet at $R_{{\\text{{DH}}}}$ = {args.truth_radius}", fontsize=13)
    fig.tight_layout()
    save(fig, name)


def stacked_display(views, title, name):
    """All truth jets stacked around their axes, each normalized to its own visible pT."""
    views = ak.flatten(views)
    if not len(views):
        return
    p = views.particles
    near = p.dr < 2 * DISPLAY_LIM
    w = (p.pt / views.truth_pt / len(views))[near]
    flat = {key: ak.to_numpy(ak.flatten(p[key][near])) for key in ("deta", "dphi", "dr", *ORIGINS)}
    w = ak.to_numpy(ak.flatten(w))
    mean_r = {pct: ak.mean(views[f"r{pct}"]) for pct in radius.SHAPE_PCTS}
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(args.display_radii)))
    order = ("truth_dh", "other_dh", "no_dh")

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.85])
    bins = np.linspace(-DISPLAY_LIM, DISPLAY_LIM, 101)
    for k, key in enumerate(order):
        ax = fig.add_subplot(grid[0, k])
        sel = flat[key]
        _, _, _, image = ax.hist2d(flat["deta"][sel], flat["dphi"][sel], bins=bins, weights=w[sel], norm=LogNorm(),
                                   cmap=ORIGINS[key]["cmap"])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03,
                     label=f"mean fraction of truth $p_{{\\text{{T}}}}$ per bin (total {w[sel].sum():.2f})")
        for pct, style in TRUTH_STYLES.items():
            circle(ax, (0, 0), mean_r[pct], color="k", ls=style, lw=1.8, label=f"mean truth $\\Delta R_{{{pct}}}$ = {mean_r[pct]:.2f}")
        for r, color in zip(args.display_radii, colors):
            circle(ax, (0, 0), r, color=color, lw=1.5, label=f"R = {r}")
        ax.set(xlim=(-DISPLAY_LIM, DISPLAY_LIM), ylim=(-DISPLAY_LIM, DISPLAY_LIM), aspect="equal",
               xlabel="$\\Delta\\eta$ to dark-hadron jet axis", ylabel="$\\Delta\\phi$ to dark-hadron jet axis")
        ax.set_title(ORIGINS[key]["label"], fontsize=12)
        if k == 0:
            ax.legend(loc="upper left", framealpha=0.8)

    ax = fig.add_subplot(grid[1, :])
    rbins = np.linspace(0, DISPLAY_LIM, 51)
    ax.hist([flat["dr"][flat[key]] for key in order], bins=rbins, weights=[w[flat[key]] for key in order], stacked=True,
            color=[ORIGINS[key]["color"] for key in order], alpha=0.8, label=[ORIGINS[key]["label"] for key in order])
    for pct, style in TRUTH_STYLES.items():
        ax.axvline(mean_r[pct], color="k", ls=style, lw=1.8, label=f"mean truth $\\Delta R_{{{pct}}}$ = {mean_r[pct]:.2f}")
    for r, color in zip(args.display_radii, colors):
        ax.axvline(r, color=color, lw=1.5, label=f"R = {r}")
    ax_psi = ax.twinx()   # cumulative fraction of the truth jets' own pT (Psi)
    own = np.histogram(flat["dr"][flat["truth_dh"]], bins=rbins, weights=w[flat["truth_dh"]])[0]
    ax_psi.plot(rbins[1:], np.cumsum(own) / own.sum(), "r-", lw=1.2)
    ax_psi.set_ylim(0, 1.02)
    ax_psi.set_ylabel("cumulative fraction of truth visible $p_{\\text{T}}$ (red line)", color="tab:red")
    ax.set(yscale="log", xlim=(0, DISPLAY_LIM), xlabel="$\\Delta R$ to dark-hadron jet axis",
           ylabel="mean $p_{\\text{T}}$ per bin / truth visible $p_{\\text{T}}$")
    ax.set_title("$p_{\\text{T}}$ vs distance from the dark-hadron axis, stacked by origin", fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.75), framealpha=0.9)
    fig.suptitle(f"{title}: {len(views)} truth jets at $R_{{\\text{{DH}}}}$ = {args.truth_radius}", fontsize=13)
    fig.tight_layout()
    save(fig, name)


def displays(model_dir, title, name, pt_bins, pt_label):
    """Event displays and stacked truth jets for one model, from its events.root."""
    with plt.rc_context(DISPLAY_STYLE):
        _displays(model_dir, title, name, pt_bins, pt_label)


def _displays(model_dir, title, name, pt_bins, pt_label):
    events = radius.load(os.path.join(model_dir, "events.root"))[: args.display_max_events]
    masks = dark_hadron_ancestry(events)
    truth = radius.cluster(events.DarkHadronCandidate, args.truth_radius)
    visible = {r: radius.cluster(events.GenCandidate, r) for r in args.display_radii}
    views = truth_views(events, masks, truth)
    leading = ak.to_numpy(ak.fill_none(ak.max(views.truth_pt, axis=1), 0))
    for (lo, hi), pt_name in zip(pt_bins, pt_label):
        in_bin = np.flatnonzero((leading >= lo) & (leading < hi))
        if len(in_bin):
            event_display(events, masks, truth, visible, views, int(in_bin[0]),
                          f"{title}, leading truth $p_{{\\text{{T}}}}$ {pt_name}", f"event_{in_bin[0]}_{name}")
    stacked_display(views, title, f"stacked_{name}")
    for (lo, hi), pt_name in zip(pt_bins, pt_label):
        stacked_display(views[(views.truth_pt >= lo) & (views.truth_pt < hi)], f"{title}, truth $p_{{\\text{{T}}}}$ {pt_name}",
                        f"stacked_{name}_pt{lo:g}-{hi:g}")


def save(fig, name):
    fig.savefig(os.path.join(args.dir, f"{name}.{args.format}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("-m", "--models", required=True, help="pattern of model directories")
    parser.add_argument("-x", type=str, default="rinv", help="model parameter used to label the curves")
    parser.add_argument("-y", type=str, nargs="+", default=list(LABELS), choices=list(LABELS), help="metrics to plot vs R")
    parser.add_argument("--truth-radius", type=float, default=0.8, help="truth radius R_DH for the plots vs R")
    parser.add_argument("--dist-truth-radii", type=float, nargs="+", default=[0.4, 0.8, 1.0, 1.2, 1.6, 2.0], help="truth radii R_DH for the truth-radius distributions")
    parser.add_argument("--dist-radii", type=float, nargs="+", default=[0.4, 0.8, 1.0, 1.2, 1.6, 2.0], help="radii R for the distributions")
    parser.add_argument("--pt-bins", type=float, nargs="+", default=[0, 50, 100, 200, np.inf], help="edges of the truth-jet visible pT bins [GeV]")
    parser.add_argument("--min-jets", type=int, default=100, help="fewest truth jets in a pT bin for its best R to be shown")
    parser.add_argument("--displays", action="store_true", help="also make event displays and stacked truth jets (needs events.root)")
    parser.add_argument("--display-radii", type=float, nargs="+", default=[0.4, 0.8, 1.0, 1.2, 1.6], help="radii R drawn in the displays")
    parser.add_argument("--display-max-events", type=int, default=1000, help="events read for the displays")
    parser.add_argument("--dir", type=str, default="JetRadius_plots", help="output directory")
    parser.add_argument("--format", type=str, default="pdf", help="plot file format")
    args = parser.parse_args()

    scans = []
    for model_dir in resolve_models(args.models):
        with open(os.path.join(model_dir, "JetRadius.pkl"), "rb") as f:
            scans.append(dict(pickle.load(f), dir=model_dir))
    scans.sort(key=lambda s: s["model"][args.x])

    def label(scan):
        return f"{args.x} = {scan['model'][args.x]}"

    set_plot_style()
    plt.rcParams["axes.prop_cycle"] = STYLE_CYCLE
    os.makedirs(args.dir, exist_ok=True)
    xs = [scan["model"][args.x] for scan in scans]
    at_t = [scan["truth_radii"].index(args.truth_radius) for scan in scans]
    rdh_title = f"truth radius $R_{{\\text{{DH}}}}$ = {args.truth_radius}"

    for metric in args.y:
        fig, ax = plt.subplots()
        for scan, t in zip(scans, at_t):
            ax.plot(scan["radii"], mean_grid(scan, metric)[t], label=label(scan))
        ax.set_xlabel("visible jet radius R")
        ax.set_ylabel(f"mean {LABELS[metric]}")
        ax.set_title(rdh_title, fontsize=14)
        legend(ax)
        save(fig, metric)

    # each score vs the model parameter: means at several R and at the best R, then its distribution
    for metric in SCORES:
        best = [best_radii(scan, metric)[0][t] for scan, t in zip(scans, at_t)]
        fig, ax = plt.subplots()
        for r in args.dist_radii:
            ax.plot(xs, [mean_grid(scan, metric)[t][scan["radii"].index(r)] if r in scan["radii"] else np.nan
                         for scan, t in zip(scans, at_t)], label=f"R = {r}")
        ax.plot(xs, [np.nanmax(mean_grid(scan, metric)[t]) for scan, t in zip(scans, at_t)], "k*-", ms=11, label="best R")
        ax.set(xlabel=args.x, ylabel=f"mean {LABELS[metric]}")
        ax.set_title(rdh_title, fontsize=14)
        legend(ax)
        save(fig, f"summary_{metric}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        for ax, title, radii in ((axes[0], f"R = {args.truth_radius}", [args.truth_radius] * len(scans)),
                                 (axes[1], "best R (labeled)", best)):
            samples = [values_at(scan, metric, args.truth_radius, r) if np.isfinite(r) else np.array([])
                       for scan, r in zip(scans, radii)]
            q = np.array([np.percentile(v, QUANTILES) if len(v) else [np.nan] * len(QUANTILES) for v in samples])
            ax.fill_between(xs, q[:, 0], q[:, 4], color="tab:blue", alpha=0.15, lw=0, label="5-95%")
            ax.fill_between(xs, q[:, 1], q[:, 3], color="tab:blue", alpha=0.3, lw=0, label="16-84%")
            ax.plot(xs, q[:, 2], "o-", color="tab:blue", label="median")
            ax.plot(xs, [v.mean() if len(v) else np.nan for v in samples], "k^--", label="mean")
            if radii is best:
                for x, r, y in zip(xs, best, q[:, 4]):
                    ax.annotate(f"{r:g}", (x, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
            ax.set(xlabel=args.x, title=f"{title}, $R_{{\\text{{DH}}}}$ = {args.truth_radius}")
        axes[0].set_ylabel(LABELS[metric])
        axes[0].legend(fontsize="small")
        fig.tight_layout()
        save(fig, f"summary_{metric}_quantiles")

    # the truth jets' visible pT: how similar the models' jet pT spectra are
    first = [(s["metrics"]["truth_radius"] == args.truth_radius) & (s["metrics"]["radius"] == s["radii"][0]) for s in scans]
    truth_pts = [s["metrics"]["truth_pt"][f] for s, f in zip(scans, first)]
    fig, ax = plt.subplots()
    bins = np.linspace(0, np.percentile(np.concatenate(truth_pts), 99.5), 51)
    for scan, v, style in zip(scans, truth_pts, styles(len(scans))):
        ax.hist(v, bins=bins, histtype="step", density=True, color=style["color"], ls=style["ls"], lw=1.5,
                label=f"{label(scan)}: {len(v)} jets, mean {v.mean():.0f} GeV")
    ax.set(yscale="log", xlabel="truth jet visible $p_{\\text{T}}$ [GeV]", ylabel="truth jets (normalized)")
    ax.set_title(rdh_title, fontsize=14)
    legend(ax)
    save(fig, "truth_pt_dist")

    pt_bins = list(zip(args.pt_bins[:-1], args.pt_bins[1:]))
    pt_label = [f"{lo:g}-{hi:g} GeV" if np.isfinite(hi) else f"> {lo:g} GeV" for lo, hi in pt_bins]
    for metric in SCORES:
        fig_best, ax_best = plt.subplots()
        for scan, t in zip(scans, at_t):
            fig, ax = plt.subplots()
            best = []
            for pt_range, name in zip(pt_bins, pt_label):
                r_best, grid = best_radii(scan, metric, pt_range, args.min_jets)
                ax.plot(scan["radii"], grid[t], label=f"truth $p_{{\\text{{T}}}}$ {name}")
                best.append(r_best[t])
            ax.set_xlabel("visible jet radius R")
            ax.set_ylabel(f"mean {LABELS[metric]}")
            ax.set_title(f"{label(scan)}, $R_{{\\text{{DH}}}}$ = {args.truth_radius}", fontsize=14)
            legend(ax)
            save(fig, f"{metric}_ptbins_{args.x}_{scan['model'][args.x]}")
            ax_best.plot(pt_label, best, label=label(scan))
        ax_best.set_xlabel("truth jet visible $p_{\\text{T}}$")
        ax_best.set_ylabel(f"best visible jet radius R ({metric})")
        ax_best.set_title(f"$R_{{\\text{{DH}}}}$ = {args.truth_radius}; not shown: < {args.min_jets} jets or zero score", fontsize=11)
        legend(ax_best)
        save(fig_best, f"best_radius_vs_pt_{metric}")

    for scan in scans:
        m, name = scan["metrics"], f"{args.x}_{scan['model'][args.x]}"
        at_truth = m["truth_radius"] == args.truth_radius
        for q in DIST:
            values = m[q][at_truth & np.isfinite(m[q])]
            if not len(values):
                continue
            bins = np.linspace(0, np.percentile(values, 99.5), 41) if q not in DIST[:5] else np.linspace(0, 1, 41)
            fig, ax = plt.subplots()
            for r in args.dist_radii:
                ax.hist(m[q][at_truth & (m["radius"] == r)], bins=bins, histtype="step", density=True, label=f"R = {r}")
            ax.set_yscale("log")
            ax.set_xlabel(LABELS[q])
            ax.set_ylabel("truth jets (normalized)")
            ax.set_title(f"{label(scan)}, $R_{{\\text{{DH}}}}$ = {args.truth_radius}", fontsize=14)
            legend(ax)
            save(fig, f"{q}_dist_{name}")
        fig, ax = plt.subplots()
        first = at_truth & (m["radius"] == scan["radii"][0])
        for pct in radius.SHAPE_PCTS:
            ax.hist(m[f"truth_r{pct}"][first], bins=np.linspace(0, 3, 61), histtype="step", density=True, label=f"$\\Delta R_{{{pct}}}$")
        ax.set_yscale("log")
        ax.set_xlabel("radius around the dark-hadron jet axis containing x% of its visible $p_{\\text{T}}$")
        ax.set_ylabel("truth jets (normalized)")
        ax.set_title(f"{label(scan)}, $R_{{\\text{{DH}}}}$ = {args.truth_radius}", fontsize=14)
        legend(ax)
        save(fig, f"truth_radius_dist_{name}")
        first = m["radius"] == scan["radii"][0]   # one row per truth jet
        for pct in radius.SHAPE_PCTS:
            fig, ax = plt.subplots()
            for t in args.dist_truth_radii:
                if t in scan["truth_radii"]:
                    ax.hist(m[f"truth_r{pct}"][first & (m["truth_radius"] == t)], bins=np.linspace(0, 3, 61),
                            histtype="step", density=True, label=f"$R_{{\\text{{DH}}}}$ = {t}")
            ax.set_yscale("log")
            ax.set_xlabel(f"truth $\\Delta R_{{{pct}}}$: radius containing {pct}% of the visible $p_{{\\text{{T}}}}$")
            ax.set_ylabel("truth jets (normalized)")
            ax.set_title(label(scan), fontsize=14)
            legend(ax)
            save(fig, f"truth_r{pct}_by_rdh_{name}")

    fig, ax = plt.subplots()
    for scan, color in zip(scans, plt.cm.viridis(np.linspace(0, 0.9, len(scans)))):
        for pct, style in TRUTH_STYLES.items():
            ax.plot(scan["truth_radii"], truth_radius(scan, pct), style, marker="o", ms=3, color=color,
                    label=f"{label(scan)}, {pct}%" if len(scans) == 1 else None)
        if len(scans) > 1:
            ax.plot([], [], "-", color=color, label=label(scan))
    if len(scans) > 1:   # line styles once, colors per model
        for pct, style in TRUTH_STYLES.items():
            ax.plot([], [], style, color="k", marker="none", label=f"$\\Delta R_{{{pct}}}$")
    ax.set_xlabel("truth radius $R_{\\text{DH}}$")
    ax.set_ylabel("mean truth $\\Delta R_{90/95/99}$")
    legend(ax)
    save(fig, "truth_radius_vs_rdh")

    # inputs for choosing R_DH first: how many truth jets an event has, and how much of the
    # truth visible pT the two leading ones carry (two dark quarks per event)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for scan, style in zip(scans, styles(len(scans))):
        n_jets, lead2 = truth_jet_counts(scan)
        axes[0].plot(scan["truth_radii"], n_jets, label=label(scan), **style)
        axes[1].plot(scan["truth_radii"], lead2, label=label(scan), **style)
    axes[0].set(xlabel="truth radius $R_{\\text{DH}}$", ylabel="mean truth jets per event (with visible decays)")
    axes[0].axhline(2, color="0.5", lw=0.8)
    axes[1].set(xlabel="truth radius $R_{\\text{DH}}$", ylabel="fraction of truth visible $p_{\\text{T}}$ in the 2 leading truth jets")
    legend(axes[1])
    fig.tight_layout()
    save(fig, "truth_jets_vs_rdh")

    multi = [s for s in scans if len(s["truth_radii"]) > 1]
    for metric in SCORES if multi else ():
        fig_best, ax_best = plt.subplots()
        fig_ratio, ax_ratio = plt.subplots()
        # spread the models horizontally within each R_DH column so equal best-R points stay visible
        step = np.min(np.diff(multi[0]["truth_radii"]))
        nudge = (np.arange(len(multi)) - (len(multi) - 1) / 2) * 0.6 * step / max(len(multi), 1)
        for scan, style, dx in zip(multi, styles(len(multi), plt.cm.viridis), nudge):
            best, grid = best_radii(scan, metric)
            truth_r = np.array(scan["truth_radii"])
            if metric in ("quality", "iou"):
                fig, ax = plt.subplots()
                image = ax.pcolormesh(truth_r, scan["radii"], grid.T, shading="nearest")
                ax.plot(truth_r, best, "w*", ms=10)
                for pct, ls in TRUTH_STYLES.items():
                    ax.plot(truth_r, truth_radius(scan, pct), ls, color="w", lw=1.5, label=f"truth $\\Delta R_{{{pct}}}$")
                square_axes(ax, scan, "w")
                # stars and lines can fill any corner, so the legend goes below; dark frame for the white lines
                ax.legend(fontsize="x-small", ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                          facecolor="0.2", labelcolor="w", framealpha=0.9)
                ax.set_title(f"{label(scan)} (star = best R)", fontsize=14)
                fig.colorbar(image, ax=ax, label=f"mean {metric}")
                save(fig, f"{metric}_2d_{args.x}_{scan['model'][args.x]}")
            ax_best.plot(truth_r + dx, best, "*-", color=style["color"], ms=9, lw=1, label=label(scan))
            for pct, ls in TRUTH_STYLES.items() if len(multi) == 1 else [(95, TRUTH_STYLES[95])]:
                ax_best.plot(truth_r, truth_radius(scan, pct), ls, color=style["color"], lw=1, marker="none",
                             label=f"truth $\\Delta R_{{{pct}}}$" if len(multi) == 1 else None)
            ax_ratio.plot(truth_r, best / truth_r, label=label(scan), **style)
            for t, r, q in zip(scan["truth_radii"], best, np.nanmax(grid, axis=1)):
                print(f"{label(scan)}, R_DH = {t}: best R by {metric} = {r} (mean {q:.3f})")
        square_axes(ax_best, multi[0], "k", pad=0.06)
        truth_lines = "truth $\\Delta R_{90/95/99}$ (dotted/dashed/dash-dot)" if len(multi) == 1 else "truth $\\Delta R_{95}$ (dashed)"
        ax_best.set_title(f"best R by {metric} (stars), {truth_lines}" + (", columns spread" if len(multi) > 1 else ""), fontsize=11)
        if len(multi) == 1:
            ax_best.legend(fontsize="x-small", loc="upper left")
        else:
            legend(ax_best)
        save(fig_best, f"best_radius_{metric}")
        ax_ratio.axhline(1, color="0.5", lw=0.8)
        ax_ratio.set(xlabel="truth radius $R_{\\text{DH}}$", ylabel=f"best R / $R_{{\\text{{DH}}}}$ (by {metric})")
        legend(ax_ratio)
        save(fig_ratio, f"best_ratio_{metric}")

    # per model: how far the jet spreads after the decay, R / R_DH, for each score and for the truth containment
    for scan in multi:
        truth_r = np.array(scan["truth_radii"])
        fig, ax = plt.subplots()
        for metric in SCORES:
            ax.plot(truth_r, best_radii(scan, metric)[0] / truth_r, label=f"best R by {metric}")
        for pct in (95, 99):
            ax.plot(truth_r, truth_radius(scan, pct) / truth_r, TRUTH_STYLES[pct], color="k", marker="none",
                    label=f"truth $\\Delta R_{{{pct}}}$ / $R_{{\\text{{DH}}}}$")
        ax.axhline(1, color="0.5", lw=0.8)
        ax.set(xlabel="truth radius $R_{\\text{DH}}$", ylabel="R / $R_{\\text{DH}}$")
        ax.set_title(label(scan), fontsize=14)
        legend(ax)
        save(fig, f"best_ratio_{args.x}_{scan['model'][args.x]}")

    # dijet, transverse and soft-drop mass (event table from scan.py): mjj and mt divided by mmed,
    # the soft-drop mass of the two leading jets by the dark-hadron mass (mpi = mrho), when every model has it
    with_events = [s for s in scans if "events" in s]
    masses = {   # quantity: (label, what it is of, reference mass in the model, its label)
        "mjj": ("$m_{\\text{jj}}$", "two leading visible jets", "mmed", "$m_{\\text{med}}$"),
        "mt": ("$m_{\\text{T}}$", "two leading visible jets", "mmed", "$m_{\\text{med}}$"),
        "msd": ("$m_{\\text{SD}}$", "each of the two leading visible jets", "mpi", "$m_{\\text{dark}}$"),
    }

    def unit(q):
        ref, ref_label = masses[q][2:]
        return f" / {ref_label}" if all(s["model"].get(ref) for s in with_events) else " [GeV]"

    def event_values(scan, q, truth, r):
        e = scan["events"]
        sel = (e["truth"] == truth) & (e["radius"] == r)
        v = np.concatenate([e[c][sel] for c in (("msd1", "msd2") if q == "msd" else (q,)) if c in e])
        v = v[np.isfinite(v)]
        return v / scan["model"][masses[q][2]] if unit(q).startswith(" /") else v

    if with_events:
        ev_styles = styles(len(with_events))
        fig, ax = plt.subplots()
        for scan, style in zip(with_events, ev_styles):
            band(ax, scan["truth_radii"], [event_values(scan, "mjj", 1, t) for t in scan["truth_radii"]], style, label(scan))
        ax.set(xlabel="truth radius $R_{\\text{DH}}$", ylabel=f"two leading truth jets: {masses['mjj'][0]}{unit('mjj')}")
        ax.set_title("dark-hadron jets (incl. stable dark hadrons): median, 16-84%", fontsize=11)
        legend(ax)
        save(fig, "mjj_truth_vs_rdh")
        for q, (q_label, of, _, _) in masses.items():
            if not any(len(event_values(s, q, 0, s["radii"][0])) for s in with_events):
                continue   # e.g. no soft-drop columns in older scans
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            for scan, style in zip(with_events, ev_styles):
                samples = [event_values(scan, q, 0, r) for r in scan["radii"]]
                band(axes[0], scan["radii"], samples, style, label(scan), alpha=0.1)
                qs = np.array([np.percentile(s, (16, 50, 84)) if len(s) else (np.nan,) * 3 for s in samples])
                axes[1].plot(scan["radii"], (qs[:, 2] - qs[:, 0]) / 2 / qs[:, 1], label=label(scan), **style)
            axes[0].set(xlabel="visible jet radius R", ylabel=f"{q_label}{unit(q)} (median, 16-84%)")
            axes[1].set(xlabel="visible jet radius R", ylabel=f"{q_label} relative width (84-16%) / 2 / median")
            legend(axes[1])
            fig.tight_layout()
            save(fig, f"{q}_vs_r")
            for scan in with_events:
                fig, ax = plt.subplots()
                values = np.concatenate([event_values(scan, q, 0, r) for r in args.dist_radii])
                bins = np.linspace(0, np.percentile(values, 99.5), 51) if len(values) else 50
                for r in args.dist_radii:
                    ax.hist(event_values(scan, q, 0, r), bins=bins, histtype="step", density=True, label=f"R = {r}")
                if unit(q).startswith(" /"):
                    ax.axvline(1, color="0.5", lw=0.8)
                ax.set(yscale="log", xlabel=f"{of}: {q_label}{unit(q)}", ylabel="jets (normalized)" if q == "msd" else "events (normalized)")
                ax.set_title(label(scan), fontsize=14)
                legend(ax)
                save(fig, f"{q}_dist_{args.x}_{scan['model'][args.x]}")

    for scan in scans if args.displays else ():
        if os.path.exists(os.path.join(scan["dir"], "events.root")):
            displays(scan["dir"], label(scan), f"{args.x}_{scan['model'][args.x]}", pt_bins, pt_label)
        else:
            print(f"{scan['dir']}: no events.root, no displays")
