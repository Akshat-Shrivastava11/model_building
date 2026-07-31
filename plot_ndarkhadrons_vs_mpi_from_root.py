#!/usr/bin/env python3
import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import awkward as ak
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages


# ==============================================================================
# Produces ONE merged multi-page PDF:
#
#   page 1: piD0 event-vs-approved-jet multiplicity
#   page 2: piDplus event-vs-approved-jet multiplicity
#   page 3: piDminus event-vs-approved-jet multiplicity
#   page 4: rhoD0 event-vs-approved-jet multiplicity
#   page 5: rhoDplus event-vs-approved-jet multiplicity
#   page 6: rhoDminus event-vs-approved-jet multiplicity
#   page 7: total dark-hadron event-vs-approved-jet multiplicity
#
# Each page:
#   left  subplot: x = mdark, y = count per event, color = events
#   right subplot: x = mdark, y = count per approved reco FatJet, color = reco FatJets
#
# Jet approval follows the DRACON training builder:
#   1. dark hadron -> nearest GenFatJet within fatjet_r
#   2. classify dark-hadron visible-descendant containment: full / partial / none
#   3. reco FatJet pT > pt_cut
#   4. reco FatJet -> nearest GenFatJet within matching_r
#   5. keep only categories requested by keep_mode
#
# Default:
#   Count only dark mesons with status 83/84 for multiplicity.
#
# Use --all-statuses to count all generator-record copies with these PIDs.
# ==============================================================================


PID_ORDER = [
    4900111,    # piD0
    4900211,    # piD+
    -4900211,   # piD-
    4900113,    # rhoD0
    4900213,    # rhoD+
    -4900213,   # rhoD-
]

PID_NAMES = {
    4900111: "piD0",
    4900211: "piDplus",
    -4900211: "piDminus",
    4900113: "rhoD0",
    4900213: "rhoDplus",
    -4900213: "rhoDminus",
}

PID_LABELS = {
    4900111: r"$\pi_D^0$",
    4900211: r"$\pi_D^+$",
    -4900211: r"$\pi_D^-$",
    4900113: r"$\rho_D^0$",
    4900213: r"$\rho_D^+$",
    -4900213: r"$\rho_D^-$",
}

DARK_HADRONS = {
    4900111,
    4900211,
    -4900211,
    4900113,
    4900213,
    -4900213,
}

INVISIBLE_PIDS_ABS = {
    12,
    14,
    16,
    18,
    4900101,
    4900102,
    4900103,
    4900104,
    4900105,
    4900106,
}

FATJET_R_DEFAULT = 0.8
MATCHING_R_DEFAULT = 0.4
PT_CUT_DEFAULT = 100.0


def pid_tag(pid):
    return PID_NAMES.get(pid, str(pid).replace("-", "m"))


def parse_mass_from_path(path):
    """
    Parse target dark-hadron mass from path.

    Supported examples:
      point_mPI_12.34
      mpi-12.34
      mpiEqMrho_12p345
      mdark_20
      mDark_20
    """
    path = str(path)

    patterns = [
        r"point_mPI_([0-9]+(?:\.[0-9]+)?)",
        r"mpi-([0-9]+(?:\.[0-9]+)?)",
        r"mpiEqMrho_([0-9]+(?:p[0-9]+)?)",
        r"mDark_([0-9]+(?:\.[0-9]+)?)",
        r"mdark[_-]([0-9]+(?:\.[0-9]+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, path, flags=re.IGNORECASE)
        if m:
            return float(m.group(1).replace("p", "."))

    return None


def dphi(phi1, phi2):
    out = phi1 - phi2
    out = np.where(out > np.pi, out - 2.0 * np.pi, out)
    out = np.where(out < -np.pi, out + 2.0 * np.pi, out)
    return out


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + dphi(phi1, phi2) ** 2)


def is_visible_final_state(pid, status):
    apid = abs(int(pid))

    if int(status) != 1:
        return False

    if apid in INVISIBLE_PIDS_ABS:
        return False

    return True


def get_descendants(idx, d1s, d2s, n_parts, visited=None):
    if visited is None:
        visited = set()

    out = []

    if idx < 0 or idx >= n_parts or idx in visited:
        return out

    visited.add(idx)

    d1 = int(d1s[idx])
    d2 = int(d2s[idx])

    if d1 < 0:
        return out

    if d2 < d1:
        d2 = d1

    for child in range(d1, d2 + 1):
        if 0 <= child < n_parts:
            out.append(child)
            out.extend(get_descendants(child, d1s, d2s, n_parts, visited))

    return out


def containment_category_code(cat):
    return {"full": 2, "partial": 1, "none": 0}.get(cat, -1)


def should_keep_category(cat, keep_mode):
    if keep_mode == "full_only":
        return cat == "full"

    if keep_mode == "full_plus_partial":
        return cat in ["full", "partial"]

    if keep_mode == "all":
        return True

    raise ValueError(f"Unknown keep_mode: {keep_mode}")


def read_filelist(filelist, eos_host):
    files = []

    with open(filelist) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("/store/"):
                line = eos_host.rstrip("/") + "/" + line

            files.append(line)

    return sorted(files)


def discover_root_files(base_dirs):
    files = []

    for base in base_dirs:
        files.extend(
            glob.glob(
                os.path.join(base, "**", "events.root"),
                recursive=True,
            )
        )

    return sorted(files)


def get_delphes_tree(root_file):
    f = uproot.open(root_file)

    if "Delphes;1" in f:
        return f, f["Delphes;1"]

    if "Delphes" in f:
        return f, f["Delphes"]

    f.close()
    raise RuntimeError("Could not find Delphes tree")


def get_branch(tree, possible_names):
    keys = set(tree.keys())

    for name in possible_names:
        if name in keys:
            return tree[name].array(library="ak")

    raise KeyError(f"Could not find any branch from: {possible_names}")


def build_genfatjet_containment_map(
    pids_evt,
    status_evt,
    eta_evt,
    phi_evt,
    mass_evt,
    d1_evt,
    d2_evt,
    gf_eta_evt,
    gf_phi_evt,
    fatjet_r,
):
    """
    This mirrors the training builder's containment logic.

    For each dark hadron:
      - find nearest GenFatJet within fatjet_r
      - collect visible stable descendants
      - measure fraction of visible descendants inside that GenFatJet
      - categorize as full / partial / none
      - keep best dark-hadron candidate per GenFatJet
    """
    genfatjet_best = {}

    dh_indices = np.where(np.isin(pids_evt, list(DARK_HADRONS)))[0]

    for dh_idx in dh_indices:
        dh_eta = eta_evt[dh_idx]
        dh_phi = phi_evt[dh_idx]
        dh_mass = mass_evt[dh_idx]

        if not np.isfinite(dh_eta) or not np.isfinite(dh_phi):
            continue

        if len(gf_eta_evt) == 0:
            continue

        drs_dh_gf = delta_r(dh_eta, dh_phi, gf_eta_evt, gf_phi_evt)
        best_gf_idx = int(np.argmin(drs_dh_gf))
        best_dr = float(drs_dh_gf[best_gf_idx])

        if best_dr >= fatjet_r:
            continue

        desc = sorted(set(get_descendants(dh_idx, d1_evt, d2_evt, len(pids_evt))))

        if len(desc) == 0:
            continue

        vis_desc = []

        for child_idx in desc:
            if is_visible_final_state(pids_evt[child_idx], status_evt[child_idx]):
                if np.isfinite(eta_evt[child_idx]) and np.isfinite(phi_evt[child_idx]):
                    vis_desc.append(child_idx)

        if len(vis_desc) == 0:
            continue

        drs_desc = delta_r(
            eta_evt[vis_desc],
            phi_evt[vis_desc],
            gf_eta_evt[best_gf_idx],
            gf_phi_evt[best_gf_idx],
        )

        inside = drs_desc < fatjet_r
        n_inside = int(np.sum(inside))
        n_total = len(vis_desc)
        frac_inside = n_inside / max(n_total, 1)

        if n_inside == n_total:
            cat = "full"
        elif n_inside > 0:
            cat = "partial"
        else:
            cat = "none"

        rank = {"full": 2, "partial": 1, "none": 0}[cat]

        candidate = {
            "dh_idx": int(dh_idx),
            "dh_mass": float(dh_mass),
            "gf_idx": int(best_gf_idx),
            "category": cat,
            "rank": int(rank),
            "frac_inside": float(frac_inside),
            "n_vis": int(n_total),
        }

        if best_gf_idx not in genfatjet_best:
            genfatjet_best[best_gf_idx] = candidate
        else:
            old = genfatjet_best[best_gf_idx]
            old_key = (old["rank"], old["frac_inside"], old["n_vis"])
            new_key = (candidate["rank"], candidate["frac_inside"], candidate["n_vis"])

            if new_key > old_key:
                genfatjet_best[best_gf_idx] = candidate

    return genfatjet_best


def process_file(
    root_file,
    pids,
    fatjet_r=FATJET_R_DEFAULT,
    matching_r=MATCHING_R_DEFAULT,
    pt_cut=PT_CUT_DEFAULT,
    max_events=None,
    require_status_8384=True,
    keep_mode="full_only",
):
    mass = parse_mass_from_path(root_file)

    if mass is None:
        print(f"[SKIP] Could not parse target mass from path: {root_file}")
        return None

    try:
        f, tree = get_delphes_tree(root_file)

        gen_pid = get_branch(
            tree,
            [
                "GenParticle/GenParticle.PID",
                "GenParticle.PID",
            ],
        )
        gen_status = get_branch(
            tree,
            [
                "GenParticle/GenParticle.Status",
                "GenParticle.Status",
            ],
        )
        gen_eta = get_branch(
            tree,
            [
                "GenParticle/GenParticle.Eta",
                "GenParticle.Eta",
            ],
        )
        gen_phi = get_branch(
            tree,
            [
                "GenParticle/GenParticle.Phi",
                "GenParticle.Phi",
            ],
        )
        gen_mass = get_branch(
            tree,
            [
                "GenParticle/GenParticle.Mass",
                "GenParticle.Mass",
            ],
        )
        gen_d1 = get_branch(
            tree,
            [
                "GenParticle/GenParticle.D1",
                "GenParticle.D1",
            ],
        )
        gen_d2 = get_branch(
            tree,
            [
                "GenParticle/GenParticle.D2",
                "GenParticle.D2",
            ],
        )

        gf_eta = get_branch(
            tree,
            [
                "GenFatJet/GenFatJet.Eta",
                "GenFatJet.Eta",
            ],
        )
        gf_phi = get_branch(
            tree,
            [
                "GenFatJet/GenFatJet.Phi",
                "GenFatJet.Phi",
            ],
        )

        fat_pt = get_branch(
            tree,
            [
                "FatJet/FatJet.PT",
                "FatJet.PT",
            ],
        )
        fat_eta = get_branch(
            tree,
            [
                "FatJet/FatJet.Eta",
                "FatJet.Eta",
            ],
        )
        fat_phi = get_branch(
            tree,
            [
                "FatJet/FatJet.Phi",
                "FatJet.Phi",
            ],
        )

        f.close()

    except Exception as e:
        print(f"[SKIP] Failed reading {root_file}: {e}")
        return None

    n_events = len(gen_pid)

    if max_events is not None:
        n_events = min(n_events, max_events)

    out = {
        "event_mass": [],
        "jet_mass": [],
        "jet_pt": [],
        "n_reco_jets_per_event": [],
        "n_approved_reco_jets_per_event": [],
        "jet_containment_category": [],
        "jet_containment_fraction": [],
    }

    for pid in pids:
        tag = pid_tag(pid)
        out[f"event_count_{tag}"] = []
        out[f"jet_count_{tag}"] = []

    for iev in range(n_events):
        pids_evt = ak.to_numpy(gen_pid[iev])
        status_evt = ak.to_numpy(gen_status[iev])
        eta_evt = ak.to_numpy(gen_eta[iev])
        phi_evt = ak.to_numpy(gen_phi[iev])
        mass_evt = ak.to_numpy(gen_mass[iev])
        d1_evt = ak.to_numpy(gen_d1[iev])
        d2_evt = ak.to_numpy(gen_d2[iev])

        gf_eta_evt = ak.to_numpy(gf_eta[iev])
        gf_phi_evt = ak.to_numpy(gf_phi[iev])

        jpts = ak.to_numpy(fat_pt[iev])
        jetas = ak.to_numpy(fat_eta[iev])
        jphis = ak.to_numpy(fat_phi[iev])

        finite_gen = np.isfinite(eta_evt) & np.isfinite(phi_evt)

        if require_status_8384:
            status_mask_for_counts = np.isin(status_evt, [83, 84])
        else:
            status_mask_for_counts = np.ones_like(pids_evt, dtype=bool)

        # ============================================================
        # Event-level counts: global event count, independent of jet approval.
        # ============================================================
        out["event_mass"].append(mass)

        per_pid_indices_for_counts = {}

        for pid in pids:
            tag = pid_tag(pid)

            mask_pid = (pids_evt == pid) & status_mask_for_counts & finite_gen
            idx = np.where(mask_pid)[0]

            per_pid_indices_for_counts[pid] = idx
            out[f"event_count_{tag}"].append(int(len(idx)))

        # ============================================================
        # Build GenFatJet containment map exactly like the training builder.
        # ============================================================
        genfatjet_best = build_genfatjet_containment_map(
            pids_evt=pids_evt,
            status_evt=status_evt,
            eta_evt=eta_evt,
            phi_evt=phi_evt,
            mass_evt=mass_evt,
            d1_evt=d1_evt,
            d2_evt=d2_evt,
            gf_eta_evt=gf_eta_evt,
            gf_phi_evt=gf_phi_evt,
            fatjet_r=fatjet_r,
        )

        # ============================================================
        # Reco FatJet approval:
        #   pT cut
        #   reco -> nearest GenFatJet within matching_r
        #   matched GenFatJet must have containment classification
        #   keep according to keep_mode
        # ============================================================
        n_good_jets = 0
        n_approved_jets = 0

        for j in range(len(jpts)):
            if not np.isfinite(jpts[j]) or jpts[j] < pt_cut:
                continue

            if not np.isfinite(jetas[j]) or not np.isfinite(jphis[j]):
                continue

            n_good_jets += 1

            if len(gf_eta_evt) == 0:
                continue

            drs_rg = delta_r(jetas[j], jphis[j], gf_eta_evt, gf_phi_evt)
            closest_gen = int(np.argmin(drs_rg))
            closest_dr = float(drs_rg[closest_gen])

            if closest_dr >= matching_r:
                continue

            if closest_gen not in genfatjet_best:
                continue

            match_info = genfatjet_best[closest_gen]
            cat = match_info["category"]

            if not should_keep_category(cat, keep_mode):
                continue

            # This reco FatJet is approved for the jet-level plot.
            n_approved_jets += 1

            out["jet_mass"].append(mass)
            out["jet_pt"].append(float(jpts[j]))
            out["jet_containment_category"].append(containment_category_code(cat))
            out["jet_containment_fraction"].append(float(match_info["frac_inside"]))

            # Count selected dark-hadron PIDs inside this approved reco FatJet.
            for pid in pids:
                tag = pid_tag(pid)
                idx = per_pid_indices_for_counts[pid]

                if len(idx) == 0:
                    n_inside = 0
                else:
                    dr = delta_r(
                        eta_evt[idx],
                        phi_evt[idx],
                        jetas[j],
                        jphis[j],
                    )
                    n_inside = int(np.sum(dr < fatjet_r))

                out[f"jet_count_{tag}"].append(n_inside)

        out["n_reco_jets_per_event"].append(n_good_jets)
        out["n_approved_reco_jets_per_event"].append(n_approved_jets)

    for key in out:
        if "mass" in key or key == "jet_pt" or key == "jet_containment_fraction":
            out[key] = np.asarray(out[key], dtype=np.float32)
        else:
            out[key] = np.asarray(out[key], dtype=np.int32)

    return out


def concatenate_chunks(chunks, pids):
    out = {}

    base_keys = [
        "event_mass",
        "jet_mass",
        "jet_pt",
        "n_reco_jets_per_event",
        "n_approved_reco_jets_per_event",
        "jet_containment_category",
        "jet_containment_fraction",
    ]

    for key in base_keys:
        vals = [c[key] for c in chunks if key in c and len(c[key]) > 0]

        if vals:
            out[key] = np.concatenate(vals)
        else:
            dtype = np.float32 if (
                "mass" in key
                or key == "jet_pt"
                or key == "jet_containment_fraction"
            ) else np.int32
            out[key] = np.array([], dtype=dtype)

    for pid in pids:
        tag = pid_tag(pid)

        for prefix in ["event_count", "jet_count"]:
            key = f"{prefix}_{tag}"
            vals = [c[key] for c in chunks if key in c and len(c[key]) > 0]

            if vals:
                out[key] = np.concatenate(vals)
            else:
                out[key] = np.array([], dtype=np.int32)

    return out


def add_total_counts(arrays, pids):
    n_event = len(arrays["event_mass"])
    n_jet = len(arrays["jet_mass"])

    total_event = np.zeros(n_event, dtype=np.int32)
    total_jet = np.zeros(n_jet, dtype=np.int32)

    for pid in pids:
        tag = pid_tag(pid)
        total_event += arrays[f"event_count_{tag}"]
        total_jet += arrays[f"jet_count_{tag}"]

    arrays["event_count_total_darkhadrons"] = total_event
    arrays["jet_count_total_darkhadrons"] = total_jet

    return arrays


def make_count_bins(y):
    y = np.asarray(y)

    if len(y) == 0:
        return np.arange(-0.5, 1.5, 1.0)

    ymax = int(np.nanmax(y))

    if ymax < 1:
        ymax = 1

    return np.arange(-0.5, ymax + 1.5, 1.0)


def make_heatmap_axis(
    ax,
    x,
    y,
    xbins,
    ylabel,
    title,
    cbar_label,
    logz=True,
    mass_range=None,
):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Force each heatmap panel to be landscape 4:3.
    ax.set_box_aspect(3 / 4)

    if len(x) == 0:
        ax.text(
            0.5,
            0.5,
            "No entries",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return None

    ybins = make_count_bins(y)

    if mass_range is None:
        xrange = (float(np.min(x)), float(np.max(x)))
    else:
        xrange = mass_range

    kwargs = dict(
        bins=[xbins, ybins],
        range=[xrange, (ybins[0], ybins[-1])],
    )

    if logz:
        h = ax.hist2d(x, y, norm=LogNorm(), **kwargs)
    else:
        h = ax.hist2d(x, y, **kwargs)

    cbar = plt.colorbar(
        h[3],
        ax=ax,
        pad=0.018,
        fraction=0.045,
        shrink=0.94,
    )
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel(r"Target dark-hadron mass $m_\pi = m_\rho$ [GeV]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=15)

    xmin, xmax = xrange
    ax.set_xlim(xmin, xmax)

    ymax = int(np.nanmax(y))
    if ymax < 1:
        ymax = 1

    ax.set_ylim(-0.5, ymax + 0.5)

    x_major_start = 0 if xmin < 5 else np.floor(xmin / 25) * 25
    ax.set_xticks(np.arange(x_major_start, xmax + 1, 25))
    ax.set_xticks(np.arange(x_major_start, xmax + 1, 5), minor=True)

    y_major_step = 1 if ymax <= 12 else 5
    ax.set_yticks(np.arange(0, ymax + 1, y_major_step))
    ax.set_yticks(np.arange(0, ymax + 1, 1), minor=True)

    # No white horizontal bin lines.
    ax.grid(which="major", axis="x", alpha=0.12, linewidth=0.6)
    ax.grid(which="minor", axis="x", alpha=0.05, linewidth=0.3)
    ax.grid(which="major", axis="y", alpha=0.08, linewidth=0.5)

    ax.tick_params(which="major", length=5, width=0.9, labelsize=12)
    ax.tick_params(which="minor", length=3, width=0.6)

    return h


def make_pid_figure(
    arrays,
    pid,
    xbins,
    fatjet_r,
    matching_r,
    pt_cut,
    keep_mode,
    status_label,
    logz=True,
    mass_range=None,
):
    tag = pid_tag(pid)
    label = PID_LABELS.get(pid, str(pid))

    event_x = arrays["event_mass"]
    event_y = arrays[f"event_count_{tag}"]

    jet_x = arrays["jet_mass"]
    jet_y = arrays[f"jet_count_{tag}"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(22, 9),
        constrained_layout=True,
    )

    ax0, ax1 = axes

    make_heatmap_axis(
        ax0,
        event_x,
        event_y,
        xbins=xbins,
        ylabel=rf"Number of {label} per event",
        title=rf"Event-level {label} count",
        cbar_label="Events",
        logz=logz,
        mass_range=mass_range,
    )

    make_heatmap_axis(
        ax1,
        jet_x,
        jet_y,
        xbins=xbins,
        ylabel=rf"Number of {label} per approved reco FatJet",
        title=(
            rf"Jet-level {label} count"
            "\n"
            rf"approved: $p_T>{pt_cut:.0f}$ GeV, "
            rf"$\Delta R_{{reco,gen}}<{matching_r}$, "
            rf"keep={keep_mode}, count $\Delta R<{fatjet_r}$"
        ),
        cbar_label="Approved reco FatJets",
        logz=logz,
        mass_range=mass_range,
    )

    fig.suptitle(
        rf"{label} multiplicity vs dark-hadron mass"
        "\n"
        f"PID {pid}, {status_label}",
        fontsize=20,
        y=1.02,
    )

    return fig


def make_total_figure(
    arrays,
    xbins,
    fatjet_r,
    matching_r,
    pt_cut,
    keep_mode,
    status_label,
    logz=True,
    mass_range=None,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(22, 9),
        constrained_layout=True,
    )

    ax0, ax1 = axes

    make_heatmap_axis(
        ax0,
        arrays["event_mass"],
        arrays["event_count_total_darkhadrons"],
        xbins=xbins,
        ylabel="Total number of dark hadrons per event",
        title="Event-level total dark-hadron count",
        cbar_label="Events",
        logz=logz,
        mass_range=mass_range,
    )

    make_heatmap_axis(
        ax1,
        arrays["jet_mass"],
        arrays["jet_count_total_darkhadrons"],
        xbins=xbins,
        ylabel="Total number of dark hadrons per approved reco FatJet",
        title=(
            "Jet-level total dark-hadron count"
            "\n"
            rf"approved: $p_T>{pt_cut:.0f}$ GeV, "
            rf"$\Delta R_{{reco,gen}}<{matching_r}$, "
            rf"keep={keep_mode}, count $\Delta R<{fatjet_r}$"
        ),
        cbar_label="Approved reco FatJets",
        logz=logz,
        mass_range=mass_range,
    )

    fig.suptitle(
        f"Total dark-hadron multiplicity vs dark-hadron mass\n{status_label}",
        fontsize=20,
        y=1.02,
    )

    return fig


def print_summary(arrays, pids):
    print("=" * 80)
    print("SUMMARY")
    print(f"N event rows               : {len(arrays['event_mass'])}")
    print(f"N approved reco FatJets    : {len(arrays['jet_mass'])}")

    if len(arrays["event_mass"]):
        print(
            f"event mass range           : "
            f"{float(np.min(arrays['event_mass'])):.6g} to "
            f"{float(np.max(arrays['event_mass'])):.6g}"
        )

    if len(arrays["jet_mass"]):
        print(
            f"approved jet mass range    : "
            f"{float(np.min(arrays['jet_mass'])):.6g} to "
            f"{float(np.max(arrays['jet_mass'])):.6g}"
        )
        print(
            f"approved jet pT range      : "
            f"{float(np.min(arrays['jet_pt'])):.6g} to "
            f"{float(np.max(arrays['jet_pt'])):.6g}"
        )

    if len(arrays["n_reco_jets_per_event"]):
        print(
            f"mean reco jets/event pTcut : "
            f"{float(np.mean(arrays['n_reco_jets_per_event'])):.3f}"
        )

    if len(arrays["n_approved_reco_jets_per_event"]):
        print(
            f"mean approved jets/event   : "
            f"{float(np.mean(arrays['n_approved_reco_jets_per_event'])):.3f}"
        )

    if len(arrays["jet_containment_category"]):
        vals, counts = np.unique(arrays["jet_containment_category"], return_counts=True)
        label_map = {2: "full", 1: "partial", 0: "none"}
        print("approved jet categories    :")
        for v, c in zip(vals, counts):
            print(f"  {label_map.get(int(v), str(v)):8s}: {int(c)}")

    print("-" * 80)
    print(
        f"Total dark hadrons/event   : "
        f"mean={float(np.mean(arrays['event_count_total_darkhadrons'])):.3f}, "
        f"min={int(np.min(arrays['event_count_total_darkhadrons']))}, "
        f"max={int(np.max(arrays['event_count_total_darkhadrons']))}"
    )

    if len(arrays["jet_count_total_darkhadrons"]):
        print(
            f"Total dark hadrons/jet     : "
            f"mean={float(np.mean(arrays['jet_count_total_darkhadrons'])):.3f}, "
            f"min={int(np.min(arrays['jet_count_total_darkhadrons']))}, "
            f"max={int(np.max(arrays['jet_count_total_darkhadrons']))}"
        )

    for pid in pids:
        tag = pid_tag(pid)
        label = PID_LABELS.get(pid, str(pid))

        e = arrays[f"event_count_{tag}"]
        j = arrays[f"jet_count_{tag}"]

        print("-" * 80)
        print(f"PID {pid} {label}")

        if len(e):
            print(
                f"  event count: "
                f"mean={float(np.mean(e)):.3f}, "
                f"min={int(np.min(e))}, "
                f"max={int(np.max(e))}, "
                f"zero_frac={float(np.mean(e == 0)):.3f}"
            )

        if len(j):
            print(
                f"  approved jet count: "
                f"mean={float(np.mean(j)):.3f}, "
                f"min={int(np.min(j))}, "
                f"max={int(np.max(j))}, "
                f"zero_frac={float(np.mean(j == 0)):.3f}"
            )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Make one merged PDF with event-vs-approved-jet multiplicity plots "
            "for each dark-hadron PID plus total dark-hadron count."
        )
    )

    parser.add_argument("--filelist", default=None)
    parser.add_argument("--eos-host", default="root://cmseos.fnal.gov")
    parser.add_argument("--base-dirs", nargs="+", default=None)
    parser.add_argument("--outdir", required=True)

    parser.add_argument(
        "--pids",
        nargs="+",
        type=int,
        default=PID_ORDER,
        help="Exact signed PIDs to count.",
    )

    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Count selected PIDs at all statuses. Default requires status 83 or 84.",
    )

    parser.add_argument("--pt-cut", type=float, default=PT_CUT_DEFAULT)

    parser.add_argument(
        "--fatjet-r",
        "--jet-r",
        dest="fatjet_r",
        type=float,
        default=FATJET_R_DEFAULT,
        help="FatJet radius used for containment and per-jet dark-hadron counting.",
    )

    parser.add_argument(
        "--matching-r",
        type=float,
        default=MATCHING_R_DEFAULT,
        help="Reco FatJet to GenFatJet matching radius, matching the training builder.",
    )

    parser.add_argument(
        "--keep-mode",
        default="full_only",
        choices=["full_only", "full_plus_partial", "all"],
        help="Which containment categories to keep for jet-level plots.",
    )

    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-events-per-file", type=int, default=None)

    parser.add_argument("--xbins", type=int, default=None)
    parser.add_argument("--linear-z", action="store_true")

    parser.add_argument("--mass-min", type=float, default=None)
    parser.add_argument("--mass-max", type=float, default=None)

    parser.add_argument(
        "--pdf-name",
        default="merged_pid_and_total_darkhadron_multiplicity.pdf",
        help="Name of the single merged output PDF.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    require_status_8384 = not args.all_statuses
    status_label = "status 83/84" if require_status_8384 else "all statuses"

    mass_range = None
    if args.mass_min is not None or args.mass_max is not None:
        if args.mass_min is None or args.mass_max is None:
            raise RuntimeError("Use both --mass-min and --mass-max, or neither.")
        mass_range = (args.mass_min, args.mass_max)

    if args.xbins is None:
        if mass_range is not None:
            args.xbins = int(round(args.mass_max - args.mass_min))
            print(f"[INFO] Using automatic 1 GeV mass binning: xbins = {args.xbins}")
        else:
            args.xbins = 100
            print(f"[INFO] Using default xbins = {args.xbins}")

    if args.filelist is not None:
        root_files = read_filelist(args.filelist, args.eos_host)
    else:
        if args.base_dirs is None:
            raise RuntimeError("Provide either --filelist or --base-dirs")
        root_files = discover_root_files(args.base_dirs)

    if args.max_files is not None:
        root_files = root_files[:args.max_files]

    if len(root_files) == 0:
        raise RuntimeError("No ROOT files found.")

    print("=" * 80)
    print("PID-by-PID dark-hadron multiplicity plots")
    print(f"N ROOT files          : {len(root_files)}")
    print(f"Output directory      : {outdir}")
    print(f"PIDs                  : {args.pids}")
    print(f"Status selection      : {status_label}")
    print(f"Reco FatJet pT cut    : {args.pt_cut}")
    print(f"FatJet R              : {args.fatjet_r}")
    print(f"Reco-gen matching R   : {args.matching_r}")
    print(f"Jet keep mode         : {args.keep_mode}")
    print(f"Mass bins             : {args.xbins}")
    print("=" * 80)

    chunks = []

    for i, rf in enumerate(root_files):
        print(f"[{i + 1}/{len(root_files)}] {rf}")

        chunk = process_file(
            rf,
            pids=args.pids,
            fatjet_r=args.fatjet_r,
            matching_r=args.matching_r,
            pt_cut=args.pt_cut,
            max_events=args.max_events_per_file,
            require_status_8384=require_status_8384,
            keep_mode=args.keep_mode,
        )

        if chunk is not None:
            chunks.append(chunk)

    if len(chunks) == 0:
        raise RuntimeError("No valid ROOT files processed.")

    arrays = concatenate_chunks(chunks, args.pids)
    arrays = add_total_counts(arrays, args.pids)

    npz_path = outdir / "pid_multiplicity_per_pid_and_total_arrays.npz"
    np.savez(
        npz_path,
        **arrays,
        pids=np.asarray(args.pids, dtype=np.int32),
    )
    print(f"[SAVE] {npz_path}")

    print_summary(arrays, args.pids)

    logz = not args.linear_z
    merged_pdf_path = outdir / args.pdf_name

    with PdfPages(merged_pdf_path) as pdf:
        # --------------------------------------------------------------
        # One PDF page per PID
        # --------------------------------------------------------------
        for pid in args.pids:
            tag = pid_tag(pid)

            fig = make_pid_figure(
                arrays=arrays,
                pid=pid,
                xbins=args.xbins,
                fatjet_r=args.fatjet_r,
                matching_r=args.matching_r,
                pt_cut=args.pt_cut,
                keep_mode=args.keep_mode,
                status_label=status_label,
                logz=logz,
                mass_range=mass_range,
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            print(f"[PDF PAGE] PID {pid} -> {tag}")

        # --------------------------------------------------------------
        # Final PDF page: total dark-hadron count
        # --------------------------------------------------------------
        fig = make_total_figure(
            arrays=arrays,
            xbins=args.xbins,
            fatjet_r=args.fatjet_r,
            matching_r=args.matching_r,
            pt_cut=args.pt_cut,
            keep_mode=args.keep_mode,
            status_label=status_label,
            logz=logz,
            mass_range=mass_range,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        print("[PDF PAGE] total dark hadrons")

    print(f"[SAVE MERGED PDF] {merged_pdf_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()