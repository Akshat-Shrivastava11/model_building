#!/usr/bin/env python3
"""
build_dracon_complete_genmatch_dataset_multiproc.py

Multiprocessing version of the complete genmatch DRACON dataset builder.

Builds DRACON training dataset from:
  SVJ_Training_2D_20260328_1534/
  SVJ_Training_2D_20260331_1625/

Uses complete dark-hadron matching logic:
  - identify dark hadrons
  - collect all visible stable descendants
  - match dark hadron to nearest GenFatJet
  - classify containment:
      full / partial / none
  - match Reco FatJet to GenFatJet
  - optionally keep only fully matched jets for training

Parallelization:
  - file-level multiprocessing with ProcessPoolExecutor

SoftDrop support:
  - tries FatJet/FatJet.SoftDroppedJet.Mass and .PT
  - falls back to FatJet/FatJet.SoftDroppedP4[5]

Output:
  NPZ with:
    X                -> micro constituent tensor, shape (N, MAX_CONSTITUENTS, 16)
    y                -> labels
    kinematics       -> [pt, eta, phi, mass]
    masses           -> [m_pi]
    macro            -> macro features
    containment_fraction
    containment_category
    matched_genfatjet_mass
    matched_genfatjet_pt
    visible_desc_multiplicity
    file_source
"""

import os
import re
import glob
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================
# DEFAULT CONFIG
# =========================================================
DEFAULT_BASE_DIRS = [
    "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Smooth_Dracon_20260325_1238",
    "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Smooth_Dracon_20260325_1323",
    "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Training_2D_20260328_1534",
    "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Training_2D_20260331_1625",
    "/lustre/research/hep/akshriva/SVJ_RandD/DRACON/training_data/SVJ_Training_2D_20260316_2324",

]

DEFAULT_OUTDIR = "/lustre/research/hep/akshriva/SVJ_RandD/DRACON/dataset_complete_genmatch_(moredata)_study_mpi=mrho_and_mpi=mrhoover2"
DEFAULT_OUTPUT_NAME = "dragon_complete_(moredata)_genmatch_fullonly_mpi=mrho_and_mpi=mrhoover2_sdmass.npz"



MAX_CONSTITUENTS = 100
FATJET_R = 0.8
MATCHING_R = 0.4
PT_CUT = 100.0

DARK_HADRONS = {4900111, 4900211, -4900211, 4900113, 4900213, -4900213}
INVISIBLE_PIDS_ABS = {12, 14, 16, 18, 4900101, 4900102, 4900103, 4900104, 4900105, 4900106}


# =========================================================
# HELPERS
# =========================================================
def calc_dphi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = calc_dphi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def center_coords(etas, phis, center_eta, center_phi):
    d_eta = etas - center_eta
    d_phi = phis - center_phi
    d_phi = np.where(d_phi > np.pi, d_phi - 2 * np.pi, d_phi)
    d_phi = np.where(d_phi < -np.pi, d_phi + 2 * np.pi, d_phi)
    return d_eta, d_phi


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


def parse_mpi_from_path(path):
    # Old formats
    m = re.search(r"point_mPI_([0-9]+(?:\.[0-9]+)?)", path)
    if m:
        return float(m.group(1))

    m = re.search(r"mpi-([0-9]+(?:\.[0-9]+)?)", path)
    if m:
        return float(m.group(1))

    # New corrected EOS format:
    #   point_000123_mpiEqMrho_12p345/...
    m = re.search(r"mpiEqMrho_([0-9]+(?:p[0-9]+)?)", path)
    if m:
        return float(m.group(1).replace("p", "."))

    return None


def category_code(cat):
    return {"full": 2, "partial": 1, "none": 0}.get(cat, -1)


def safe_hist(ax, data, bins=50, range=None, label=None, density=False, histtype="step", lw=2):
    data = np.asarray(data)
    if len(data) == 0:
        return
    ax.hist(data, bins=bins, range=range, label=label, density=density, histtype=histtype, linewidth=lw)


def maybe_subsample(*arrays, max_n=50000, seed=42):
    n = len(arrays[0])
    if n <= max_n:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return [np.asarray(a)[idx] for a in arrays]


def ensure_dirs(outdir):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "pre_dataset"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "post_dataset"), exist_ok=True)


def safe_softdrop_mass(rf_sdmass, event_idx, jet_idx):
    if rf_sdmass is None:
        return -1.0
    if len(rf_sdmass[event_idx]) <= jet_idx:
        return -1.0
    val = rf_sdmass[event_idx][jet_idx]
    try:
        return float(val)
    except Exception:
        return -1.0


def safe_softdrop_pt(rf_sdpt, event_idx, jet_idx):
    if rf_sdpt is None:
        return -1.0
    if len(rf_sdpt[event_idx]) <= jet_idx:
        return -1.0
    val = rf_sdpt[event_idx][jet_idx]
    try:
        return float(val)
    except Exception:
        return -1.0


# =========================================================
# PER-FILE WORKER
# =========================================================
def process_one_root_file(args):
    """
    Process one ROOT file independently.
    """
    root_file, file_idx, keep_mode, max_events_per_file, fatjet_r, matching_r, pt_cut = args

    result = {
        "ok": False,
        "root_file": root_file,
        "file_idx": file_idx,
        "error": None,

        # counters
        "n_total_recojets": 0,
        "n_ptcut_recojets": 0,
        "n_recojets_matched_to_gen": 0,
        "n_recojets_kept": 0,

        # pre-dataset diagnostics
        "pre_full_frac": [],
        "pre_partial_frac": [],
        "pre_none_frac": [],
        "pre_full_gf_mass": [],
        "pre_partial_gf_mass": [],
        "pre_none_gf_mass": [],
        "pre_full_gf_pt": [],
        "pre_partial_gf_pt": [],
        "pre_none_gf_pt": [],
        "pre_full_nvis": [],
        "pre_partial_nvis": [],
        "pre_none_nvis": [],
        "pre_full_dh_mass": [],
        "pre_partial_dh_mass": [],
        "pre_none_dh_mass": [],
        "pre_best_dh_to_gf_dr": [],
        "pre_reco_to_gen_dr": [],

        # dataset pieces
        "X_jets": [],
        "Y_labels": [],
        "Jet_kinematics": [],
        "Mass_params": [],
        "Macro_features": [],
        "Containment_fraction": [],
        "Containment_category": [],
        "Matched_genfatjet_mass": [],
        "Matched_genfatjet_pt": [],
        "Visible_desc_multiplicity": [],
        "File_source": [],
    }

    target_mpi = parse_mpi_from_path(root_file)
    if target_mpi is None:
        result["error"] = f"Could not parse m_pi from path: {root_file}"
        return result

    try:
        f = uproot.open(root_file)
    except Exception as e:
        result["error"] = f"Failed opening file: {e}"
        return result

    if "Delphes;1" not in f:
        result["error"] = "No Delphes;1 tree found"
        return result

    try:
        tree = f["Delphes;1"]

        # -------------------------------------------------
        # Gen particles
        # -------------------------------------------------
        gen_pid    = tree["GenParticle/GenParticle.PID"].array()
        gen_eta    = tree["GenParticle/GenParticle.Eta"].array()
        gen_phi    = tree["GenParticle/GenParticle.Phi"].array()
        gen_pt     = tree["GenParticle/GenParticle.PT"].array()
        gen_mass   = tree["GenParticle/GenParticle.Mass"].array()
        gen_status = tree["GenParticle/GenParticle.Status"].array()
        gen_d1     = tree["GenParticle/GenParticle.D1"].array()
        gen_d2     = tree["GenParticle/GenParticle.D2"].array()

        # -------------------------------------------------
        # GenFatJets
        # -------------------------------------------------
        gf_eta  = tree["GenFatJet/GenFatJet.Eta"].array()
        gf_phi  = tree["GenFatJet/GenFatJet.Phi"].array()
        gf_pt   = tree["GenFatJet/GenFatJet.PT"].array()
        gf_mass = tree["GenFatJet/GenFatJet.Mass"].array()

        # -------------------------------------------------
        # Reco FatJets
        # -------------------------------------------------
        rf_pt     = tree["FatJet/FatJet.PT"].array()
        rf_eta    = tree["FatJet/FatJet.Eta"].array()
        rf_phi    = tree["FatJet/FatJet.Phi"].array()
        rf_m      = tree["FatJet/FatJet.Mass"].array()
        rf_ptd    = tree["FatJet/FatJet.PTD"].array()
        rf_msqdr  = tree["FatJet/FatJet.MeanSqDeltaR"].array()
        rf_nef    = tree["FatJet/FatJet.NeutralEnergyFraction"].array()
        rf_cef    = tree["FatJet/FatJet.ChargedEnergyFraction"].array()
        rf_tau    = tree["FatJet/FatJet.Tau[5]"].array()

        # -------------------------------------------------
        # SoftDrop branches
        # -------------------------------------------------
        rf_sdmass = None
        rf_sdpt = None

        try:
            rf_sdmass = tree["FatJet/FatJet.SoftDroppedJet.Mass"].array()
            print(f"[INFO][{os.path.basename(root_file)}] Using SoftDrop mass branch: FatJet/FatJet.SoftDroppedJet.Mass")
        except Exception:
            pass

        try:
            rf_sdpt = tree["FatJet/FatJet.SoftDroppedJet.PT"].array()
            print(f"[INFO][{os.path.basename(root_file)}] Using SoftDrop pt branch: FatJet/FatJet.SoftDroppedJet.PT")
        except Exception:
            pass

        if rf_sdmass is None or rf_sdpt is None:
            try:
                rf_sdp4_raw = tree["FatJet/FatJet.SoftDroppedP4[5]"].array()
                print(f"[INFO][{os.path.basename(root_file)}] Using SoftDrop packed branch: FatJet/FatJet.SoftDroppedP4[5]")

                tmp_pt = []
                tmp_mass = []

                for i_evt in range(len(rf_sdp4_raw)):
                    evt_pt = []
                    evt_mass = []

                    for j in range(len(rf_sdp4_raw[i_evt])):
                        jet_entry = rf_sdp4_raw[i_evt][j]

                        if len(jet_entry) == 0:
                            evt_pt.append(-1.0)
                            evt_mass.append(-1.0)
                            continue

                        vec = jet_entry[0]

                        px = vec["fP"]["fX"]
                        py = vec["fP"]["fY"]
                        pz = vec["fP"]["fZ"]
                        E  = vec["fE"]

                        pt = np.sqrt(px**2 + py**2)
                        m2 = E**2 - px**2 - py**2 - pz**2
                        mass = np.sqrt(max(m2, 0.0))

                        evt_pt.append(float(pt))
                        evt_mass.append(float(mass))

                    tmp_pt.append(evt_pt)
                    tmp_mass.append(evt_mass)

                if rf_sdpt is None:
                    rf_sdpt = ak.Array(tmp_pt)
                if rf_sdmass is None:
                    rf_sdmass = ak.Array(tmp_mass)

            except Exception as e:
                print(f"[WARNING][{os.path.basename(root_file)}] Failed to build SoftDrop from SoftDroppedP4[5]: {e}")

        if rf_sdmass is None:
            print(f"[WARNING][{os.path.basename(root_file)}] No SoftDrop mass branch found")
        if rf_sdpt is None:
            print(f"[WARNING][{os.path.basename(root_file)}] No SoftDrop pt branch found")

        # -------------------------------------------------
        # PF candidates
        # -------------------------------------------------
        pf_pt     = tree["ParticleFlowCandidate/ParticleFlowCandidate.PT"].array()
        pf_eta    = tree["ParticleFlowCandidate/ParticleFlowCandidate.Eta"].array()
        pf_phi    = tree["ParticleFlowCandidate/ParticleFlowCandidate.Phi"].array()
        pf_e      = tree["ParticleFlowCandidate/ParticleFlowCandidate.E"].array()
        pf_charge = tree["ParticleFlowCandidate/ParticleFlowCandidate.Charge"].array()
        pf_pid    = tree["ParticleFlowCandidate/ParticleFlowCandidate.PID"].array()
        pf_d0     = tree["ParticleFlowCandidate/ParticleFlowCandidate.D0"].array()
        pf_dz     = tree["ParticleFlowCandidate/ParticleFlowCandidate.DZ"].array()
        pf_eem    = tree["ParticleFlowCandidate/ParticleFlowCandidate.Eem"].array()
        pf_ehad   = tree["ParticleFlowCandidate/ParticleFlowCandidate.Ehad"].array()

        num_events = len(rf_pt)
        if max_events_per_file is not None:
            num_events = min(num_events, max_events_per_file)

        for i in range(num_events):
            pids_evt    = ak.to_numpy(gen_pid[i])
            eta_evt     = ak.to_numpy(gen_eta[i])
            phi_evt     = ak.to_numpy(gen_phi[i])
            pt_evt      = ak.to_numpy(gen_pt[i])
            mass_evt    = ak.to_numpy(gen_mass[i])
            status_evt  = ak.to_numpy(gen_status[i])
            d1_evt      = ak.to_numpy(gen_d1[i])
            d2_evt      = ak.to_numpy(gen_d2[i])

            gf_eta_evt  = ak.to_numpy(gf_eta[i])
            gf_phi_evt  = ak.to_numpy(gf_phi[i])
            gf_pt_evt   = ak.to_numpy(gf_pt[i])
            gf_mass_evt = ak.to_numpy(gf_mass[i])

            rfs_pt    = ak.to_numpy(rf_pt[i])
            rfs_eta   = ak.to_numpy(rf_eta[i])
            rfs_phi   = ak.to_numpy(rf_phi[i])
            rfs_m     = ak.to_numpy(rf_m[i])
            rfs_ptd   = ak.to_numpy(rf_ptd[i])
            rfs_msqdr = ak.to_numpy(rf_msqdr[i])
            rfs_nef   = ak.to_numpy(rf_nef[i])
            rfs_cef   = ak.to_numpy(rf_cef[i])
            rfs_tau   = ak.to_numpy(rf_tau[i])

            pfs_pt     = ak.to_numpy(pf_pt[i])
            pfs_eta    = ak.to_numpy(pf_eta[i])
            pfs_phi    = ak.to_numpy(pf_phi[i])
            pfs_e      = ak.to_numpy(pf_e[i])
            pfs_charge = ak.to_numpy(pf_charge[i])
            pfs_pid    = ak.to_numpy(pf_pid[i])
            pfs_d0     = ak.to_numpy(pf_d0[i])
            pfs_dz     = ak.to_numpy(pf_dz[i])
            pfs_eem    = ak.to_numpy(pf_eem[i])
            pfs_ehad   = ak.to_numpy(pf_ehad[i])

            # -------------------------------------------------
            # classify dark hadrons by containment in GenFatJets
            # -------------------------------------------------
            genfatjet_best = {}
            dh_indices = np.where(np.isin(pids_evt, list(DARK_HADRONS)))[0]

            for dh_idx in dh_indices:
                dh_eta = eta_evt[dh_idx]
                dh_phi = phi_evt[dh_idx]
                dh_mass = mass_evt[dh_idx]

                if np.isnan(dh_eta) or np.isnan(dh_phi) or np.isinf(dh_eta) or np.isinf(dh_phi):
                    continue
                if len(gf_eta_evt) == 0:
                    continue

                drs_dh_gf = calc_dr(dh_eta, dh_phi, gf_eta_evt, gf_phi_evt)
                best_gf_idx = int(np.argmin(drs_dh_gf))
                best_dr = float(drs_dh_gf[best_gf_idx])
                result["pre_best_dh_to_gf_dr"].append(best_dr)

                if best_dr >= fatjet_r:
                    continue

                desc = sorted(set(get_descendants(dh_idx, d1_evt, d2_evt, len(pids_evt))))
                if len(desc) == 0:
                    continue

                vis_desc = []
                for j in desc:
                    if is_visible_final_state(pids_evt[j], status_evt[j]):
                        if not (
                            np.isnan(eta_evt[j]) or np.isnan(phi_evt[j]) or
                            np.isinf(eta_evt[j]) or np.isinf(phi_evt[j])
                        ):
                            vis_desc.append(j)

                if len(vis_desc) == 0:
                    continue

                drs_desc = calc_dr(
                    eta_evt[vis_desc], phi_evt[vis_desc],
                    gf_eta_evt[best_gf_idx], gf_phi_evt[best_gf_idx]
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

                if cat == "full":
                    result["pre_full_frac"].append(frac_inside)
                    result["pre_full_gf_mass"].append(gf_mass_evt[best_gf_idx])
                    result["pre_full_gf_pt"].append(gf_pt_evt[best_gf_idx])
                    result["pre_full_nvis"].append(n_total)
                    result["pre_full_dh_mass"].append(dh_mass)
                elif cat == "partial":
                    result["pre_partial_frac"].append(frac_inside)
                    result["pre_partial_gf_mass"].append(gf_mass_evt[best_gf_idx])
                    result["pre_partial_gf_pt"].append(gf_pt_evt[best_gf_idx])
                    result["pre_partial_nvis"].append(n_total)
                    result["pre_partial_dh_mass"].append(dh_mass)
                else:
                    result["pre_none_frac"].append(frac_inside)
                    result["pre_none_gf_mass"].append(gf_mass_evt[best_gf_idx])
                    result["pre_none_gf_pt"].append(gf_pt_evt[best_gf_idx])
                    result["pre_none_nvis"].append(n_total)
                    result["pre_none_dh_mass"].append(dh_mass)

                rank = {"full": 2, "partial": 1, "none": 0}[cat]
                candidate = {
                    "dh_idx": dh_idx,
                    "dh_mass": float(dh_mass),
                    "gf_idx": best_gf_idx,
                    "category": cat,
                    "rank": rank,
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

            # -------------------------------------------------
            # loop reco jets
            # -------------------------------------------------
            for r_idx in range(len(rfs_pt)):
                result["n_total_recojets"] += 1

                j_pt = rfs_pt[r_idx]
                j_eta = rfs_eta[r_idx]
                j_phi = rfs_phi[r_idx]
                j_m = rfs_m[r_idx]

                if j_pt < pt_cut:
                    continue
                result["n_ptcut_recojets"] += 1

                if len(gf_eta_evt) == 0:
                    continue

                drs_rg = calc_dr(j_eta, j_phi, gf_eta_evt, gf_phi_evt)
                closest_gen = int(np.argmin(drs_rg))
                closest_dr = float(drs_rg[closest_gen])
                result["pre_reco_to_gen_dr"].append(closest_dr)

                if closest_dr >= matching_r:
                    continue
                result["n_recojets_matched_to_gen"] += 1

                if closest_gen not in genfatjet_best:
                    continue

                match_info = genfatjet_best[closest_gen]
                cat = match_info["category"]
                frac_inside = match_info["frac_inside"]
                n_vis = match_info["n_vis"]

                keep = False
                if keep_mode == "full_only":
                    keep = (cat == "full")
                elif keep_mode == "full_plus_partial":
                    keep = (cat in ["full", "partial"])
                elif keep_mode == "all":
                    keep = True

                if not keep:
                    continue

                # -------------------------------------------------
                # PF candidates in reco jet cone
                # -------------------------------------------------
                pf_drs = calc_dr(j_eta, j_phi, pfs_eta, pfs_phi)
                pf_in = pf_drs < fatjet_r

                c_pt     = pfs_pt[pf_in]
                c_eta    = pfs_eta[pf_in]
                c_phi    = pfs_phi[pf_in]
                c_e      = pfs_e[pf_in]
                c_charge = pfs_charge[pf_in]
                c_pid    = pfs_pid[pf_in]
                c_d0     = pfs_d0[pf_in]
                c_dz     = pfs_dz[pf_in]
                c_eem    = pfs_eem[pf_in]
                c_ehad   = pfs_ehad[pf_in]

                c_abs_pid = np.abs(c_pid)
                is_photon = (c_abs_pid == 22).astype(np.float32)
                is_chad   = (c_abs_pid == 211).astype(np.float32)
                is_nhad   = (c_abs_pid == 130).astype(np.float32)
                is_elec   = (c_abs_pid == 11).astype(np.float32)
                is_muon   = (c_abs_pid == 13).astype(np.float32)

                sort_idx = np.argsort(c_pt)[::-1]
                c_pt      = c_pt[sort_idx]
                c_eta     = c_eta[sort_idx]
                c_phi     = c_phi[sort_idx]
                c_e       = c_e[sort_idx]
                c_charge  = c_charge[sort_idx]
                c_d0      = c_d0[sort_idx]
                c_dz      = c_dz[sort_idx]
                c_eem     = c_eem[sort_idx]
                c_ehad    = c_ehad[sort_idx]
                is_photon = is_photon[sort_idx]
                is_chad   = is_chad[sort_idx]
                is_nhad   = is_nhad[sort_idx]
                is_elec   = is_elec[sort_idx]
                is_muon   = is_muon[sort_idx]

                c_deta, c_dphi = center_coords(c_eta, c_phi, j_eta, j_phi)
                n_const = min(len(c_pt), MAX_CONSTITUENTS)

                jet_matrix = np.zeros((MAX_CONSTITUENTS, 16), dtype=np.float32)
                j_e = np.sqrt((j_pt * np.cosh(j_eta))**2 + j_m**2)

                if n_const > 0:
                    jet_matrix[:n_const, 0]  = c_e[:n_const]
                    jet_matrix[:n_const, 1]  = c_pt[:n_const]
                    jet_matrix[:n_const, 2]  = c_deta[:n_const]
                    jet_matrix[:n_const, 3]  = c_dphi[:n_const]
                    jet_matrix[:n_const, 4]  = c_charge[:n_const]
                    jet_matrix[:n_const, 5]  = c_d0[:n_const]
                    jet_matrix[:n_const, 6]  = c_dz[:n_const]
                    jet_matrix[:n_const, 7]  = c_eem[:n_const]
                    jet_matrix[:n_const, 8]  = c_ehad[:n_const]
                    jet_matrix[:n_const, 9]  = is_photon[:n_const]
                    jet_matrix[:n_const, 10] = is_chad[:n_const]
                    jet_matrix[:n_const, 11] = is_nhad[:n_const]
                    jet_matrix[:n_const, 12] = is_elec[:n_const]
                    jet_matrix[:n_const, 13] = is_muon[:n_const]
                    jet_matrix[:n_const, 14] = np.log(np.clip(c_pt[:n_const] / j_pt, 1e-8, 1.0))
                    jet_matrix[:n_const, 15] = np.log(np.clip(c_e[:n_const] / j_e, 1e-8, 1.0))

                # -------------------------------------------------
                # macro features
                # -------------------------------------------------
                taus = rfs_tau[r_idx]
                tau1 = taus[0] if len(taus) > 0 else -1.0
                tau2 = taus[1] if len(taus) > 1 else -1.0
                tau3 = taus[2] if len(taus) > 2 else -1.0
                tau21 = tau2 / tau1 if tau1 > 0 else -1.0
                tau32 = tau3 / tau2 if tau2 > 0 else -1.0

                j_sdmass = safe_softdrop_mass(rf_sdmass, i, r_idx)
                j_sdpt   = safe_softdrop_pt(rf_sdpt, i, r_idx)

                macro_vec = [
                    j_m,
                    j_sdmass,
                    j_sdpt,
                    rfs_ptd[r_idx],
                    rfs_msqdr[r_idx],
                    rfs_nef[r_idx],
                    rfs_cef[r_idx],
                    tau1,
                    tau2,
                    tau3,
                    tau21,
                    tau32,
                ]

                result["X_jets"].append(jet_matrix)
                result["Y_labels"].append(1.0)
                result["Jet_kinematics"].append([j_pt, j_eta, j_phi, j_m])
                result["Mass_params"].append([target_mpi])
                result["Macro_features"].append(macro_vec)

                result["Containment_fraction"].append(frac_inside)
                result["Containment_category"].append(category_code(cat))
                result["Matched_genfatjet_mass"].append(gf_mass_evt[closest_gen])
                result["Matched_genfatjet_pt"].append(gf_pt_evt[closest_gen])
                result["Visible_desc_multiplicity"].append(n_vis)
                result["File_source"].append(file_idx)

                result["n_recojets_kept"] += 1

        result["ok"] = True
        return result

    except Exception as e:
        result["error"] = f"{e}\n{traceback.format_exc()}"
        return result


# =========================================================
# MERGING
# =========================================================
def merge_worker_results(results):
    merged = {
        "pre_full_frac": [],
        "pre_partial_frac": [],
        "pre_none_frac": [],
        "pre_full_gf_mass": [],
        "pre_partial_gf_mass": [],
        "pre_none_gf_mass": [],
        "pre_full_gf_pt": [],
        "pre_partial_gf_pt": [],
        "pre_none_gf_pt": [],
        "pre_full_nvis": [],
        "pre_partial_nvis": [],
        "pre_none_nvis": [],
        "pre_full_dh_mass": [],
        "pre_partial_dh_mass": [],
        "pre_none_dh_mass": [],
        "pre_best_dh_to_gf_dr": [],
        "pre_reco_to_gen_dr": [],

        "X_jets": [],
        "Y_labels": [],
        "Jet_kinematics": [],
        "Mass_params": [],
        "Macro_features": [],
        "Containment_fraction": [],
        "Containment_category": [],
        "Matched_genfatjet_mass": [],
        "Matched_genfatjet_pt": [],
        "Visible_desc_multiplicity": [],
        "File_source": [],

        "n_total_recojets": 0,
        "n_ptcut_recojets": 0,
        "n_recojets_matched_to_gen": 0,
        "n_recojets_kept": 0,
    }

    for r in results:
        merged["n_total_recojets"] += r["n_total_recojets"]
        merged["n_ptcut_recojets"] += r["n_ptcut_recojets"]
        merged["n_recojets_matched_to_gen"] += r["n_recojets_matched_to_gen"]
        merged["n_recojets_kept"] += r["n_recojets_kept"]

        for key in [
            "pre_full_frac", "pre_partial_frac", "pre_none_frac",
            "pre_full_gf_mass", "pre_partial_gf_mass", "pre_none_gf_mass",
            "pre_full_gf_pt", "pre_partial_gf_pt", "pre_none_gf_pt",
            "pre_full_nvis", "pre_partial_nvis", "pre_none_nvis",
            "pre_full_dh_mass", "pre_partial_dh_mass", "pre_none_dh_mass",
            "pre_best_dh_to_gf_dr", "pre_reco_to_gen_dr",
            "X_jets", "Y_labels", "Jet_kinematics", "Mass_params",
            "Macro_features", "Containment_fraction", "Containment_category",
            "Matched_genfatjet_mass", "Matched_genfatjet_pt",
            "Visible_desc_multiplicity", "File_source"
        ]:
            merged[key].extend(r[key])

    return merged


# =========================================================
# PLOTTING
# =========================================================
def make_plots(outdir, merged, fatjet_r, matching_r):
    pre_dir = os.path.join(outdir, "pre_dataset")
    post_dir = os.path.join(outdir, "post_dataset")

    pre_full_frac = merged["pre_full_frac"]
    pre_partial_frac = merged["pre_partial_frac"]
    pre_none_frac = merged["pre_none_frac"]

    pre_full_nvis = merged["pre_full_nvis"]
    pre_partial_nvis = merged["pre_partial_nvis"]
    pre_none_nvis = merged["pre_none_nvis"]

    pre_full_gf_mass = merged["pre_full_gf_mass"]
    pre_partial_gf_mass = merged["pre_partial_gf_mass"]
    pre_none_gf_mass = merged["pre_none_gf_mass"]

    pre_full_gf_pt = merged["pre_full_gf_pt"]
    pre_partial_gf_pt = merged["pre_partial_gf_pt"]
    pre_none_gf_pt = merged["pre_none_gf_pt"]

    pre_full_dh_mass = merged["pre_full_dh_mass"]
    pre_partial_dh_mass = merged["pre_partial_dh_mass"]
    pre_none_dh_mass = merged["pre_none_dh_mass"]

    pre_best_dh_to_gf_dr = merged["pre_best_dh_to_gf_dr"]
    pre_reco_to_gen_dr = merged["pre_reco_to_gen_dr"]

    # 1
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, pre_full_frac, bins=20, range=(0, 1.05), label="Fully matched", density=True)
    safe_hist(ax, pre_partial_frac, bins=20, range=(0, 1.05), label="Partially matched", density=True)
    safe_hist(ax, pre_none_frac, bins=20, range=(0, 1.05), label="Unmatched", density=True)
    ax.set_xlabel("Fraction of visible descendants inside matched GenFatJet")
    ax.set_ylabel("Normalized entries")
    ax.set_title("Pre-dataset: Containment Fraction")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pre_dir, "containment_fraction.pdf"))
    plt.close(fig)

    # 2
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, pre_full_nvis, bins=np.arange(0, 26) - 0.5, label="Fully matched", density=True)
    safe_hist(ax, pre_partial_nvis, bins=np.arange(0, 26) - 0.5, label="Partially matched", density=True)
    safe_hist(ax, pre_none_nvis, bins=np.arange(0, 26) - 0.5, label="Unmatched", density=True)
    ax.set_xlabel("Number of visible stable descendants")
    ax.set_ylabel("Normalized entries")
    ax.set_title("Pre-dataset: Visible Descendant Multiplicity")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pre_dir, "visible_desc_multiplicity.pdf"))
    plt.close(fig)

    # 3
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, pre_full_gf_mass, bins=60, label="Fully matched", density=True)
    safe_hist(ax, pre_partial_gf_mass, bins=60, label="Partially matched", density=True)
    safe_hist(ax, pre_none_gf_mass, bins=60, label="Unmatched", density=True)
    ax.set_xlabel("Matched GenFatJet mass [GeV]")
    ax.set_ylabel("Normalized entries")
    ax.set_title("Pre-dataset: Matched GenFatJet Mass")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pre_dir, "matched_genfatjet_mass.pdf"))
    plt.close(fig)

    # 4
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, pre_full_gf_pt, bins=60, label="Fully matched", density=True)
    safe_hist(ax, pre_partial_gf_pt, bins=60, label="Partially matched", density=True)
    safe_hist(ax, pre_none_gf_pt, bins=60, label="Unmatched", density=True)
    ax.set_xlabel("Matched GenFatJet pT [GeV]")
    ax.set_ylabel("Normalized entries")
    ax.set_title("Pre-dataset: Matched GenFatJet pT")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pre_dir, "matched_genfatjet_pt.pdf"))
    plt.close(fig)

    # 5
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, pre_full_dh_mass, bins=50, label="Fully matched", density=True)
    safe_hist(ax, pre_partial_dh_mass, bins=50, label="Partially matched", density=True)
    safe_hist(ax, pre_none_dh_mass, bins=50, label="Unmatched", density=True)
    ax.set_xlabel("Dark hadron mass [GeV]")
    ax.set_ylabel("Normalized entries")
    ax.set_title("Pre-dataset: Dark Hadron Mass")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pre_dir, "darkhadron_mass.pdf"))
    plt.close(fig)

    # 6
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, pre_best_dh_to_gf_dr, bins=50, range=(0, 1.5), label="Best DR", density=False, histtype="stepfilled", lw=1)
    ax.axvline(fatjet_r, color="red", linestyle="--", label=f"R={fatjet_r}")
    ax.set_xlabel(r"Best $\Delta R$(dark hadron, GenFatJet)")
    ax.set_ylabel("Counts")
    ax.set_title("Pre-dataset: Dark Hadron to Best GenFatJet Axis Distance")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pre_dir, "darkhadron_to_genfatjet_bestdr.pdf"))
    plt.close(fig)

    # 7
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, pre_reco_to_gen_dr, bins=50, range=(0, 1.5), label="Closest Reco-Gen DR", density=False, histtype="stepfilled", lw=1)
    ax.axvline(matching_r, color="red", linestyle="--", label=f"matching R={matching_r}")
    ax.set_xlabel(r"Best $\Delta R$(Reco FatJet, GenFatJet)")
    ax.set_ylabel("Counts")
    ax.set_title("Pre-dataset: Reco to GenFatJet Matching Distance")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pre_dir, "reco_to_genfatjet_bestdr.pdf"))
    plt.close(fig)

    # -------------------------------------------------
    # post-dataset plots
    # -------------------------------------------------
    if len(merged["X_jets"]) == 0:
        return

    X_jets = np.asarray(merged["X_jets"], dtype=np.float32)
    Mass_params = np.asarray(merged["Mass_params"], dtype=np.float32)
    Jet_kinematics = np.asarray(merged["Jet_kinematics"], dtype=np.float32)
    Containment_fraction = np.asarray(merged["Containment_fraction"], dtype=np.float32)
    Macro_features = np.asarray(merged["Macro_features"], dtype=np.float32) 
    Containment_category = np.asarray(merged["Containment_category"], dtype=np.int32)
    Matched_genfatjet_mass = np.asarray(merged["Matched_genfatjet_mass"], dtype=np.float32)
    Matched_genfatjet_pt = np.asarray(merged["Matched_genfatjet_pt"], dtype=np.float32)
    Visible_desc_multiplicity = np.asarray(merged["Visible_desc_multiplicity"], dtype=np.int32)


    
    masses_1d = Mass_params.reshape(-1)
    jet_pt = Jet_kinematics[:, 0]
    jet_mass = Jet_kinematics[:, 3]

    # 1
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, masses_1d, bins=80, label="Kept dataset", density=False)
    ax.set_xlabel(r"Target $m_\pi$ [GeV]")
    ax.set_ylabel("Entries")
    ax.set_title("Post-dataset: Target Mass Spectrum")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "target_mass_spectrum.pdf"))
    plt.close(fig)

    # 2
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, jet_pt, bins=80, label="Kept dataset", density=True)
    ax.set_xlabel("Reco FatJet pT [GeV]")
    ax.set_ylabel("Normalized entries")
    ax.set_title("Post-dataset: Reco FatJet pT")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "recojet_pt.pdf"))
    plt.close(fig)

    # 3
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, jet_mass, bins=80, label="Kept dataset", density=True)
    ax.set_xlabel("Reco FatJet mass [GeV]")
    ax.set_ylabel("Normalized entries")
    ax.set_title("Post-dataset: Reco FatJet Mass")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "recojet_mass.pdf"))
    plt.close(fig)

    # 4
    fig, ax = plt.subplots(figsize=(8, 6))
    safe_hist(ax, Containment_fraction, bins=20, range=(0, 1.05), label="Kept dataset", density=True)
    ax.set_xlabel("Containment fraction")
    ax.set_ylabel("Normalized entries")
    ax.set_title("Post-dataset: Containment Fraction")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "containment_fraction_kept.pdf"))
    plt.close(fig)

    # 5
    fig, ax = plt.subplots(figsize=(8, 6))
    vals, counts = np.unique(Containment_category, return_counts=True)
    label_map = {2: "full", 1: "partial", 0: "none"}
    ax.bar([label_map.get(v, str(v)) for v in vals], counts)
    ax.set_ylabel("Entries")
    ax.set_title("Post-dataset: Containment Category Composition")
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "containment_category_counts.pdf"))
    plt.close(fig)

    # 6
    x, y = maybe_subsample(Containment_fraction, Matched_genfatjet_mass)
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, gridsize=40, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")
    ax.set_xlabel("Containment fraction")
    ax.set_ylabel("Matched GenFatJet mass [GeV]")
    ax.set_title("Post-dataset: Matched GenFatJet Mass vs Containment")
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "matched_genfatjet_mass_vs_containment.pdf"))
    plt.close(fig)

    # 7
    x, y = maybe_subsample(Containment_fraction, Matched_genfatjet_pt)
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, gridsize=40, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")
    ax.set_xlabel("Containment fraction")
    ax.set_ylabel("Matched GenFatJet pT [GeV]")
    ax.set_title("Post-dataset: Matched GenFatJet pT vs Containment")
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "matched_genfatjet_pt_vs_containment.pdf"))
    plt.close(fig)

    # 8
    x, y = maybe_subsample(Containment_fraction, Visible_desc_multiplicity)
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, gridsize=35, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")
    ax.set_xlabel("Containment fraction")
    ax.set_ylabel("Visible descendant multiplicity")
    ax.set_title("Post-dataset: Visible Multiplicity vs Containment")
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "visible_mult_vs_containment.pdf"))
    plt.close(fig)

    # 9
    x, y = maybe_subsample(masses_1d, jet_pt)
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, gridsize=40, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")
    ax.set_xlabel(r"Target $m_\pi$ [GeV]")
    ax.set_ylabel("Reco FatJet pT [GeV]")
    ax.set_title("Post-dataset: Reco Jet pT vs Target Mass")
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "recojet_pt_vs_target_mass.pdf"))
    plt.close(fig)

    # 10
    x, y = maybe_subsample(masses_1d, jet_mass)
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, gridsize=40, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")
    ax.set_xlabel(r"Target $m_\pi$ [GeV]")
    ax.set_ylabel("Reco FatJet mass [GeV]")
    ax.set_title("Post-dataset: Reco Jet Mass vs Target Mass")
    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "recojet_mass_vs_target_mass.pdf"))
    plt.close(fig)
    

    # 11: SD mass vs true mass (THIS IS THE KEY BASELINE)
    sd_mass = Macro_features[:, 1]

    # remove invalid SD mass entries
    valid = sd_mass > 0
    x, y = maybe_subsample(masses_1d[valid], sd_mass[valid])

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, gridsize=45, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")

    ax.set_xlabel(r"True $m_\pi$ [GeV]")
    ax.set_ylabel("SoftDrop mass [GeV]")
    ax.set_title("Post-dataset: SoftDrop Mass vs True Mass")

    # Optional: add y=x line for reference
    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "sdmass_vs_true_mass.pdf"))
    plt.close(fig)

    #12: SD mass resolution vs true mass
    residual = sd_mass[valid] - masses_1d[valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(masses_1d[valid], residual, gridsize=45, mincnt=1)
    fig.colorbar(hb, ax=ax, label="Counts")

    ax.set_xlabel(r"True $m_\pi$ [GeV]")
    ax.set_ylabel(r"SD Mass - True Mass [GeV]")
    ax.set_title("SoftDrop Mass Residuals")

    ax.axhline(0, linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(post_dir, "sdmass_residuals.pdf"))
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Multiprocessing complete-genmatch DRACON dataset builder.")
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        default=DEFAULT_BASE_DIRS,
        help="Base directories to recursively search for events.root files",
    )
    parser.add_argument(
        "--filelist",
        default=None,
        help="Optional text file containing events.root paths. Supports EOS paths from xrdfs.",
    )
    parser.add_argument(
        "--eos-host",
        default="root://cmseos.fnal.gov",
        help="EOS XRootD host prefix used when --filelist contains /store/... paths.",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Output NPZ file name")
    parser.add_argument(
        "--keep-mode",
        default="full_only",
        choices=["full_only", "full_plus_partial", "all"],
        help="Which containment categories to keep",
    )
    parser.add_argument("--max-events-per-file", type=int, default=None, help="Optional cap on events per file")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    parser.add_argument("--fatjet-r", type=float, default=FATJET_R, help="FatJet radius")
    parser.add_argument("--matching-r", type=float, default=MATCHING_R, help="Reco-to-gen matching radius")
    parser.add_argument("--pt-cut", type=float, default=PT_CUT, help="Reco jet pT cut")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable final global shuffle")
    args = parser.parse_args()

    ensure_dirs(args.outdir)
    output_file = os.path.join(args.outdir, args.output_name)

    root_files = []

    if args.filelist is not None:
        with open(args.filelist) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                
                if line.startswith("/store/"):
                    line = args.eos_host.rstrip("/") + "/" + line

                root_files.append(line)

        root_files = sorted(root_files)
        print(f"Loaded {len(root_files)} ROOT files from filelist: {args.filelist}")

    else:
        for b in args.base_dirs:
            root_files.extend(glob.glob(os.path.join(b, "**", "events.root"), recursive=True))
        root_files = sorted(root_files)
        print(f"Found {len(root_files)} ROOT files total from local base dirs")

    if len(root_files) == 0:
        raise RuntimeError("No ROOT files found.")

    worker_args = [
        (
            root_file,
            file_idx,
            args.keep_mode,
            args.max_events_per_file,
            args.fatjet_r,
            args.matching_r,
            args.pt_cut,
        )
        for file_idx, root_file in enumerate(root_files)
    ]

    good_results = []
    bad_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one_root_file, wa) for wa in worker_args]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing ROOT files"):
            res = fut.result()
            if res["ok"]:
                good_results.append(res)
            else:
                bad_results.append(res)
                print(f"[WARNING] Failed: {res['root_file']}")
                print(f"          Reason: {res['error']}")

    if len(good_results) == 0:
        raise RuntimeError("All files failed. No dataset produced.")

    merged = merge_worker_results(good_results)

    X_jets = np.array(merged["X_jets"], dtype=np.float32)
    Y_labels = np.array(merged["Y_labels"], dtype=np.float32)
    Jet_kinematics = np.array(merged["Jet_kinematics"], dtype=np.float32)
    Mass_params = np.array(merged["Mass_params"], dtype=np.float32)
    Macro_features = np.array(merged["Macro_features"], dtype=np.float32)
    Containment_fraction = np.array(merged["Containment_fraction"], dtype=np.float32)
    Containment_category = np.array(merged["Containment_category"], dtype=np.int32)
    Matched_genfatjet_mass = np.array(merged["Matched_genfatjet_mass"], dtype=np.float32)
    Matched_genfatjet_pt = np.array(merged["Matched_genfatjet_pt"], dtype=np.float32)
    Visible_desc_multiplicity = np.array(merged["Visible_desc_multiplicity"], dtype=np.int32)
    File_source = np.array(merged["File_source"], dtype=np.int32)

    print("\n" + "=" * 60)
    print("DATASET EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total reco jets seen                : {merged['n_total_recojets']}")
    print(f"Reco jets passing pT cut            : {merged['n_ptcut_recojets']}")
    print(f"Reco jets matched to GenFatJets     : {merged['n_recojets_matched_to_gen']}")
    print(f"Reco jets kept in dataset           : {merged['n_recojets_kept']}")
    print(f"Final dataset size                  : {len(X_jets)}")
    print(f"X shape                             : {X_jets.shape}")
    print(f"macro shape                         : {Macro_features.shape}")
    print(f"masses shape                        : {Mass_params.shape}")
    print(f"failed files                        : {len(bad_results)}")

    if len(Y_labels) > 0 and not args.no_shuffle:
        print("Applying global shuffle...")
        idx = np.random.permutation(len(Y_labels))
        X_jets = X_jets[idx]
        Y_labels = Y_labels[idx]
        Jet_kinematics = Jet_kinematics[idx]
        Mass_params = Mass_params[idx]
        Macro_features = Macro_features[idx]
        Containment_fraction = Containment_fraction[idx]
        Containment_category = Containment_category[idx]
        Matched_genfatjet_mass = Matched_genfatjet_mass[idx]
        Matched_genfatjet_pt = Matched_genfatjet_pt[idx]
        Visible_desc_multiplicity = Visible_desc_multiplicity[idx]
        File_source = File_source[idx]

    print(f"\nSaving dataset to {output_file} ...")
    np.savez_compressed(
        output_file,
        X=X_jets,
        y=Y_labels,
        kinematics=Jet_kinematics,
        masses=Mass_params,
        macro=Macro_features,
        containment_fraction=Containment_fraction,
        containment_category=Containment_category,
        matched_genfatjet_mass=Matched_genfatjet_mass,
        matched_genfatjet_pt=Matched_genfatjet_pt,
        visible_desc_multiplicity=Visible_desc_multiplicity,
        file_source=File_source,
    )
    print("Saved dataset.")

    make_plots(
        outdir=args.outdir,
        merged=merged,
        fatjet_r=args.fatjet_r,
        matching_r=args.matching_r,
    )

    print(f"\nDone. Outputs saved in: {args.outdir}")
    print("\nMacro feature order is:")
    print("  [0]  Jet_mass")
    print("  [1]  Jet_sdmass")
    print("  [2]  Jet_sdpt")
    print("  [3]  Jet_ptD")
    print("  [4]  Jet_msqdr")
    print("  [5]  Jet_nef")
    print("  [6]  Jet_cef")
    print("  [7]  Jet_tau1")
    print("  [8]  Jet_tau2")
    print("  [9]  Jet_tau3")
    print("  [10] Jet_tau21")
    print("  [11] Jet_tau32")


if __name__ == "__main__":
    main()