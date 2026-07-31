#!/usr/bin/env python3
"""
build_dracon_complete_genmatch_dataset_multiproc_EOS_mpiEqMrho_constituentmatch_bestdh_multidh_pftruth_noptcut.py

DRAGON complete-genmatch NPZ builder using GenFatJet constituent ancestry
with old splitmatch-style best-dark-hadron category semantics.

This version:
  - starts from GenFatJet constituents, climbs to their dark-hadron ancestors,
    then descends through each dark hadron to check final-state visible daughter
    containment in the same GenFatJet
  - assigns each GenFatJet category from the best single dark-hadron candidate,
    ranked like the old splitmatch_pt20 script: full > partial > none, then
    containment fraction, then visible daughter multiplicity
  - ignores stable/invisible dark hadrons in the containment procedure
  - defaults to no reco FatJet pT threshold
  - processes EOS filelists with multiprocessing
  - saves THREE NPZ files per run, split by containment category:
      <output-prefix>_full.npz
      <output-prefix>_partial.npz
      <output-prefix>_none.npz
  - saves per-jet visible unstable dark-hadron multiplicity labels:
      n_dark_hadrons_per_jet
      n_visible_dark_hadrons
      ndark_piD0_per_jet, ndark_piDplus_per_jet, ndark_piDminus_per_jet,
      ndark_rhoD0_per_jet, ndark_rhoDplus_per_jet, ndark_rhoDminus_per_jet
  - saves padded visible unstable dark-hadron 4-vectors and metadata. Slot 0 is
    the best dark hadron used for category assignment; later slots hold the
    remaining GenFatJet-level visible dark-hadron candidates.
      dark_hadron_p4                  [px, py, pz, e]
      dark_hadron_kinematics          [pt, eta, phi, mass]
      dark_hadron_pid, dark_hadron_status
      dark_hadron_full_containment
      dark_hadron_visible_daughter_counts
      dark_hadron_visible_daughter_inside_counts
  - adds one PF-level dark-hadron decay-tree feature:
      X[..., 16] = dark_hadron_descendant_id
      0 means no linked visible dark-hadron descendant.
      k > 0 means this PF candidate descends from the dark hadron stored at
      dark_hadron_* slot k-1 for this jet. For example, if ndark=2, visible PF
      descendants of the two dark hadrons are marked 1 and 2 respectively.

Main NPZ keys:
  X, y, kinematics, masses, macro,
  containment_fraction, containment_category,
  matched_genfatjet_mass, matched_genfatjet_pt,
  visible_desc_multiplicity,
  n_dark_hadrons_per_jet, n_visible_dark_hadrons,
  per-PID dark-hadron counts, dark-hadron 4-vectors,
  file_source, root_file_index
"""

import os
import re
import csv
import glob
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import awkward as ak
import uproot
from tqdm import tqdm


# =============================================================================
# Defaults
# =============================================================================
DEFAULT_BASE_DIRS = [
    "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Smooth_Dracon_20260325_1238",
    "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Smooth_Dracon_20260325_1323",
    "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Training_2D_20260328_1534",
    "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Training_2D_20260331_1625",
    "/lustre/research/hep/akshriva/SVJ_RandD/DRACON/training_data/SVJ_Training_2D_20260316_2324",
]

DEFAULT_OUTDIR = "./dragon_npz_constituentmatch_bestdh_multidh_pftruth_noptcut"
DEFAULT_OUTPUT_PREFIX = "dragon_mpiEqMrho_constituentmatch_bestdh_multidh_pftruth_noptcut"

MAX_CONSTITUENTS_DEFAULT = 100
MAX_DARK_HADRONS_DEFAULT = 64
N_CONSTITUENT_FEATURES = 17
PF_DARK_DESCENDANT_FEATURE_INDEX = 16
FATJET_R_DEFAULT = 0.8
MATCHING_R_DEFAULT = 0.4
PT_CUT_DEFAULT = 0.0

DARK_HADRONS = {
    4900111,
    4900211,
    -4900211,
    4900113,
    4900213,
    -4900213,
}

PID_ORDER = [
    4900111,
    4900211,
    -4900211,
    4900113,
    4900213,
    -4900213,
]

PID_TAGS = {
    4900111: "piD0",
    4900211: "piDplus",
    -4900211: "piDminus",
    4900113: "rhoD0",
    4900213: "rhoDplus",
    -4900213: "rhoDminus",
}

# Default dark-hadron count uses the final dark-hadron copies in this signal record.
DARK_HADRON_COUNT_STATUSES = {83, 84}

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

BUCKETS = {
    "full": "fully matched visible descendants",
    "partial": "partially matched visible descendants",
    "none": "unmatched visible descendants",
}

DATA_KEYS = [
    "X_jets",
    "Y_labels",
    "Jet_kinematics",
    "Mass_params",
    "Macro_features",
    "Containment_fraction",
    "Containment_category",
    "Matched_genfatjet_mass",
    "Matched_genfatjet_pt",
    "Visible_desc_multiplicity",
    "NDark_hadrons_per_jet",
    "NVisible_dark_hadrons",
    "NDark_piD0_per_jet",
    "NDark_piDplus_per_jet",
    "NDark_piDminus_per_jet",
    "NDark_rhoD0_per_jet",
    "NDark_rhoDplus_per_jet",
    "NDark_rhoDminus_per_jet",
    "NDark_hadrons_complete",
    "NDark_hadrons_partial",
    "Dark_hadron_p4",
    "Dark_hadron_kinematics",
    "Dark_hadron_pid",
    "Dark_hadron_status",
    "Dark_hadron_full_containment",
    "Dark_hadron_visible_daughter_counts",
    "Dark_hadron_visible_daughter_inside_counts",
    "File_source",
    "Root_file_index",
]


# =============================================================================
# Geometry and parsing helpers
# =============================================================================
def calc_dphi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2.0 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2.0 * np.pi, dphi)
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + calc_dphi(phi1, phi2) ** 2)


def center_coords(etas, phis, center_eta, center_phi):
    d_eta = etas - center_eta
    d_phi = phis - center_phi
    d_phi = np.where(d_phi > np.pi, d_phi - 2.0 * np.pi, d_phi)
    d_phi = np.where(d_phi < -np.pi, d_phi + 2.0 * np.pi, d_phi)
    return d_eta, d_phi


def parse_mpi_from_path(path):
    """Parse target m_pi/m_dark from common SVJ file path formats."""
    path = str(path)

    patterns = [
        r"point_mPI_([0-9]+(?:\.[0-9]+)?)",
        r"mpi-([0-9]+(?:\.[0-9]+)?)",
        r"mpiEqMrho_([0-9]+(?:p[0-9]+)?)",
        r"mDark_([0-9]+(?:\.[0-9]+)?)",
        r"mdark[_-]([0-9]+(?:\.[0-9]+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, path, flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace("p", "."))

    return None


def category_code(cat):
    return {"full": 2, "partial": 1, "none": 0}.get(cat, -1)


def category_from_code(code):
    return {2: "full", 1: "partial", 0: "none"}.get(int(code), "unknown")


def should_keep_category(cat, keep_mode):
    if keep_mode == "full_only":
        return cat == "full"
    if keep_mode == "partial_only":
        return cat == "partial"
    if keep_mode == "none_only":
        return cat == "none"
    if keep_mode == "full_plus_partial":
        return cat in ["full", "partial"]
    if keep_mode == "partial_plus_none":
        return cat in ["partial", "none"]
    if keep_mode == "not_full":
        return cat in ["partial", "none"]
    if keep_mode == "all":
        return True
    raise ValueError(f"Unknown keep_mode: {keep_mode}")


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


def get_visible_final_descendants(idx, pids, status, d1s, d2s, n_parts):
    """Final-state visible descendants of a generator particle."""
    desc = sorted(set(get_descendants(idx, d1s, d2s, n_parts)))
    out = []
    for child_idx in desc:
        if is_visible_final_state(pids[child_idx], status[child_idx]):
            out.append(int(child_idx))
    return out


def get_refs_from_tref_array_entry(tref_entry):
    """Extract raw TRef ids from one Delphes TRefArray entry."""
    def normalize_refs(value):
        if isinstance(value, dict) and "refs" in value:
            value = value["refs"]
        if value is None:
            return []
        if isinstance(value, (int, np.integer)):
            return [int(value)]
        if isinstance(value, (float, np.floating)) and np.isfinite(value):
            return [int(value)]
        return [int(ref) for ref in value]

    try:
        return normalize_refs(ak.to_list(tref_entry["refs"]))
    except Exception:
        try:
            return normalize_refs(ak.to_list(tref_entry.refs))
        except Exception:
            try:
                return normalize_refs(ak.to_list(tref_entry))
            except Exception:
                return []


def build_ref_to_index_map(unique_ids_evt):
    if unique_ids_evt is None:
        return None
    return {int(uid): int(idx) for idx, uid in enumerate(unique_ids_evt)}


def refs_to_indices(raw_refs, ref_to_index, n_parts):
    indices = []
    for ref in raw_refs:
        idx = None
        if ref_to_index is not None:
            idx = ref_to_index.get(int(ref))
        elif 1 <= int(ref) <= n_parts:
            idx = int(ref) - 1
        elif 0 <= int(ref) < n_parts:
            idx = int(ref)

        if idx is not None and 0 <= idx < n_parts:
            indices.append(int(idx))

    return sorted(set(indices))


def first_ref_to_index(tref_entry, ref_to_index, n_parts):
    indices = refs_to_indices(get_refs_from_tref_array_entry(tref_entry), ref_to_index, n_parts)
    if not indices:
        return -1
    return int(indices[0])


def build_pf_gen_indices(pf_particles_evt, ref_to_index, n_parts, n_pf):
    """Map PF candidates to their first referenced GenParticle index, when available."""
    out = np.full(n_pf, -1, dtype=np.int32)
    if pf_particles_evt is None:
        return out

    n_refs = min(len(pf_particles_evt), n_pf)
    for pf_idx in range(n_refs):
        out[pf_idx] = first_ref_to_index(pf_particles_evt[pf_idx], ref_to_index, n_parts)

    return out


def sort_dark_hadrons_for_slots(dark_hadrons):
    return sorted(
        dark_hadrons,
        key=lambda item: (
            item.get("is_best", False),
            item.get("rank", -1),
            item.get("frac_inside", -1.0),
            item.get("n_daughters", -1),
            item.get("pt", -1.0),
        ),
        reverse=True,
    )


def build_pf_dark_descendant_ids(
    pf_gen_indices,
    dark_hadrons,
    max_dark_hadrons,
    pids_evt,
    status_evt,
    d1_evt,
    d2_evt,
    n_parts,
):
    """Return one PF feature linking each visible descendant to its dark-hadron slot."""
    out = np.zeros(len(pf_gen_indices), dtype=np.float32)
    if len(pf_gen_indices) == 0 or not dark_hadrons:
        return out

    gen_to_dark_slot = {}
    sorted_dh = sort_dark_hadrons_for_slots(dark_hadrons)
    for slot, dh in enumerate(sorted_dh[:max_dark_hadrons], start=1):
        for gen_idx in get_visible_final_descendants(dh["idx"], pids_evt, status_evt, d1_evt, d2_evt, n_parts):
            gen_to_dark_slot.setdefault(int(gen_idx), float(slot))

    if not gen_to_dark_slot:
        return out

    for pf_idx, gen_idx in enumerate(pf_gen_indices):
        out[pf_idx] = gen_to_dark_slot.get(int(gen_idx), 0.0)

    return out


def find_dark_ancestors(
    idx,
    pids_evt,
    status_evt,
    m1_evt,
    m2_evt,
    eligible_dark_hadrons,
    n_parts,
):
    """Walk up both mother links and collect eligible dark-hadron ancestors."""
    stack = [int(idx)]
    visited = set()
    out = set()

    while stack:
        cur = stack.pop()
        if cur < 0 or cur >= n_parts or cur in visited:
            continue
        visited.add(cur)

        if cur in eligible_dark_hadrons:
            out.add(int(cur))

        for mother in (int(m1_evt[cur]), int(m2_evt[cur])):
            if 0 <= mother < n_parts and mother not in visited:
                stack.append(mother)

    return out


def make_dark_hadron_arrays(dark_hadrons, max_dark_hadrons):
    p4 = np.zeros((max_dark_hadrons, 4), dtype=np.float32)
    kin = np.zeros((max_dark_hadrons, 4), dtype=np.float32)
    pid = np.zeros(max_dark_hadrons, dtype=np.int32)
    status = np.zeros(max_dark_hadrons, dtype=np.int32)
    full = np.zeros(max_dark_hadrons, dtype=np.int32)
    ndaughters = np.zeros(max_dark_hadrons, dtype=np.int32)
    ninside = np.zeros(max_dark_hadrons, dtype=np.int32)

    sorted_dh = sort_dark_hadrons_for_slots(dark_hadrons)
    for out_idx, dh in enumerate(sorted_dh[:max_dark_hadrons]):
        p4[out_idx] = [dh["px"], dh["py"], dh["pz"], dh["e"]]
        kin[out_idx] = [dh["pt"], dh["eta"], dh["phi"], dh["mass"]]
        pid[out_idx] = dh["pid"]
        status[out_idx] = dh["status"]
        full[out_idx] = 1 if dh["is_full"] else 0
        ndaughters[out_idx] = dh["n_daughters"]
        ninside[out_idx] = dh["n_inside"]

    return {
        "p4": p4,
        "kinematics": kin,
        "pid": pid,
        "status": status,
        "full_containment": full,
        "visible_daughter_counts": ndaughters,
        "visible_daughter_inside_counts": ninside,
    }


# =============================================================================
# ROOT helpers
# =============================================================================
def get_delphes_tree(root_file):
    f = uproot.open(root_file)
    if "Delphes;1" in f:
        return f, f["Delphes;1"]
    if "Delphes" in f:
        return f, f["Delphes"]
    f.close()
    raise RuntimeError("Could not find Delphes tree")


def get_branch(tree, possible_names, required=True):
    keys = set(tree.keys())
    for name in possible_names:
        if name in keys:
            return tree[name].array(library="ak")
    if required:
        raise KeyError(f"Could not find any branch from: {possible_names}")
    return None


def safe_softdrop_mass(rf_sdmass, event_idx, jet_idx):
    if rf_sdmass is None:
        return -1.0
    try:
        if len(rf_sdmass[event_idx]) <= jet_idx:
            return -1.0
        return float(rf_sdmass[event_idx][jet_idx])
    except Exception:
        return -1.0


def safe_softdrop_pt(rf_sdpt, event_idx, jet_idx):
    if rf_sdpt is None:
        return -1.0
    try:
        if len(rf_sdpt[event_idx]) <= jet_idx:
            return -1.0
        return float(rf_sdpt[event_idx][jet_idx])
    except Exception:
        return -1.0


def build_softdrop_from_packed(tree):
    """Fallback for FatJet/FatJet.SoftDroppedP4[5]."""
    rf_sdp4_raw = get_branch(
        tree,
        ["FatJet/FatJet.SoftDroppedP4[5]", "FatJet.SoftDroppedP4[5]"],
        required=False,
    )
    if rf_sdp4_raw is None:
        return None, None

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
            e = vec["fE"]

            pt = np.sqrt(px**2 + py**2)
            m2 = e**2 - px**2 - py**2 - pz**2
            mass = np.sqrt(max(m2, 0.0))

            evt_pt.append(float(pt))
            evt_mass.append(float(mass))

        tmp_pt.append(evt_pt)
        tmp_mass.append(evt_mass)

    return ak.Array(tmp_mass), ak.Array(tmp_pt)


# =============================================================================
# Dataset containers
# =============================================================================
def empty_bucket_container():
    return {key: [] for key in DATA_KEYS}


def empty_worker_result(root_file, file_idx):
    return {
        "ok": False,
        "root_file": root_file,
        "file_idx": file_idx,
        "error": None,
        "n_total_recojets": 0,
        "n_ptcut_recojets": 0,
        "n_recojets_matched_to_gen": 0,
        "n_recojets_approved": 0,
        "n_recojets_saved_full": 0,
        "n_recojets_saved_partial": 0,
        "n_recojets_saved_none": 0,
        "n_recojets_saved": 0,
        "bucket_data": {name: empty_bucket_container() for name in BUCKETS},
    }


def append_jet_to_bucket(
    bucket_data,
    bucket_name,
    jet_matrix,
    target_mpi,
    j_pt,
    j_eta,
    j_phi,
    j_m,
    macro_vec,
    match_info,
    gf_mass_val,
    gf_pt_val,
    file_idx,
    n_dark_total,
    n_visible_dark_hadrons,
    per_pid_counts,
    dark_hadron_arrays,
):
    bucket = bucket_data[bucket_name]

    bucket["X_jets"].append(jet_matrix)
    bucket["Y_labels"].append(1.0)
    bucket["Jet_kinematics"].append([j_pt, j_eta, j_phi, j_m])
    bucket["Mass_params"].append([target_mpi])
    bucket["Macro_features"].append(macro_vec)
    bucket["Containment_fraction"].append(match_info["frac_inside"])
    bucket["Containment_category"].append(category_code(match_info["category"]))
    bucket["Matched_genfatjet_mass"].append(gf_mass_val)
    bucket["Matched_genfatjet_pt"].append(gf_pt_val)
    bucket["Visible_desc_multiplicity"].append(match_info["n_vis"])
    bucket["NDark_hadrons_per_jet"].append(n_dark_total)
    bucket["NVisible_dark_hadrons"].append(n_visible_dark_hadrons)
    bucket["File_source"].append(file_idx)
    bucket["Root_file_index"].append(file_idx)

    bucket["NDark_piD0_per_jet"].append(per_pid_counts[4900111])
    bucket["NDark_piDplus_per_jet"].append(per_pid_counts[4900211])
    bucket["NDark_piDminus_per_jet"].append(per_pid_counts[-4900211])
    bucket["NDark_rhoD0_per_jet"].append(per_pid_counts[4900113])
    bucket["NDark_rhoDplus_per_jet"].append(per_pid_counts[4900213])
    bucket["NDark_rhoDminus_per_jet"].append(per_pid_counts[-4900213])
    bucket["NDark_hadrons_complete"].append(match_info["n_complete"])
    bucket["NDark_hadrons_partial"].append(match_info["n_partial"])
    bucket["Dark_hadron_p4"].append(dark_hadron_arrays["p4"])
    bucket["Dark_hadron_kinematics"].append(dark_hadron_arrays["kinematics"])
    bucket["Dark_hadron_pid"].append(dark_hadron_arrays["pid"])
    bucket["Dark_hadron_status"].append(dark_hadron_arrays["status"])
    bucket["Dark_hadron_full_containment"].append(dark_hadron_arrays["full_containment"])
    bucket["Dark_hadron_visible_daughter_counts"].append(dark_hadron_arrays["visible_daughter_counts"])
    bucket["Dark_hadron_visible_daughter_inside_counts"].append(
        dark_hadron_arrays["visible_daughter_inside_counts"]
    )


# =============================================================================
#  logic
# =============================================================================
def build_genfatjet_containment_map(
    pids_evt,
    status_evt,
    m1_evt,
    m2_evt,
    eta_evt,
    phi_evt,
    mass_evt,
    pt_evt,
    px_evt,
    py_evt,
    pz_evt,
    e_evt,
    d1_evt,
    d2_evt,
    gf_particles_evt,
    ref_to_index,
    require_status_8384=True,
):
    """
    Build per-GenFatJet containment using generator ancestry:
      1. Convert GenFatJet TRef constituents to GenParticle indices.
      2. Walk from each constituent up to dark-hadron ancestors.
      3. For each unstable dark hadron, check whether all final-state visible
         descendants are in the same GenFatJet constituent set.
      4. Use the best dark-hadron candidate for the GenFatJet category,
         matching the old splitmatch_pt20 category semantics.
      5. Save all visible unstable dark-hadron candidates found from the
         GenFatJet constituent ancestry.
    """
    n_parts = len(pids_evt)
    dh_indices = np.where(np.isin(pids_evt, list(DARK_HADRONS)))[0]
    dh_daughters = {}

    for dh_idx in dh_indices:
        if require_status_8384 and int(status_evt[dh_idx]) not in DARK_HADRON_COUNT_STATUSES:
            continue

        daughters = get_visible_final_descendants(
            int(dh_idx),
            pids_evt,
            status_evt,
            d1_evt,
            d2_evt,
            n_parts,
        )
        daughters = [
            int(child_idx)
            for child_idx in daughters
            if np.isfinite(eta_evt[child_idx]) and np.isfinite(phi_evt[child_idx])
        ]

        # Stable dark hadrons and fully invisible dark decays are ignored.
        if len(daughters) == 0:
            continue

        dh_daughters[int(dh_idx)] = sorted(set(daughters))

    genfatjet_best = {}
    for gf_idx in range(len(gf_particles_evt)):
        raw_refs = get_refs_from_tref_array_entry(gf_particles_evt[gf_idx])
        gf_constituents = set(refs_to_indices(raw_refs, ref_to_index, n_parts))

        dh_in_gf = set()
        for const_idx in gf_constituents:
            dh_in_gf.update(
                find_dark_ancestors(
                    const_idx,
                    pids_evt,
                    status_evt,
                    m1_evt,
                    m2_evt,
                    dh_daughters,
                    n_parts,
                )
            )

        candidates = []
        all_dark_hadrons = []
        per_pid_counts = {pid_key: 0 for pid_key in PID_ORDER}
        n_complete_total = 0
        n_partial_total = 0

        for dh_idx in sorted(dh_in_gf):
            daughters = dh_daughters[dh_idx]
            daughter_set = set(daughters)
            inside = daughter_set & gf_constituents
            n_inside = len(inside)
            n_daughters = len(daughters)

            is_full = n_inside == n_daughters
            pid = int(pids_evt[dh_idx])
            frac_inside = float(n_inside / max(n_daughters, 1))

            if is_full:
                category = "full"
                rank = 2
                n_complete = 1
                n_partial = 0
            elif n_inside > 0:
                category = "partial"
                rank = 1
                n_complete = 0
                n_partial = 1
            else:
                category = "none"
                rank = 0
                n_complete = 0
                n_partial = 0

            if pid in per_pid_counts:
                per_pid_counts[pid] += 1
            n_complete_total += int(n_complete)
            n_partial_total += int(n_partial)

            dark_hadron = {
                "idx": int(dh_idx),
                "pid": pid,
                "status": int(status_evt[dh_idx]),
                "pt": float(pt_evt[dh_idx]),
                "eta": float(eta_evt[dh_idx]),
                "phi": float(phi_evt[dh_idx]),
                "mass": float(mass_evt[dh_idx]),
                "px": float(px_evt[dh_idx]),
                "py": float(py_evt[dh_idx]),
                "pz": float(pz_evt[dh_idx]),
                "e": float(e_evt[dh_idx]),
                "n_daughters": int(n_daughters),
                "n_inside": int(n_inside),
                "is_full": bool(is_full),
                "category": category,
                "rank": int(rank),
                "frac_inside": frac_inside,
                "is_best": False,
            }
            all_dark_hadrons.append(dark_hadron)

            candidate = {
                "dh_idx": int(dh_idx),
                "gf_idx": int(gf_idx),
                "category": category,
                "rank": int(rank),
                "frac_inside": frac_inside,
                "n_vis": int(n_daughters),
                "dark_hadron": dark_hadron,
            }
            candidates.append(candidate)

        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda item: (
                item["rank"],
                item["frac_inside"],
                item["n_vis"],
                item["dark_hadron"]["pt"],
            ),
        )
        best["dark_hadron"]["is_best"] = True
        best["n_dark"] = len(all_dark_hadrons)
        best["n_visible_dark_hadrons"] = len(all_dark_hadrons)
        best["n_complete"] = int(n_complete_total)
        best["n_partial"] = int(n_partial_total)
        best["per_pid_counts"] = per_pid_counts
        best["dark_hadrons"] = all_dark_hadrons

        genfatjet_best[int(gf_idx)] = best

    return genfatjet_best


def count_dark_hadrons_in_reco_jet(
    pids_evt,
    status_evt,
    eta_evt,
    phi_evt,
    j_eta,
    j_phi,
    fatjet_r,
    require_status_8384=True,
):
    """Count generator-level dark mesons inside the approved reco FatJet cone."""
    finite = np.isfinite(eta_evt) & np.isfinite(phi_evt)
    pid_mask = np.isin(pids_evt, list(DARK_HADRONS))

    if require_status_8384:
        status_mask = np.isin(status_evt, list(DARK_HADRON_COUNT_STATUSES))
    else:
        status_mask = np.ones_like(pid_mask, dtype=bool)

    base_mask = finite & pid_mask & status_mask

    per_pid_counts = {pid: 0 for pid in PID_ORDER}
    total = 0

    for pid in PID_ORDER:
        idx = np.where(base_mask & (pids_evt == pid))[0]
        if len(idx) == 0:
            continue

        dr = calc_dr(eta_evt[idx], phi_evt[idx], j_eta, j_phi)
        n_inside = int(np.sum(dr < fatjet_r))
        per_pid_counts[pid] = n_inside
        total += n_inside

    return total, per_pid_counts


def count_visible_dark_hadrons_in_reco_jet(
    pids_evt,
    status_evt,
    eta_evt,
    phi_evt,
    d1_evt,
    d2_evt,
    j_eta,
    j_phi,
    fatjet_r,
    require_status_8384=True,
):
    """
    Count dark hadrons inside the reco FatJet cone that have at least one visible
    stable descendant. This is a per-jet target-style label, separate from the
    visible_desc_multiplicity of the matched dark-hadron candidate.
    """
    finite = np.isfinite(eta_evt) & np.isfinite(phi_evt)
    pid_mask = np.isin(pids_evt, list(DARK_HADRONS))

    if require_status_8384:
        status_mask = np.isin(status_evt, list(DARK_HADRON_COUNT_STATUSES))
    else:
        status_mask = np.ones_like(pid_mask, dtype=bool)

    dh_indices = np.where(finite & pid_mask & status_mask)[0]
    n_visible_dark = 0

    for dh_idx in dh_indices:
        dr = calc_dr(eta_evt[dh_idx], phi_evt[dh_idx], j_eta, j_phi)
        if dr >= fatjet_r:
            continue

        desc = sorted(set(get_descendants(dh_idx, d1_evt, d2_evt, len(pids_evt))))
        has_visible_desc = False

        for child_idx in desc:
            if is_visible_final_state(pids_evt[child_idx], status_evt[child_idx]):
                if np.isfinite(eta_evt[child_idx]) and np.isfinite(phi_evt[child_idx]):
                    has_visible_desc = True
                    break

        if has_visible_desc:
            n_visible_dark += 1

    return n_visible_dark


def build_jet_matrix(
    j_pt,
    j_eta,
    j_phi,
    j_m,
    pfs_pt,
    pfs_eta,
    pfs_phi,
    pfs_e,
    pfs_charge,
    pfs_pid,
    pfs_d0,
    pfs_dz,
    pfs_eem,
    pfs_ehad,
    pf_dark_descendant_ids,
    fatjet_r,
    max_constituents,
):
    pf_drs = calc_dr(j_eta, j_phi, pfs_eta, pfs_phi)
    pf_in = pf_drs < fatjet_r

    c_pt = pfs_pt[pf_in]
    c_eta = pfs_eta[pf_in]
    c_phi = pfs_phi[pf_in]
    c_e = pfs_e[pf_in]
    c_charge = pfs_charge[pf_in]
    c_pid = pfs_pid[pf_in]
    c_d0 = pfs_d0[pf_in]
    c_dz = pfs_dz[pf_in]
    c_eem = pfs_eem[pf_in]
    c_ehad = pfs_ehad[pf_in]
    c_dark_descendant_id = pf_dark_descendant_ids[pf_in]

    c_abs_pid = np.abs(c_pid)
    is_photon = (c_abs_pid == 22).astype(np.float32)
    is_chad = (c_abs_pid == 211).astype(np.float32)
    is_nhad = (c_abs_pid == 130).astype(np.float32)
    is_elec = (c_abs_pid == 11).astype(np.float32)
    is_muon = (c_abs_pid == 13).astype(np.float32)

    sort_idx = np.argsort(c_pt)[::-1]
    c_pt = c_pt[sort_idx]
    c_eta = c_eta[sort_idx]
    c_phi = c_phi[sort_idx]
    c_e = c_e[sort_idx]
    c_charge = c_charge[sort_idx]
    c_d0 = c_d0[sort_idx]
    c_dz = c_dz[sort_idx]
    c_eem = c_eem[sort_idx]
    c_ehad = c_ehad[sort_idx]
    c_dark_descendant_id = c_dark_descendant_id[sort_idx]
    is_photon = is_photon[sort_idx]
    is_chad = is_chad[sort_idx]
    is_nhad = is_nhad[sort_idx]
    is_elec = is_elec[sort_idx]
    is_muon = is_muon[sort_idx]

    c_deta, c_dphi = center_coords(c_eta, c_phi, j_eta, j_phi)
    n_const = min(len(c_pt), max_constituents)

    jet_matrix = np.zeros((max_constituents, N_CONSTITUENT_FEATURES), dtype=np.float32)
    j_e = np.sqrt((j_pt * np.cosh(j_eta)) ** 2 + j_m**2)

    if n_const > 0:
        jet_matrix[:n_const, 0] = c_e[:n_const]
        jet_matrix[:n_const, 1] = c_pt[:n_const]
        jet_matrix[:n_const, 2] = c_deta[:n_const]
        jet_matrix[:n_const, 3] = c_dphi[:n_const]
        jet_matrix[:n_const, 4] = c_charge[:n_const]
        jet_matrix[:n_const, 5] = c_d0[:n_const]
        jet_matrix[:n_const, 6] = c_dz[:n_const]
        jet_matrix[:n_const, 7] = c_eem[:n_const]
        jet_matrix[:n_const, 8] = c_ehad[:n_const]
        jet_matrix[:n_const, 9] = is_photon[:n_const]
        jet_matrix[:n_const, 10] = is_chad[:n_const]
        jet_matrix[:n_const, 11] = is_nhad[:n_const]
        jet_matrix[:n_const, 12] = is_elec[:n_const]
        jet_matrix[:n_const, 13] = is_muon[:n_const]
        jet_matrix[:n_const, 14] = np.log(np.clip(c_pt[:n_const] / j_pt, 1e-8, 1.0))
        jet_matrix[:n_const, 15] = np.log(np.clip(c_e[:n_const] / j_e, 1e-8, 1.0))
        jet_matrix[:n_const, PF_DARK_DESCENDANT_FEATURE_INDEX] = c_dark_descendant_id[:n_const]

    return jet_matrix


# =============================================================================
# Per-file worker
# =============================================================================
def process_one_root_file(args_tuple):
    (
        root_file,
        file_idx,
        keep_mode,
        max_events_per_file,
        fatjet_r,
        matching_r,
        pt_cut,
        max_constituents,
        max_dark_hadrons,
        require_status_8384,
    ) = args_tuple

    result = empty_worker_result(root_file, file_idx)

    target_mpi = parse_mpi_from_path(root_file)
    if target_mpi is None:
        result["error"] = f"Could not parse target mass from path: {root_file}"
        return result

    f = None
    try:
        f, tree = get_delphes_tree(root_file)

        # Gen particles
        gen_pid = get_branch(tree, ["GenParticle/GenParticle.PID", "GenParticle.PID"])
        gen_uid = get_branch(tree, ["GenParticle/GenParticle.fUniqueID", "GenParticle.fUniqueID"], required=False)
        gen_eta = get_branch(tree, ["GenParticle/GenParticle.Eta", "GenParticle.Eta"])
        gen_phi = get_branch(tree, ["GenParticle/GenParticle.Phi", "GenParticle.Phi"])
        gen_mass = get_branch(tree, ["GenParticle/GenParticle.Mass", "GenParticle.Mass"])
        gen_pt = get_branch(tree, ["GenParticle/GenParticle.PT", "GenParticle.PT"])
        gen_px = get_branch(tree, ["GenParticle/GenParticle.Px", "GenParticle.Px"])
        gen_py = get_branch(tree, ["GenParticle/GenParticle.Py", "GenParticle.Py"])
        gen_pz = get_branch(tree, ["GenParticle/GenParticle.Pz", "GenParticle.Pz"])
        gen_e = get_branch(tree, ["GenParticle/GenParticle.E", "GenParticle.E"])
        gen_status = get_branch(tree, ["GenParticle/GenParticle.Status", "GenParticle.Status"])
        gen_m1 = get_branch(tree, ["GenParticle/GenParticle.M1", "GenParticle.M1"])
        gen_m2 = get_branch(tree, ["GenParticle/GenParticle.M2", "GenParticle.M2"])
        gen_d1 = get_branch(tree, ["GenParticle/GenParticle.D1", "GenParticle.D1"])
        gen_d2 = get_branch(tree, ["GenParticle/GenParticle.D2", "GenParticle.D2"])

        # GenFatJets
        gf_eta = get_branch(tree, ["GenFatJet/GenFatJet.Eta", "GenFatJet.Eta"])
        gf_phi = get_branch(tree, ["GenFatJet/GenFatJet.Phi", "GenFatJet.Phi"])
        gf_pt = get_branch(tree, ["GenFatJet/GenFatJet.PT", "GenFatJet.PT"])
        gf_mass = get_branch(tree, ["GenFatJet/GenFatJet.Mass", "GenFatJet.Mass"])
        gf_particles = get_branch(
            tree,
            ["GenFatJet/GenFatJet.Particles", "GenFatJet.Particles", "GenFatJet/GenFatJet.Constituents", "GenFatJet.Constituents"],
        )

        # Reco FatJets
        rf_pt = get_branch(tree, ["FatJet/FatJet.PT", "FatJet.PT"])
        rf_eta = get_branch(tree, ["FatJet/FatJet.Eta", "FatJet.Eta"])
        rf_phi = get_branch(tree, ["FatJet/FatJet.Phi", "FatJet.Phi"])
        rf_m = get_branch(tree, ["FatJet/FatJet.Mass", "FatJet.Mass"])
        rf_ptd = get_branch(tree, ["FatJet/FatJet.PTD", "FatJet.PTD"])
        rf_msqdr = get_branch(tree, ["FatJet/FatJet.MeanSqDeltaR", "FatJet.MeanSqDeltaR"])
        rf_nef = get_branch(tree, ["FatJet/FatJet.NeutralEnergyFraction", "FatJet.NeutralEnergyFraction"])
        rf_cef = get_branch(tree, ["FatJet/FatJet.ChargedEnergyFraction", "FatJet.ChargedEnergyFraction"])
        rf_tau = get_branch(tree, ["FatJet/FatJet.Tau[5]", "FatJet.Tau[5]"])

        # SoftDrop
        rf_sdmass = get_branch(
            tree,
            ["FatJet/FatJet.SoftDroppedJet.Mass", "FatJet.SoftDroppedJet.Mass"],
            required=False,
        )
        rf_sdpt = get_branch(
            tree,
            ["FatJet/FatJet.SoftDroppedJet.PT", "FatJet.SoftDroppedJet.PT"],
            required=False,
        )
        if rf_sdmass is None or rf_sdpt is None:
            packed_sdmass, packed_sdpt = build_softdrop_from_packed(tree)
            if rf_sdmass is None:
                rf_sdmass = packed_sdmass
            if rf_sdpt is None:
                rf_sdpt = packed_sdpt

        # PF candidates
        pf_pt = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.PT", "ParticleFlowCandidate.PT"])
        pf_eta = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.Eta", "ParticleFlowCandidate.Eta"])
        pf_phi = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.Phi", "ParticleFlowCandidate.Phi"])
        pf_e = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.E", "ParticleFlowCandidate.E"])
        pf_charge = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.Charge", "ParticleFlowCandidate.Charge"])
        pf_pid = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.PID", "ParticleFlowCandidate.PID"])
        pf_d0 = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.D0", "ParticleFlowCandidate.D0"])
        pf_dz = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.DZ", "ParticleFlowCandidate.DZ"])
        pf_eem = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.Eem", "ParticleFlowCandidate.Eem"])
        pf_ehad = get_branch(tree, ["ParticleFlowCandidate/ParticleFlowCandidate.Ehad", "ParticleFlowCandidate.Ehad"])
        pf_particles = get_branch(
            tree,
            [
                "ParticleFlowCandidate/ParticleFlowCandidate.Particles",
                "ParticleFlowCandidate.Particles",
                "ParticleFlowCandidate/ParticleFlowCandidate.Particle",
                "ParticleFlowCandidate.Particle",
            ],
            required=False,
        )

        n_events = len(rf_pt)
        if max_events_per_file is not None:
            n_events = min(n_events, max_events_per_file)

        for iev in range(n_events):
            pids_evt = ak.to_numpy(gen_pid[iev])
            uid_evt = ak.to_numpy(gen_uid[iev]) if gen_uid is not None else None
            eta_evt = ak.to_numpy(gen_eta[iev])
            phi_evt = ak.to_numpy(gen_phi[iev])
            mass_evt = ak.to_numpy(gen_mass[iev])
            pt_evt = ak.to_numpy(gen_pt[iev])
            px_evt = ak.to_numpy(gen_px[iev])
            py_evt = ak.to_numpy(gen_py[iev])
            pz_evt = ak.to_numpy(gen_pz[iev])
            e_evt = ak.to_numpy(gen_e[iev])
            status_evt = ak.to_numpy(gen_status[iev])
            m1_evt = ak.to_numpy(gen_m1[iev])
            m2_evt = ak.to_numpy(gen_m2[iev])
            d1_evt = ak.to_numpy(gen_d1[iev])
            d2_evt = ak.to_numpy(gen_d2[iev])
            ref_to_index = build_ref_to_index_map(uid_evt)

            gf_eta_evt = ak.to_numpy(gf_eta[iev])
            gf_phi_evt = ak.to_numpy(gf_phi[iev])
            gf_pt_evt = ak.to_numpy(gf_pt[iev])
            gf_mass_evt = ak.to_numpy(gf_mass[iev])

            rfs_pt = ak.to_numpy(rf_pt[iev])
            rfs_eta = ak.to_numpy(rf_eta[iev])
            rfs_phi = ak.to_numpy(rf_phi[iev])
            rfs_m = ak.to_numpy(rf_m[iev])
            rfs_ptd = ak.to_numpy(rf_ptd[iev])
            rfs_msqdr = ak.to_numpy(rf_msqdr[iev])
            rfs_nef = ak.to_numpy(rf_nef[iev])
            rfs_cef = ak.to_numpy(rf_cef[iev])
            rfs_tau = ak.to_numpy(rf_tau[iev])

            pfs_pt = ak.to_numpy(pf_pt[iev])
            pfs_eta = ak.to_numpy(pf_eta[iev])
            pfs_phi = ak.to_numpy(pf_phi[iev])
            pfs_e = ak.to_numpy(pf_e[iev])
            pfs_charge = ak.to_numpy(pf_charge[iev])
            pfs_pid = ak.to_numpy(pf_pid[iev])
            pfs_d0 = ak.to_numpy(pf_d0[iev])
            pfs_dz = ak.to_numpy(pf_dz[iev])
            pfs_eem = ak.to_numpy(pf_eem[iev])
            pfs_ehad = ak.to_numpy(pf_ehad[iev])
            pf_gen_indices = build_pf_gen_indices(
                pf_particles_evt=pf_particles[iev] if pf_particles is not None else None,
                ref_to_index=ref_to_index,
                n_parts=len(pids_evt),
                n_pf=len(pfs_pt),
            )

            genfatjet_best = build_genfatjet_containment_map(
                pids_evt=pids_evt,
                status_evt=status_evt,
                m1_evt=m1_evt,
                m2_evt=m2_evt,
                eta_evt=eta_evt,
                phi_evt=phi_evt,
                mass_evt=mass_evt,
                pt_evt=pt_evt,
                px_evt=px_evt,
                py_evt=py_evt,
                pz_evt=pz_evt,
                e_evt=e_evt,
                d1_evt=d1_evt,
                d2_evt=d2_evt,
                gf_particles_evt=gf_particles[iev],
                ref_to_index=ref_to_index,
                require_status_8384=require_status_8384,
            )

            for r_idx in range(len(rfs_pt)):
                result["n_total_recojets"] += 1

                j_pt = float(rfs_pt[r_idx])
                j_eta = float(rfs_eta[r_idx])
                j_phi = float(rfs_phi[r_idx])
                j_m = float(rfs_m[r_idx])

                if not (np.isfinite(j_pt) and np.isfinite(j_eta) and np.isfinite(j_phi) and np.isfinite(j_m)):
                    continue

                if pt_cut is not None and pt_cut > 0.0 and j_pt < pt_cut:
                    continue
                result["n_ptcut_recojets"] += 1

                if len(gf_eta_evt) == 0:
                    continue

                drs_rg = calc_dr(j_eta, j_phi, gf_eta_evt, gf_phi_evt)
                closest_gen = int(np.argmin(drs_rg))
                closest_dr = float(drs_rg[closest_gen])

                if closest_dr >= matching_r:
                    continue
                result["n_recojets_matched_to_gen"] += 1

                if closest_gen not in genfatjet_best:
                    continue

                match_info = genfatjet_best[closest_gen]
                cat = match_info["category"]

                if not should_keep_category(cat, keep_mode):
                    continue
                if cat not in BUCKETS:
                    continue

                result["n_recojets_approved"] += 1

                n_dark_total = match_info["n_dark"]
                n_visible_dark_hadrons = match_info["n_visible_dark_hadrons"]
                per_pid_counts = match_info["per_pid_counts"]
                dark_hadron_arrays = make_dark_hadron_arrays(
                    match_info["dark_hadrons"],
                    max_dark_hadrons=max_dark_hadrons,
                )
                pf_dark_descendant_ids = build_pf_dark_descendant_ids(
                    pf_gen_indices=pf_gen_indices,
                    dark_hadrons=match_info["dark_hadrons"],
                    max_dark_hadrons=max_dark_hadrons,
                    pids_evt=pids_evt,
                    status_evt=status_evt,
                    d1_evt=d1_evt,
                    d2_evt=d2_evt,
                    n_parts=len(pids_evt),
                )

                jet_matrix = build_jet_matrix(
                    j_pt=j_pt,
                    j_eta=j_eta,
                    j_phi=j_phi,
                    j_m=j_m,
                    pfs_pt=pfs_pt,
                    pfs_eta=pfs_eta,
                    pfs_phi=pfs_phi,
                    pfs_e=pfs_e,
                    pfs_charge=pfs_charge,
                    pfs_pid=pfs_pid,
                    pfs_d0=pfs_d0,
                    pfs_dz=pfs_dz,
                    pfs_eem=pfs_eem,
                    pfs_ehad=pfs_ehad,
                    pf_dark_descendant_ids=pf_dark_descendant_ids,
                    fatjet_r=fatjet_r,
                    max_constituents=max_constituents,
                )

                taus = rfs_tau[r_idx]
                tau1 = float(taus[0]) if len(taus) > 0 else -1.0
                tau2 = float(taus[1]) if len(taus) > 1 else -1.0
                tau3 = float(taus[2]) if len(taus) > 2 else -1.0
                tau21 = tau2 / tau1 if tau1 > 0 else -1.0
                tau32 = tau3 / tau2 if tau2 > 0 else -1.0

                j_sdmass = safe_softdrop_mass(rf_sdmass, iev, r_idx)
                j_sdpt = safe_softdrop_pt(rf_sdpt, iev, r_idx)

                macro_vec = [
                    j_m,
                    j_sdmass,
                    j_sdpt,
                    float(rfs_ptd[r_idx]),
                    float(rfs_msqdr[r_idx]),
                    float(rfs_nef[r_idx]),
                    float(rfs_cef[r_idx]),
                    tau1,
                    tau2,
                    tau3,
                    tau21,
                    tau32,
                ]

                append_jet_to_bucket(
                    bucket_data=result["bucket_data"],
                    bucket_name=cat,
                    jet_matrix=jet_matrix,
                    target_mpi=target_mpi,
                    j_pt=j_pt,
                    j_eta=j_eta,
                    j_phi=j_phi,
                    j_m=j_m,
                    macro_vec=macro_vec,
                    match_info=match_info,
                    gf_mass_val=float(gf_mass_evt[closest_gen]),
                    gf_pt_val=float(gf_pt_evt[closest_gen]),
                    file_idx=file_idx,
                    n_dark_total=n_dark_total,
                    n_visible_dark_hadrons=n_visible_dark_hadrons,
                    per_pid_counts=per_pid_counts,
                    dark_hadron_arrays=dark_hadron_arrays,
                )

                result[f"n_recojets_saved_{cat}"] += 1
                result["n_recojets_saved"] += 1

        result["ok"] = True
        return result

    except Exception as exc:
        result["error"] = f"{exc}\n{traceback.format_exc()}"
        return result
    finally:
        try:
            if f is not None:
                f.close()
        except Exception:
            pass


# =============================================================================
# Merge and save
# =============================================================================
def merge_results(results):
    merged = {
        "n_total_recojets": 0,
        "n_ptcut_recojets": 0,
        "n_recojets_matched_to_gen": 0,
        "n_recojets_approved": 0,
        "n_recojets_saved_full": 0,
        "n_recojets_saved_partial": 0,
        "n_recojets_saved_none": 0,
        "n_recojets_saved": 0,
        "bucket_data": {name: empty_bucket_container() for name in BUCKETS},
    }

    for result in results:
        for key in [
            "n_total_recojets",
            "n_ptcut_recojets",
            "n_recojets_matched_to_gen",
            "n_recojets_approved",
            "n_recojets_saved_full",
            "n_recojets_saved_partial",
            "n_recojets_saved_none",
            "n_recojets_saved",
        ]:
            merged[key] += result[key]

        for bucket_name in BUCKETS:
            for data_key in DATA_KEYS:
                merged["bucket_data"][bucket_name][data_key].extend(result["bucket_data"][bucket_name][data_key])

    return merged


def bucket_to_arrays(bucket, max_constituents, max_dark_hadrons):
    n = len(bucket["Y_labels"])

    arrays = {
        "X": np.asarray(bucket["X_jets"], dtype=np.float32).reshape(n, max_constituents, N_CONSTITUENT_FEATURES),
        "y": np.asarray(bucket["Y_labels"], dtype=np.float32),
        "kinematics": np.asarray(bucket["Jet_kinematics"], dtype=np.float32).reshape(n, 4),
        "masses": np.asarray(bucket["Mass_params"], dtype=np.float32).reshape(n, 1),
        "macro": np.asarray(bucket["Macro_features"], dtype=np.float32).reshape(n, 12),
        "containment_fraction": np.asarray(bucket["Containment_fraction"], dtype=np.float32),
        "containment_category": np.asarray(bucket["Containment_category"], dtype=np.int32),
        "matched_genfatjet_mass": np.asarray(bucket["Matched_genfatjet_mass"], dtype=np.float32),
        "matched_genfatjet_pt": np.asarray(bucket["Matched_genfatjet_pt"], dtype=np.float32),
        "visible_desc_multiplicity": np.asarray(bucket["Visible_desc_multiplicity"], dtype=np.int32),
        "n_dark_hadrons_per_jet": np.asarray(bucket["NDark_hadrons_per_jet"], dtype=np.int32),
        "n_visible_dark_hadrons": np.asarray(bucket["NVisible_dark_hadrons"], dtype=np.int32),
        "ndark_piD0_per_jet": np.asarray(bucket["NDark_piD0_per_jet"], dtype=np.int32),
        "ndark_piDplus_per_jet": np.asarray(bucket["NDark_piDplus_per_jet"], dtype=np.int32),
        "ndark_piDminus_per_jet": np.asarray(bucket["NDark_piDminus_per_jet"], dtype=np.int32),
        "ndark_rhoD0_per_jet": np.asarray(bucket["NDark_rhoD0_per_jet"], dtype=np.int32),
        "ndark_rhoDplus_per_jet": np.asarray(bucket["NDark_rhoDplus_per_jet"], dtype=np.int32),
        "ndark_rhoDminus_per_jet": np.asarray(bucket["NDark_rhoDminus_per_jet"], dtype=np.int32),
        "n_dark_hadrons_complete": np.asarray(bucket["NDark_hadrons_complete"], dtype=np.int32),
        "n_dark_hadrons_partial": np.asarray(bucket["NDark_hadrons_partial"], dtype=np.int32),
        "dark_hadron_p4": np.asarray(bucket["Dark_hadron_p4"], dtype=np.float32).reshape(n, max_dark_hadrons, 4),
        "dark_hadron_kinematics": np.asarray(bucket["Dark_hadron_kinematics"], dtype=np.float32).reshape(
            n, max_dark_hadrons, 4
        ),
        "dark_hadron_pid": np.asarray(bucket["Dark_hadron_pid"], dtype=np.int32).reshape(n, max_dark_hadrons),
        "dark_hadron_status": np.asarray(bucket["Dark_hadron_status"], dtype=np.int32).reshape(n, max_dark_hadrons),
        "dark_hadron_full_containment": np.asarray(
            bucket["Dark_hadron_full_containment"], dtype=np.int32
        ).reshape(n, max_dark_hadrons),
        "dark_hadron_visible_daughter_counts": np.asarray(
            bucket["Dark_hadron_visible_daughter_counts"], dtype=np.int32
        ).reshape(n, max_dark_hadrons),
        "dark_hadron_visible_daughter_inside_counts": np.asarray(
            bucket["Dark_hadron_visible_daughter_inside_counts"], dtype=np.int32
        ).reshape(n, max_dark_hadrons),
        "file_source": np.asarray(bucket["File_source"], dtype=np.int32),
        "root_file_index": np.asarray(bucket["Root_file_index"], dtype=np.int32),
    }

    return arrays


def shuffle_arrays(arrays, rng):
    n = len(arrays["y"])
    if n == 0:
        return arrays
    idx = rng.permutation(n)
    return {key: value[idx] for key, value in arrays.items()}


def save_bucket_npz(outdir, output_prefix, bucket_name, bucket, max_constituents, max_dark_hadrons, no_shuffle, rng):
    arrays = bucket_to_arrays(bucket, max_constituents=max_constituents, max_dark_hadrons=max_dark_hadrons)
    if not no_shuffle:
        arrays = shuffle_arrays(arrays, rng)

    output_path = os.path.join(outdir, f"{output_prefix}_{bucket_name}.npz")
    np.savez_compressed(output_path, **arrays)

    n = len(arrays["y"])
    print(f"[SAVE] {bucket_name:7s}: {n:8d} jets -> {output_path}")
    if n > 0:
        vals, counts = np.unique(arrays["n_visible_dark_hadrons"], return_counts=True)
        count_str = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
        print(f"       n_visible_dark_hadrons counts: {count_str}")

    return output_path, n


def write_root_file_map(outdir, root_files):
    path = os.path.join(outdir, "root_file_index_map.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["root_file_index", "root_file"])
        for idx, root_file in enumerate(root_files):
            writer.writerow([idx, root_file])
    return path


def write_summary(outdir, output_prefix, merged, saved_info, bad_results):
    path = os.path.join(outdir, f"{output_prefix}_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        for key in [
            "n_total_recojets",
            "n_ptcut_recojets",
            "n_recojets_matched_to_gen",
            "n_recojets_approved",
            "n_recojets_saved_full",
            "n_recojets_saved_partial",
            "n_recojets_saved_none",
            "n_recojets_saved",
        ]:
            writer.writerow([key, merged[key]])
        for bucket_name, (output_path, n_saved) in saved_info.items():
            writer.writerow([f"output_{bucket_name}", output_path])
            writer.writerow([f"n_saved_{bucket_name}", n_saved])
        writer.writerow(["n_failed_files", len(bad_results)])

    if bad_results:
        bad_path = os.path.join(outdir, f"{output_prefix}_failed_files.txt")
        with open(bad_path, "w") as f:
            for result in bad_results:
                f.write(f"{result['root_file']}\n")
                f.write(f"{result['error']}\n\n")
        print(f"[WARNING] Failed-file list written to: {bad_path}")

    return path


# =============================================================================
# IO discovery
# =============================================================================
def read_filelist(filelist, eos_host):
    files = []
    with open(filelist) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("/store/") or line.startswith("/eos/"):
                line = eos_host.rstrip("/") + "/" + line
            files.append(line)
    return sorted(files)


def discover_root_files(base_dirs):
    files = []
    for base in base_dirs:
        files.extend(glob.glob(os.path.join(base, "**", "events.root"), recursive=True))
    return sorted(files)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build DRACON NPZ datasets split by complete-genmatch containment category."
    )
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        default=DEFAULT_BASE_DIRS,
        help="Base directories to recursively search for events.root files when --filelist is not used.",
    )
    parser.add_argument(
        "--filelist",
        default=None,
        help="Optional text file containing events.root paths. Supports /store/... paths with --eos-host.",
    )
    parser.add_argument(
        "--eos-host",
        default="root://cmseos.fnal.gov",
        help="EOS XRootD host prefix used when --filelist contains /store/... paths.",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for output NPZ files. Outputs are <prefix>_full.npz, <prefix>_partial.npz, <prefix>_none.npz.",
    )
    parser.add_argument(
        "--keep-mode",
        default="all",
        choices=[
            "full_only",
            "partial_only",
            "none_only",
            "full_plus_partial",
            "partial_plus_none",
            "not_full",
            "all",
        ],
        help="Which containment categories to keep before writing split NPZs. Use all for the full campaign.",
    )
    parser.add_argument("--max-events-per-file", type=int, default=None, help="Optional cap on events per ROOT file.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of ROOT files for tests.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes.")
    parser.add_argument("--fatjet-r", type=float, default=FATJET_R_DEFAULT, help="FatJet radius.")
    parser.add_argument("--matching-r", type=float, default=MATCHING_R_DEFAULT, help="Reco-to-GenFatJet matching radius.")
    parser.add_argument(
        "--pt-cut",
        type=float,
        default=PT_CUT_DEFAULT,
        help="Optional reco FatJet pT cut. Default 0 disables the cut.",
    )
    parser.add_argument("--max-constituents", type=int, default=MAX_CONSTITUENTS_DEFAULT, help="Max PF constituents per jet.")
    parser.add_argument(
        "--max-dark-hadrons",
        type=int,
        default=MAX_DARK_HADRONS_DEFAULT,
        help="Max dark hadrons per jet to save in padded dark_hadron_* arrays.",
    )
    parser.add_argument(
        "--all-dark-statuses",
        action="store_true",
        help="Use all generator-record dark mesons in constituent matching. Default uses only status 83/84.",
    )
    parser.add_argument("--no-shuffle", action="store_true", help="Disable per-bucket final shuffle.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.filelist is not None:
        root_files = read_filelist(args.filelist, args.eos_host)
        print(f"[INFO] Loaded {len(root_files)} ROOT files from filelist: {args.filelist}")
    else:
        root_files = discover_root_files(args.base_dirs)
        print(f"[INFO] Found {len(root_files)} ROOT files from local base dirs")

    if args.max_files is not None:
        root_files = root_files[: args.max_files]
        print(f"[INFO] Applying --max-files: using {len(root_files)} files")

    if len(root_files) == 0:
        raise RuntimeError("No ROOT files found.")

    map_path = write_root_file_map(args.outdir, root_files)
    print(f"[INFO] Wrote root-file index map: {map_path}")
    if args.pt_cut is not None and args.pt_cut > 0.0:
        print(f"[INFO] pT cut: {args.pt_cut} GeV")
    else:
        print("[INFO] pT cut: disabled")
    print(f"[INFO] keep-mode: {args.keep_mode}")

    require_status_8384 = not args.all_dark_statuses
    if require_status_8384:
        print("[INFO] Dark-hadron constituent matching uses status 83/84 only")
    else:
        print("[INFO] Dark-hadron constituent matching uses all generator-record dark-hadron statuses")

    worker_args = [
        (
            root_file,
            file_idx,
            args.keep_mode,
            args.max_events_per_file,
            args.fatjet_r,
            args.matching_r,
            args.pt_cut,
            args.max_constituents,
            args.max_dark_hadrons,
            require_status_8384,
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
        raise RuntimeError("All files failed. No datasets produced.")

    merged = merge_results(good_results)

    print("\n" + "=" * 72)
    print("DRACON COMPLETE-GENMATCH SPLIT DATASET EXTRACTION COMPLETE")
    print("=" * 72)
    print(f"Total reco jets seen                 : {merged['n_total_recojets']}")
    print(f"Reco jets passing pT selection       : {merged['n_ptcut_recojets']}")
    print(f"Reco jets matched to GenFatJets      : {merged['n_recojets_matched_to_gen']}")
    print(f"Reco jets approved by keep-mode      : {merged['n_recojets_approved']}")
    print(f"Saved full jets                      : {merged['n_recojets_saved_full']}")
    print(f"Saved partial jets                   : {merged['n_recojets_saved_partial']}")
    print(f"Saved none jets                      : {merged['n_recojets_saved_none']}")
    print(f"Saved total jets                     : {merged['n_recojets_saved']}")
    print(f"Failed files                         : {len(bad_results)}")

    rng = np.random.default_rng(args.seed)
    saved_info = {}
    for bucket_name in ["full", "partial", "none"]:
        output_path, n_saved = save_bucket_npz(
            outdir=args.outdir,
            output_prefix=args.output_prefix,
            bucket_name=bucket_name,
            bucket=merged["bucket_data"][bucket_name],
            max_constituents=args.max_constituents,
            max_dark_hadrons=args.max_dark_hadrons,
            no_shuffle=args.no_shuffle,
            rng=rng,
        )
        saved_info[bucket_name] = (output_path, n_saved)

    summary_path = write_summary(args.outdir, args.output_prefix, merged, saved_info, bad_results)
    print(f"[INFO] Summary written to: {summary_path}")

    print("\nNPZ feature notes:")
    print(f"  X shape: (N, max_constituents, {N_CONSTITUENT_FEATURES})")
    print("  X[..., 16]: dark_hadron_descendant_id")
    print("    0 = no linked visible DH descendant")
    print("    k > 0 = descendant of dark_hadron_* slot k-1 for this jet")
    print("  kinematics order: [pt, eta, phi, mass]")
    print("  masses: true target m_pi parsed from the file path")
    print("  containment_category: full=2, partial=1, none=0")
    print("  n_dark_hadrons_per_jet: visible unstable dark mesons found from GenFatJet constituent ancestry")
    print("  n_visible_dark_hadrons: same count after stable/invisible dark hadrons are ignored")
    print("  dark_hadron_p4 order: [px, py, pz, e], padded to max_dark_hadrons; slot 0 is the best category DH")
    print("  dark_hadron_kinematics order: [pt, eta, phi, mass], padded to max_dark_hadrons; slot 0 is the best category DH")
    print("  macro order:")
    print("    [0]  Jet_mass")
    print("    [1]  Jet_sdmass")
    print("    [2]  Jet_sdpt")
    print("    [3]  Jet_ptD")
    print("    [4]  Jet_msqdr")
    print("    [5]  Jet_nef")
    print("    [6]  Jet_cef")
    print("    [7]  Jet_tau1")
    print("    [8]  Jet_tau2")
    print("    [9]  Jet_tau3")
    print("    [10] Jet_tau21")
    print("    [11] Jet_tau32")

    print(f"\nDone. Outputs saved in: {args.outdir}")


if __name__ == "__main__":
    main()
