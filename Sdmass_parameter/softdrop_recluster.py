"""Recompute soft drop mass at an arbitrary (beta, z_cut) directly from the
PF-constituent 4-vectors already stored in the strategy npz datasets (the
`X` array: up to 100 constituents x 17 features per jet, see
build_dracon_..._pftruth_noptcut.py). No Pythia/Delphes rerun needed.

Constituent features used here (indices into X's last axis):
  0: E (absolute energy)
  1: pt (absolute)
  2: deta  (eta relative to jet axis, i.e. c_eta - j_eta)
  3: dphi  (phi relative to jet axis, wrapped c_phi - j_phi)
Absolute eta/phi are recovered as j_eta + deta, j_phi + dphi (see
center_coords() in build_dracon_..._pftruth_noptcut.py) -- this matters:
plugging deta/dphi directly into cos/sin/sinh instead of the *absolute*
eta/phi silently wrecks all inter-constituent angular separations (only
correct when j_eta==j_phi==0), which was caught by the validation gate
below producing >100% mass disagreement before this fix.

Procedure follows Sec. 2.1 of the Soft Drop paper (arXiv:1402.2657):
  1. recluster the jet's constituents with Cambridge/Aachen (R0)
  2. undo the last C/A merging; if the softer branch fails
     min(pT1,pT2)/(pT1+pT2) > z_cut * (dR12/R0)^beta, drop it and recurse on
     the harder branch; otherwise stop (this is the final soft-drop jet)
  3. if declustering bottoms out at a single particle, keep it as-is
     (grooming mode -- matches ComputeSoftDrop in the Delphes card, which is
     not run in tagging mode).
"""
import numpy as np
import fastjet as fj

import config


def constituents_to_pseudojets(X_row, j_eta, j_phi):
    """X_row: (max_constituents, 17) array for one jet. j_eta/j_phi: that
    jet's own (absolute) axis, needed to undo the deta/dphi centering.
    Returns a list of fastjet.PseudoJet, one per valid (pt>0) constituent,
    each tagged with user_index = its row index in X_row so soft-drop-
    dropped constituents can be identified later by index."""
    pt = X_row[:, 1]
    valid = np.nonzero(pt > 0)[0]
    pjs = []
    for i in valid:
        e = float(X_row[i, 0])
        pt_i = float(X_row[i, 1])
        eta_i = j_eta + float(X_row[i, 2])
        phi_i = j_phi + float(X_row[i, 3])
        px = pt_i * np.cos(phi_i)
        py = pt_i * np.sin(phi_i)
        pz = pt_i * np.sinh(eta_i)
        pj = fj.PseudoJet(px, py, pz, e)
        pj.set_user_index(int(i))
        pjs.append(pj)
    return pjs


def cluster_ca(pjs, R0=config.R0):
    """Recluster with Cambridge/Aachen, return the hardest inclusive jet
    (None if there are no constituents)."""
    if len(pjs) == 0:
        return None
    jetdef = fj.JetDefinition(fj.cambridge_algorithm, R0)
    cs = fj.ClusterSequence(pjs, jetdef)
    jets = fj.sorted_by_pt(cs.inclusive_jets())
    if not jets:
        return None
    # Keep the cluster sequence alive by stashing it on the returned jet, in
    # case anything downstream needs associated-structure lookups; the
    # declustering walk below no longer relies on this (see _collect_leaves)
    # but it's a harmless safety net.
    jets[0]._cs_keepalive = cs
    return jets[0]


def _collect_leaves(jet, out_indices):
    """Recursively walk a PseudoJet's declustering history down to the
    original input particles, appending each leaf's user_index to
    out_indices. Deliberately avoids PseudoJet.constituents(): in this
    fastjet SWIG binding, PseudoJets returned by has_parents() do not carry
    the "structure" pointer .constituents() needs (confirmed live -- it
    raises FastJetError "no associated structure" a few has_parents() calls
    deep), whereas has_parents()/user_index()/pt() keep working at any
    depth, which is all a manual leaf-walk needs."""
    p1, p2 = fj.PseudoJet(), fj.PseudoJet()
    if jet.has_parents(p1, p2):
        _collect_leaves(p1, out_indices)
        _collect_leaves(p2, out_indices)
    else:
        out_indices.append(jet.user_index())


def soft_drop_decluster(hardest_jet, z_cut, beta, R0=config.R0):
    """Walk the C/A declustering history applying the soft drop condition.

    Returns (groomed_pt, groomed_mass, retained_indices, dropped_indices).
    `retained_indices`/`dropped_indices` are lists of original-constituent
    row indices (the user_index values set in constituents_to_pseudojets).
    """
    current = hardest_jet
    dropped = []
    while True:
        p1, p2 = fj.PseudoJet(), fj.PseudoJet()
        has_parents = current.has_parents(p1, p2)
        if not has_parents:
            break
        if p1.pt() < p2.pt():
            p1, p2 = p2, p1
        pt1, pt2 = p1.pt(), p2.pt()
        if pt1 + pt2 <= 0:
            break
        dr12 = p1.delta_R(p2)
        z = min(pt1, pt2) / (pt1 + pt2)
        threshold = z_cut * (dr12 / R0) ** beta
        if z > threshold:
            # Soft drop condition passed: `current` is already p1+p2 (that's
            # how it got produced during clustering) -- do NOT reassign it to
            # a freshly-summed p1+p2 PseudoJet, which would be a brand new
            # object with no declustering history, breaking the
            # _collect_leaves() walk below.
            break
        _collect_leaves(p2, dropped)
        current = p1

    retained = []
    _collect_leaves(current, retained)
    return current.pt(), current.m(), retained, dropped


def jet_softdrop_mass(X_row, j_eta, j_phi, z_cut, beta, R0=config.R0):
    """Convenience one-shot: X_row -> groomed mass (GeV). Returns np.nan if
    the jet has no valid constituents."""
    pjs = constituents_to_pseudojets(X_row, j_eta, j_phi)
    hardest = cluster_ca(pjs, R0=R0)
    if hardest is None:
        return np.nan
    _, m, _, _ = soft_drop_decluster(hardest, z_cut, beta, R0=R0)
    return m


def jet_softdrop_full(X_row, j_eta, j_phi, z_cut, beta, R0=config.R0):
    """Like jet_softdrop_mass but also returns which constituent indices
    (into X_row's first axis) were retained vs dropped -- used by the
    event-display plots."""
    pjs = constituents_to_pseudojets(X_row, j_eta, j_phi)
    hardest = cluster_ca(pjs, R0=R0)
    if hardest is None:
        return np.nan, np.nan, [], []
    pt, m, retained, dropped = soft_drop_decluster(hardest, z_cut, beta, R0=R0)
    return pt, m, retained, dropped


def batch_softdrop_mass(X, kinematics, z_cut, beta, R0=config.R0):
    """X: (n_jets, max_constituents, 17). kinematics: (n_jets, 4) with
    columns [pt, eta, phi, mass] (as stored in every strategy npz's
    'kinematics' array). Returns (n_jets,) array of groomed masses."""
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        out[i] = jet_softdrop_mass(X[i], float(kinematics[i, 1]), float(kinematics[i, 2]), z_cut, beta, R0=R0)
    return out


def plain_jet_mass_from_constituents(X_row, j_eta, j_phi):
    """Sum of all valid constituent 4-vectors -> invariant mass, with no
    grooming at all. Used only as a cross-check against the stored
    macro['jet_mass'] value, to confirm the X constituent set faithfully
    represents the jet fastjet would have found."""
    pt = X_row[:, 1]
    valid = pt > 0
    if not np.any(valid):
        return np.nan
    e = X_row[valid, 0]
    eta = j_eta + X_row[valid, 2]
    phi = j_phi + X_row[valid, 3]
    ptv = X_row[valid, 1]
    px = ptv * np.cos(phi)
    py = ptv * np.sin(phi)
    pz = ptv * np.sinh(eta)
    E = np.sum(e)
    PX = np.sum(px)
    PY = np.sum(py)
    PZ = np.sum(pz)
    m2 = E * E - PX * PX - PY * PY - PZ * PZ
    return float(np.sqrt(max(m2, 0.0)))
