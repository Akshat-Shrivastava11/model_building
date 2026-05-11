import awkward as ak
import pickle
import numpy as np
import hist
import matplotlib as mpl
from coffea.nanoevents import NanoEventsFactory
from common import load_events
from collections import defaultdict
from itertools import chain
from scipy.stats import sem

def ET(vec):
    return np.sqrt(vec.px**2+vec.py**2+vec.mass**2)

def deltaR(jet):
    return jet.deltaR(jet.Constituents)

def calculate_girth(jet):
    particle_dR = deltaR(jet)
    girth = ak.sum(jet.Constituents.pt * particle_dR, axis=-1)
    girth = np.divide(girth,jet.pt) #normalize wrt jet pt
    return girth

def calculate_ptD(jet):
    sum_pt = ak.sum(jet.Constituents.pt,axis=-1)
    sum_pt2 = ak.sum(jet.Constituents.pt ** 2,axis=-1)
    ptD = np.sqrt(sum_pt2) / sum_pt

    # Return the result (ptD)
    return ptD

def calc_axis1_axis2(jet):
    jet_constpt = jet.Constituents.pt
    deta_particle = np.abs(jet.eta-jet.Constituents.eta)
    dphi_particle = np.abs(jet.deltaphi(jet.Constituents))

    # Calculate weights (pt^2) for each constituent
    weights_pt = jet_constpt**2

    # Calculate weighted sums for each event
    sum_weight = ak.sum(weights_pt, axis=1)  # Sum of weights (pt^2) for each event

    sum_deta = ak.sum(deta_particle * weights_pt, axis=1)
    sum_dphi = ak.sum(dphi_particle * weights_pt, axis=1)
    sum_deta2 = ak.sum(deta_particle**2 * weights_pt, axis=1)
    sum_dphi2 = ak.sum(dphi_particle**2 * weights_pt, axis=1)
    sum_detadphi = ak.sum(deta_particle * dphi_particle * weights_pt, axis=1)

    # Calculate averages
    ave_deta = sum_deta / sum_weight
    ave_dphi = sum_dphi / sum_weight
    ave_deta2 = sum_deta2 / sum_weight
    ave_dphi2 = sum_dphi2 / sum_weight

    # Calculate covariance matrix components
    a = ave_deta2 - ave_deta**2
    b = ave_dphi2 - ave_dphi**2
    c = -(sum_detadphi / sum_weight - ave_deta * ave_dphi)

    # Calculate the discriminant (delta) for each event
    delta = np.sqrt(np.abs((a - b)**2 + 4 * c**2))

    # Calculate axis1 (major) and axis2 (minor) for each event
    axis1 = np.sqrt(0.5 * (a + b + delta))
    axis2 = np.sqrt(0.5 * (a + b - delta))

    return axis1, axis2

def getTau(events):
    # we have tau1 to tau5
    for i in range(1,6):
        events[f"Jet12_tau{i}"] = events["Jet12"].Tau_5[:,:,i-1]
        if i != 1: events[f"Jet12_tau{i}{i-1}"] = events[f"Jet12_tau{i}"] / events[f"Jet12_tau{i-1}"]
    return events

def calc_rinv(events, helper, meta_dict, debug):
    pid = events.GenParticle["PID"]

    def dprint(*args):
        if debug:
            print(*args)

    # Stable inv frac
    dark_hadron_ids = helper.darkHadronIDs
    dprint('dark_hadron_ids',dark_hadron_ids)
    dark_hadron_final_ids = helper.darkHadronFinalIDs
    dprint('dark_hadron_final_ids',dark_hadron_final_ids)
    stable_particle_ids = helper.stableIDs
    dprint('stable_particle_ids',stable_particle_ids)

    def printer(name, arr):
        dprint(f'{name:<30}',ak.sum(arr, axis=1).to_numpy().tolist())

    # Boolean array of whether a particle is dark
    is_dark = ak.zeros_like(pid)
    for dhid in dark_hadron_ids:
        is_dark = is_dark | (np.abs(pid)==dhid)
    is_dark = is_dark==1
    printer('is_dark',is_dark)

    # Boolean array of whether a particle is dark
    is_dark_final = ak.zeros_like(pid)
    for dhid in dark_hadron_final_ids:
        is_dark_final = is_dark_final | (np.abs(pid)==dhid)
    is_dark_final = is_dark_final==1
    printer('is_dark_final',is_dark_final)

    # exclude dark hadrons resulting from mixed decay of another dark hadron
    m1 = events.GenParticle["M1"]
    m2 = events.GenParticle["M2"]
    d1 = events.GenParticle["D1"]
    d2 = events.GenParticle["D2"]

    m1_dark = (m1!=-1) & (is_dark[m1])
    m1_d1_sm = (d1[m1]!=-1) & (~is_dark[d1[m1]])
    m1_d2_sm = (d2[m1]!=-1) & (~is_dark[d2[m1]])
    m2_dark = (m2!=-1) & (is_dark[m2])
    m2_d1_sm = (d1[m2]!=-1) & (~is_dark[d1[m2]])
    m2_d2_sm = (d2[m2]!=-1) & (~is_dark[d2[m2]])

    def make_table(mask):
        table = ak.zip({
            "index": ak.local_index(pid)[mask],
            "pid": pid[mask],
            "final": is_dark_final[mask],
            "i_m1": m1[mask],
            "m1": pid[m1[mask]],
            "m1_dark": m1_dark[mask],
            "m1_d1": pid[d1[m1[mask]]],
            "m1_d1_sm": m1_d1_sm[mask],
            "m1_d2": pid[d2[m1[mask]]],
            "m1_d2_sm": m1_d2_sm[mask],
            "i_m2": m2[mask],
            "m2": pid[m2[mask]],
            "m2_dark": m2_dark[mask],
            "m2_d1": pid[d1[m2[mask]]],
            "m2_d1_sm": m2_d1_sm[mask],
            "m2_d2": pid[d2[m2[mask]]],
            "m2_d2_sm": m2_d2_sm[mask],
            "i_d1": d1[mask],
            "d1": pid[d1[mask]],
            "i_d2": d2[mask],
            "d2": pid[d2[mask]],
        })
        return table

    # for debugging, show only dark hadron entries
    if debug:
        table_debug = make_table(mask=is_dark)
        import pandas as pd
        with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.width', None, 'display.max_colwidth', None):
            dprint(ak.to_pandas(table_debug))

    m1_dark_d_sm = m1_dark & (m1_d1_sm | m1_d2_sm)
    m2_dark_d_sm = m2_dark & (m2_d1_sm | m2_d2_sm)
    for name,arr in [('m1_dark',m1_dark),
                     ('m1_dark_d1_sm',m1_dark & m1_d1_sm),
                     ('m1_dark_d2_sm',m1_dark & m1_d2_sm),
                     ('m1_dark_d1_d2_sm',m1_dark_d_sm),
                     ('m2_dark',m2_dark),
                     ('m2_dark_d1_sm',m2_dark & m2_d1_sm),
                     ('m2_dark_d2_sm',m2_dark & m2_d2_sm),
                     ('m2_dark_d1_d2_sm',m2_dark_d_sm)]:
        printer(name,arr)
    dark_mother_sm_sibling = (m1_dark_d_sm) | (m2_dark_d_sm)
    dark_mother_sm_sibling = dark_mother_sm_sibling==1
    printer('dark_mother_sm_sibling',dark_mother_sm_sibling)

    # quick diversion here to measure alpha = E_pi / m_rho for 3-body decays
    is_dark_3body = is_dark_final & dark_mother_sm_sibling
    pi_3body = events.GenParticle[is_dark_3body]
    rho_3body = events.GenParticle[m1[is_dark_3body]]
    pi_3body_restframe = pi_3body.boostCM_of_beta3(rho_3body.to_beta3())
    E_pi_3body = pi_3body_restframe.energy
    m_rho_3body = rho_3body.mass
    alpha_3body = E_pi_3body/m_rho_3body
    meta_dict["alpha_3body"] = fill_stats(alpha_3body)
    print(f"Average alpha_3body = {meta_dict['alpha_3body']['mean']:.3} ({meta_dict['alpha_3body']['stdev']:.3})")
    events["alpha_3body"] = alpha_3body

    is_dark_final = is_dark_final & ~dark_mother_sm_sibling
    printer('is_dark_final',is_dark_final)

    # PIDs of dark daughter
    dark_final_daughter = pid[d1[is_dark_final]]
    is_dark_final_daughter = ak.zeros_like(dark_final_daughter) | (d1[is_dark_final]==-1)
    printer('is_dark_final_daughter',is_dark_final_daughter)

    for dsid in stable_particle_ids:
        printer(f'dark_final_daughter=={dsid}', (np.abs(dark_final_daughter)==dsid))
        is_dark_final_daughter = is_dark_final_daughter | (np.abs(dark_final_daughter)==dsid)
    printer('is_dark_final_daughter',is_dark_final_daughter)

    numer = ak.sum(is_dark_final_daughter, axis=1).to_numpy()
    denom = ak.sum(is_dark_final, axis=1).to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        stable_invisible_fraction = np.where(denom>0, numer/denom, 0)
    dprint('stable_invisible_fraction',stable_invisible_fraction.tolist())
    meta_dict["stable_invisible_fraction"] = fill_stats(stable_invisible_fraction)
    print(f"Average computed rinv value (pions) = {meta_dict['stable_invisible_fraction']['mean']:.5} ({meta_dict['stable_invisible_fraction']['stdev']:.5})")

    events["stable_invisible_fraction"] = stable_invisible_fraction

def calc_mt(jet, met):
    # transverse mass calculation
    E1 = ET(jet)
    E2 = met.MET
    MTsq = (E1+E2)**2-(jet.px+met.px)**2-(jet.py+met.py)**2
    MTsq = MTsq.to_numpy(allow_missing=True)
    return np.sqrt(MTsq, where=MTsq>=0)

def proj(events, jet, const):
    return events[jet, const].dot(events[jet]) / events[jet].mass

def jet_const_cumsum(array):
    counts = ak.num(array, axis=-1)
    flat_counts = ak.flatten(counts, axis=None)
    flat_array = ak.flatten(array, axis=None)
    global_cumsum = np.cumsum(flat_array)
    offsets = np.zeros(len(flat_counts)+1, dtype=int)
    offsets[1:] = np.cumsum(flat_counts)
    start_sums = np.zeros_like(global_cumsum)
    prev_tot = np.zeros(len(flat_counts))
    prev_tot[1:] = global_cumsum[offsets[1:-1]-1]
    subtractions = np.repeat(prev_tot, flat_counts)
    # unflatten in two stages
    per_jet_flat = global_cumsum - subtractions
    jets_unflat = ak.unflatten(per_jet_flat, flat_counts)
    return ak.unflatten(jets_unflat, ak.num(counts, axis=1))

def fill_stats(array):
    nparray = ak.to_numpy(ak.drop_none(ak.nan_to_none(ak.flatten(array, axis=None))))
    return {
        "N": len(nparray),
        "mean": np.mean(nparray),
        "stdev": np.std(nparray),
        "stderr": sem(nparray),
    }

def histogram(filename, helper, with_constituents=True, debug=False):
    events = load_events(filename, with_constituents=with_constituents)

    # output dictionary with histograms and metadata
    output = {}
    output["model"] = helper.metadata()
    meta_dict = {}

    # require two jets
    mask = ak.num(events.FatJet)>=2
    events = events[mask]

    #get rid of None Events
    mask2 = ~ak.is_none(events.Event.Number)
    events = events[mask2]

    # Dijet
    events["Dijet"] = events.FatJet[:,0]+events.FatJet[:,1]

    # transverse mass calculation
    events["MT"] = calc_mt(events.Dijet, events.MissingET)

    # 4-vectors for dijet
    events["Dijet_pt"] = events.Dijet.pt
    events["Dijet_eta"] = events.Dijet.eta
    events["Dijet_phi"] = events.Dijet.phi
    events["Dijet_mass"] = events.Dijet.mass

    events["MET"] = events.MissingET.MET

    ## For plotting individually for jet1 and jet2
    events["Jet12"] = ak.pad_none(events.FatJet[:,0:2], target=2, axis=1)

    # 4-vectors for jet1 and jet2
    events["Jet12_pt"] = events["Jet12"].pt
    events["Jet12_eta"] = events["Jet12"].eta
    events["Jet12_phi"] = events["Jet12"].phi
    events["Jet12_mass"] = events["Jet12"].mass

    events["DeltaEta"] = np.abs(events["Jet12_eta"][:,0] - events["Jet12_eta"][:,1])
    events["DeltaPhi"] = np.abs(events["Jet12"][:,0].deltaphi(events["Jet12"][:,1]))

    events["DeltaPhi_MET_Jet12"] = np.abs(events.MissingET.deltaphi(events["Jet12"]))

    # add substructure quantities
    if with_constituents:
        events["Jet12_girth"] = calculate_girth(events["Jet12"])
        events["Jet12_ptD"] = calculate_ptD(events["Jet12"])
        events["Jet12_majoraxis"], events["Jet12_minoraxis"] = calc_axis1_axis2(events["Jet12"])
    events["Jet12_sdmass"] = events["Jet12"].SoftDroppedJet.mass
    events["Jet12_sdpt"] = events["Jet12"].SoftDroppedJet.pt

    events = getTau(events)

    # gen-level info
    pid = events.GenParticle["PID"]

    # mediator (gen-level)
    mmed = helper.mmed
    mediator_id = helper.mediatorID
    is_med = pid==mediator_id
    meds = events.GenParticle[is_med]

    # final version of mediator decays to other particles
    # intermediate versions "decay" to same particle (radiation)
    med_d1 = meds["D1"]
    is_final = pid[med_d1]!=mediator_id
    meds_final = meds[is_final][:,0]
    events["mMediator"] = meds_final.mass

    # Add the invisible fraction to the events
    print(f"Predicted rinv = {output['model'].get('rinv_3body',output['model'].get('rinv',-1)):.5}")
    calc_rinv(events, helper, meta_dict, debug)

    # dark hadron jets and corresponding visible and invisible+visible jets
    events["DHJet12"] = ak.pad_none(events.DarkHadronJet[:,0:2], target=2, axis=1)
    events["DHVJet12"] = ak.pad_none(events.DarkHadronVisibleJet[:,0:2], target=2, axis=1)
    events["DHIVJet12"] = ak.pad_none(events.DarkHadronStableJet[:,0:2], target=2, axis=1)
    dhj_pre = ["DH", "DHV", "DHIV"]
    jet_inds = [0, 1, slice(0, 2)]
    jet_ind_names = ["1","2","1,2"]
    jet_ind_keys = ["1","2","12"]

    if with_constituents:
        # per-jet calculation of invisible fraction based on momentum projection
        # pick out dark hadron constituents (stability already checked in Delphes)
        dark_hadron_final_ids = helper.darkHadronFinalIDs
        is_dark = ak.zeros_like(events["DHIVJet12"].Constituents.PID)
        for dhid in dark_hadron_final_ids:
            is_dark = is_dark | (np.abs(events["DHIVJet12"].Constituents.PID)==dhid)
        is_dark = is_dark==1
        events["DHIVJet12", "DHConstituents"] = events["DHIVJet12", "Constituents"][is_dark]

        def fill_DHIVJet_rinv(numer, denom, suff):
            events[f"DHIVJet12_rinv_{suff}"] = ak.sum(numer, axis=-1)/ak.sum(denom, axis=-1)
            for ind,key in zip(jet_inds, jet_ind_keys):
                meta_dict[f"DHIVJet{key}_rinv_{suff}"] = fill_stats(events[f"DHIVJet12_rinv_{suff}"][:, ind])
            print(f"Average jet-level rinv ({suff}) =", ", ".join(
                [f"{meta_dict[f'DHIVJet{key}_rinv_{suff}']['mean']:.5} ({meta_dict[f'DHIVJet{key}_rinv_{suff}']['stdev']:.5})" for key in jet_ind_keys]
            ))
            # summing jets
            events[f"DiDHIVJet_rinv_{suff}"] = ak.sum(ak.flatten(numer, axis=2), axis=-1)/ak.sum(ak.flatten(denom, axis=2), axis=-1)
            meta_dict[f"DiDHIVJet_rinv_{suff}"] = fill_stats(events[f"DiDHIVJet_rinv_{suff}"])
            print(f"Average dijet-level rinv ({suff}) =",
                f"{meta_dict[f'DiDHIVJet_rinv_{suff}']['mean']:.5} ({meta_dict[f'DiDHIVJet_rinv_{suff}']['stdev']:.5})"
            )
            # global version, summing all events
            rinv_global = ak.sum(ak.flatten(numer, axis=1))/ak.sum(ak.flatten(denom, axis=1))
            meta_dict[f"DHIVJet12_rinv_{suff}_global"] = {"N": 1, "mean": rinv_global, "stdev": 0, "stderr": 0}
            print(f"Global jet-level rinv ({suff}) =",
                f"{meta_dict[f'DHIVJet12_rinv_{suff}_global']['mean']:.5}"
            )

        # project constituent momentum onto jet axis
        proj_numer = proj(events, "DHIVJet12", "DHConstituents")
        proj_denom = proj(events, "DHIVJet12", "Constituents")
        fill_DHIVJet_rinv(proj_numer, proj_denom, 'proj')

        # alternative: use scalar pT sum (related to "jet shape" below)
        shape_numer = events["DHIVJet12"].DHConstituents.pt
        shape_denom = events["DHIVJet12"].Constituents.pt
        fill_DHIVJet_rinv(shape_numer, shape_denom, 'shape')

        # dark jet and visible jet radius
        dr_pcts = [90,95,99]
        for pre in dhj_pre:
            print(f"\n{pre}Jet")

            # also compute nconst and girth
            events[f"{pre}Jet12_girth"] = calculate_girth(events[f"{pre}Jet12"])
            events[f"{pre}Jet12_nconst"] = ak.num(events[f"{pre}Jet12"].Constituents, axis=-1)

            # pt-weighted percentile per jet
            # scalar sum of constituent pT within DeltaR / scalar sum of all constituent pT = "jet shape"
            for pct in dr_pcts:
                none_mask = ak.is_none(events[f"{pre}Jet12"], axis=1)
                const_prop = ak.zip({
                    'dr': deltaR(events[f"{pre}Jet12"]),
                    'pt': events[f"{pre}Jet12"].Constituents.pt,
                })[~none_mask]
                sort_indices = ak.argsort(const_prop.dr, axis=-1)
                sorted_prop = const_prop[sort_indices]
                cumul_pt = jet_const_cumsum(sorted_prop.pt)
                # last element of sum is total from all constituents
                total_pt = ak.fill_none(ak.pad_none(cumul_pt, 1, axis=-1)[:, :, -1], 0)
                target_pt = pct/100 * total_pt
                mask_pt = cumul_pt >= target_pt
                events[f"{pre}Jet12_radius{pct}"] = ak.pad_none(ak.firsts(sorted_prop[mask_pt].dr, axis=-1), target=2, axis=1)
                for ind,key in zip(jet_inds, jet_ind_keys):
                    r_pct_pt = events[f"{pre}Jet12_radius{pct}"][:, ind]
                    meta_dict[f"DHJet{key}_radius{pct}"] = fill_stats(r_pct_pt)
                print(f"{pct}% radius (jet shape):", ", ".join(
                    [f"{meta_dict[f'DHJet{key}_radius{pct}']['mean']:.2} ({meta_dict[f'DHJet{key}_radius{pct}']['stdev']:.2})" for key in jet_ind_keys]
                ))

    # dark hadron jet mass and pt
    events["DHJet12_pt"] = events["DHJet12"].pt
    events["DiDHJet"] = events["DHJet12"][:,0] + events["DHJet12"][:,1]
    events["DiDHJet_mass"] = events["DiDHJet"].mass

    events["DHVJet12_pt"] = events["DHVJet12"].pt
    events["DiDHVJet"] = events["DHVJet12"][:,0] + events["DHVJet12"][:,1]
    events["DiDHVJet_mass"] = events["DiDHVJet"].mass
    events["DiDHVJet_MT"] = calc_mt(events["DiDHVJet"], events.GenMissingET)

    events["DHIVJet12_pt"] = events["DHIVJet12"].pt
    events["DiDHIVJet"] = events["DHIVJet12"][:,0] + events["DHIVJet12"][:,1]
    events["DiDHIVJet_mass"] = events["DiDHIVJet"].mass

    # bind events into filling functions
    def fill_single_hist(var,nbins,bmin,bmax,label,ind=None):
        h = (
            hist.Hist.new
            .Reg(nbins, bmin, bmax, label=label)
            .Double()
        )
        event_var = events[var]
        if ind is not None:
            event_var = events[var][:,ind]
        h.fill(ak.flatten(event_var,axis=None))
        return h

    def fill_hist(var,nbins,bmin,bmax,label):
        split = "JETIND" in label
        if not split:
            # still return a list of pairs for consistency w/ below usage of chain.from_iterable
            return [(var,fill_single_hist(var,nbins,bmin,bmax,label))]
        else:
            results = []
            for ind,key,name in zip(jet_inds, jet_ind_keys, jet_ind_names):
                results.append((var.replace("Jet12",f"Jet{key}"), fill_single_hist(var,nbins,bmin,bmax,label.replace("JETIND",name),ind)))
            return results

    # Creating hist objects
    hist_dict = dict(chain.from_iterable([
        fill_hist("MT",50,0,mmed*1.5,r"$m_{\text{T}}$ [GeV]"),
        fill_hist("Dijet_pt",50,0,mmed*0.75,r"$p_{\text{T}}(JJ)$ [GeV]"),
        fill_hist("Dijet_eta",50,-10,10,r"$\eta(JJ)$ [GeV]"),
        fill_hist("Dijet_phi",25,-3.15,3.15,r"$\phi(JJ)$"),
        fill_hist("Dijet_mass",50,0,mmed*1.5,r"$m_{JJ}$ [GeV]"),
        fill_hist("Jet12_pt",50,0,mmed*0.75,r"$p_{\text{T}}(J_{JETIND})$ [GeV]"),
        fill_hist("Jet12_eta",50,-6,6,r"$\eta(J_{JETIND})$"),
        fill_hist("Jet12_phi",25,-3.15,3.15,r"$\phi(J_{JETIND})$"),
        fill_hist("Jet12_mass",50,0,250,r"$m_{J_{JETIND}}$ [GeV]"),
        fill_hist("MET",50,0,mmed*0.75,r"$p_{\text{T}}^{\text{miss}}$ [GeV]"),
        fill_hist("DeltaEta",35,0,8.0,r"$\Delta\eta(JJ)$"),
        fill_hist("DeltaPhi",20,0,3.15,r"$\Delta\phi(JJ)$"),
        fill_hist("DeltaPhi_MET_Jet12",25,0,3.15,r"$\Delta\phi(J_{JETIND},p_{\text{T}}^{\text{miss}})$"),
    ]))
    if with_constituents:
        hist_dict.update(chain.from_iterable([
            fill_hist("Jet12_girth",50,0,1,r"$g_{\text{jet}}(J_{JETIND})$"),
            fill_hist("Jet12_ptD",50,0,1.01,r"$D_{p_{\text{T}}}(J_{JETIND})$"),
            fill_hist("Jet12_majoraxis",50,0,0.5,r"$\sigma_{\text{major}}(J_{JETIND})$"),
            fill_hist("Jet12_minoraxis",50,0,0.5,r"$\sigma_{\text{minor}}(J_{JETIND})$"),
            fill_hist("DHIVJet12_rinv_proj",25,0,1,r"$r_{\text{inv}}^{\text{kin}}(J_{JETIND}^{\text{stable}})$"),
            fill_hist("DiDHIVJet_rinv_proj",25,0,1,r"$r_{\text{inv}}^{\text{kin}}(J^{\text{stable}}J^{\text{stable}})$"),
            fill_hist("DHIVJet12_rinv_shape",25,0,1,r"$r_{\text{inv}}^{\text{kin(alt)}}(J_{JETIND}^{\text{stable}})$"),
            fill_hist("DiDHIVJet_rinv_shape",25,0,1,r"$r_{\text{inv}}^{\text{kin(alt)}}(J^{\text{stable}}J^{\text{stable}})$"),
        ]))
        dhj_labels = ["DH", "vis", "stable"]
        dhj_nmax = [24.5, 199.5, 199.5]
        dhj_nbin = [25, 50, 50]
        for pre,label,nmax,nbin in zip(dhj_pre, dhj_labels,dhj_nmax,dhj_nbin):
            hist_dict.update(chain.from_iterable([
                fill_hist(f"{pre}Jet12_radius90",50,0,2,r"${\Delta}R_{90}(J_{JETIND}^{\text{"+label+"}})$"),
                fill_hist(f"{pre}Jet12_radius95",50,0,2,r"${\Delta}R_{95}(J_{JETIND}^{\text{"+label+"}})$"),
                fill_hist(f"{pre}Jet12_radius99",50,0,2,r"${\Delta}R_{99}(J_{JETIND}^{\text{"+label+"}})$"),
                fill_hist(f"{pre}Jet12_girth",50,0,1,r"$g_{\text{jet}}(J_{JETIND}^{\text{"+label+"}})$"),
                fill_hist(f"{pre}Jet12_nconst",nbin,-0.5,nmax,r"$n_{\text{const}}(J_{JETIND}^{\text{"+label+"}})$"),
            ]))
    hist_dict.update(chain.from_iterable([
        fill_hist("Jet12_sdmass",50,0,150,r"$m_{\text{SD}}(J_{JETIND})$ [GeV]"),
        fill_hist("Jet12_sdpt",50,0,mmed*0.75,r"$p^{\text{SD}}_{\text{T}}(J_{JETIND})$ [GeV]"),
        fill_hist("stable_invisible_fraction",25,0,1,r"$r_{\text{inv}}^{\text{gen}}$"),
        fill_hist("alpha_3body",50,0,1,r"$\alpha_{\text{3body}}$"),
        fill_hist("mMediator",50,0,mmed*1.5,r"$m_{\text{mediator}}$ [GeV]"),
        fill_hist("DHJet12_pt",50,0,mmed*0.75,r"$p_{\text{T}}(J_{JETIND}^{\text{DH}})$ [GeV]"),
        fill_hist("DHVJet12_pt",50,0,mmed*0.75,r"$p_{\text{T}}(J_{JETIND}^{\text{vis}})$ [GeV]"),
        fill_hist("DHIVJet12_pt",50,0,mmed*0.75,r"$p_{\text{T}}(J_{JETIND}^{\text{stable}})$ [GeV]"),
        fill_hist("DiDHJet_mass",50,0,mmed*1.5,r"$m_{J^{\text{DH}}J^{\text{DH}}}$ [GeV]"),
        fill_hist("DiDHVJet_mass",50,0,mmed*1.5,r"$m_{J^{\text{vis}}J^{\text{vis}}}$ [GeV]"),
        fill_hist("DiDHVJet_MT",50,0,mmed*1.5,r"$m_{\text{T}}^{J^{\text{vis}}J^{\text{vis}}}$ [GeV]"),
        fill_hist("DiDHIVJet_mass",50,0,mmed*1.5,r"$m_{J^{\text{stable}}J^{\text{stable}}}$ [GeV]"),
    ]))

    for t in events.fields:
        if 'tau' not in t: continue
        l = t.split('_')
        label = l[1].replace('tau', '$\\tau_{')+','+l[0].replace('et12', '_{JETIND}')+'}$'
        hist_dict.update(chain.from_iterable([
            fill_hist(t,40,0,1,label)
        ]))

    # finish output dictionary
    output["hist"] = hist_dict
    output["analysis"] = meta_dict

    # alternative 3body rinv calculation using alpha measured from Pythia
    if helper.mrho < 2*helper.mpi:
        from svjHelper import fcdc_rinv_3body, fcdc_rinv_3body_simp
        if helper.Ns is not None:
            output["model"]['rinv_3body_gen'] = fcdc_rinv_3body(Nf=helper.Nf, Ns=helper.Ns, mrho=helper.mrho, mpi=helper.mpi, pvector=helper.pvector, alpha=meta_dict['alpha_3body']['mean'])
        else:
            output["model"]['rinv_3body_gen'] = fcdc_rinv_3body_simp(rinv=helper.rinv, Nf=helper.Nf, mrho=helper.mrho, mpi=helper.mpi, pvector=helper.pvector, alpha=meta_dict['alpha_3body']['mean'])

    # Saving the histograms
    with open("Hists.pkl", "wb") as out:
        pickle.dump(output, out)
