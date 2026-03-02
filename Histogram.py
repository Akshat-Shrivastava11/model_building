import awkward as ak
import pickle
import numpy as np
import hist
import matplotlib as mpl
from coffea.nanoevents import NanoEventsFactory
from common import load_events

def ET(vec):
    return np.sqrt(vec.px**2+vec.py**2+vec.mass**2)

def normalize_angle(angle):
    angle = np.mod(angle, 2 * np.pi)
    angle = np.where(angle >= np.pi, angle - 2 * np.pi, angle)
    return angle

def deltaR(jet):
    deta_particle = np.abs(jet.eta-jet.Constituents.eta)
    dphi_particle = np.abs(normalize_angle(jet.phi-jet.Constituents.phi))
    dR = np.sqrt(deta_particle**2+dphi_particle**2)
    return dR

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
    dphi_particle = np.abs(normalize_angle(jet.phi-jet.Constituents.phi))

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
    for j in ['Jet1', 'Jet2']:
        # we have tau1 to tau5
        for i in range(1,6):
            events[j+'_tau{:d}'.format(i)] = events[j].Tau_5[:,i-1]
            if i != 1: events[j+'_tau{:d}{:d}'.format(i, i-1)] = events[j+'_tau{:d}'.format(i)] / events[j+'_tau{:d}'.format(i-1)]
    return events

def calc_rinv(events, helper, debug):
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
        stability = np.where(denom>0, numer/denom, 0)
    dprint('stability',stability.tolist())
    print(f"Average computed rinv value = {np.mean(stability):.5} ({np.std(stability):.5})")

    return stability

def histogram(filename, helper, with_constituents=True, debug=False):
    events = load_events(filename, with_constituents=with_constituents)

    # require two jets
    mask = ak.num(events.FatJet)>=2
    events = events[mask]

    #get rid of None Events
    mask2 = ~ak.is_none(events.Event.Number)
    events = events[mask2]

    # Dijet
    events["Dijet"] = events.FatJet[:,0]+events.FatJet[:,1]

    # transverse mass calculation
    E1 = ET(events.Dijet)
    E2 = events.MissingET.MET
    MTsq = (E1+E2)**2-(events.Dijet.px+events.MissingET.px)**2-(events.Dijet.py+events.MissingET.py)**2
    MTsq = MTsq.to_numpy(allow_missing=True)
    events["MT"] = np.sqrt(MTsq, where=MTsq>=0)

    # 4-vectors for dijet
    events["Dijet_pt"] = events.Dijet.pt
    events["Dijet_eta"] = events.Dijet.eta
    events["Dijet_phi"] = events.Dijet.phi
    events["Dijet_mass"] = events.Dijet.mass

    events["MET"] = events.MissingET.MET

    ## For plotting individually for jet1 and jet2

    events["Jet1"] = events.FatJet[:,0]
    events["Jet2"] = events.FatJet[:,1]

    # 4-vectors for jet1 and jet2
    events["Jet1_pt"] = events["Jet1"].pt
    events["Jet2_pt"] = events["Jet2"].pt

    events["Jet1_eta"] = events["Jet1"].eta
    events["Jet2_eta"] = events["Jet2"].eta

    events["Jet1_phi"] = events["Jet1"].phi
    events["Jet2_phi"] = events["Jet2"].phi

    events["Jet1_mass"] = events["Jet1"].mass
    events["Jet2_mass"] = events["Jet2"].mass

    events["DeltaEta"] = np.abs(events["Jet2_eta"] - events["Jet1_eta"])
    events["DeltaPhi"] = np.abs(normalize_angle(events["Jet1_phi"] - events["Jet2_phi"]))

    events["DeltaPhi_MET_Jet1"] = np.abs(normalize_angle(events.MissingET.phi - events["Jet1_phi"]))
    events["DeltaPhi_MET_Jet2"] = np.abs(normalize_angle(events.MissingET.phi - events["Jet2_phi"]))

    # add substructure quantities
    if with_constituents:
        events["Jet1_girth"] = calculate_girth(events["Jet1"])
        events["Jet2_girth"] = calculate_girth(events["Jet2"])
        events["Jet1_ptD"] = calculate_ptD(events["Jet1"])
        events["Jet2_ptD"] = calculate_ptD(events["Jet2"])
        events["Jet1_majoraxis"], events["Jet1_minoraxis"] = calc_axis1_axis2(events["Jet1"])
        events["Jet2_majoraxis"], events["Jet2_minoraxis"] = calc_axis1_axis2(events["Jet2"])
    events["Jet1_sdmass"] = events["Jet1"].SoftDroppedJet.mass
    events["Jet2_sdmass"] = events["Jet2"].SoftDroppedJet.mass
    events["Jet1_sdpt"] = events["Jet1"].SoftDroppedJet.pt
    events["Jet2_sdpt"] = events["Jet2"].SoftDroppedJet.pt

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
    events["stable_invisible_fraction"] = calc_rinv(events, helper, debug)

    # bind events into filling functions
    def get_values(var):
        return ak.flatten(events[var],axis=None)

    def fill_hist(var,nbins,bmin,bmax,label):
        h = (
            hist.Hist.new
            .Reg(nbins, bmin, bmax, label=label)
            .Double()
        )
        h.fill(get_values(var))
        return h

    # Creating hist objects
    hist_dict = {
        "MT": fill_hist("MT",50,0,mmed*1.5,r"$m_{\text{T}}$ [GeV]"),
        "Dijet_pt": fill_hist("Dijet_pt",50,0,mmed*0.75,r"$p_{\text{T}}(JJ)$ [GeV]"),
        "Dijet_eta": fill_hist("Dijet_eta",50,-10,10,r"$\eta(JJ)$ [GeV]"),
        "Dijet_phi": fill_hist("Dijet_phi",25,-3.15,3.15,r"$\phi(JJ)$"),
        "Dijet_mass": fill_hist("Dijet_mass",50,0,mmed*1.5,r"$m_{JJ}$ [GeV]"),
        "Jet1_pt": fill_hist("Jet1_pt",50,0,mmed*0.75,r"$p_{\text{T}}(J_1)$ [GeV]"),
        "Jet2_pt": fill_hist("Jet2_pt",50,0,mmed*0.75,r"$p_{\text{T}}(J_2)$ [GeV]"),
        "Jet1_eta": fill_hist("Jet1_eta",50,-6,6,r"$\eta(J_1)$"),
        "Jet2_eta": fill_hist("Jet2_eta",50,-6,6,r"$\eta(J_2)$"),
        "Jet1_phi": fill_hist("Jet1_phi",25,-3.15,3.15,r"$\phi(J_1)$"),
        "Jet2_phi": fill_hist("Jet2_phi",25,-3.15,3.15,r"$\phi(J_2)$"),
        "Jet1_mass": fill_hist("Jet1_mass",50,0,250,r"$m_{J_1}$ [GeV]"),
        "Jet2_mass": fill_hist("Jet2_mass",50,0,250,r"$m_{J_2}$ [GeV]"),
        "MET": fill_hist("MET",50,0,mmed*0.75,r"$p_{\text{T}}^{\text{miss}}$ [GeV]"),
        "DeltaEta": fill_hist("DeltaEta",35,0,8.0,r"$\Delta\eta(JJ)$"),
        "DeltaPhi": fill_hist("DeltaPhi",20,0,3.15,r"$\Delta\phi(JJ)$"),
        "DeltaPhi_MET_Jet1": fill_hist("DeltaPhi_MET_Jet1",25,0,3.15,r"$\Delta\phi(J_1,p_{\text{T}}^{\text{miss}})$"),
        "DeltaPhi_MET_Jet2": fill_hist("DeltaPhi_MET_Jet2",25,0,3.15,r"$\Delta\phi(J_2,p_{\text{T}}^{\text{miss}})$"),
    }
    if with_constituents:
        hist_dict.update({
            "Jet1_girth": fill_hist("Jet1_girth",50,0,1,r"$g_{\text{jet}}(J_1)$"),
            "Jet2_girth": fill_hist("Jet2_girth",50,0,1,r"$g_{\text{jet}}(J_2)$"),
            "Jet1_ptD": fill_hist("Jet1_ptD",50,0,1.01,r"$D_{p_{\text{T}}}(J_1)$"),
            "Jet2_ptD": fill_hist("Jet2_ptD",50,0,1.01,r"$D_{p_{\text{T}}}(J_2)$"),
            "Jet1_major": fill_hist("Jet1_majoraxis",50,0,0.5,r"$\sigma_{\text{major}}(J_1)$"),
            "Jet2_major": fill_hist("Jet2_majoraxis",50,0,0.5,r"$\sigma_{\text{major}}(J_2)$"),
            "Jet1_minor": fill_hist("Jet1_minoraxis",50,0,0.5,r"$\sigma_{\text{minor}}(J_1)$"),
            "Jet2_minor": fill_hist("Jet2_minoraxis",50,0,0.5,r"$\sigma_{\text{minor}}(J_2)$"),
        })
    hist_dict.update({
        "Jet1_sdmass": fill_hist("Jet1_sdmass",50,0,250,r"$m_{\text{SD}}(J_1)$ [GeV]"),
        "Jet2_sdmass": fill_hist("Jet2_sdmass",50,0,250,r"$m_{\text{SD}}(J_2)$ [GeV]"),
        "Jet1_sdpt" : fill_hist("Jet1_sdpt",50,0,mmed*0.75,r"$p^{\text{SD}}_{\text{T}}(J_1)$ [GeV]"),
        "Jet2_sdpt" : fill_hist("Jet2_sdpt",50,0,mmed*0.75,r"$p^{\text{SD}}_{\text{T}}(J_2)$ [GeV]"),
        "stable_invisible_fraction": fill_hist("stable_invisible_fraction",25,0,1,r"$\overline{r}_{\text{inv}}$"),
        "mMediator": fill_hist("mMediator",50,0,mmed*1.5,r"$m_{\text{mediator}}$ [GeV]"),
    })

    for t in events.fields:
        if 'tau' not in t: continue
        l = t.split('_')
        label = l[1].replace('tau', '$\\tau_{')+','+l[0].replace('et', '_')+'}$'
        hist_dict[t] = fill_hist(t,40,0,1,label)

    # Saving the histograms
    with open("Hists.pkl", "wb") as out:
        pickle.dump(hist_dict, out)
