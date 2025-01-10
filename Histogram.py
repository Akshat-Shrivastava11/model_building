import awkward as ak
import pickle
import numpy as np
import hist
import matplotlib as mpl
from coffea.nanoevents import NanoEventsFactory
from common import load_events

def ET(vec):
    return np.sqrt(vec.px**2+vec.py**2+vec.mass**2)

def get_values(var,Events):
    return ak.flatten(Events[var],axis=None)

def fill_hist(var,nbins,bmin,bmax,label,Events):
    h = (
        hist.Hist.new
        .Reg(nbins, bmin, bmax, label=label)
        .Double()
    )
    h.fill(get_values(var,Events))
    return h

def normalize_angle(angle):
    angle = np.mod(angle, 2 * np.pi)
    angle = np.where(angle >= np.pi, angle - 2 * np.pi, angle)
    return angle

def deltaR(jet):
    deta_particle = np.abs(jet.eta-jet.Constituents.eta)
    dphi_particle = np.abs(normalize_angle(jet.phi-jet.Constituents.phi))
    dR = np.sqrt(deta_particle**2+dphi_particle)
    return dR


def calculate_girth(jet):
    particle_dR = deltaR(jet)
    girth_val = ak.sum(jet.Constituents.pt * particle_dR, axis=-1)
    # Normalize by jet pt                                                                                                                                                                  
    girth_val_normalized = girth_val / jet.pt
    return girth_val_normalized

def calculate_girth(jet):
    girth = []
    particle_dR = deltaR(jet)
    girth_val =  ak.sum(jet.Constituents.pt * particle_dR, axis=-1)
    #girth_val /= jet.pt #normalization wrt jet                                                                                                                                            
    girth.append(girth_val)

    np.divide(girth,jet.pt)
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

def histogram(filename, helper):
    Events = load_events(filename, with_constituents=True)

    # require two jets
    events = Events
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

    events["Jet1_NSubJetsPruned"] = events["Jet1"].NSubJetsPruned
    evetns["Jet2_NSubJetsPruned"] = events["Jet2"].NSubJetsPruned

    events["Jet1_NSubJetsSoftDropped"] = events["Jet1"].NSubJetsSoftDropped
    evetns["Jet2_NSubJetsSoftDropped"] = events["Jet2"].NSubJetsSoftDropped

    events["Jet1_NSubJetsTrimmed"] = events["Jet1"].NSubJetsTrimmed
    evetns["Jet2_NSubJetsTrimmed"] = events["Jet2"].NSubJetsTrimmed

    events["Jet1_NeutralEnergyFraction"] = events["Jet1"].NeutralEnergyFraction
    evetns["Jet2_NeutralEnergyFraction"] = events["Jet2"].NeutralEnergyFraction
    
    events["Jet1_girth"] = calculate_girth(events["Jet1"])
    events["Jet2_girth"]= calculate_girth(events["Jet2"])

    events["Jet1_ptD"]=calculate_ptD(events["Jet1"])
    events["Jet2_ptD"]=calculate_ptD(events["Jet2"])

    
    events["Jet1_majoraxis"], events["Jet1_minoraxis"] = calc_axis1_axis2(events["Jet1"])
    events["Jet2_majoraxis"], events["Jet2_minoraxis"] = calc_axis1_axis2(events["Jet2"])


    # Stable inv frac
    dark_hadron_ids = helper.darkHadronFinalIDs
    stable_particle_ids = helper.stableIDs
    pid = events.GenParticle["PID"]
    d1 = events.GenParticle["D1"]

    # Boolean array of whether a particle is dark
    is_dark = ak.zeros_like(pid)
    for dhid in dark_hadron_ids:
        is_dark = is_dark | (np.abs(pid)==dhid)
    is_dark = is_dark==1

    # PIDs of dark daughter
    dark_daughter = pid[d1[is_dark]]
    is_dark_daughter = ak.zeros_like(dark_daughter) | (d1[is_dark]==-1)

    for dsid in stable_particle_ids:
        is_dark_daughter = is_dark_daughter | (np.abs(dark_daughter)==dsid)

    numer = ak.sum(is_dark_daughter, axis=1).to_numpy().astype(float)
    denom = ak.sum(is_dark, axis=1).to_numpy().astype(float)
    stability = np.divide(numer, denom, out=np.zeros_like(numer), where=denom>0)
    print(f"Average computed rinv value = {np.mean(stability)}")
    
    # Add the invisible fraction to the events
    events["stable_invisible_fraction"] = stability
    
    # store modified events array
    Events = events
    
    
    # Histogram

    # Creating hist objects

    hist_dict = {
        "Jet_mt": fill_hist("MT",25,0,1500,r"$m_{\text{T}}$ [GeV]",Events),
        "Dijet_pt": fill_hist("Dijet_pt",50,0,1000,r"$p_{\text{T}}(JJ)$ [GeV]",Events),
        "Jet1_pt": fill_hist("Jet1_pt",50,0,1000,r"$p_{\text{T}}(J_1)$ [GeV]",Events),
        "Jet2_pt": fill_hist("Jet2_pt",50,0,1000,r"$p_{\text{T}}(J_2)$ [GeV]",Events),
        "MET": fill_hist("MET",50,0,1000,r"$p_{\text{T}}^{\text{miss}}$ [GeV]",Events),
        "Dijet_eta": fill_hist("Dijet_eta",50,-10,10,r"$\eta_{JJ}$ [GeV]",Events),
        "Jet1_eta": fill_hist("Jet1_eta",50,-6,6,r"$\eta_{J_1}$",Events),
        "Jet2_eta": fill_hist("Jet2_eta",50,-6,6,r"$\eta_{J_2}$",Events),
        "Dijet_phi": fill_hist("Dijet_phi",25,-4,4,r"$\phi_{JJ}$",Events),
        "Jet1_phi": fill_hist("Jet1_phi",25,-4,4,r"$\phi_{J_1}$",Events),
        "Jet2_phi": fill_hist("Jet2_phi",25,-4,4,r"$\phi_{J_2}$",Events),
        "Dijet_mass": fill_hist("Dijet_mass",50,0,2300,r"$m_{JJ}$ [GeV]",Events),
        "Jet1_mass": fill_hist("Jet1_mass",50,0,250,r"$m_{J_1}$ [GeV]",Events),
        "Jet2_mass": fill_hist("Jet2_mass",50,0,250,r"$m_{J_2}$ [GeV]",Events),
        "DeltaEta": fill_hist("DeltaEta",35,0,8.0,r"$\Delta\eta(JJ)$",Events),
        "DeltaPhi": fill_hist("DeltaPhi",20,0,3.5,r"$\Delta\phi(JJ)$",Events),
        "DeltaPhi_MET_Jet1": fill_hist("DeltaPhi_MET_Jet1",25,0,3.5,r"$\Delta\phi(J_1,p_{\text{T}}^{\text{miss}})$",Events),
        "DeltaPhi_MET_Jet2": fill_hist("DeltaPhi_MET_Jet2",25,0,3.5,r"$\Delta\phi(J_2,p_{\text{T}}^{\text{miss}})$",Events),
        "Jet1_NSubJetsPruned": fill_hist("Jet1_NSubJetsPruned", 20, 0, 20, r"$N_{\text{subjets, pruned}}(J_1)$", Events),
        "Jet2_NSubJetsPruned": fill_hist("Jet2_NSubJetsPruned", 20, 0, 20, r"$N_{\text{subjets, pruned}}(J_2)$", Events),
        "Jet1_NSubJetsSoftDropped": fill_hist("Jet1_NSubJetsSoftDropped", 20, 0, 20, r"$N_{\text{subjets, soft-dropped}}(J_1)$", Events),
        "Jet2_NSubJetsSoftDropped": fill_hist("Jet2_NSubJetsSoftDropped", 20, 0, 20, r"$N_{\text{subjets, soft-dropped}}(J_2)$", Events),
        "Jet1_NSubJetsTrimmed": fill_hist("Jet1_NSubJetsTrimmed", 20, 0, 20, r"$N_{\text{subjets, trimmed}}(J_1)$", Events),
        "Jet2_NSubJetsTrimmed": fill_hist("Jet2_NSubJetsTrimmed", 20, 0, 20, r"$N_{\text{subjets, trimmed}}(J_2)$", Events),
        "Jet1_NeutralEnergyFraction": fill_hist("Jet1_NeutralEnergyFraction", 50, 0, 1, r"Neutral Energy Fraction $(J_1)$", Events),
        "Jet1_girth": fill_hist("Jet1_girth",50,0,1,r"DeltaR",Events),
        "Jet2_girth": fill_hist("Jet2_girth",50,0,1,r"DeltaR",Events),
        "Jet1_ptD": fill_hist("Jet1_ptD",50,0,1.01,r"$p_{\text{tD}}(J_1)$ [GeV]",Events),
        "Jet2_ptD": fill_hist("Jet2_ptD",50,0,1.01,r"$p_{\text{tD}}(J_2)$ [GeV]",Events),
        "Jet1_major": fill_hist("Jet1_majoraxis",50,0,1,r"Major Axis",Events),
        "Jet1_minor": fill_hist("Jet1_minoraxis",50,0,1,r"Minor Axis",Events),
        "Jet2_major": fill_hist("Jet2_majoraxis",50,0,1,r"Major Axis",Events),
        "Jet2_minor": fill_hist("Jet2_minoraxis",50,0,1,r"Minor Axis",Events),
        "stable_invisible_fraction": fill_hist("stable_invisible_fraction",25,0,1,r"$\overline{r}_{\text{inv}}$",Events)
    }
    

    # Saving the histograms
    with open("Hists.pkl", "wb") as out:
        pickle.dump(hist_dict, out)




