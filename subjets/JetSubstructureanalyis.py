import awkward as ak
import numpy as np
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory
from common import load_events
import ROOT
import os

filename = "/uscms/home/ashrivas/nobackup/Dark_Sector/SVJ/model_building/models/s-channel_mmed-1000_Nc-3_Nf-3_scale-50_mq-50.595_mpi-30_mrho-125.499_pvector-0.5_spectrum-snowmass_rinv-0.333333/events.root"
Events = load_events(filename, with_constituents=True)
events = Events

print(events)
print(type(events))
print(Events.Event.Number)

mask = ak.num(Events.FatJet) >= 2
events = Events[mask]

mask2 = ~ak.is_none(events.Event.Number)
events = events[mask2]

#cleaned_event_no = [item for item in events.Event.Number if item is not None]
print("after both masks",events.Event.Number)


Jet1 = events.FatJet[:,0]
Jet2 = events.FatJet[:,1]


PFcandidates = events.ParticleFlowCandidate
events["Jet1"] = events.FatJet[:,0]
events["Jet2"] = events.FatJet[:,1]



eventnumber = events.Event.Number
print("event no:",eventnumber)
print("Jet1.Particles PT:", Jet1.Constituents.PT)

'''
calculate 
girth 
ptD transverse momentum dispersion 
axis major 
axis minor
'''
'''
girth measurement for jet = sum(particle_pt*particle_dR)
'''


def calculate_ptD(jet):
    sum_pt = ak.sum(jet.Constituents.pt,axis=-1)
    sum_pt2 = ak.sum(jet.Constituents.pt ** 2,axis=-1)
    ptD = np.sqrt(sum_pt2) / sum_pt

    # Return the result (ptD)
    return ptD

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
    girth_val =  ak.sum(jet.Constituents.pt * particle_dR, axis=-1)
    girth_val = ak.to_numpy(girth_val, allow_missing=True)
    girth = np.divide(girth_val,jet.pt)
    #girth = np.array(girth)
    return girth

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

axis1_jet1, axis2_jet1 = calc_axis1_axis2(Jet1)
axis1_jet2, axis2_jet2 = calc_axis1_axis2(Jet2)

Jet1_ptD = calculate_ptD(Jet1)
Jet2_ptD = calculate_ptD(Jet2)


girth_jet1 = calculate_girth(Jet1)
girth_jet2 = calculate_girth(Jet2)

print("Jet1 pT:",Jet1.pt)
print("Jet1 particles pT",Jet1.Constituents.pt)
print("Jet1ptd:",Jet1_ptD)
print("Jet particles dR", deltaR(Jet1))
print("Jet1 girth :",girth_jet1)
print("Jet1 pt:",Jet1.pt)
print(f"Axis1 jet1 (major) for each event: {axis1_jet1}")
print(f"Axis2 jet1 (minor) for each event: {axis2_jet1}")


#Jet1_NSubJetsPruned =Jet1.NSubJetsPruned'NSubJetsPruned', 'NSubJetsSoftDropped', 'NSubJetsTrimmed', 'NeutralEnergyFraction'

if not os.path.exists('subjet_plots'):
    os.makedirs('subjet_plots')


plt.figure(figsize=(6, 6))


plt.hist(girth_jet1, bins=20, color=['blue'], label=['Jet1 Girth'],histtype='step')
plt.title("Girths Jet1 ")
plt.xlabel("Girth ")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/jet_girth_histogram.jpg')

plt.hist(girth_jet2, bins=20, color=['red'], label=['Jet2 Girth'],histtype='step')
plt.title("Girths Jet2")
plt.xlabel("Girth")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/jet_girth_histogram.jpg')
plt.clf() 
print("Girth histogram saved as subjet_plots/jet_girth_histogram.pdf")

#################################################################################################
plt.hist(Jet1_ptD, bins=20, color=['blue'], label=['Jet1 ptD'],histtype='step')
plt.title("pT Distribution")
plt.xlabel("ptD")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/jet_ptD_histogram.jpg')

plt.hist(Jet2_ptD, bins=20, color=['red'], label=['Jet2 ptD'],histtype='step')
plt.title("pT Distribution")
plt.xlabel("ptD")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/jet_ptD_histogram.jpg')
plt.clf() 

#################################################################################################
plt.hist(axis1_jet1, bins=20, color=['blue'], label=['Jet 1'],histtype='step')
plt.title("Major Axis")
plt.xlabel("Axis 1")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/major_axis_histogram.jpg')

plt.hist(axis1_jet2, bins=20, color=['red'], label=['Jet 2'],histtype='step')
plt.title("Major Axis")
plt.xlabel("Axis 1")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/major_axis_histogram.jpg')
plt.clf()  


###################################################################################################
plt.hist(axis2_jet1, bins=20, color=['blue'], label=['Jet 1'],histtype='step')
plt.title("Minor Axis")
plt.xlabel("Axis 2")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/minor_axis_histogram.jpg')

plt.hist(axis2_jet2, bins=20, color=['red'], label=['Jet 2'],histtype='step')
plt.title("Minor Axis")
plt.xlabel("Axis 2")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/minor_axis_histogram.jpg')
plt.clf()  

###################################################################################################                                                                                                                 
print('done')
