import awkward as ak
print(ak.__version__)
import numpy as np
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory
from common import load_events
#import ROOT
import os
import fastjet
print(fastjet.__version__)

#from fastjet import contrib


'''
To run this script activate 
creates env python3 -m venv fastjetenv
pip install awkward==1.10.3 fastjet==3.4.0.1 coffea uproot
**** activate env source fastjet_env/bin/activate

Also used this https://github.com/delphes/delphes/blob/master/external/fastjet/contribs/Nsubjettiness/Nsubjettiness.cc
From my understanding  N-subjettiness is a way to measure how "jet-like" a jet is by checking how well it can be divided into smaller subjets
N-subjettiness is a number that tells you:
If the jet can be split into N subjets (like splitting a jet into 2, 3, or more smaller parts).
Lower values = a simpler jet, easier to split.
Higher values = a more complicated jet, harder to split.
tau_1 measures how the jet can be split into 1 part (i.e., how well the jet is a single, compact structure).
tau_2 measures how the jet can be split into 2 parts (i.e., how well the jet can be divided into two subjets).
tau_3 measures how the jet can be split into 3 parts, and so on.
Key points:
Smaller tau values mean that the jet is easier to split into multiple subjets (so it’s a "cleaner" jet).
Larger tau values mean that the jet is more complicated and harder to split into subjets (so it’s a "messier" jet).


1. T21 (Tau_2 / Tau_1)
T21 is the ratio of tau_2 to tau_1.
It compares how well a jet can be split into 2 subjets versus how well it can be split into 1 subjet.
Interpretation:

Small T21: A small value means that the jet can easily be split into 2 subjets, but it's still fairly compact (not too messy). It’s like a clean, two-part structure.
Large T21: A large value suggests that the jet is more complex, and it's harder to split into two subjets compared to one. It’s a more messy or complicated jet.
'''

filename = "/uscms/home/ashrivas/nobackup/Dark_Sector/SVJ/model_building/models/s-channel_mmed-1000_Nc-3_Nf-3_scale-50_mq-50.595_mpi-30_mrho-125.499_pvector-0.5_spectrum-snowmass_rinv-0.333333/events.root"
Events = load_events(filename, with_constituents=True)
events = Events

def get_momentum_array(particles):
    # Extract px, py, pz, E from each particle in the ParticleArray
    px = particles.PT * np.cos(particles.Phi)
    py = particles.PT * np.sin(particles.Phi)
    pz = particles.PT * np.sinh(particles.Eta)
    E = particles.E
    return ak.zip({"px": px, "py": py, "pz": pz, "E": E}, with_name="FourVector")

def get_subjet_momentum_array(subjet):
    # Extract px, py, pz, E from each particle in the ParticleArray
    px = subjet.px
    py = subjet.py
    pz = subjet.pz
    E = subjet.E  # Assuming energy is stored in 'E'
    
    return ak.zip({"px": px, "py": py, "pz": pz, "E": E}, with_name="FourVector")


def get_subjets(jet,subjet_rad):
    constituents_momentum = get_momentum_array(jet.Constituents)
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm,subjet_rad)
    cluster_sequence_subjets = fastjet.ClusterSequence(constituents_momentum, jetdef)
    subjets = cluster_sequence_subjets.inclusive_jets()

    return subjets

def get_energy_correlators(jet, beta=1, npoint=0, angles=-1):
    constituents_momentum = get_momentum_array(jet.Constituents)
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.2)
    cluster_sequence_subjets = fastjet.ClusterSequence(constituents_momentum, jetdef)
    energy_corr = cluster_sequence_subjets.exclusive_jets_energy_correlator(njets=1, beta=1, npoint=0)
    
    return energy_corr

def get_lund_declustering(jet,subjet_rad,njets=2):
    constituents_momentum = get_momentum_array(jet.Constituents)

    # Check if the number of constituents is less than the requested number of subjets
    if len(constituents_momentum) < njets:
        print(f"Warning: Requested {njets} subjets, but the jet has only {len(constituents_momentum)} constituents.")
        return None, None
    
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, subjet_rad)
    cluster_sequence_subjets = fastjet.ClusterSequence(constituents_momentum, jetdef)
    # Extract the Lund declustering parameters (Delta and k_T)
    lund_params = cluster_sequence_subjets.exclusive_jets_lund_declusterings(njets=njets)# Get Lund declustering parameters for the leading 2 jets
    

    print(f"Type of Jet1_lund: {type(lund_params)}")
    print(f"Jet1_lund layout: {lund_params.layout}")
    print(f"Jet1_lund fields: {lund_params.fields}")
    
    # Extract Delta and kt
    delta_values = lund_params["Delta"].to_list()  # Convert to a Python list for easier handling
    kt_values = lund_params["kt"].to_list()

    return delta_values, kt_values

def get_softdrop_groomed_jets(jet, beta=0, symmetry_cut=0.1, R0=0.8):
    constituents_momentum = get_momentum_array(jet.Constituents)
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.2)
    cluster_sequence_subjets = fastjet.ClusterSequence(constituents_momentum, jetdef)
    # Perform SoftDrop grooming
    groomed_jets = cluster_sequence_subjets.exclusive_jets_softdrop_grooming(njets=1, beta=beta, symmetry_cut=symmetry_cut, R0=R0)
    
    return groomed_jets



# Print or process Lund declustering results

mask = ak.num(Events.FatJet) >= 2
events = Events[mask]

mask2 = ~ak.is_none(events.Event.Number)
events = events[mask2]

Jet1 = events.FatJet[:,0]
Jet2 = events.FatJet[:,1]



#print("Available attributes of the subjet:")                                                                                                                    
#print(dir(Jet1[0])) 

Jet1_subjets = get_subjets(Jet1,0.2)
Jet2_subjets = get_subjets(Jet2,0.2)

#print(f"Number of subjets in Jet1: {ak.num(Jet1_subjets)}")
#print(f"Number of subjets in Jet2: {ak.num(Jet2_subjets)}")

subjet_radius = 0.2

# Process Jet1
delta_jet1, kt_jet1 = get_lund_declustering(Jet1, subjet_radius)
print("Jet1 Lund declustering parameters:")
print("Delta values:", delta_jet1)
print("k_T values:", kt_jet1)

# Process Jet2
delta_jet2, kt_jet2 = get_lund_declustering(Jet2, subjet_radius)
print("\nJet2 Lund declustering parameters:")
print("Delta values:", ak.to_list(delta_jet2))
print("k_T values:", ak.to_list(kt_jet2))

#'NSubJetsPruned', 'NSubJetsSoftDropped', 'NSubJetsTrimmed', 'NeutralEnergyFraction'
#Jet1_NSubJetsPruned = Jet1.NSubJetsPruned
#Jet2_NSubJetsPruned = Jet2.NSubJetsPruned


#Jet1_ecf = get_energy_correlators(Jet1)
#Jet2_ecf = get_energy_correlators(Jet2)

#Jet1_groomed = get_softdrop_groomed_jets(Jet1)
#Jet2_groomed = get_softdrop_groomed_jets(Jet2)

#Jet1_subjets, Jet1_ecf, Jet1_lund, Jet1_groomed = analyze_jet_substructure(Jet1, 0.2)
#Jet2_subjets, Jet2_ecf, Jet2_lund, Jet2_groomed = analyze_jet_substructure(Jet2, 0.2)


#print(Jet1_subjets)
#Jet1_subjets = ak.fill_none(Jet1_subjets, None) 
#Jet1_subjets = [subjet for subjet in Jet1_subjets if len(subjet) > 0]
#for i, subjet in enumerate(Jet1_subjets):
#    print(f"Subjet {i}: {subjet}")


Jet1_subjets_count = ak.num(Jet1_subjets)
Jet2_subjets_count = ak.num(Jet2_subjets)

print(f"Type of Jet1_lund: {type(Jet1_lund)}")
print(f"Jet1_lund layout: {Jet1_lund.layout}")
print(f"Jet1_lund fields: {Jet1_lund.fields}")

#print(f"lund in Jet2: {Jet2_lund}")
# Plot number of subjets for Jet1
plt.hist(Jet1_subjets_count, bins=6, color='blue', label='Jet 1', histtype='step')
plt.hist(Jet2_subjets_count, bins=6, color='red', label='Jet 2', histtype='step')
plt.title("Number of Subjets")
plt.xlabel("Number of Subjets")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/Jet_nsubjets.jpg')
plt.clf()




'''
# Plot Energy Correlators (ECF) for Jet1 and Jet2
plt.hist(Jet1_ecf, bins=30, color='blue', label='Jet 1', histtype='step')
plt.hist(Jet2_ecf, bins=30, color='red', label='Jet 2', histtype='step')
plt.title("Energy Correlators")
plt.xlabel("Energy Correlators")
plt.ylabel("Arbitrary units")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('subjet_plots/Jet_ecf.jpg')
plt.clf()


print('done')

#for subjet in Jet1_subjets:
#    print("Available attributes of the subjet:")
#    print(dir(subjet))









def calculate_nsubjettiness(jet, nsubjettiness_n=2):
    # Get subjets for the jet
    subjets = get_subjets(jet, 0.4)  # Use appropriate radius for subjets
    subjets_momentum = get_subjet_momentum_array(subjets)  # Get momentum for subjets

    # Create a JetDefinition for subjets (using the same clustering algorithm)
    jetdef_subjets = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

    # Initialize RestFrameNSubjettinessTagger
    tau2cut = 0.2  # Example cut for Tau2 (you can adjust this based on your needs)
    costhetascut = 0.5  # Example cosine cut for the angle between subjets
    use_exclusive = False  # Set to True if you want exclusive clustering

    # Create RestFrameNSubjettinessTagger
    nsubjettiness = fastjet.RestFrameNSubjettinessTagger(jetdef_subjets, tau2cut, costhetascut, use_exclusive)

    # Calculate Tau values (e.g., Tau1, Tau2)
    tau_values = nsubjettiness.result(subjets)

    return tau_values

# Calculate Nsubjettiness for Jet1 and Jet2
Jet1_tau = calculate_nsubjettiness(Jet1, 2)  # Calculate Tau1 and Tau2 for Jet1
Jet2_tau = calculate_nsubjettiness(Jet2, 2)  # Calculate Tau1 and Tau2 for Jet2

# Print Tau1 and Tau2 values
print(f"Jet1 Tau1: {Jet1_tau[0]}, Tau2: {Jet1_tau[1]}")
print(f"Jet2 Tau1: {Jet2_tau[0]}, Tau2: {Jet2_tau[1]}")

for subjet in Jet1_subjets:
    # Extract momentum components
    px = subjet.px
    py = subjet.py
    pz = subjet.pz
    E = subjet.E  # Energy

    # Calculate pt, eta, phi
    pt = np.sqrt(px**2 + py**2)
    eta = 0.5 * np.log((E + pz) / (E - pz))  # Alternatively, eta = np.arctanh(pz / E)
    phi = np.arctan2(py, px)

    # Print the properties
    print(f"Subjet:")
    print(f"  pt: {pt}")
    print(f"  eta: {eta}")
    print(f"  phi: {phi}")
    print(f"  mass: {np.sqrt(E**2 - pt**2 - pz**2)}")  # Mass from 4-momentum
    print(f"  energy: {E}")
    print(f"  px: {px}")
    print(f"  py: {py}")
    print(f"  pz: {pz}")
    print("-" * 40)
'''
