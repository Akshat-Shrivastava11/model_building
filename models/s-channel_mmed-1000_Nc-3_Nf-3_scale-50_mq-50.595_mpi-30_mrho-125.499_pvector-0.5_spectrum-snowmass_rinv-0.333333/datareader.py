
import ROOT


file = ROOT.TFile("events.root", "READ")
events = file.Get("Delphes")
branches = events.GetListOfBranches()
nevents = events.GetEntries()

#for branch in branches:
#    print(branch.GetName())

jet = events.GetBranch("Jet")
pfcandidate = events.GetBranch("ParticleFlowCandidate")
MissingET = events.GetBranch("MissingET")
GenJet = events.GetBranch("GenJet")
Event = events.GetBranch("Event")
FatJet = events.GetBranch("FatJet")
nevents = events.GetEntries()
jet = events.GetBranch("Jet")
jet_size = events.GetBranch("Jet_size")

for i in range(nevents):
    events.GetEntry(i) 
    
    jet_count = jet_size.GetLeaf("Jet_size").GetValue(0)  # Get the size for the current event   
    print(f"Event {i}: Number of jets = {jet_count}")


