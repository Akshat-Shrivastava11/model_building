// Includes
#include "Pythia8/Pythia.h"
#include "Pythia8/UserHooks.h"
#include "Pythia8/Event.h"

#ifndef HEPMC2
#include "Pythia8Plugins/HepMC3.h"
#else
#include "Pythia8Plugins/HepMC2.h"
#endif

using namespace Pythia8;

//based on Pythia8Plugins/ResonanceDecayFilterHook.h
//accepts decays to any combination of any subset of the specified daughter particles
//also related to https://github.com/cms-sw/cmssw/blob/master/GeneratorInterface/GenFilters/plugins/MCParticleModuloFilter.cc

class AllowResonanceDecays : public UserHooks {
public:
  // Constructor.
  AllowResonanceDecays(Settings &settings);

  // Override base class methods.
  bool canVetoResonanceDecays() override {return true;}
  bool doVetoResonanceDecays(Event& process) override {
    return checkVetoResonanceDecays(process);}
  bool initAfterBeams() override;

  // Class specific.
  bool checkVetoResonanceDecays(const Event& process);
  bool enabled() const {return filter;}
  unsigned long int returnCounter() {return counter;}

private:

  // Data members.
  bool filter, absID;
  unsigned long int counter;
  set<int> mothers, daughters;
  unordered_map<int, int> requestedDaughters, observedDaughters;
};

AllowResonanceDecays::AllowResonanceDecays(Settings &settings) {
  counter = 0;
  settings.addFlag("AllowResonanceDecays:filter", false);
  settings.addFlag("AllowResonanceDecays:absID", false);
  settings.addMVec("AllowResonanceDecays:mothers", vector<int>(), false, false, 0, 0);
  settings.addMVec("AllowResonanceDecays:daughters", vector<int>(), false, false, 0, 0);
}

bool AllowResonanceDecays::initAfterBeams() {
  filter = settingsPtr->flag("AllowResonanceDecays:filter");
  absID = settingsPtr->flag("AllowResonanceDecays:absID");
  auto mothersIn = settingsPtr->mvec("AllowResonanceDecays:mothers");
  mothers.clear();
  mothers.insert(mothersIn.begin(), mothersIn.end());
  auto daughtersIn = settingsPtr->mvec("AllowResonanceDecays:daughters");
  daughters.clear();
  daughters.insert(daughtersIn.begin(), daughtersIn.end());

  return true;
}

bool AllowResonanceDecays::checkVetoResonanceDecays(const Event &process) {
  if (!filter) return false;

  // Count the number of times hook is called.
  counter++;

  // Loop over particles
  bool found_disallowed_daughter = false;
  for (int i = 0; i < process.size(); ++i) {
    const Particle &p = process[i];
    int mid = p.mother1() > 0 ? abs(process[p.mother1()].id()) : 0;

    // If no list of mothers is provided, then all particles
    // in hard process and resonance decays are counted together
    if (mothers.empty() || mothers.count(mid) || mothers.count(-mid)) {
      int pid = p.id();
      if (absID) pid = abs(pid);
      if (!daughters.count(pid)) {
        found_disallowed_daughter = true;
        break;
      }
    }
  }

  return found_disallowed_daughter;
}

//this is a straight copy-paste of examples/main42.cc
//with the above filter/hook added in, following examples/main103.cc

int main(int argc, char* argv[]) {

  // Check that correct number of command-line arguments
  if (argc != 3) {
    cerr << " Unexpected number of command-line arguments. \n You are"
         << " expected to provide one input and one output file name. \n"
         << " Program stopped! " << endl;
    return 1;
  }

  // Check that the provided input name corresponds to an existing file.
  ifstream is(argv[1]);
  if (!is) {
    cerr << " Command-line file " << argv[1] << " was not found. \n"
         << " Program stopped! " << endl;
    return 1;
  }

  // Confirm that external files will be used for input and output.
  cout << "\n >>> PYTHIA settings will be read from file " << argv[1]
       << " <<< \n >>> HepMC events will be written to file "
       << argv[2] << " <<< \n" << endl;

  // Interface for conversion from Pythia8::Event to HepMC event.
  // Specify file where HepMC events will be stored.
  Pythia8ToHepMC toHepMC(argv[2]);

  // Generator.
  Pythia pythia;

  // hook
  auto myUserHooks = make_shared<AllowResonanceDecays>(pythia.settings);
  pythia.setUserHooksPtr(myUserHooks);

  // Read in commands from external file.
  pythia.readFile(argv[1]);

  // Extract settings to be used in the main program.
  int    nEvent    = pythia.mode("Main:numberOfEvents");
  int    nAbort    = pythia.mode("Main:timesAllowErrors");

  // Initialization.
  pythia.init();

  // Begin event loop.
  int iAbort = 0;
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    // Generate event.
    if (!pythia.next()) {

      // If failure because reached end of file then exit event loop.
      if (pythia.info.atEndOfFile()) {
        cout << " Aborted since reached end of Les Houches Event File\n";
        break;
      }

      // First few failures write off as "acceptable" errors, then quit.
      if (++iAbort < nAbort) continue;
      cout << " Event generation aborted prematurely, owing to error!\n";
      break;
    }

    // Construct new empty HepMC event, fill it and write it out.
    toHepMC.writeNextEvent( pythia );

  // End of event loop. Statistics.
  }
  pythia.stat();

  if (myUserHooks->enabled()) {
    double filterEfficiency = (double) pythia.info.getCounter(4) / (double) myUserHooks->returnCounter();
    cout << "AllowResonanceDecays efficiency = " << filterEfficiency << "\n";
  }

  // Done.
  return 0;
}
