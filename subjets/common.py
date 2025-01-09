from coffea.nanoevents import DelphesSchema
import numpy as np
import numba as nb
from numpy.typing import NDArray
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.delphes import behavior, _set_repr_name, Particle

DelphesSchema.mixins.update({
    "ParticleFlowCandidate": "Particle",
    "FatJet": "Jet",
    "GenFatJet": "Jet",
    "DarkHadronJet": "Jet",
})

# workaround for https://cp3.irmp.ucl.ac.be/projects/delphes/ticket/1170
# manually fix mass units

@ak.mixin_class(behavior)
class GenParticle(Particle):
    @property
    def mass(self):
        return self["Mass"]*0.001

_set_repr_name("GenParticle")

# propagate usage to schema, only for generator particles
DelphesSchema.mixins.update({
    "GenParticle": "GenParticle",
    "GenCandidate": "GenParticle",
    "DarkHadronCandidate": "GenParticle",
})

class DelphesSchema2(DelphesSchema):
    jet_const_pairs = {
        "FatJet" : "ParticleFlowCandidate",
        "Jet" : "ParticleFlowCandidate",
        "DarkHadronJet" : "DarkHadronCandidate",
        "GenFatJet" : "GenCandidate",
        "GenJet" : "GenCandidate",
    }

    # avoid weird error when adding constituents
    def __init__(self, base_form):
        for key in list(base_form["contents"].keys()):
            if "fBits" in key:
                base_form["contents"].pop(key, None)
        super().__init__(base_form)

# ignore unnecessary warning
from numba.core.errors import NumbaTypeSafetyWarning
import warnings
warnings.simplefilter('ignore',category=NumbaTypeSafetyWarning)
# optimized kernel for jet:constituent matching within an event
@nb.njit("i8[:](i8[:],u4[:])")
def get_constituents_kernel(jet_refs: NDArray[np.int64], cand_ids: NDArray[np.uint32]) -> NDArray[np.int64]:
    # get hash table mapping global index : global unique ID
    hash_table = {k:v for v,k in enumerate(cand_ids)}
    # apply hash map
    output = [hash_table[ref] for ref in jet_refs]
    return np.asarray(output)

# apply kernel to events
def get_constituents(events, jetsname, candsname):
    output = []
    for jets,cands in zip(events[jetsname], events[candsname]):
        indices = get_constituents_kernel(ak.flatten(jets.Constituents.refs).to_numpy(), cands.fUniqueID.to_numpy()) if jets is not None else []
        if len(indices)==0:
            unflattened = None
        else:
            unflattened = ak.unflatten(cands[indices],ak.count(jets.Constituents.refs,axis=1),behavior=cands.behavior)[None]
        output.append(unflattened)
    # very important for performance to call ak.concatenate only once at the end
    return ak.with_name(ak.concatenate(output), DelphesSchema2.mixins[candsname])

def init_constituents(events):
    for jet,const in DelphesSchema2.jet_const_pairs.items():
        events[jet,"ConstituentsOrig"] = events[jet,"Constituents"]
        events[jet,"Constituents"] = get_constituents(events,jet,const)
    return events

# helper to test that all jet constituents were found
def sum_4vec(vec):
    summed_vec = {
        "t": ak.sum(vec.energy,axis=-1),
        "x": ak.sum(vec.px,axis=-1),
        "y": ak.sum(vec.py,axis=-1),
        "z": ak.sum(vec.pz,axis=-1),
    }
    return ak.zip(summed_vec,with_name="LorentzVector")

def test_constituents(events):
    for jet in DelphesSchema2.jet_const_pairs:
        check_jets = sum_4vec(events[jet,"Constituents"])
        print(jet, check_jets.mass-events[jet].mass)

def load_sample(sample,helper=None,schema=DelphesSchema,with_constituents=False):
    from coffea.nanoevents import NanoEventsFactory
    path = f'models/{sample["model"]}'
    if helper is None:
        from svjHelper import svjHelper
        sample["helper"] = svjHelper.build(f'{path}/config.py')
    else:
        sample["helper"] = helper
    metadict = sample["helper"].metadata()
    metadict["dataset"] = sample["name"]
    sample["events"] = load_events(f'{path}/events.root',schema=schema,metadict=metadict,with_constituents=with_constituents)

def load_events(filename,schema=DelphesSchema,metadict=None,with_constituents=False):
    from coffea.nanoevents import NanoEventsFactory
    if with_constituents and schema==DelphesSchema:
        schema = DelphesSchema2
    events = NanoEventsFactory.from_root(
        file=filename,
        treepath="Delphes",
        schemaclass=schema,
        metadata=metadict,
    ).events()
    if with_constituents:
        events = init_constituents(events)
    return events

