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
		# these two lists have to be kept in sync: zip, drop, unzip
        base_form["fields"], base_form["contents"] = zip(*[entry for entry in zip(base_form["fields"], base_form["contents"]) if not "fBits" in entry[0]])
        super().__init__(base_form)

# ignore unnecessary warning
from numba.core.errors import NumbaTypeSafetyWarning
import warnings
warnings.simplefilter('ignore',category=NumbaTypeSafetyWarning)
# optimized kernel for jet:constituent matching within an event
@nb.njit("i8[:](i4[:],u4[:])")
def get_constituents_kernel(jet_refs: NDArray[np.int32], cand_ids: NDArray[np.uint32]) -> NDArray[np.int64]:
    # get hash table mapping global index : global unique ID
    hash_table = {k:v for v,k in enumerate(cand_ids)}
    # apply hash map
    output = [hash_table[ref] for ref in jet_refs]
    return np.asarray(output)

# apply kernel to events (in chunks)
def get_constituents_chunk(events, jetsname, candsname):
    jets = events[jetsname]
    cands = events[candsname]

    flat_indices = []
    counts_all = []
    jets_per_event = []

    # compute candidate offsets
    cand_counts = ak.num(cands, axis=1)
    cand_offsets = np.cumsum(np.concatenate([[0], ak.to_numpy(cand_counts[:-1])]))

    for i, (jets_evt, cands_evt) in enumerate(zip(jets, cands)):
        if jets_evt is None:
            jets_per_event.append(0)
            continue

        refs = ak.flatten(jets_evt.Constituents.refs)
        counts = ak.num(jets_evt.Constituents.refs, axis=1)

        jets_per_event.append(len(counts))
        counts_all.extend(counts.tolist())

        if len(refs) == 0:
            flat_indices.append(np.array([], dtype=np.int64))
            continue

        # local indices within event
        local_idx = get_constituents_kernel(
            ak.to_numpy(refs),
            ak.to_numpy(cands_evt.fUniqueID),
        )

        # convert to global indices
        global_idx = local_idx + cand_offsets[i]

        flat_indices.append(global_idx)

    # concatenate indices only (cheap)
    flat_indices = np.concatenate(flat_indices) if flat_indices else np.array([], dtype=np.int64)

    # zero-copy flatten of candidates
    flat_cands = ak.flatten(cands)

    # single gather
    gathered = flat_cands[flat_indices]

    # rebuild structure (one level at a time)
    jets_level = ak.unflatten(gathered, counts_all)
    events_level = ak.unflatten(jets_level, jets_per_event)

    return ak.with_name(events_level, DelphesSchema2.mixins[candsname])

def get_constituents(events, jetsname, candsname, chunk_size=500):
    outputs = []

	# chunking avoids memory overusage
    for start in range(0, len(events), chunk_size):
        stop = start + chunk_size
        chunk = events[start:stop]

        outputs.append(
            get_constituents_chunk(chunk, jetsname, candsname)
        )

    return ak.with_name(
        ak.concatenate(outputs),
        DelphesSchema2.mixins[candsname]
    )

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
        file={filename : "Delphes"},
        schemaclass=schema,
        metadata=metadict,
    ).events()

    if with_constituents:
        events = init_constituents(events)
    return events

