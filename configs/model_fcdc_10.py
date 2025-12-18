from svjHelper import masses_snowmass, gchi_lhcdm
from magiconfig import MagiConfig

# Common configs
common = MagiConfig()
common.channel = 's'
common.mmed = 1000
common.pvector = 0.5
common.spectrum = 'fcdc'
common = masses_snowmass(config=common,scale=10,mpi_over_scale=0.6)
common.gq = 0.25

# generate new configs for each input
# only consider Nc=Nf case here
Nf_min = 3
Nf_max = 8
Ns_min = 1

for Nf_val in range(Nf_min,Nf_max+1):
    Ns_max = Nf_val - 2
    for Ns_val in range(Ns_min, Ns_max+1):
        configName = 'configNc{:d}Nf{:d}Ns{:d}'.format(Nf_val, Nf_val, Ns_val)
        globals()[configName] = MagiConfig(Nc=Nf_val, Nf=Nf_val, Ns=Ns_val, gchi=gchi_lhcdm(gDM=1.0, Nc=Nf_val, Nf=Nf_val))
        globals()[configName].join(common)
