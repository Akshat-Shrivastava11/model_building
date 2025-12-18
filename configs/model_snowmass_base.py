from magiconfig import MagiConfig
from svjHelper import gchi_lhcdm

config = MagiConfig()
config.channel = 's'
config.mmed = 1000
config.Nc = 3
config.Nf = 3
config.pvector = 0.5
# this rinv is only applied to diagonal pions; overall rinv value is (6+k)/9
k = 1
config.rinv = k/3
config.spectrum = 'snowmass'
config.gq = 0.25
config.gchi = gchi_lhcdm(gDM=1.0, Nc=config.Nc, Nf=config.Nf)
