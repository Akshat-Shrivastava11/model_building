from magiconfig import MagiConfig
from svjHelper import scale_cms, gchi_lhcdm

config = MagiConfig()
config.channel = 's'
config.mmed = 1000
config.Nc = 2
config.Nf = 2
config.mpi = 20
config.mrho = config.mpi
config.scale = scale_cms(mpi=config.mpi)
config.mq = config.mpi/2.
config.pvector = 0.75
config.rinv = 0.3
config.spectrum = 'cms'
config.gq = 0.25
config.gchi = gchi_lhcdm(gDM=1.0, Nc=config.Nc, Nf=config.Nf)
