from model_matt_base import config
from svjHelper import masses_matt
config = masses_matt(config=config,scale=10,mpi_over_scale=0.6)
config.Nc = 6
config.Nf = 6
config.Ns = 4
Nu = 6 - 4
Nf = config.Nf
config.rinv = (Nf * (Nf-1) - Nu * (Nu-1)) / (Nf**2-1)
