from model_matt_base import config
from svjHelper import masses_matt
config = masses_matt(config=config,scale=10,mpi_over_scale=0.6)
config.Nc = 7
config.Nf = 7
config.Ns = 2
Nu = 7 - 2
Nf = config.Nf
config.rinv = (Nf * (Nf-1) - Nu * (Nu-1)) / (Nf**2-1)
