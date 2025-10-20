from model_matt_base import config
from svjHelper import masses_matt
config = masses_matt(config=config,scale=10,mpi_over_scale=0.6)
config.Nc = 8
config.Nf = 8
config.Ns = 5
Nu = 8 - 5
Nf = config.Nf
config.rinv = (Nf * (Nf-1) - Nu * (Nu-1)) / (Nf**2-1)
