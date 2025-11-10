from model_snowmass_base import config
from svjHelper import masses_snowmass

config = masses_snowmass(config=config,scale=10,mpi_over_scale=0.6)

config.rinv = 0.86
config.spectrum = 'snowmass_cmslike'
