from model_snowmass_base import config
from svjHelper import masses_snowmass

# from mrho_snowmass, mrho > 2mpi -> mpi/scale < sqrt(5.76/2.5) ~ 1.518
config = masses_snowmass(config=config,scale=10,mpi_over_scale=1.7)

config.rinv = 0.5
config.spectrum = 'fcdcSimp'
