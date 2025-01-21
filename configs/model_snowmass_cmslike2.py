from model_snowmass_base import config
from svjHelper import masses_snowmass

mpi = 20
mpi_over_scale = 0.6
config = masses_snowmass(config=config,scale=mpi/mpi_over_scale,mpi_over_scale=mpi_over_scale)
# this rinv is only applied to diagonal pions; overall rinv value is (6+k)/9 (w/ k=0 here)
config.rinv = 0
