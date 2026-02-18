from model_fcdc_base import config as common
from svjHelper import masses_snowmass, fcdc_configs_NcNf1

# from mrho_snowmass, mrho > 2mpi -> mpi/scale < sqrt(5.76/2.5) ~ 1.518
common = masses_snowmass(config=common,scale=10,mpi_over_scale=1.7)
# generate new configs for each input
config = fcdc_configs_NcNf1(common=common)
