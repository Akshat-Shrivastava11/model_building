from model_fcdc_base import config as common
from svjHelper import masses_snowmass, fcdc_configs_NcNf1, fcdc_configs_NcNfAll

# generate new configs for each input
common = masses_snowmass(config=common,scale=10,mpi_over_scale=0.6)
config = fcdc_configs_NcNfAll(common=common)
