from model_snowmass_base import config as common
from svjHelper import masses_snowmass, fcdc_configs_NcNf1

common = masses_snowmass(config=common,scale=6/0.6,mpi_over_scale=0.6)
common.spectrum = 'fcdcSimp'
config = fcdc_configs_NcNf1(common=common, simp=True)
