from magiconfig import MagiConfig

config = MagiConfig()
config.Nc = 3
config.Nf = 3
config.channel = 's'
config.delphes = 'cards/delphes_card_CMS.tcl'
config.dir = 'models'
config.events = 1000
config.mmed = 1000.0
config.model_type = 'helper'
config.mpi = 30.0
config.mq = 50.59504132231405
config.mrho = 125.49900398011134
config.pvector = 0.5
config.pythia = ['cards/CMS_Common.txt', 'cards/CMS_Tune_CP5.txt']
config.quiet = False
config.rinv = 0.3333333333333333
config.scale = 50.0
config.spectrum = 'snowmass'
config.steps = ['all']
config.verbose = True