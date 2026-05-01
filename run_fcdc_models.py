import os, sys, imp

config_dir = os.path.join(os.getcwd(), "configs")
sys.path.append(config_dir)

nEvents = 1000
config_name = "configs/model_fcdc_10.py"
configs_fcdc = imp.load_source("configs_fcdc", config_name)
objs = [obj for obj in vars(configs_fcdc.config)]

for obj in objs:
    logName = f'model_{obj}'
    print(obj)
    cmd = f'./run_model helper -C {config_name} -O {obj} --dir models/fcdc --steps all --events {nEvents} --verbose > models/fcdc/{logName}.log'
    print(cmd)
    os.system(cmd)
