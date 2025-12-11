import os, sys, imp

config_dir = os.path.join(os.getcwd(), "configs")
sys.path.append(config_dir)

nEvents = 1000
config_name = "configs/model_fcdc_10.py"
path = os.path.join(config_dir, "fcdc")
configs_fcdc = imp.load_source("configs_fcdc", config_name)
prefix = "config"
objs = [obj for obj in dir(configs_fcdc) if obj.startswith(prefix) and len(obj)>len(prefix)]

for obj in objs:
    logName = obj.replace(prefix,"model_")
    print(obj)
    cmd = f'./run_model helper -C {config_name} -O {obj} --dir models/fcdc --steps all --events {nEvents} --verbose > models/fcdc/{logName}.log'
    print(cmd)
    os.system(cmd)
