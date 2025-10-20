import os, sys

nEvents = 1000
path = 'configs/matt/'
sys.path.append(os.path.dirname(sys.path[0]+'/'+path))

files = [f for f in os.listdir(path) if os.path.isfile(path+f) and ('base' not in f) and ('make' not in f)]

for f in files:
    logName = f.replace('.py','')
    print(f)
    print(logName)
    cmd = './run_model helper -C {:s} --dir models/matt --steps all --events 1000 --verbose > models/matt/{:s}.log'.format(path+f, logName)
    print(cmd)
    os.system(cmd)
