import os, sys, imp
from magiconfig import ArgumentParser, ArgumentDefaultsRawHelpFormatter

config_dir = os.path.join(os.getcwd(), "configs")
sys.path.append(config_dir)

def run_objs(config_name, args, suffix, dryrun):
    configs_fcdc = imp.load_source("configs_fcdc", config_name)
    objs = [obj for obj in vars(configs_fcdc.config)]

    for obj in objs:
        logName = f'model_{obj}'
        if suffix: logName += f'_{suffix}'
        print(obj)
        cmd = f'./run_model helper -C {config_name} -O config.{obj} {args} > models/fcdc/{logName}.log'
        print(cmd)
        if not dryrun: os.system(cmd)

if __name__=="__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsRawHelpFormatter
    )
    parser.add_argument("--config-name", type=str, default="configs/model_fcdc_10.py", help="config file containing multiple models")
    parser.add_argument("--args", type=str, default="--dir models/fcdc --steps all --events 1000 --verbose", help="arguments to pass to run_model")
    parser.add_argument("--suffix", type=str, default="", help="suffix for log file names")
    parser.add_argument("--dryrun", default=False, action="store_true", help="print commands but don't run")
    args = parser.parse_args()
    run_objs(args.config_name, args.args, args.suffix, args.dryrun)
