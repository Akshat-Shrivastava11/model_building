import os, sys, imp
from magiconfig import ArgumentParser, ArgumentDefaultsRawHelpFormatter
from multiprocessing import Pool

config_dir = os.path.join(os.getcwd(), "configs")
if config_dir not in sys.path:
    sys.path.append(config_dir)

def run_objs(config_name, args, suffix, outdir, nproc, dryrun):
    os.makedirs(outdir, exist_ok=True)
    configs_fcdc = imp.load_source("configs_fcdc", config_name)
    objs = [obj for obj in vars(configs_fcdc.config)]

    cmds = []
    for obj in objs:
        logName = f'model_{obj}'
        if suffix: logName += f'_{suffix}'
        cmd = f'./run_model helper -C {config_name} -O config.{obj} --dir {outdir} {args} > {outdir}/{logName}.log'
        cmds.append(cmd)

    for (obj,cmd) in zip(objs,cmds):
        print(obj)
        print(cmd)
        # serial processing
        if not dryrun and nproc<=1:
            os.system(cmd)

    # parallel processing
    if not dryrun and nproc>1:
        with Pool(nproc) as pool:
            exit_codes = pool.map(os.system, cmds)
        print(f"Exit codes: {exit_codes}")

if __name__=="__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsRawHelpFormatter
    )
    parser.add_argument("--config-name", type=str, default="configs/model_fcdc_10.py", help="config file containing multiple models")
    parser.add_argument("--args", type=str, default="--steps all --events 1000 --verbose", help="arguments to pass to run_model")
    parser.add_argument("--suffix", type=str, default="", help="suffix for log file names")
    parser.add_argument("--outdir", type=str, default="models/fcdc/", help="outdir for logfile and events")
    parser.add_argument("--nproc", type=int, default=1, help="run in parallel (if > 1)")
    parser.add_argument("--dryrun", default=False, action="store_true", help="print commands but don't run")
    args = parser.parse_args()
    run_objs(args.config_name, args.args, args.suffix, args.outdir, args.nproc, args.dryrun)
