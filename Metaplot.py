import os
import hist
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
from magiconfig import ArgumentParser, ArgumentDefaultsRawHelpFormatter
from glob import glob
import itertools

samples = [
    {"name": "FCDC", "models": glob("models/fcdc/s-channel_mmed-1000_Nc-*_Nf-*_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdc_gq-0.25_gchi-*_Ns-*")},
    {"name": "simp", "models": glob("models/fcdc/s-channel_mmed-1000_Nc-3_Nf-3_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdcSimp_gq-0.25_gchi-0.333333_rinv-*")},
    {"name": "FCDC (3-body)", "models": glob("models/fcdc/s-channel_mmed-1000_Nc-*_Nf-*_scale-3.52941_mq-3.8666_mpi-6_mrho-11.2139_pvector-0.5_spectrum-fcdc_gq-0.25_gchi-*_Ns-*")},
    {"name": "simp (3-body)", "models": glob("models/fcdc/s-channel_mmed-1000_Nc-3_Nf-3_scale-3.52941_mq-3.8666_mpi-6_mrho-11.2139_pvector-0.5_spectrum-fcdcSimp_gq-0.25_gchi-0.333333_rinv-*")},
]

# stylistic options
mpl.rcParams.update({
    "axes.labelsize" : 18,
    "legend.fontsize" : 16,
    "xtick.labelsize" : 14,
    "ytick.labelsize" : 14,
    "font.size" : 18,
    "legend.frameon": True,
})
# based on https://github.com/mpetroff/accessible-color-cycles
# red, blue, mauve, orange, purple, gray,
colors = ["#e42536", "#5790fc", "#964a8b", "#f89c20", "#7a21dd", "#9c9ca1"]

# last two are dashdotdot and dashdashdot
lines = ["solid", "dashed", "dotted", "dashdot", (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 3, 5, 1, 5))]
markers = ['o', 's', 'D', 'v', '^', '*']
custom_cycler = mpl.cycler(color=colors) + mpl.cycler(linestyle=lines) + mpl.cycler(marker=markers)

# helper to make a plot
def stat_plot(data, x, qname, outdir, offset):
    fig, ax = plt.subplots(figsize=(8,6))
    # iterator for manual control
    props = iter(custom_cycler)
    # advance iterator if requested
    next(itertools.islice(props, offset, offset), None)
    for sample, models in data.items():
        style = next(props)
        xvals = np.array([model['meta'][x] for model in models])
        means, stdevs, stderrs = zip(*[
            (model['meta'][qname]['mean'], model['meta'][qname]['stdev'], model['meta'][qname]['stderr']) for model in models
        ])
        # order by rinv_pred
        order = np.argsort(xvals)
        xvals = xvals[order]
        means = np.array(means)[order]
        stdevs = np.array(stdevs)[order]
        stderrs = np.array(stderrs)[order]
        # means
        line, = ax.plot(xvals, means, label=sample, fillstyle='none', **style)
        color = line.get_color()
        # stderr as errorbar
        ax.errorbar(xvals, means, yerr=stderrs, fmt='none', ecolor=style['color'], capsize=3, **style)
        # stdev as filled
        ax.fill_between(xvals, means-stdevs, means+stdevs, color=style['color'], alpha=0.15)
    ax.axline((0,0), slope=1, color='black', linestyle=':')
    ax.set_xlabel(r'$r_{\text{inv}}^{\text{pred}}$')
    ax.set_ylabel(qname)
    ax.legend(framealpha=0.5)
    plt.savefig('{}/stat_{}.pdf'.format(outdir,qname),bbox_inches='tight')
    plt.close(fig)

def make_all_plots(outdir, sample_list, x, y, offset):
    data = {} # hists + metadata for all models

    for sample in samples:
        if sample_list and sample["name"] not in sample_list: continue
        data[sample["name"]] = []
        for model in sample['models']:
            file = f'{model}/Hists.pkl'

            with open(file, "rb") as inp:
                data_model = pickle.load(inp)
                # join these for ease of use
                data_model['meta'] = data_model['model'] | data_model['analysis']

            data[sample["name"]].append(data_model)

    os.makedirs(outdir, exist_ok=True)
    for qname in y:
        stat_plot(data, x, qname, outdir, offset)

if __name__=="__main__":
    qtys_default = ['stability','DHIVJet12_rinv','DiDHIVJet_rinv','DHIVJet12_rinv_global']

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsRawHelpFormatter
    )
    parser.add_argument("--dir", type=str, default="All_metaplots", help="output directory")
    parser.add_argument("--samples", type=str, default=[], nargs='*', help="list of samples to plot")
    parser.add_argument("-x", type=str, default='rinv', help="x variable")
    parser.add_argument("-y", type=str, default=qtys_default, nargs='*', help="y variable(s)")
    parser.add_argument("--offset", type=int, default=0, help="offset for color/style cycler")
    args = parser.parse_args()

    unknown_samples = [s for s in args.samples if not any([s==sm['name'] for sm in samples])]
    if unknown_samples:
        raise ValueError("Unknown sample(s) requested:",','.join(unknown_samples))

    make_all_plots(args.dir, args.samples, args.x, args.y, args.offset)
