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
# account for two complete models with rinv=0.75
for sample in samples:
    if "simp" in sample["name"]:
        idx = next((i for i, val in enumerate(sample["models"]) if 'rinv-0.75' in val), None)
        if idx is not None:
            sample["models"].insert(idx+1, sample["models"][idx])

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

def process_data(data, x, qname, forcex):
    processed = []
    for sample, models in data.items():
        xvals = np.array([model['meta'][x] for model in models])
        means, stdevs, stderrs, hists = zip(*[
            (model['meta'][qname]['mean'], model['meta'][qname]['stdev'], model['meta'][qname]['stderr'], model['hist'].get(qname,None)) for model in models
        ])
        # order by rinv_pred
        order = np.argsort(xvals)
        processed_dict = dict(
            sample = sample,
            xvals = xvals[order],
            means = np.array(means)[order],
            stdevs = np.array(stdevs)[order],
            stderrs = np.array(stderrs)[order],
            hists = [hists[i] for i in order],
        )
        hists_missing = [h is None for h in processed_dict['hists']]
        if all(hists_missing):
            processed_dict['hists'] = []
        elif any(hists_missing):
            raise RuntimeError(f"Missing histogram {qname} in:\n"+','.join([model[file] for i,model in enumerate(models) if hists_missing[i]]))
        processed.append(processed_dict)
    if forcex:
        if forcex not in data:
            raise ValueError(f"Unknown forcex sample {forcex}")
        forcexvals = next((pd['xvals'] for pd in processed if pd['sample']==forcex))
        for processed_dict in processed:
            processed_dict['xvals'] = forcexvals
    return processed

# helper to make a plot
def make_plot(type, data, x, qname, outdir, offset):
    fig, ax = plt.subplots(figsize=(8,6))
    # iterator for manual control
    props = iter(custom_cycler)
    # advance iterator if requested
    next(itertools.islice(props, offset, offset), None)
    ylim = None
    if type=='stat':
        for entry in data:
            style = next(props)
            # means
            line, = ax.plot(entry['xvals'], entry['means'], label=entry['sample'], fillstyle='none', **style)
            # stderr as errorbar
            ax.errorbar(entry['xvals'], entry['means'], yerr=entry['stderrs'], fmt='none', ecolor=style['color'], capsize=3, **style)
            # stdev as filled
            ax.fill_between(entry['xvals'], entry['means']-entry['stdevs'], entry['means']+entry['stdevs'], color=style['color'], alpha=0.15)
    elif type=='violin':
        # get global max hist height for each xval
        all_hists = []
        samples = []
        labels = []
        bin_edges = []
        heights = []
        centers = []
        max_per_label = []
        for entry in data:
            if len(entry['hists'])==0:
                continue

            # these are assumed to be shared across all samples
            if len(labels)==0:
                labels = entry['xvals']
                bin_edges = entry['hists'][0].axes[0].edges
                ylim = (bin_edges[0], bin_edges[-1])
                heights = np.diff(bin_edges)
                centers = bin_edges[:-1] + heights/2
            # extract values from histograms
            all_hists.append({x:h.values() for x,h in zip(entry['xvals'],entry['hists'])})
            samples.append(entry['sample'])

        # get global max hist height for each xval
        for label in labels:
            max_per_label.append(max([max(h[label]) for h in all_hists]))
        widths = np.array(max_per_label)
        x_locs = np.cumsum(widths) - 0.5 * widths

        for hists,sample in zip(all_hists, samples):
            style = next(props)
            # plot all hists from this sample
            style.update({
                'fc': 'none',
                'ec': style['color'],
                'label': sample,
            })
            style.pop('marker')
            for x_loc, label in zip(x_locs, labels):
                hist = hists[label]
                lefts = x_loc - 0.5 * hist
                ax.barh(centers, hist, height=heights, left=lefts, **style)
                # only apply label once
                style.pop('label', None)
            ax.set_xticks(x_locs, [f'{label:.3f}' for label in labels], rotation=45)
    if type=='stat':
        ax.axline((0,0), slope=1, color='black', linestyle=':')
    ax.set_xlabel(r'$r_{\text{inv}}^{\text{pred}}$')
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(qname)
    ax.legend(framealpha=0.5)
    plt.savefig(f'{outdir}/{type}_{qname}.pdf',bbox_inches='tight')
    plt.close(fig)

def make_all_plots(outdir, types, sample_list, x, y, forcex, offset):
    data = {} # hists + metadata for all models

    for sample in samples:
        if sample_list and sample["name"] not in sample_list: continue
        data[sample["name"]] = []
        for model in sample['models']:
            file = f'{model}/Hists.pkl'

            with open(file, "rb") as inp:
                data_model = pickle.load(inp)
                # track filename
                data_model['file'] = file
                # join these for ease of use
                data_model['meta'] = data_model['model'] | data_model['analysis']

            data[sample["name"]].append(data_model)

    os.makedirs(outdir, exist_ok=True)
    for qname in y:
        processed = process_data(data, x, qname, forcex)
        for plot_type in types:
            make_plot(plot_type, processed, x, qname, outdir, offset)

if __name__=="__main__":
    allowed_types = ['stat', 'violin']
    qtys_default = ['stable_invisible_fraction','DHIVJet12_rinv','DiDHIVJet_rinv','DHIVJet12_rinv_global']

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsRawHelpFormatter
    )
    parser.add_argument("--dir", type=str, default="All_metaplots", help="output directory")
    parser.add_argument("--types", type=str, default=allowed_types, nargs='*', choices=allowed_types, help="plot types")
    parser.add_argument("--samples", type=str, default=[], nargs='*', help="list of samples to plot")
    parser.add_argument("-x", type=str, default='rinv', help="x variable")
    parser.add_argument("--forcex", type=str, default=None, help="force use of x values from specified sample")
    parser.add_argument("-y", type=str, default=qtys_default, nargs='*', help="y variable(s)")
    parser.add_argument("--offset", type=int, default=0, help="offset for color/style cycler")
    args = parser.parse_args()

    unknown_samples = [s for s in args.samples if not any([s==sm['name'] for sm in samples])]
    if unknown_samples:
        raise ValueError("Unknown sample(s) requested:",','.join(unknown_samples))

    make_all_plots(args.dir, args.types, args.samples, args.x, args.y, args.forcex, args.offset)
