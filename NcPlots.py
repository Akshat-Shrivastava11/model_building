import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.patches import Patch
import pickle
import numpy as np
import itertools
from glob import glob
from svjHelper import get_pred_rinv

def _parse_path_param(path, key, cast):
    for part in path.split('_'):
        if part.startswith(f'{key}-'):
            try:
                return cast(part[len(key) + 1:])
            except (ValueError, TypeError):
                return None
    return None

outdir = "plots_AllNcNf_10k"
os.makedirs(outdir,exist_ok=True)

samples = [
    {"name": "FCDC Nc=3", "models": glob("/eos/uscms/store/user/easmith/svj/fcdc/s-channel_mmed-1000_Nc-3_Nf-*_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdc_gq-0.25_gchi-*")},
    {"name": "FCDC Nc=4", "models": glob("/eos/uscms/store/user/easmith/svj/fcdc/s-channel_mmed-1000_Nc-4_Nf-*_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdc_gq-0.25_gchi-*")},
    {"name": "FCDC Nc=5", "models": glob("/eos/uscms/store/user/easmith/svj/fcdc/s-channel_mmed-1000_Nc-5_Nf-*_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdc_gq-0.25_gchi-*")},
    {"name": "FCDC Nc=6", "models": glob("/eos/uscms/store/user/easmith/svj/fcdc/s-channel_mmed-1000_Nc-6_Nf-*_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdc_gq-0.25_gchi-*")},
    {"name": "FCDC Nc=7", "models": glob("/eos/uscms/store/user/easmith/svj/fcdc/s-channel_mmed-1000_Nc-7_Nf-*_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdc_gq-0.25_gchi-*")},
    {"name": "FCDC Nc=8", "models": glob("/eos/uscms/store/user/easmith/svj/fcdc/s-channel_mmed-1000_Nc-8_Nf-*_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdc_gq-0.25_gchi-*")},
    {"name": "Simple", "models": glob("/eos/uscms/store/user/easmith/svj/fcdc/s-channel_mmed-1000_Nc-3_Nf-3_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-fcdcSimp_gq-0.25_gchi-0.333333_rinv-*")},    
]

data = {} # hists + metadata for all models
for sample in samples:
    #if sample_list and sample["name"] not in sample_list: continue
    data[sample["name"]] = []
    for model in sample['models']:
        file = f'{model}/Hists.pkl'

        with open(file, "rb") as inp:
            data_model = pickle.load(inp)
            # track filename
            data_model['file'] = file

        nc = _parse_path_param(file, 'Nc', int)
        nf = _parse_path_param(file, 'Nf', int)
        ns = _parse_path_param(file, 'Ns', int)
        data_model['_nc'] = nc
        data_model['_nf'] = nf
        data_model['_ns'] = ns
        data_model['_rinvpred'] = get_pred_rinv(nc, nf, ns) if (nc and nf and ns) else None
        data_model['_rinv'] = _parse_path_param(file, 'rinv', float)
        data_model['_is_simple'] = 'fcdcSimp' in file

        data[sample["name"]].append(data_model)

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
colors = ["#e42536", "#5790fc",  "#f89c20", "#9c9ca1", "#964a8b", "#7a21dd"]
# last two are dashdotdot and dashdashdot
lines = ["solid", "dashed", "dotted", "dashdot", (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 3, 5, 1, 5))]
markers = ['o', 's', 'D', 'v', '^', '*']
custom_cycler = mpl.cycler(color=colors) + mpl.cycler(linestyle=lines) + mpl.cycler(marker=markers)


def make_plot(hname, nf_nc_ratio=None):
    all_models = []
    for group_models in data.values():
        for md in group_models:
            nc, nf, ns = md['_nc'], md['_nf'], md['_ns']
            if not (nc and nf and ns) or hname not in md:
                continue
            ratio = nf / nc
            if nf_nc_ratio is not None and ratio != nf_nc_ratio:
                continue
            all_models.append({
                'nc': nc, 'nf': nf, 'ns': ns,
                'ratio': ratio, 'md': md,
            })
    if not all_models:
        return

    first_hist = all_models[0]['md'][hname]
    bin_edges = first_hist.axes[0].edges
    xlabel = first_hist.axes[0].label

    all_nc = sorted(set(m['nc'] for m in all_models))
    props = itertools.cycle(custom_cycler)
    nc_styles = {nc: next(props) for nc in all_nc}

    ratio_str = f'{nf_nc_ratio:.4g}'.replace('.', 'p') if nf_nc_ratio is not None else None

    for nf in sorted(set(m['nf'] for m in all_models)):
        nf_models = [m for m in all_models if m['nf'] == nf]
        for ns in sorted(set(m['ns'] for m in nf_models)):
            group = [m for m in nf_models if m['ns'] == ns]
            if not group:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            for m in sorted(group, key=lambda x: x['nc']):
                style = nc_styles[m['nc']]
                hep.histplot(m['md'][hname], density=True, ax=ax,
                             label=f'$N_c={m["nc"]}$',
                             color=style['color'], linestyle='solid',
                             flow="none", yerr=False)
            ax.set_xlim(bin_edges[0], bin_edges[-1])
            ax.set_yscale("log")
            ax.set_xlabel(xlabel if xlabel else hname)
            ax.set_ylabel("Arbitrary units")
            if nf_nc_ratio is not None:
                ax.set_title(f'$N_f={nf},\\ N_f/N_c={nf_nc_ratio:.4g},\\ N_s={ns}$')
                fname = f'{outdir}/plot_ratio{ratio_str}_Nf{nf}_Ns{ns}_{hname}.png'
            else:
                ax.set_title(f'$N_f={nf},\\ N_s={ns}$')
                fname = f'{outdir}/plot_Nf{nf}_Ns{ns}_{hname}.png'
            ax.legend(framealpha=0.5)
            plt.tight_layout()
            plt.savefig(fname, bbox_inches='tight')
            plt.close(fig)




def make_violin_plots_rinv(hname):
    all_models = []
    for group_models in data.values():
        for md in group_models:
            nc, nf, ns = md['_nc'], md['_nf'], md['_ns']
            if not (nc and nf and ns) or hname not in md:
                continue
            all_models.append({
                'nc': nc, 'nf': nf, 'ns': ns, 'md': md,
            })
    if not all_models:
        return

    first_hist = all_models[0]['md'][hname]
    bin_edges = first_hist.axes[0].edges
    heights = np.diff(bin_edges)
    centers = bin_edges[:-1] + heights / 2
    ylabel = first_hist.axes[0].label
    ylim = (bin_edges[0], bin_edges[-1])
    violin_half_width = 0.48

    all_nc = sorted(set(m['nc'] for m in all_models))
    props = itertools.cycle(custom_cycler)
    nc_colors = {nc: next(props)['color'] for nc in all_nc}

    nf_ns_pairs = sorted(set((m['nf'], m['ns']) for m in all_models))

    y_max = ylim[0]
    for m in all_models:
        nonzero = np.where(m['md'][hname].values().astype(float) > 0)[0]
        if len(nonzero):
            y_max = max(y_max, bin_edges[nonzero[-1] + 1])

    violin_dir = f'{outdir}/violin'
    os.makedirs(violin_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    nc_legend_handles = {}
    for i, (nf, ns) in enumerate(nf_ns_pairs):
        for m in (m for m in all_models if m['nf'] == nf and m['ns'] == ns):
            vals = m['md'][hname].values().astype(float)
            max_val = vals.max()
            if max_val > 0:
                vals = vals / max_val * violin_half_width
            color = nc_colors[m['nc']]
            ax.barh(centers, vals, height=heights, left=i - 0.5 * vals,
                    fc='none', ec=color, linewidth=0.5)
            if m['nc'] not in nc_legend_handles:
                nc_legend_handles[m['nc']] = Patch(fc='none', ec=color, label=f'$N_c={m["nc"]}$')

    nf_groups = {}
    for i, (nf, ns) in enumerate(nf_ns_pairs):
        nf_groups.setdefault(nf, []).append(i)
    nf_vals = sorted(nf_groups)

    ax.set_xticks(range(len(nf_ns_pairs)))
    ax.set_xticklabels([f'$N_s={ns}$' for _, ns in nf_ns_pairs], fontsize=9)
    ax.set_xlim(-0.5, len(nf_ns_pairs) - 0.5)

    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for j, nf in enumerate(nf_vals):
        indices = nf_groups[nf]
        ax.text(np.mean(indices), -0.10, f'$N_f={nf}$', transform=trans,
                ha='center', va='top', fontsize=11)
        if j < len(nf_vals) - 1:
            boundary = (indices[-1] + nf_groups[nf_vals[j + 1]][0]) / 2
            ax.axvline(boundary, color='gray', linewidth=0.8, linestyle='--', zorder=0, ymin=-0.1, clip_on=False)
    ax.set_ylim(ylim[0], y_max)
    ax.set_ylabel(ylabel if ylabel else hname)
    ax.legend(handles=[nc_legend_handles[k] for k in sorted(nc_legend_handles)], framealpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{violin_dir}/violin_{hname}.png', bbox_inches='tight')
    plt.close(fig)


def make_band_plots(hname, rinv_decimals=2):
    all_models = []
    for group_models in data.values():
        for md in group_models:
            if hname not in md:
                continue
            nc, nf = md['_nc'], md['_nf']
            if not (nc and nf):
                continue
            if md['_is_simple']:
                if md['_rinv'] is None:
                    continue
                all_models.append({
                    'nc': nc, 'nf': nf, 'ns': None,
                    'rinv': round(md['_rinv'], rinv_decimals), 'md': md, 'is_simple': True,
                })
            else:
                if md['_ns'] is None or md['_rinvth'] is None:
                    continue
                all_models.append({
                    'nc': nc, 'nf': nf, 'ns': md['_ns'],
                    'rinv': round(md['_rinvth'], rinv_decimals), 'md': md, 'is_simple': False,
                })

    if not all_models:
        return

    first_hist = all_models[0]['md'][hname]
    bin_edges = first_hist.axes[0].edges
    widths = np.diff(bin_edges)
    xlabel = first_hist.axes[0].label

    band_dir = f'{outdir}/band'
    os.makedirs(band_dir, exist_ok=True)

    simp_color = '#333333'

    simp_rinv_bins = sorted(set(m['rinv'] for m in all_models if m['is_simple']))
    for rinv_bin in simp_rinv_bins:
        fcdc_at_rinv = [m for m in all_models if not m['is_simple'] and abs(m['rinv'] - rinv_bin) <= 0.05]
        simp_at_rinv  = [m for m in all_models if m['is_simple'] and m['rinv'] == rinv_bin]
        if not fcdc_at_rinv:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        legend_handles, legend_labels = [], []

        nf_ns_pairs = sorted(set((m['nf'], m['ns']) for m in fcdc_at_rinv))
        props = itertools.cycle(custom_cycler)
        nf_ns_colors = {pair: next(props)['color'] for pair in nf_ns_pairs}
        for nf, ns in nf_ns_pairs:
            group = [m for m in fcdc_at_rinv if m['nf'] == nf and m['ns'] == ns]
            densities = []
            for m in group:
                vals = m['md'][hname].values().astype(float)
                total = (vals * widths).sum()
                if total > 0:
                    densities.append(vals / total)
            if not densities:
                continue
            densities = np.array(densities)
            y_min = densities.min(axis=0)
            y_max = densities.max(axis=0)
            color = nf_ns_colors[(nf, ns)]
            x_step = np.append(bin_edges[:-1], bin_edges[-1])
            poly = ax.fill_between(x_step, np.append(y_min, y_min[-1]),
                                   np.append(y_max, y_max[-1]),
                                   step='post', alpha=0.4, color=color)
            legend_handles.append(poly)
            group_rinv = round(float(np.mean([m['rinv'] for m in group])), 2)
            legend_labels.append(f'FCDC $N_f={nf},\\ N_s={ns},\\ r_{{\\rm inv}}\\approx{group_rinv:.2f}$')

        for m in simp_at_rinv:
            vals = m['md'][hname].values().astype(float)
            total = (vals * widths).sum()
            density = vals / total if total > 0 else vals
            simp_handle = ax.stairs(density, bin_edges, color=simp_color, linewidth=1.5)
            legend_handles.append(simp_handle)
            legend_labels.append(f'Simple $N_c={m["nc"]},\\ N_f={m["nf"]},\\ r_{{\\rm inv}}={rinv_bin}$')
            break

        ax.legend(handles=legend_handles, labels=legend_labels, framealpha=0.5)
        ax.set_xlabel(xlabel if xlabel else hname)
        ax.set_ylabel("Density")
        ax.set_yscale('log')
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        rinv_str = f'{rinv_bin:.2f}'.replace('.', 'p')
        plt.tight_layout()
        plt.savefig(f'{band_dir}/band_rinv{rinv_str}_{hname}.png', bbox_inches='tight')
        plt.close(fig)


def make_ratio_rinv_plots(hname, rinv_decimals=1):
    all_models = []
    for group_models in data.values():
        for md in group_models:
            nc, nf, ns, rinvth = md['_nc'], md['_nf'], md['_ns'], md['_rinvth']
            if not (nc and nf and ns and rinvth is not None) or hname not in md:
                continue
            all_models.append({
                'nc': nc, 'nf': nf, 'ns': ns,
                'rinv': round(rinvth, rinv_decimals), 'rinv_raw': rinvth,
                'nf_nc': round(nf / nc, 4), 'md': md,
            })
    if not all_models:
        return

    first_hist = all_models[0]['md'][hname]
    bin_edges = first_hist.axes[0].edges
    xlabel = first_hist.axes[0].label

    all_nc = sorted(set(m['nc'] for m in all_models))
    props = itertools.cycle(custom_cycler)
    nc_styles = {nc: next(props) for nc in all_nc}

    ratio_rinv_dir = f'{outdir}/ConstNcNf'
    os.makedirs(ratio_rinv_dir, exist_ok=True)

    groups = sorted(set((m['rinv'], m['nf_nc']) for m in all_models))
    for rinv_bin, nf_nc in groups:
        group = [m for m in all_models if m['rinv'] == rinv_bin and m['nf_nc'] == nf_nc]
        if len(set(m['nc'] for m in group)) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        for m in sorted(group, key=lambda x: x['nc']):
            style = nc_styles[m['nc']]
            hep.histplot(m['md'][hname], density=True, ax=ax,
                         label=f'$N_c={m["nc"]},\\ N_f={m["nf"]},\\ N_s={m["ns"]},\\ r_{{\\rm inv}}={m["rinv_raw"]:.2f}$',
                         color=style['color'], linestyle='solid',
                         flow="none", yerr=False)
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_yscale("log")
        ax.set_xlabel(xlabel if xlabel else hname)
        ax.set_ylabel("Arbitrary units")
        ax.set_title(f'$N_f/N_c={nf_nc:.4g},\\ r_{{\\rm inv}}\\approx{rinv_bin:.{rinv_decimals}f}$')
        ax.legend(framealpha=0.5)
        plt.tight_layout()
        ratio_str = f'{nf_nc:.4g}'.replace('.', 'p')
        rinv_str = f'{rinv_bin:.{rinv_decimals}f}'.replace('.', 'p')
        plt.savefig(f'{ratio_rinv_dir}/plot_ratio{ratio_str}_rinv{rinv_str}_{hname}.png', bbox_inches='tight')
        plt.close(fig)


def make_nc_rinv_comparison(hname):
    targets = [
        {'nc': 3, 'nf': 5, 'ns': 1},
        {'nc': 8, 'nf': 5, 'ns': 1},
        {'nc': 3, 'nf': 7, 'ns': 4},
        {'nc': 8, 'nf': 7, 'ns': 4},
    ]
    all_mds = [md for group in data.values() for md in group]
    models = []
    for t in targets:
        found = next((md for md in all_mds if md['_nc'] == t['nc'] and md['_nf'] == t['nf']
                      and md['_ns'] == t['ns'] and hname in md), None)
        if found:
            models.append(found)
    if len(models) != 4:
        return

    first_hist = models[0][hname]
    bin_edges = first_hist.axes[0].edges
    xlabel = first_hist.axes[0].label

    rinv_colors = {5: colors[0], 7: colors[1]}
    nc_lines   = {3: 'solid',   8: 'dashed'}

    fig, ax = plt.subplots(figsize=(8, 6))
    for md in models:
        hep.histplot(md[hname], density=True, ax=ax,
                     label=f'$N_c={md["_nc"]},\\ N_f={md["_nf"]},\\ N_s={md["_ns"]},\\ r_{{\\rm inv}}={md["_rinvth"]:.2f}$',
                     color=rinv_colors[md['_nf']], linestyle=nc_lines[md['_nc']],
                     flow="none", yerr=False)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_yscale("log")
    ax.set_xlabel(xlabel if xlabel else hname)
    ax.set_ylabel("Arbitrary units")
    ax.legend(framealpha=0.5)
    plt.tight_layout()
    comp_dir = f'{outdir}/comparison'
    os.makedirs(comp_dir, exist_ok=True)
    plt.savefig(f'{comp_dir}/comparison_{hname}.png', bbox_inches='tight')
    plt.close(fig)


def make_all_plots():
    all_loaded = [md for group in data.values() for md in group]
    if not all_loaded:
        print("No models loaded — check that EOS paths in samples are accessible.")
        return
    first_model = all_loaded[0]
    hnames = [k for k in first_model if not k.startswith('_') and k != 'file']

    for hname in hnames:
        make_violin_plots_rinv(hname)
        make_band_plots(hname)
        make_plot(hname)
        make_ratio_rinv_plots(hname)
        make_nc_rinv_comparison(hname)

if __name__=="__main__":
    make_all_plots()
