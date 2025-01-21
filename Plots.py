import os
import hist
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import pickle

samples = [
    {"name": r"CMS ($r_{\text{inv}} = 0.667$)", "model": "s-channel_mmed-1000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-20_mrho-20_pvector-0.75_spectrum-cms_rinv-0.666667"},
    {"name": r"Snowmass ($m_{\text{dark}} = 20\,\text{GeV}$)", "model": "s-channel_mmed-1000_Nc-3_Nf-3_scale-33.3333_mq-33.73_mpi-20_mrho-83.666_pvector-0.5_spectrum-snowmass_rinv-0"},
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
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)



hists = {}      # Contains the lists of histos for all models

for sample in samples:
    path = f'models/{sample["model"]}'
    file=f'{path}/Hists.pkl'

    with open(file, "rb") as inp:
        hists_model=pickle.load(inp)                # Dict Contains all the histos for 1 model

    hists[sample["name"]] = hists_model

# helper to make a plot
def make_plot(hname):                         # hists is a dict containing
    fig, ax = plt.subplots(figsize=(8,6))
    for l,h in hists.items():                       # h is a list of hist objects
        hep.histplot(h[hname],density=True,ax=ax,label=l,flow="none",yerr=0)
    ax.set_xlim(h[hname].axes[0].edges[0],h[hname].axes[0].edges[-1])
    ax.set_yscale("log")
    ax.set_ylabel("Arbitrary units")
    ax.legend(framealpha=0.5)
    outdir = "All_plots"
    os.makedirs(outdir,exist_ok=True)
    plt.savefig('{}/{}.pdf'.format(outdir,hname),bbox_inches='tight')
    plt.close(fig)

def make_all_plots():
    for hname in hists[samples[0]["name"]]:
        make_plot(hname)

if __name__=="__main__":
    make_all_plots()
