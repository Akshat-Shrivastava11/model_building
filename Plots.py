import os
import hist
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import pickle

samples = [
    #{"name": r"CMS", "model": "s-channel_mmed-1000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-20_mrho-20_pvector-0.75_spectrum-cms_rinv-0.3/"}
    # {"name": r"CMS ($r_{\text{inv}} = 0.667$)", "model": "s-channel_mmed-1000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-20_mrho-20_pvector-0.75_spectrum-cms_rinv-0.666667"},
    # {"name": r"Snowmass ($m_{\text{dark}} = 20\,\text{GeV}$)", "model": "s-channel_mmed-1000_Nc-3_Nf-3_scale-33.3333_mq-33.73_mpi-20_mrho-83.666_pvector-0.5_spectrum-snowmass_rinv-0"},
    {"name": r"Matt_rinv0p5", "model": "s-channel_mmed-1000_Nc-3_Nf-3_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.5"},
    # {"name": r"Matt_rinv0p4", "model": "s-channel_mmed-1000_Nc-4_Nf-4_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.4"},
    # {"name": r"Matt_rinv0p67", "model": "s-channel_mmed-1000_Nc-4_Nf-4_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.666667"},
    # {"name": r"Matt_rinv0p33", "model": "s-channel_mmed-1000_Nc-5_Nf-5_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.333333"},
    # {"name": r"Matt_rinv0p58", "model": "s-channel_mmed-1000_Nc-5_Nf-5_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.583333"},
    {"name": r"Matt_rinv0p75", "model": "s-channel_mmed-1000_Nc-5_Nf-5_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.75"},
    # {"name": r"Matt_rinv0p28", "model": "s-channel_mmed-1000_Nc-6_Nf-6_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.285714"},
    # {"name": r"Matt_rinv0p51", "model": "s-channel_mmed-1000_Nc-6_Nf-6_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.514286"},
    # {"name": r"Matt_rinv0p69", "model": "s-channel_mmed-1000_Nc-6_Nf-6_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.685714"},
    # {"name": r"Matt_rinv0p8", "model": "s-channel_mmed-1000_Nc-6_Nf-6_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.8"},
    # {"name": r"Matt_rinv0p25", "model": "s-channel_mmed-1000_Nc-7_Nf-7_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.25"},
    # {"name": r"Matt_rinv0p46", "model": "s-channel_mmed-1000_Nc-7_Nf-7_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.458333"},
    # {"name": r"Matt_rinv0p625", "model": "s-channel_mmed-1000_Nc-7_Nf-7_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.625"},
    # {"name": r"Matt_rinv0p75", "model": "s-channel_mmed-1000_Nc-7_Nf-7_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.75"},
    # {"name": r"Matt_rinv0p83", "model": "s-channel_mmed-1000_Nc-7_Nf-7_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.833333"},
    {"name": r"Matt_rinv0p22", "model": "s-channel_mmed-1000_Nc-8_Nf-8_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.222222"},
    # {"name": r"Matt_rinv0p41", "model": "s-channel_mmed-1000_Nc-8_Nf-8_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.412698"},
    # {"name": r"Matt_rinv0p57", "model": "s-channel_mmed-1000_Nc-8_Nf-8_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.571429"},
    # {"name": r"Matt_rinv0p7", "model": "s-channel_mmed-1000_Nc-8_Nf-8_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.698413"},
    # {"name": r"Matt_rinv0p79", "model": "s-channel_mmed-1000_Nc-8_Nf-8_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.793651"},
    {"name": r"Matt_rinv0p85", "model": "s-channel_mmed-1000_Nc-8_Nf-8_scale-10_mq-10.119_mpi-6_mrho-25.0998_pvector-0.5_spectrum-matt_rinv-0.857143"},
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
    path = f'models/matt/{sample["model"]}'
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
