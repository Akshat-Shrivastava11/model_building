import os

Nf_all = [3,4,5,6,7,8]

for Nf in Nf_all:
    Nc = Nf
    for Ns in range(1, Nf-1):
        Nu = Nf - Ns
        r_inv = round((Nf * (Nf-1) - Nu * (Nu-1)) / (Nf**2-1), 2)

        filename = 'model_matt_10_Nf{:g}_Ns{:g}_rinv{:s}.py'.format(Nf, Ns, str(r_inv).replace('.','p'))
        print(filename)
        lines = [
            'from model_matt_base import config',
            'from svjHelper import masses_matt',
            'config = masses_matt(config=config,scale=10,mpi_over_scale=0.6)',
            'config.Nc = ' + str(Nc),
            'config.Nf = ' + str(Nf),
            'config.Ns = ' + str(Ns),
            'Nu = ' + str(Nf) + ' - ' + str(Ns),
            'Nf = config.Nf',
            'config.rinv = (Nf * (Nf-1) - Nu * (Nu-1)) / (Nf**2-1)'
            ]

        if os.path.exists(filename): os.system('mv {:s} {:s}'.format(filename, filename+'.old'))
        with open(filename,'w') as file: # (fileinput.input(args["common"].pythia) if len(args["common"].pythia)>0 else nullcontext()) as inputs:
            file.write('\n'.join(lines))
            file.write('\n')
        # if inputs is not None:
        #     for line in inputs:
        #         file.write(line)
