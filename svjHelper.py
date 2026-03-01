import math, os
from string import Template
from pathlib import Path
from magiconfig import MagiConfig

# utilities for simple calculations

# "empirical" formula from CMS search to maximize dark hadron yield
# (see 2203.09503 Section 2.2.3 for details)
def scale_cms(*, mpi):
    return 3.2*math.pow(mpi,0.8)

# current quark mass, 2203.09503 eq. 14
def mq_snowmass(*, mpi, scale):
    return (mpi/5.5)**2/scale

# constituent quark mass (used in Pythia), 2203.09503 Section 4.1.4
def mqconst_snowmass(*, mpi, scale):
    return mq_snowmass(mpi=mpi, scale=scale) + scale

# 2203.09503 eq. 14
def mrho_snowmass(*, mpi, scale):
    return scale*math.sqrt(5.76+1.5*mpi**2/scale**2)

# combining these into snowmass mass scheme
def masses_snowmass(*, config, scale, mpi_over_scale):
    config.scale = scale
    config.mpi = mpi_over_scale * scale
    config.mq = mqconst_snowmass(mpi=config.mpi, scale=config.scale)
    config.mrho = mrho_snowmass(mpi=config.mpi, scale=config.scale)
    return config

# dark quark coupling calculation for consistency with LHC DM WG mediator width
# (accounting for dark flavor and dark color)
def gchi_lhcdm(*, gDM, Nc, Nf):
    return gDM/math.sqrt(Nc*Nf)

# create a single fcdc config using input numbers
def fcdc_config(*, common, Nc, Nf, Ns):
    configName = 'Nc{:d}Nf{:d}Ns{:d}'.format(Nc, Nf, Ns)
    config = MagiConfig(Nc=Nc, Nf=Nf, Ns=Ns, gchi=gchi_lhcdm(gDM=1.0, Nc=Nc, Nf=Nf))
    config.join(common)
    return configName, config

# create spread of fcdc configs for Nc=Nf case
def fcdc_configs_NcNf1(*, common):
    Nf_min = 3
    Nf_max = 8
    Ns_min = 1

    config = MagiConfig()
    for Nf_val in range(Nf_min,Nf_max+1):
        Ns_max = Nf_val - 2
        for Ns_val in range(Ns_min, Ns_max+1):
            this_name, this_config = fcdc_config(common=common, Nc=Nf_val, Nf=Nf_val, Ns=Ns_val)
            setattr(config, this_name, this_config)
    return config

# classes for helper

class quark(object):
    def __init__(self,*,id,mass):
        self.id = id
        self.mass = mass
        self.massrun = mass
        self.bf = 1
        self.on = True
        self.active = True # for running nf

    def __repr__(self):
        return str(self.id)+": m = "+str(self.mass)+", mr = "+str(self.massrun)+", on = "+str(self.on)+", bf = "+str(self.bf)

# follows Ellis, Stirling, Webber calculations
class massRunner(object):
    def __init__(self):
        # QCD scale in GeV
        self.Lambda = 0.218

    # RG terms, assuming nc = 3 (QCD)
    def c(self): return 1./math.pi
    def cp(self,nf): return (303.-10.*nf)/(72.*math.pi)
    def b(self,nf): return (33.-2.*nf)/(12.*math.pi)
    def bp(self,nf): return (153.-19.*nf)/(2.*math.pi*(33.-2.*nf))
    def alphaS(self,Q,nf): return 1./(self.b(nf)*math.log(Q**2/self.Lambda**2))

    # derived terms
    def cb(self,nf): return 12./(33.-2.*nf)
    def one_c_cp_bp_b(self,nf): return 1.+self.cb(nf)*(self.cp(nf)-self.bp(nf))

    # constant of normalization
    def mhat(self,mq,nfq):
        return mq/math.pow(self.alphaS(mq,nfq),self.cb(nfq))/self.one_c_cp_bp_b(nfq)

    # mass formula
    def m(self,mq,nfq,Q,nf):
        # temporary hack: exclude quarks w/ mq < Lambda
        alphaq = self.alphaS(mq,nfq)
        if alphaq < 0: return 0
        else: return self.mhat(mq,nfq)*math.pow(self.alphaS(Q,nf),self.cb(nf))*self.one_c_cp_bp_b(nf)

    # operation
    def run(self,quark,nfq,scale,nf):
        # run to specified scale and nf
        return self.m(quark.mass,nfq,scale,nf)

class quarklist(object):
    def __init__(self):
        # mass-ordered
        # using Pythia masses
        # would be nice to get these directly from Pythia to ensure consistency
        self.qlist = [
            quark(id=1,mass=0.33), # down
            quark(id=2,mass=0.33), # up
            quark(id=3,mass=0.5),  # strange
            quark(id=4,mass=1.5),  # charm
            quark(id=5,mass=4.8),  # bottom
            quark(id=6,mass=173.0), # top
        ]
        self.scale = None
        self.runner = massRunner()

    def set(self,scale):
        self.scale = scale
        # mask quarks above scale
        for q in self.qlist:
            # for decays
            if scale is None or 2*q.mass < scale: q.on = True
            else: q.on = False
            # for nf running
            if scale is None or q.mass < scale: q.active = True
            else: q.active = False
        # compute running masses
        if scale is not None:
            qtmp = self.get(active=True)
            nf = len(qtmp)
            for iq,q in enumerate(qtmp):
                q.massrun = self.runner.run(q,iq,scale,nf)
        # or undo running
        else:
            for q in self.qlist:
                q.massrun = q.mass

    def reset(self):
        self.set(None)

    def get(self,active=False):
        return [q for q in self.qlist if (q.active if active else q.on)]

class darkHadron():
    def __init__(self, helper, *, id, mass, decay, props=[], rinv=None, dm=None, placeholder=False, auto_rinv=True):
        self.id = id
        self.mass = mass
        self.decay = decay
        self.helper = helper
        self.props = props
        self.placeholder = placeholder
        if not hasattr(self, self.decay+'Decay'):
            raise ValueError("unknown decay {} for id {}".format(self.decay, self.id))

        self.quarks = quarklist()
        # get limited set of quarks for decays (check mDark against quark masses, compute running)
        self.quarks.set(self.mass)

        self.dm = dm
        self.rinv = rinv
        if self.rinv is None:
            self.rvis = 1
        else:
            self.rvis = 1 - self.rinv
        self.auto_rinv = auto_rinv

    def getLines(self):
        lines = []
        lines += ['{:d}:'.format(self.id)+prop for prop in self.props]
        lines += ['{:d}:m0 = {:g}'.format(self.id,self.mass)]
        if self.rinv is not None and self.auto_rinv:
            lines += self.invisibleDecay()
        lines += getattr(self, self.decay+'Decay')()
        # first channel should be oneChannel
        for i,line in enumerate(lines):
            if "Channel" in line:
                lines[i] = line.replace("addChannel","oneChannel")
                break
        return lines

    def stableDecay(self):
        lines = ['{:d}:onMode = 0'.format(self.id)]
        return lines

    def invisibleDecay(self):
        lines = ['{:d}:addChannel = 1 {:g} 0 {:d} -{:d}'.format(self.id,self.rinv,self.dm,self.dm)]
        return lines

    def simpleDecay(self):
        theQuarks = self.quarks.get()
        # just pick down quarks
        theQuarks = [q for q in theQuarks if q.id==1]
        theQuarks[0].bf = self.rvis
        return self.visibleLines(theQuarks)

    def democraticDecay(self):
        theQuarks = self.quarks.get()
        bfQuarks = (self.rvis)/float(len(theQuarks))
        for iq,q in enumerate(theQuarks):
            theQuarks[iq].bf = bfQuarks
        return self.visibleLines(theQuarks)

    def massInsertionDecay(self):
        theQuarks = self.quarks.get()
        denom = sum([q.massrun**2 for q in theQuarks])
        # hack for really low masses
        if denom==0.: return self.democraticDecay()
        for q in theQuarks:
            q.bf = self.rvis*(q.massrun**2)/denom
        return self.visibleLines(theQuarks)

    def visibleLines(self,theQuarks):
        lines = ['{:d}:addChannel = 1 {:g} 91 {:d} -{:d}'.format(self.id,q.bf,q.id,q.id) for q in theQuarks if q.bf>0]
        return lines

    def getDarkQuark(self):
        quarkIndex = str(self.id)[4]
        return int(quarkIndex)

    def getAntiDarkQuark(self):
        antiQuarkIndex = str(self.id)[5]
        return int(antiQuarkIndex)

    @staticmethod
    def getDarkMeson(*, dq, adq, spin):
        spin_final_digit = {
            0: 1,
            1: 3,
        }
        dq1 = max(dq, adq)
        dq2 = min(dq, adq)
        final = spin_final_digit[spin]
        id = f'4900{dq1}{dq2}{final}'
        return int(id)

    def darkRhoDecay(self):
        if self.mass > 2*self.helper.mpi:
            return self.darkRho2BodyDecay()
        else:
            # if a) diagonal or b) off-diagonal but with FCDCs (unstable dark quark), rho goes directly to qq (SM) when pipi not allowed
            return self.democraticDecay()

    def darkRhoProtectedDecay(self):
        if self.mass > 2*self.helper.mpi:
            return self.darkRho2BodyDecay()
        else:
            # "protected" rho (off-diagonal, no FCDCs) can only decay through 3-body
            return self.darkRho3BodyDecay()

    def darkRho2BodyDecay(self):
        darkQuarkFromRho = self.getDarkQuark()
        antiDarkQuarkFromRho = self.getAntiDarkQuark()

        etaPrime = self.getDarkMeson(dq=self.helper.Nf, adq=self.helper.Nf, spin=0)
        allowed = []
        for n in range(1, self.helper.Nf+1):
            sign1 = -1 if n > darkQuarkFromRho else 1
            sign2 = 1 if n > antiDarkQuarkFromRho else -1
            meson1 = sign1 * self.getDarkMeson(dq=n, adq=darkQuarkFromRho, spin=0)
            meson2 = sign2 * self.getDarkMeson(dq=n, adq=antiDarkQuarkFromRho, spin=0)
            # etaPrime taken to be heavy (probKeepEta1=0), so exclude from allowed decays
            if abs(meson1)==etaPrime or abs(meson2)==etaPrime:
                continue
            allowed.append(
                (meson1, meson2)
            )

        # equal branching fraction to all allowed combinations of dark quark flavors
        lines = []
        bf = 1.0/len(allowed)
        for decay1, decay2 in allowed:
            lines.append('{:d}:addChannel = 1 {:03f} 101 {:d} {:d}'.format(self.id, bf, decay1, decay2))

        return lines

    def darkRho3BodyDecay(self):
        darkQuarkFromRho = self.getDarkQuark()
        antiDarkQuarkFromRho = self.getAntiDarkQuark()

        # this *should* only ever be a stable dark pion
        darkPion = self.getDarkMeson(dq=darkQuarkFromRho, adq=antiDarkQuarkFromRho, spin=0)
        thisBR = 1.0
        # to reuse this for simplified case
        if self.dm:
            darkPion = self.dm
            thisBR = self.rinv
        # compute allowed decays to quarks
        self.quarks.set(self.mass - self.helper.mpi)
        theQuarks = self.quarks.get()
        bfQuarks = thisBR/float(len(theQuarks))
        lines = [
            '{:d}:addChannel = 1 {:g} 101 {} {:d} -{:d}'.format(self.id,bfQuarks,darkPion,q.id,q.id) for q in theQuarks
        ]
        return lines

    def darkRhoSimpDecay(self):
        if self.mass > 2*self.helper.mpi:
            # no rinv in this case, just rho -> pi pi
            self.rinv = None
            return self.darkRho2BodySimpDecay()
        else:
            return self.darkRho3BodySimpDecay()

    def darkRho2BodySimpDecay(self):
        decay_args = []
        if self.id==4900113:
            decay_args = [4900211, -4900211]
        elif self.id==4900213:
            decay_args = [4900111, 4900211]
        lines = ['{:d}:addChannel = 1 1 101 {:d} {:d}'.format(self.id,decay_args[0],decay_args[1])]
        return lines

    def darkRho3BodySimpDecay(self):
        # in complete model, rho 3-body decays correspond to stable pions (in terms of flavor)
        # while other rhos decay fully visibly (qq)
        # therefore, in simplified model, two decay types:
        # rho -> pi qq with BR = rinv
        # rho -> qq with BR = 1-rinv
        lines = self.democraticDecay() + self.darkRho3BodyDecay()
        return lines

class hvSpectrum():
    def __init__(self, name, helper):
        self.customLines = []
        self.darkHadrons = []
        self.helper = helper

        if not hasattr(self, name+'Spectrum'):
            raise ValueError("unknown spectrum {}".format(name))
        getattr(self, name+'Spectrum')()

    # helper for common dark quark/hadron lines in simple setup
    def quarkLines(self):
        return [
            # fermionic dark quark
            '4900101:m0 = {:g}'.format(self.helper.mq),
            # define missing antiparticles
            '4900111:antiName = pivDiagbar',
            '4900113:antiName = rhovDiagbar',
        ]

    # helper for common dark quark/hadron lines in separateFlav setup
    def quarkLinesSeparate(self):
        lines = [
            'HiddenValley:separateFlav = on',
            # disable eta prime production: Nf^2-1 accessible states
            'HiddenValley:probKeepEta1 = 0',
        ]
        # for separateFlav=on, set masses of all the dark quarks
        for i in range(1, self.helper.Nf+1):
            lines.append('490010{:d}:m0 = {:g}'.format(i, self.helper.mq))

        return lines

    # helper for common invisible particles
    def dmForRinv(self):
        return [
            darkHadron(self.helper,id=51,mass=0.0,decay='stable',props=['isResonance = false'],placeholder=True),
            darkHadron(self.helper,id=52,mass=0.0,decay='stable',props=['isResonance = false'],placeholder=True),
            darkHadron(self.helper,id=53,mass=0.0,decay='stable',props=['isResonance = false'],placeholder=True),
        ]

    def cmsSpectrum(self):
        self.customLines = self.quarkLines()
        self.darkHadrons = self.dmForRinv() + [
            darkHadron(self.helper,id=4900111,mass=self.helper.mpi,decay='massInsertion',rinv=self.helper.rinv,dm=51),
            darkHadron(self.helper,id=4900211,mass=self.helper.mpi,decay='massInsertion',rinv=self.helper.rinv,dm=51),
            darkHadron(self.helper,id=4900113,mass=self.helper.mrho,decay='democratic',rinv=self.helper.rinv,dm=53),
            darkHadron(self.helper,id=4900213,mass=self.helper.mrho,decay='democratic',rinv=self.helper.rinv,dm=53),
        ]

    def snowmassSpectrum(self):
        self.customLines = self.quarkLines()
        self.darkHadrons = self.dmForRinv() + [
            darkHadron(self.helper,id=4900111,mass=self.helper.mpi,decay='massInsertion',rinv=self.helper.rinv,dm=51),
            darkHadron(self.helper,id=4900211,mass=self.helper.mpi,decay='stable'),
            darkHadron(self.helper,id=4900113,mass=self.helper.mrho,decay='darkRho2BodySimp'), # 3-body case not explored here
            darkHadron(self.helper,id=4900213,mass=self.helper.mrho,decay='darkRho2BodySimp'),
        ]

    def snowmass_cmslikeSpectrum(self):
        self.customLines = self.quarkLines()
        self.darkHadrons = self.dmForRinv() + [
            darkHadron(self.helper,id=4900111,mass=self.helper.mpi,decay='massInsertion',rinv=self.helper.rinv,dm=51),
            darkHadron(self.helper,id=4900211,mass=self.helper.mpi,decay='massInsertion',rinv=self.helper.rinv,dm=51),
            darkHadron(self.helper,id=4900113,mass=self.helper.mrho,decay='darkRho2BodySimp'), # 3-body case not explored here
            darkHadron(self.helper,id=4900213,mass=self.helper.mrho,decay='darkRho2BodySimp'),
        ]

    def fcdcSpectrum(self):
        self.customLines = self.quarkLinesSeparate()

        #meson names are pivij and rhovij, where i = j are the flavour-diagonal mesons
        #else i > j, with j representing the antiquark.
        #identity codes then are 4900ij1 for pseudoscalars and 4900ij3 for vectors.
        #An antimeson comes with an overall negative sign, and here i gives the antiquark.

        # Diagonal (same flavor) mesons unstable
        # Off diagonal states carrying a stable quark are stable, others unstable
        #Assume mrho > 2mpi

        hadronLines = []
        antiLines = []
        for i in range(1, self.helper.Nf+1):
            for j in range(1, self.helper.Nf+1):
                if i < j: continue
                pid_scalar = darkHadron.getDarkMeson(dq=i, adq=j, spin=0)
                pid_vector = darkHadron.getDarkMeson(dq=i, adq=j, spin=1)
                antiLines.extend([
                    f'{pid_scalar}:antiName = piv{i}{j}bar',
                    f'{pid_vector}:antiName = rhov{i}{j}bar',
                ])
                if i == j:
                    # diagonal scalar unstable
                    hadronLines.append(darkHadron(self.helper,id=pid_scalar,mass=self.helper.mpi,decay='massInsertion'))
                    # diagonal vector unstable; decays to scalars
                    hadronLines.append(darkHadron(self.helper,id=pid_vector,mass=self.helper.mrho,decay='darkRho'))
                else:
                    # only stable if carrying a stable quark... first Ns quarks are stable
                    if i <= self.helper.Ns or j<= self.helper.Ns:
                        hadronLines.append(darkHadron(self.helper,id=pid_scalar,mass=self.helper.mpi,decay='stable'))
                        hadronLines.append(darkHadron(self.helper,id=pid_vector,mass=self.helper.mrho,decay='darkRhoProtected'))
                    else:
                        hadronLines.append(darkHadron(self.helper,id=pid_scalar,mass=self.helper.mpi,decay='massInsertion'))
                        hadronLines.append(darkHadron(self.helper,id=pid_vector,mass=self.helper.mrho,decay='darkRho'))

        self.darkHadrons = hadronLines
        self.customLines += antiLines

    def fcdcSimpSpectrum(self):
        self.customLines = self.quarkLines()
        # for 3-body decay
        stableDarkPion = 4900210
        self.darkHadrons = self.dmForRinv() + [
            darkHadron(self.helper,id=stableDarkPion,mass=self.helper.mpi,decay='stable',props=['new = pivStable pivStablebar 1 0 0']),
            darkHadron(self.helper,id=4900111,mass=self.helper.mpi,decay='massInsertion',rinv=self.helper.rinv,dm=51),
            darkHadron(self.helper,id=4900211,mass=self.helper.mpi,decay='massInsertion',rinv=self.helper.rinv,dm=51),
            darkHadron(self.helper,id=4900113,mass=self.helper.mrho,decay='darkRhoSimp',rinv=self.helper.rinv,dm=stableDarkPion,auto_rinv=False),
            darkHadron(self.helper,id=4900213,mass=self.helper.mrho,decay='darkRhoSimp',rinv=self.helper.rinv,dm=stableDarkPion,auto_rinv=False),
        ]

class hvChannel():
    def __init__(self, name, helper):
        self.customLines = []
        self.helper = helper

        if not hasattr(self, name+'Channel'):
            raise ValueError("unknown channel {}".format(name))
        getattr(self, name+'Channel')()

    def sChannel(self):
        # leading order branching fraction calculation
        # factors of mMed/(12pi) divide out
        # correct branching fractions -> correct Zprime width and cross section
        quarks = quarklist()
        quarks.set(self.helper.mmed)
        theQuarks = quarks.get()
        NcSM = 3
        NfSM = len(theQuarks)
        Wq = NcSM*NfSM*(self.helper.gq**2)
        Wchi = self.helper.Nc*self.helper.Nf*(self.helper.gchi**2)
        Wtot = Wq + Wchi
        # get per-particle branching fractions
        Bq = Wq/Wtot/NfSM
        Bchi = Wchi/Wtot/self.helper.Nf
        # calculate total width
        Gtot = Wtot*self.helper.mmed/(12*math.pi)

        self.mediatorID = 4900023
        # cut off low mediator masses from low-momentum PDFs
        mSigma = 5 # following Pythia8 SLHA convention
        mMinMin = 50
        mMin = max(self.helper.mmed - Gtot*mSigma, mMinMin)
        mMax = self.helper.mmed + Gtot*mSigma
        self.customLines = [
            'HiddenValley:ffbar2Zv = on',
            # parameters for leptophobic Z'
            f'{self.mediatorID}:m0 = {self.helper.mmed:g}',
            f'{self.mediatorID}:mWidth = {Gtot:g}', # manual calculation
            f'{self.mediatorID}:mMin = {mMin:g}',
            f'{self.mediatorID}:mMax = {mMax:g}',
        ]

        # divide up Z' BF between the Nf quarks
        dark_quarks = []
        for i in range(1, self.helper.Nf+1):
            dq = f'490010{i}'
            dark_quarks.append(dq)
            if i==1: line = f'{self.mediatorID}:oneChannel = 1 {Bchi:3f} 102 {dq} -{dq}'
            else: line = f'{self.mediatorID}:addChannel = 1 {Bchi:3f} 102 {dq} -{dq}'
            self.customLines.append(line)

        # SM quark couplings needed to produce Zprime from pp initial state
        self.customLines.extend([
            f'{self.mediatorID}:addChannel = 1 {Bq:3f} 102 {quark.id} -{quark.id}' for quark in theQuarks
        ])

        # only save events with Zprime -> dark quarks
        self.customLines.extend([
            f'{self.mediatorID}:onMode = off',
            f'{self.mediatorID}:onIfAny = {" ".join(dark_quarks)}',
        ])

        # decouple t-channel mediator particles
        self.customLines.extend([
            '4900001:m0 = 10000',
            '4900002:m0 = 10000',
            '4900003:m0 = 10000',
            '4900004:m0 = 10000',
            '4900005:m0 = 10000',
            '4900006:m0 = 10000',
            '4900011:m0 = 10000',
            '4900012:m0 = 10000',
            '4900013:m0 = 10000',
            '4900014:m0 = 10000',
            '4900015:m0 = 10000',
            '4900016:m0 = 10000',
        ])

class baseHelper():
    @staticmethod
    def add_arguments(parser):
        pass

    @classmethod
    def build(cls,config_path):
        from magiconfig import ArgumentParser, MagiConfigOptions
        parser = ArgumentParser(config_options=MagiConfigOptions())
        cls.add_arguments(parser)
        args = parser.parse_args(['-C',config_path])
        helper = cls(args)
        return helper

    def __init__(self,args):
        for key,val in vars(args).items():
            setattr(self,key,val)

    def name(self):
        pass

    def metadata(self):
        return {}

    def getPythiaSettings(self):
        pass

    def getDelphesSettings(self,input):
        def add_neg(ids):
            return ids + [-1*id for id in ids]

        def pdg_lines(ids):
            return ["  add PdgCode {{{}}}".format(id) for id in ids]

        HVEnergyFractions = '\n'.join(["  add EnergyFraction {{{}}} {{0}}".format(id) for id in self.stableIDs])
        stableIDs_with_neg = add_neg(self.stableIDs)
        HVNuFilter = '\n'.join(pdg_lines(stableIDs_with_neg))
        darkHadronIDs_with_neg = add_neg(self.darkHadronIDs)
        HVDarkHadronFilter = '\n'.join(pdg_lines(darkHadronIDs_with_neg))

        with input.open() as infile:
            old_lines = Template(infile.read())
            new_lines = old_lines.safe_substitute(
                HVEnergyFractions = HVEnergyFractions,
                HVNuFilter = HVNuFilter,
                HVDarkHadronFilter = HVDarkHadronFilter,
            )
        return new_lines

class svjHelper(baseHelper):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--channel",type=str,required=True,help="production channel")
        parser.add_argument("--mmed",type=float,required=True,help="mediator mass [GeV]")
        parser.add_argument("--Nc",type=int,required=True,help="number of dark colors")
        parser.add_argument("--Nf",type=int,required=True,help="number of dark flavors")
        parser.add_argument("--Ns",type=int,default=None,help='number of sterile ("stable") dark quarks')
        parser.add_argument("--scale",type=float,required=True,help="dark force scale Lambda [GeV]")
        parser.add_argument("--mq",type=float,required=True,help="dark quark mass [GeV]")
        parser.add_argument("--mpi",type=float,required=True,help="dark pion mass [GeV]")
        parser.add_argument("--mrho",type=float,required=True,help="dark rho mass [GeV]")
        parser.add_argument("--pvector",type=float,required=True,help="probability of producing rho (vs. pion)")
        parser.add_argument("--rinv",type=float,default=None,help="invisible fraction")
        parser.add_argument("--spectrum",type=str,required=True,help="dark hadron spectrum scheme")
        parser.add_argument("--gq",type=float,default=True,help="Zprime quark coupling")
        parser.add_argument("--gchi",type=float,default=True,help="Zprime dark quark coupling")

    def __init__(self,args):
        super().__init__(args)

        # sanity checks
        if self.mrho is None: self.mrho = self.mpi
        if self.rinv is not None:
            if self.rinv<0 or self.rinv>1:
                raise ValueError(f'rinv {self.rinv} not allowed (0 <= rinv <= 1)')

        # set up production channel
        self.channelHelper = hvChannel(self.channel, self)
        self.channelLines = self.channelHelper.customLines
        self.mediatorID = self.channelHelper.mediatorID

        # set up spectrum
        self.spectrumHelper = hvSpectrum(self.spectrum, self)
        self.spectrumLines = self.spectrumHelper.customLines
        self.spectrumParticles = self.spectrumHelper.darkHadrons
        self.darkHadronIDs = [dh.id for dh in self.spectrumParticles if not dh.placeholder]
        self.darkHadronFinalIDs = [dh.id for dh in self.spectrumParticles if not dh.placeholder and 'darkRho' not in dh.decay]
        self.stableIDs = [dh.id for dh in self.spectrumParticles if dh.decay=='stable']

        # metadata tracking
        self.always_included = ["channel","mmed","Nc","Nf","scale","mq","mpi","mrho","pvector","spectrum","gq","gchi"]
        self.special_formats = {
            "channel": "{1}-{0}",
            "spectrum": "{}-{}",
        }

    def _param_name(self,param,form="{}-{:g}"):
        form = self.special_formats.get(param,"{}-{:g}")
        return form.format(param,getattr(self,param))

    def name(self):
        params = [self._param_name(p) for p in self.always_included]
        if self.rinv is not None:
            params.append(self._param_name("rinv"))
        if self.Ns is not None:
            params.append(self._param_name("Ns"))
        _name = '_'.join(params)
        return _name

    def metadata(self):
        metadict = {param:getattr(self,param) for param in self.always_included}
        metadict["stableIDs"] = self.stableIDs
        metadict["darkHadronIDs"] = self.darkHadronIDs
        metadict["darkHadronFinalIDs"] = self.darkHadronFinalIDs
        if self.rinv is not None:
            metadict["rinv"] = self.rinv
        return metadict

    def getPythiaSettings(self):
        lines_HV = [
            # other HV params
            'HiddenValley:Ngauge = {:d}'.format(self.Nc),
            # when Fv has spin 0, qv spin fixed at 1/2
            'HiddenValley:spinFv = 0',
            'HiddenValley:FSR = on',
            'HiddenValley:fragment = on',
            'HiddenValley:alphaOrder = 1',
            'HiddenValley:Lambda = {:g}'.format(self.scale),
            'HiddenValley:nFlav = {:d}'.format(self.Nf),
            'HiddenValley:probVector = {:g}'.format(self.pvector),
        ]
        lines_decay = [line for dh in self.spectrumParticles for line in dh.getLines()]

        lines = self.channelLines + lines_HV + self.spectrumLines + lines_decay
        return lines

class extHelper(baseHelper):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--card", type=str, required=True, help="external Pythia card")
        parser.add_argument("--stableIDs", type=int, nargs='*', default=[], help="list of stable PDG IDs")
        parser.add_argument("--darkHadronIDs", type=int, nargs='*', default=[], help="list of dark hadron PDG IDs")
        parser.add_argument("--darkHadronFinalIDs", type=int, nargs='*', default=[], help="list of final dark hadron PDG IDs")

    def __init__(self,args):
        super().__init__(args)
        self.card = Path(self.card).absolute()

    def name(self):
        return self.card.stem

    def metadata(self):
        metadict = {}
        metadict["stableIDs"] = self.stableIDs
        metadict["darkHadronIDs"] = self.darkHadronIDs
        metadict["darkHadronFinalIDs"] = self.darkHadronFinalIDs
        return metadict

    def getPythiaSettings(self):
        with self.card.open() as infile:
            lines = [line.rstrip() for line in infile]
            return lines
