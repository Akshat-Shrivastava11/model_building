import math, os
from string import Template
from pathlib import Path

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

# classes for helper

class quark(object):
    def __init__(self,id,mass):
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
        self.qlist = [
            quark(2,0.0023), # up
            quark(1,0.0048), # down
            quark(3,0.095),  # strange
            quark(4,1.275),  # charm
            quark(5,4.18),   # bottom
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
    def __init__(self, id, mass, decay, props=[], rinv=None, dm=None, decay_args=None):
        self.id = id
        self.mass = mass
        self.decay = decay
        self.decay_args = decay_args
        self.props = props
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

    def getLines(self):
        lines = ['{:d}:m0 = {:g}'.format(self.id,self.mass)]
        lines += ['{:d}:'.format(self.id)+prop for prop in self.props]
        if self.rinv is not None:
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

    def darkPionDecay(self):
        lines = ['{:d}:addChannel = 1 1 101 {:d} {:d}'.format(self.id,self.decay_args[0],self.decay_args[1])]
        return lines

class hvSpectrum():
    def populate(self, name, helper):
        if not hasattr(self, name+'Spectrum'):
            raise ValueError("unknown spectrum {}".format(name))
        return getattr(self, name+'Spectrum')(helper)

    # helper for common invisible particles
    def dmForRinv(self):
        return [
            darkHadron(51,0.0,'stable',props=['isResonance = false']),
            darkHadron(52,0.0,'stable',props=['isResonance = false']),
            darkHadron(53,0.0,'stable',props=['isResonance = false']),
        ]

    def cmsSpectrum(self, helper):
        return self.dmForRinv() + [
            darkHadron(4900111,helper.mpi,'massInsertion',rinv=helper.rinv,dm=51),
            darkHadron(4900211,helper.mpi,'massInsertion',rinv=helper.rinv,dm=51),
            darkHadron(4900113,helper.mrho,'democratic',rinv=helper.rinv,dm=53),
            darkHadron(4900213,helper.mrho,'democratic',rinv=helper.rinv,dm=53),
        ]

    def snowmassSpectrum(self, helper):
        return self.dmForRinv() + [
            darkHadron(4900111,helper.mpi,'massInsertion',rinv=helper.rinv,dm=53),
            darkHadron(4900211,helper.mpi,'stable'),
            darkHadron(4900113,helper.mpi,'darkPion',decay_args=[4900211,-4900211]),
            darkHadron(4900213,helper.mpi,'darkPion',decay_args=[4900111,4900211]),
        ]

    def snowmass_cmslikeSpectrum(self, helper):
        return self.dmForRinv() + [
            darkHadron(4900111,helper.mpi,'massInsertion',rinv=helper.rinv,dm=53),
            darkHadron(4900211,helper.mpi,'massInsertion',rinv=helper.rinv,dm=53),
            darkHadron(4900113,helper.mpi,'darkPion',decay_args=[4900211,-4900211]),
            darkHadron(4900213,helper.mpi,'darkPion',decay_args=[4900111,4900211]),
        ]

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
        stableIDs_with_neg = self.stableIDs + [-1*id for id in self.stableIDs]
        HVEnergyFractions = '\n'.join(["  add EnergyFraction {{{}}} {{0}}".format(id) for id in self.stableIDs])
        HVNuFilter = '\n'.join(["  add PdgCode {{{}}}".format(id) for id in stableIDs_with_neg])

        with input.open() as infile:
            old_lines = Template(infile.read())
            new_lines = old_lines.safe_substitute(
                HVEnergyFractions = HVEnergyFractions,
                HVNuFilter = HVNuFilter,
            )
        return new_lines

class svjHelper(baseHelper):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--channel",type=str,required=True,help="production channel")
        parser.add_argument("--mmed",type=float,required=True,help="mediator mass [GeV]")
        parser.add_argument("--Nc",type=int,required=True,help="number of dark colors")
        parser.add_argument("--Nf",type=int,required=True,help="number of dark flavors")
        parser.add_argument("--scale",type=float,required=True,help="dark force scale Lambda [GeV]")
        parser.add_argument("--mq",type=float,required=True,help="dark quark mass [GeV]")
        parser.add_argument("--mpi",type=float,required=True,help="dark pion mass [GeV]")
        parser.add_argument("--mrho",type=float,required=True,help="dark rho mass [GeV]")
        parser.add_argument("--pvector",type=float,required=True,help="probability of producing rho (vs. pion)")
        parser.add_argument("--rinv",type=float,default=None,help="invisible fraction")
        parser.add_argument("--spectrum",type=str,required=True,help="dark hadron spectrum scheme")

    def __init__(self,args):
        super().__init__(args)

        # sanity checks
        if self.mrho is None: self.mrho = self.mpi
        if self.rinv is not None:
            if self.rinv<0 or self.rinv>1:
                raise ValueError(f'rinv {self.rinv} not allowed (0 <= rinv <= 1)')

        # make this width more realistic based on couplings?
        self.mmin = self.mmed-1
        self.mmax = self.mmed+1

        # set up spectrum
        self.spectrumParticles = hvSpectrum().populate(self.spectrum, self)
        self.stableIDs = [dh.id for dh in self.spectrumParticles if dh.decay=='stable']

        # metadata tracking
        self.always_included = ["channel","mmed","Nc","Nf","scale","mq","mpi","mrho","pvector","spectrum"]
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
        _name = '_'.join(params)
        return _name

    def metadata(self):
        metadict = {param:getattr(self,param) for param in self.always_included}
        metadict["stableIDs"] = self.stableIDs
        if self.rinv is not None:
            metadict["rinv"] = self.rinv
        return metadict

    def getPythiaSettings(self):
        lines_channel = {
            's': [
                'HiddenValley:ffbar2Zv = on',
                # parameters for leptophobic Z'
                '4900023:m0 = {:g}'.format(self.mmed),
                '4900023:mMin = {:g}'.format(self.mmin),
                '4900023:mMax = {:g}'.format(self.mmax),
                '4900023:mWidth = 0.01',
                '4900023:oneChannel = 1 0.982 102 4900101 -4900101',
                # SM quark couplings needed to produce Zprime from pp initial state
                '4900023:addChannel = 1 0.003 102 1 -1',
                '4900023:addChannel = 1 0.003 102 2 -2',
                '4900023:addChannel = 1 0.003 102 3 -3',
                '4900023:addChannel = 1 0.003 102 4 -4',
                '4900023:addChannel = 1 0.003 102 5 -5',
                '4900023:addChannel = 1 0.003 102 6 -6',
                # decouple
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
            ],
        }
        if self.channel not in lines_channel:
            raise ValueError("Unknown channel {}".format(self.channel))
        lines_HV = [
            # fermionic dark quark
            '4900101:m0 = {:g}'.format(self.mq),
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

        lines = lines_channel[self.channel] + lines_HV + lines_decay
        return lines

class extHelper(baseHelper):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--card", type=str, required=True, help="external Pythia card")
        parser.add_argument("--stableIDs", type=int, nargs='*', default=[], help="list of stable PDG IDs")

    def __init__(self,args):
        super().__init__(args)
        self.card = Path(self.card).absolute()

    def name(self):
        return self.card.stem

    def metadata(self):
        metadict = {}
        metadict["stableIDs"] = self.stableIDs
        return metadict

    def getPythiaSettings(self):
        with self.card.open() as infile:
            lines = [line.rstrip() for line in infile]
            return lines
