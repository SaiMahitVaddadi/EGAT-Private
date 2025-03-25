import subprocess,sys,os,traceback,Auto3D
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign,rdDistGeom,rdMolAlign,rdMolDescriptors,Descriptors3D,rdFreeSASA,MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize

from scipy.spatial import ConvexHull

try:
    from openff.toolkit.topology import Molecule
    from openff.toolkit.utils import get_data_file_path
    from openff.interchange import Interchange
    from openff.forcefields import ForceField
except:
    pass

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except:
    pass

try:
    from molvs.tautomer import TautomerEnumerator
except:
    pass

from Auto3D.auto3D import options, main
from Auto3D.ASE.thermo import calc_thermo
from Auto3D.ASE.geometry import opt_geometry
from Auto3D.tautomer import get_stable_tautomers

from ..egat.molmatdesc import MolMatDesc


from ase.io import read
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from rdkit.Chem import rdmolfiles
from dataclasses import dataclass
from typing import Optional
from pyscf import gto, scf

def print_failure_causes(counts):
    for i,k in enumerate(rdDistGeom.EmbedFailureCauses.names):
        print(k,counts[i])
    # in v2022.03.1 two names are missing from `rdDistGeom.EmbedFailureCauses`:
    print('LINEAR_DOUBLE_BOND',counts[i+1])
    print('BAD_DOUBLE_BOND_STEREO',counts[i+2])    



@dataclass
class ConformerGeneratorParams:
    nconfs: int = 1
    method: Optional[str] = None
    seed: int = 1
    verbose: bool = False
    theory: str = 'UFF'
    tmp: str = 'tmp'
    window: Optional[int] = None
    engine: str = 'ANI2x'
    gpu: bool = False
    enumerate_tautomer: bool = True
    tauto_engine: str = "rdkit"
    pKaNorm: bool = True
    enumerate_isomer: bool = True
    max_confs: Optional[int] = None
    patience: int = 1000
    opt_steps: int = 5000
    convergence_threshold: float = .003
    threshold: float = .3
    genmode: str = 'confgen'
    geomopt: bool = False
    thermo: bool = False
    tauto_k: Optional[int] = None
    tauto_window: Optional[int] = None
    opt_tol: float = .0002


    infile: str = 'input.xyz'
    charge: int = 0
    uhf: int = 0
    solvation: Optional[str] = None
    optlev: Optional[str] = None
    sampling: Optional[str] = None
    sites: Optional[str] = None
    mdlen: Optional[str] = None
    shake: Optional[str] = None
    tstep: Optional[str] = None
    mddump: Optional[str] = None
    vbdump: Optional[str] = None
    zsort: bool = False
    genzsort: bool = True
    norotmd: bool = False
    tnmd: Optional[str] = None
    mrest: Optional[str] = None
    hflip: bool = True
    maxflip: Optional[str] = None
    gcspeed: Optional[str] = None
    props: Optional[str] = None
    origin: bool = True
    keepdir: bool = False
    noreftopo: bool = False
    noopt: bool = False
    wall: Optional[str] = None
    scthr: Optional[str] = None
    ssthr: Optional[str] = None
    trange: Optional[str] = None
    ptot: Optional[str] = None
    fscal: Optional[str] = None
    sthr: Optional[str] = None
    ithr: Optional[str] = None
    cinp: Optional[str] = None
    cbonds: Optional[str] = None
    cheavy: Optional[str] = None
    clight: Optional[str] = None
    fc: Optional[str] = None
    mdopt: Optional[str] = None
    screen: Optional[str] = None
    rrhoav: Optional[str] = None
    thermo: Optional[str] = None
    nanoreactor: Optional[str] = None
    solvent: Optional[str] = None
    testtopo: Optional[str] = None
    inputfile: Optional[str] = None

    tmp: str = 'tmp'
    solvent: str = 'h2o'
    run: str = 'grow'
    nsolv: Optional[int] = None
    nopreopt: bool = False
    keepdir: bool = False
    gfn1: bool = False
    gfn2: bool = False
    gfnff: bool = False
    samerand: bool = False
    chrg: Optional[int] = None
    uhf: Optional[int] = None
    wscal: Optional[float] = None
    fixsolute: bool = False
    nofix: bool = False
    xtbiff: bool = False
    normdock: bool = False
    directed: Optional[str] = None
    qcgmtd: bool = False
    ncimtd: bool = False
    mtd: bool = False
    md: bool = False
    enslvl: Optional[str] = None
    mdlen: Optional[str] = None
    mddump: Optional[str] = None
    tstep: Optional[str] = None
    vbdump: Optional[str] = None
    norotmd: bool = False
    tnmd: Optional[str] = None
    mreset: Optional[str] = None
    fin_opt_gfn2: bool = False
    nocff: bool = False
    esolv: bool = False
    nclus: Optional[int] = None
    freqlvl: Optional[str] = None
    freqscal: Optional[float] = None

    ewin: Optional[float] = None
    rthr: Optional[float] = None
    ethr: Optional[float] = None
    bthr: Optional[float] = None
    pthr: Optional[float] = None
    nmr: bool = False
    eqv: bool = False
    athr: Optional[float] = None
    temp: Optional[float] = None
    esort: bool = False
    nowr: bool = False
    subrmsd: bool = False
    notopo: Optional[str] = None
    cluster: Optional[str] = None
    pccap: Optional[float] = None
    nopcmin: bool = False
    pcaex: Optional[float] = None

    speed: Optional[str] = 'fast'
    children: int =5
    mutability: int =5
    converge: Optional[int]=None
    score: Optional[float]=None

    evaluate: bool = True
    ff: str ='uff'
    steps: int =1500
    crit:float =1e-06
    search: str='sd'
    cutoff: bool=True
    vdw: int=6
    rele: int=10
    freq: int=10
    
    

class RDKitFunctions:
    def __init__(self,egatecule:MolMatDesc):
        self.egatecule = egatecule
        

    def _Case1(self,seed,attempts):
        failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol)
        if failed == -1:
            failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,maxAttempts=attempts,randomSeed=seed)
            if failed == -1:
                failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,maxAttempts=attempts,randomSeed=seed,useRandomCoords=True)
                if failed == -1:
                    failed = self.EmbedwithOpenBabel(seed)
        return failed

    def _Case2(self,seed,attempts):
        failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,maxAttempts=attempts,randomSeed=seed)
        if failed == -1:
            failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,maxAttempts=attempts,randomSeed=seed,useRandomCoords=True)
            if failed == -1:
                failed = self.EmbedwithOpenBabel(seed)
        return failed

    def _Case3(self,seed,attempts):
        failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,maxAttempts=attempts,randomSeed=seed,useRandomCoords=True)
        if failed == -1:
            failed = self.EmbedwithOpenBabel(seed)
        return failed

    def EmbedMoleculewithMethod(self,seed,method,verbose):
        try:
            embedder = getattr(rdDistGeom,method,None)
        except:
            embedder = getattr(rdDistGeom,None,None)
        way = embedder(randomSeed=seed,trackFailures = True)
        failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,way)
        if not verbose: 
            print_failure_causes(way.GetFailureCounts())


    def Initialize(self,method=None,seed=1,verbose=True):
        if method == None:
            try:
                failed = self._Case1(seed,5000)
            except:
                try:
                    failed = self._Case2(seed,5000)
                except:
                    try:
                        failed = self._Case3(seed,5000)
                    except:
                        try:
                            failed = self.EmbedwithOpenBabel(seed)
                        except Exception as e:
                            raise ValueError(f'Cannot embed molecule. This is the error. \n {e}')
        else:
            self.EmbedMoleculewithMethod(seed,method,verbose)


    def ConverttoSDF(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            sdf_path = os.path.join(tmpdirname, 'molecule.sdf')
            writer = Chem.SDWriter(sdf_path)
            writer.write(self.egatecule.new_mol)
            writer.close()
        return sdf_path


    def _Case4(self,seed,attempts):
        if failed == -1:
            failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,maxAttempts=attempts,randomSeed=seed)
            if failed == -1:
                failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,maxAttempts=attempts,randomSeed=seed,useRandomCoords=True)
        return failed

    def EmbedwithOpenBabel(self,seed):
        command = f'obabel -isdf {self.ConverttoSDF()} -osdf --gen3d'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        self.egatecule.new_mol = Chem.MolFromMolBlock(result.stdout,sanitize=False)
        failed = rdDistGeom.EmbedMolecule(self.egatecule.new_mol,randomSeed=seed)
        return self._Case4(seed,5000)

    def LoadasMolObject(self,outfile):
        mol = Chem.SDMolSupplier(outfile, removeHs=False)[0]
        for mol in Chem.SDMolSupplier(outfile, removeHs=False):
            if mol is not None:
                mol.AddConformer(mol.GetConformer(), assignId=True)
        self.egatecule.new_mol = mol
        self.conf_ids = [conf.GetId() for conf in self.egatecule.new_mol.GetConformers()]
    

class RDKitPostProcessing:
    def __init__(self,egatecule:MolMatDesc):
        self.egatecule = egatecule

    def ConformerCheck(self):
        self.num_conf = self.egatecule.new_mol.GetNumConformers()
        try:
            self.threedcheck = self.egatecule.new_mol.GetConformer().Is3D()
        except:
            self._initialize(seed=self.seed+1)
            self.num_conf = self.egatecule.new_mol.GetNumConformers()
            self.threedcheck = self.egatecule.new_mol.GetConformer().Is3D()
            
    def AlignConformerstoReference(self):
        rms_list = []
        AllChem.AlignMolConformers(self.egatecule.new_mol, RMSlist=rms_list)
        return rms_list

    def OptimizeConformer(self):
        function = getattr(AllChem,f'{self.params.theory}OptimizeMolecule',None)
        if self.conf_ids != 1:
            for id in self.conf_ids:
                function(self.egatecule.new_mol,confId=id)
        else:
            function(self.egatecule.new_mol)
        
class RDKitScoring:
    def __init__(self,egatecule:MolMatDesc):
        self.egatecule = egatecule


    def CrippenProperties(self):
        try:
            self.contribs = [rdMolDescriptors._CalcCrippenContribs(mol) for mol in self.egatecule.new_mol]
            self.ref_contrib = self.crippen_contribs[0]
            self.prob_contribs = self.crippen_contribs[1:]
        except:
            self.contribs = rdMolDescriptors._CalcCrippenContribs(self.egatecule.new_mol)

    def MMFFProperties(self):
        try: 
            self.contribs = [AllChem.MMFFGetMoleculeProperties(mol) for mol in self.egatecule.new_.mol]
            self.ref_contrib = self.mmff_params[0]
            self.prob_contrib = self.mmff_params[1:]
        except:
            self.mmff_params = AllChem.MMFFGetMoleculeProperties(self.egatecule.new_mol)
    
    def Score(self,mode='Crippen'):
        function = getattr(self,f'{mode}Properties',None)
        function()
        try:
            scores = []
            base = Chem.Mol(self.egatecule.new_mol,confId=self.conf_ids[0])
            for _ in self.conf_ids:
                if _ > self.conf_ids[0]:
                    cp = Chem.Mol(self.egatecule.new_mol,confId=_)
                    if mode == 'crippen':
                        o3a = rdMolAlign.GetCrippenO3A(base,cp,self.prob_contrib[_],self.ref_contrib,_,0)
                    elif mode == 'mmff':
                        o3a = rdMolAlign.GetCrippenO3A(base,cp,self.prob_contrib[_],self.ref_contrib,_,0)
                    o3a.Align()
                    scores.append(o3a.Score())
            self.scores = scores 
        except:
            self.scores = None

class A3DHelpers:
    def __init__(self, egatecule,params):
        self.params = params

    def ConverttoSDF(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            sdf_path = os.path.join(tmpdirname, 'molecule.sdf')
            writer = Chem.SDWriter(sdf_path)
            writer.write(self.egatecule.new_mol)
            writer.close()
        return sdf_path
    
    def create_temp_dir(self):
        infile = self.ConverttoSDF()
        outfolder = tempfile.mkdtemp()

    def setup(self):
        if self.params.window == None:
            args = options(self.params.infile, k=self.params.confs, job_name=self.params.outfolder,enumerate_tautomer=self.params.enumerate_tautomer, 
                           tauto_engine=self.params.tauto_engine,pKaNorm=self.params.pKaNorm,enumerate_isomer=self.params.enumerate_isomer,
                           max_confs=self.params.max_confs,optimizing_engine=self.params.engine, use_gpu=self.params.gpu,patience=self.params.patience
                           ,opt_steps=self.params.opt_steps,convergence_threshold=self.params.convergence_threshold,threshold=self.params.threshold)
        else:
            args = options(self.params.infile, window=self.params.window, job_name=self.params.outfolder,enumerate_tautomer=self.params.enumerate_tautomer, 
                           tauto_engine=self.params.tauto_engine,pKaNorm=self.params.pKaNorm,enumerate_isomer=self.params.enumerate_isomer,max_confs=self.params.max_confs,
                           optimizing_engine=self.params.engine, use_gpu=self.params.gpu,patience=self.params.patience,opt_steps=self.params.opt_steps,
                           convergence_threshold=self.params.convergence_threshold,threshold=self.params.threshold)
        
        return args

class BabelFunctions:
    def __init__(self,egatecule:MolMatDesc,params):
        self.egatecule = egatecule
        self.params = params

    def ConverttoSDF(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            sdf_path = os.path.join(tmpdirname, 'molecule.sdf')
            writer = Chem.SDWriter(sdf_path)
            writer.write(self.egatecule.new_mol)
            writer.close()
        return sdf_path

    def Gen3DOpenBabel(self):
        infile = self.ConverttoSDF()
        command = ['obabel', '-isdf',self.smiles, '-osdf', '--gen3d',self.params.speed]
        result = subprocess.run(command)
        self.egatecule.new_mol = Chem.MolFromMolBlock(result.stdout,sanitize=False)
    

    def CreateFilesOpenBabel(self):
        infile = self.ConverttoSDF()
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = os.path.join(tmpdirname, 'new_outfile.sdf')
        return infile,outfile

    def OpenBabelGenerationByAlgorithm(self):
        infile,outfile = self.CreateFilesOpenBabel()
        command = ['obabel', infile + '"', '-O', outfile]
        command += ['--conformer', '--children',str(self.params.children),'--mutability',str(self.params.mutability),'--nconf',str(self.params.n_confs)]
        if self.params.converge is not None:
            command += ['--converge',str(self.params.converge)]
        if self.params.score is not None: command += ['--score',self.params.score]
        result = subprocess.run(command)
        self.LoadasMolObject(outfile)
        self.RemoveFiles(infile,outfile)
    
    def OpenBabelGenerationByForceField(self,system='random',ff='uff'):
        infile,outfile = self.CreateFilesOpenBabel()
        command = ['obabel', infile + '"', '-O', outfile]
        command += ['--conformer',f'--{system}','--ff',ff]
        result = subprocess.run(command)
        self.LoadasMolObject(outfile)
        self.RemoveFiles(infile,outfile)

    def OpenBabelGenerationByConfab(self,rmsd=.5,ecutoff=50,maxconfs=1000000,original=False):
        infile,outfile = self.CreateFilesOpenBabel()
        command = ['obabel', infile + '"', '-O', outfile]
        command += ['--confab',f'--rcutoff',rmsd,'--ecutoff',ecutoff,'--conf',maxconfs]
        if original: command += ['--original']
        result = subprocess.run(command)
        self.LoadasMolObject(outfile)
        self.RemoveFiles(infile,outfile)
    
    def OpenBabelEnergyMinimization(self,infile,outfile,evaluate=True,ff='uff',steps=1500,crit=1e-06,search='sd',cutoff=True,vdw=6,rele=10,freq=10):
        infile,outfile = self.CreateFilesOpenBabel()
        command = ['obabel', infile + '"', '-O', outfile]
        if evaluate: command += ['--energy', '--append','"Energy"']
        command += ['--minimize', '--steps',steps,'--crit',crit,f'--{search}']
        if cutoff: command += ['--cutoff', '--rvdw',vdw,'--rele',rele,'--freq',freq]
        result = subprocess.run(command)
        self.LoadasMolObject(outfile)
        self.RemoveFiles(infile,outfile)
    
    def OpenBabelAlignment(self,infile,outfile,smarts):
        command = ['obabel', infile + '"', '-O', outfile,'-s', smarts,'--align']
        result = subprocess.run(command)
        self.LoadasMolObject(outfile)
        self.RemoveFiles(infile,outfile)
    
    def RemoveFiles(self,infile,outfile):
        os.remove(infile)
        os.remove(outfile)

class CRESTFunctions:
    def __init__(self,egatecule:MolMatDesc,arguments):
        self.egatecule = egatecule
        self.params = arguments

    def ConverttoSDF(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            sdf_path = os.path.join(tmpdirname, 'molecule.sdf')
            writer = Chem.SDWriter(sdf_path)
            writer.write(self.egatecule.new_mol)
            writer.close()
        return sdf_path
    
    def CreateFilesOpenBabel(self):
        infile = self.ConverttoSDF()
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = os.path.join(tmpdirname, 'new_outfile.sdf')
        return infile,outfile


    def CreateCRESTInput(self):
        infile,_ = self.CreateFilesOpenBabel()
        outfolder = tempfile.mkdtemp()
        return infile,outfolder
    
    def CreateCRESTCommand(self):
        infile,outfolder = self.CreateCRESTInput(self)

        

        command = f"crest {infile} --{self.params.theory} --scratch {outfolder} "

        if self.params.optlev != None: command += f"--opt {self.params.optlev} "
        if self.params.solvation != None: command += f"--{self.params.solvation} {self.params.solvent}"
        if self.params.sampling != None:
            samplingdict = dict()
            samplingdict['MF-MD-GC'] = 'v1'
            samplingdict['MTD-GC'] = 'v2i'
            samplingdict['iMTD-GC'] = 'v3'
            samplingdict['iMTD-sMTD'] = 'v4'
            samplingdict['Entropy'] = 'entropy'
            
            command += f"--{samplingdict[self.params.sampling]} "
        
        if self.params.sites != None: command += f"--{self.params.ites} "
        if self.params.mdlen != None: command += f"--len x={self.params.mdlen} "
        if self.params.shake != None: command += f"--shake {self.params.shake} "
        if self.params.tstep != None: command += f"--tstep {self.params.tstep} "
        if self.params.mddump != None: command += f"--mddump {self.params.mddump} "
        if self.params.vbdump != None: command += f"--vbdump {self.params.vbdump} "

        if self.params.zsort: command += '--zs '
        if not self.params.genzsort: command += '--nocross '
        if self.params.norotmd: command += '--norotmd '
        if self.params.tnmd: command += f'--tnmd {self.params.tnmd} '
        if self.params.mrest: command += f'--mrest {self.params.mrest} '
        if not self.params.hflip: command += f'--noflip '
        if self.params.maxflip: command += f'--maxflip {self.params.maxflip} '
        if self.params.gcspeed: command += f'--{self.params.gcspeed} '

        if self.params.props: command += f'--prop {self.params.props}'
        if self.params.keepdir: command += f'--keepdir '
        if self.params.noreftopo: command += f'--noreftopo '
        if self.params.noopt: command += f'--noopt '
        if self.params.wall != None: command += f'--{self.params.wall} '
        if self.params.scthr != None: command += f'--scthr {self.params.scthr} '
        if self.params.ssthr != None: command += f'--ssthr {self.params.ssthr} '
        if self.params.trange != None: command += f'--trange {self.params.trange[0]} {self.params.trange[1]} {self.params.trange[2]} '
        if self.params.ptot != None: command += f'--ptot {self.params.ptot}'
        if self.params.fscal != None: command += f'--fscal {self.params.fscal} '
        if self.params.sthr != None: command += f'--sthr {self.params.sthr} '
        if self.params.ithr != None: command += f'--ithr {self.params.ithr} '

        if self.params.cinp != None: command += f'--cinp {self.params.cinp} '
        if self.params.cbonds != None: 
            if self.params.cbonds == False:
                command += '--nocbonds '
            else:
                command += f'--cbonds {self.params.cbonds} '
        

        if self.params.cheavy != None: command += f'--cheavy {self.params.cheavy} '
        if self.params.clight != None: command += f'--clight {self.params.clight} '
        if self.params.fc != None: command += f'--fc {self.params.fc} '
        if self.params.mdopt != None: command += f'--mdopt {self.params.mdopt} '
        if self.params.screen != None: command += f'--screen {self.params.screen} '
        if self.params.rrhoav != None: command += f'--rrhoav {self.params.rrhoav} '
        if self.params.thermo != None: command += f'--thermo {self.params.thermo} '
        if self.params.testtopo != None: command += f'--testtopo {self.params.testtopo} '
        if self.params.nanoreactor: command += '--nanoreactor '
        if self.params.inputfile != None: command += f'--input {self.params.inputfile}'
        return infile,outfolder,command
    
    def CreateQCGCommand(self):
        infile,outfolder = self.CreateCRESTInput(self)
        command = f"crest {infile} --scratch {os.path.join(self.params.tmp,'crest-qcg')} -qcg {self.params.solvent} --{self.params.run} "

        if self.params.run == 'grow':
            if self.params.nsolv: command += f'--nsolv {self.params.nsolv} '
            if self.params.nopreopt: command += '--nopreopt '
            if self.params.keepdir: command += '--keepdir '
            if self.params.gfn1: command += '--gfn1 '
            if self.params.gfn2: command += '--gfn2 '
            if self.params.gfnff: command += '--gfnff '
            if self.params.samerand: command += '--samerand '
            if self.params.chrg: command += f'--chrg {self.params.chrg} '
            if self.params.uhf: command += f'--uhf {self.params.uhf} '
            if self.params.wscal: command += f'--wscal {self.params.wscal} '
            if self.params.fixsolute: command += '--fixsolute '
            if self.params.nofix: command += '--nofix '
            if self.params.xtbiff: command += '--xtbiff '
            if self.params.normdock: command += '--normdock '
            if self.params.directed: command += f'--directed {self.params.directed} '
            print('Under construction')
        elif self.params.run == 'ensemble':
            if self.params.qcgmtd: command += '--qcgmtd '
            if self.params.ncimtd: command += '--ncimtd '
            if self.params.mtd: command += '--mtd '
            if self.params.md: command += '--md '
            if self.params.enslvl: command += f'--enslvl {self.params.enslvl} '
            if self.params.mdlen: command += f'--mdlen {self.params.mdlen} '
            if self.params.mddump: command += f'--mddump {self.params.mddump} '
            if self.params.tstep: command += f'--tstep {self.params.tstep} '
            if self.params.vbdump: command += f'--vbdump {self.params.vbdump} '
            if self.params.norotmd: command += '--norotmd '
            if self.params.tnmd: command += f'--tnmd {self.params.tnmd} '
            if self.params.mreset: command += f'--mreset {self.params.mreset} '
            if self.params.fin_opt_gfn2: command += '--fin_opt_gfn2 '
            print('Under construction')
        elif self.params.run == 'solvation':
            if self.params.nocff: command += '--nocff '
            if self.params.esolv: command += '--esolv '
            if self.params.nclus: command += f'--nclus {self.params.nclus} '
            if self.params.freqlvl: command += f'--freqlvl {self.params.freqlvl} '
            if self.params.freqscal: command += f'--freqscal {self.params.freqscal} '
            print('Under construction')


        return infile,outfolder,command

    def CreateCREGENcommand(self):

        infile, outfile = self.CreateFilesOpenBabel(self)
        command = f"crest {infile} --cregen {outfile} "
        #crest struc.xyz --cregen input-ensemble.xyz

        if self.params.ewin: command += f'--ewin {self.params.ewin} '
        if self.params.rthr: command += f'--rthr {self.params.rthr} '
        if self.params.ethr: command += f'--ethr {self.params.ethr} '
        if self.params.bthr: command += f'--bthr {self.params.bthr} '
        if self.params.pthr: command += f'--pthr {self.params.pthr} '
        if self.params.nmr: command += '--nmr '
        if self.params.eqv: command += '--eqv '
        if self.params.athr: command += f'--athr {self.params.athr} '
        if self.params.temp: command += f'--temp {self.params.temp} '
        if self.params.esort: command += '--esort '
        if self.params.nowr: command += '--nowr '
        if self.params.subrmsd: command += '--subrmsd '
        if self.params.notopo: command += f'--notopo {self.params.notopo} '
        if self.params.cluster: command += f'--cluster {self.params.cluster} '
        if self.params.pccap: command += f'--pccap {self.params.pccap} '
        if self.params.nopcmin: command += '--nopcmin '
        if self.params.pcaex: command += f'--pcaex {self.params.pcaex} '
        return infile,outfile,command


    def ParseCRESTOutput(self,outfolder):
        """
        Parse the output from CREST to extract conformers.
        """
        # Assuming CREST outputs a file named 'crest_conformers.xyz'
        crest_output_file = f'{outfolder}/crest_conformers.xyz'

        if not os.path.exists(crest_output_file):
            raise FileNotFoundError(f"CREST output file {crest_output_file} not found")

        # Parse the XYZ file to extract conformers
        conformers = read(crest_output_file, index=':')
        return conformers

    def ProcessCREST(self):
        if self.params.crest == 'crest':
            infile,outfile,command = self.CreateCRESTCommand()
        elif self.params.crest == 'qcg':
            infile,outfile,command = self.CreateQCGCommand()
        # Run the CREST command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check if CREST ran successfully
        if result.returncode != 0:
            raise RuntimeError(f"CREST failed with error: {result.stderr}")

        # Parse the CREST output
        self.egatecule.new_mols = self.ParseCRESTOutput()


 

class ASEFunctions:

    def __init__(self,egatecule:MolMatDesc,params):
        self.params = params
        self.egatecule = egatecule
    

    def toaseatoms(self):
        # Convert RDKit mol object to ASE Atoms object
        mol_block = Chem.MolToMolBlock(self.egatecule.new_mol)
        ase_molecule = rdmolfiles.MolFromMolBlock(mol_block, sanitize=False)

        # Set up ASE Atoms object
        symbols = [atom.GetSymbol() for atom in self.egatecule.new_mol.GetAtoms()]
        positions = self.egatecule.new_mol.GetConformer().GetPositions()
        self.ase_atoms = Atoms(symbols=symbols, positions=positions)
    

    def setupcalc(self):
        # Set up ASE calculator
        self.calculator = EMT()
        self.ase_atoms.set_calculator(self.calculator)


    def setupmd(self):
        dyn = BFGS(self.ase_atoms)
        dyn.run(fmax=0.05)

        # Generate conformers
        conformers = []
        for i in range(self.params.nconfs):
            self.ase_atoms.rattle(stdev=0.1)
            dyn.run(fmax=0.05)
            conformers.append(self.ase_atoms.copy())

        self.ase_conformers = conformers
        # Convert ASE Atoms objects back to RDKit mol object
        for i, ase_conformer in enumerate(self.ase_conformers):
            positions = ase_conformer.get_positions()
            conf = Chem.Conformer(self.egatecule.new_mol.GetNumAtoms())
            for j, pos in enumerate(positions):
                conf.SetAtomPosition(j, pos)
                conf.SetId(i)
                self.egatecule.new_mol.AddConformer(conf, assignId=True)


class OpenFFFunctions:
    def __init__(self,egatecule:MolMatDesc,params):
        self.egatecule = egatecule
        self.off_molecule = None
        self.params = params

    def convert_to_openff(self):
        # Convert RDKit mol object to OpenFF Molecule
        self.off_molecule = Molecule.from_rdkit(self.egatecule.new_mol, allow_undefined_stereo=self.params.allow_undefined_stereo)
    
    def generate_conformers(self):
        # Generate conformers
        self.off_molecule.generate_conformers(n_conformers=self.nconfs, rms_cutoff=self.params.rms_cutoff)

    def optimize_conformers(self):
        # Load the specified OpenFF force field
        ff = ForceField(self.params.force_field)
        # Create an Interchange object for energy minimization
        interchange = Interchange.from_smirnoff(ff, [self.off_molecule])
        # Optimize the conformers
        interchange.minimize()
    
    def convert_to_rdkit(self):
        # Convert back to RDKit mol object
        self.egatecule.new_mol = self.off_molecule.to_rdkit()

    def process(self):
        self.convert_to_openff()
        self.generate_conformers()
        self.optimize_conformers()
        self.convert_to_rdkit()



class OpenMMFunctions:
    def __init__(self,egatecule:MolMatDesc,params):
        self.egatecule = egatecule
        self.params = params

    def initializeff(self):
        # Create an OpenMM system
        self.forcefield = app.ForceField(self.params.force_field,self.params.solvation)


    def generate_conformers(self):
       conformer_energies = []
       ids = [conf.GetId() for conf in self.egatecule.new_mol.GetConformers()]
       for cid in ids:
           conf = self.egatecule.new_mol.GetConformer(cid)
           positions = conf.GetPositions() * unit.angstrom
           pdb = app.PDBFile.from_rdkit(self.egatecule.new_mol, confId=cid)
           system =self.forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=self.params.constraints)
           integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
           simulation = app.Simulation(pdb.topology, system, integrator)
           simulation.context.setPositions(positions)
           simulation.minimizeEnergy()
           state = simulation.context.getState(getPositions=True, getEnergy=True)
           minimized_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
           energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
           for i in range(self.egatecule.new_mol.GetNumAtoms()):
               x, y, z = minimized_positions[i]
               conf.SetAtomPosition(i, (x, y, z))
           self.egatecule.new_mol.SetProp(f"Conformer_{cid}_Energy", str(energy))
           conformer_energies.append((cid, energy))
       conformer_energies.sort(key=lambda x: x[1])
       return conformer_energies
       


    def process(self):
        self.initializeff()
        self.conformer_energies = self.generate_conformers()





class SCFunctions:
    def __init__(self,egatecule:MolMatDesc,params):
        self.egatecule = egatecule
        self.params = params

    def convert_to_scf(self):
        mols = []
        for conf in self.egatecule.new_mol.GetConformers():
            atom_str = ""
            for atom in self.egatecule.new_mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atom_str += f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}; "
            mol = gto.M(atom=atom_str, basis='sto-3g', charge=self.egatecule.new_mol.GetProp("_Charge"))
            mols.append(scf.RHF(mol).run())

        return mols

    def optimize_with_pyscf(self):
        mols = self.convert_to_scf()
        optimized_mols = []
        for mol in mols:
            mf = scf.RHF(mol)
            mf.kernel()
            optimized_mols.append(mf)
        return optimized_mols


class Psi4Functions:
    def __init__(self,egatecule:MolMatDesc,params):
        self.egatecule = egatecule
        self.params = params
    

    def convert_to_psi4(self):
        mols = []
        for conf in self.egatecule.new_mol.GetConformers():
            atom_str = ""
            for atom in self.egatecule.new_mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atom_str += f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}\n"
            mol_str = f"molecule {{\n{atom_str}\nunits angstrom\n}}\n"
            mols.append(mol_str)
        return mols



    

class ConformerGeneratorEGAT(RDKitFunctions,RDKitPostProcessing,RDKitScoring,A3DHelpers,BabelFunctions,CRESTFunctions,ASEFunctions):
    def __init__(self,egatecule:MolMatDesc,arguments):
        
        self.params = arguments
        self.methodlist = ['EDG','ETKDG','ETKDGv2','ETKDGv3','KDG','srETKDGv3']
        self.egatecule = egatecule
        self.Initialize(self.params.method,self.params.seed,self.params.verbose)

        if self.params.nconfs == 1:
            self.GeneratewithRDKit()
            self.conf_ids = 1

    def GeneratewithRDKit(self,seed=1):
        self.conf_ids = rdDistGeom.EmbedMultipleConfs(self.egatecule.new_mol,self.params.nconfs,randomSeed=seed)
        self.ConformerCheck()
        self.OptimizeConformer()
        self.rms_list = self.AlignConformerstoReference()


    
    def GeneratewithOpenBabel(self,mode='Algorithm',genspeed='fast',children=5,mutability=5,converge=None,score=None,system='random',ff='uff',rmsd=.5,ecutoff=50,maxconfs=1000000,original=False,
               evaluate=True,steps=1500,crit=1e-06,search='sd',cutoff=True,vdw=6,rele=10,freq=10):
        
        infile,outfile = self.CreateFilesOpenBabel()
        
        self.Gen3DOpenBabel(outfile=infile,speed=genspeed)
        if mode == 'Algorithm':
            self.OpenBabelGenerationByAlgorithm(infile,outfile,children=children,mutability=mutability,converge=converge,score=score)
        elif mode == 'ff':
            self.OpenBabelGenerationByForceField(infile,outfile,system,ff)
        elif mode  == 'confab':
            self.OpenBabelGenerationByConfab(infile,outfile,rmsd,ecutoff,maxconfs,original)
        self.OpenBabelEnergyMinimization(outfile,outfile,evaluate,ff,steps,crit,search,cutoff,vdw,rele,freq)
        self.AlignConformerstoReference()

    def GeneratewithAuto3D(self):
        
        self.create_temp_dir()
        args = self.setup()
        
        if self.params.genmode == 'confgen':
            outfile = main(args)
        elif self.params.genmode == 'tautomer':
            outfile = get_stable_tautomers(args, tauto_k=3)

        if self.params.geomopt == True:
            outfile = opt_geometry(outfile, model_name=self.params.engine, opt_tol=self.params.opt_tol)
        
        def custom_func(mol):
            '''The mol object is an RDKit Molecule object.'''
            id = mol.GetProp("_Name")
            t = 273
            return (id, t)

        if self.params.thermo == True:
            self.thermo = calc_thermo(outfile, self.params.engine, get_mol_idx_t=custom_func, opt_tol=self.params.opt_tol)

        self.LoadasMolObject(outfile)
        self.AlignConformerstoReference()

    def GeneratewithCREST(self):
        self.ProcessCREST()


    def GeneratewithASE(self):
        self.toaseatoms()
        self.setupcalc()
        self.setupmd()
        self.AlignConformerstoReference()
    

    def GeneratewithOpenFF(self):
        self.convert_to_openff()
        self.generate_conformers()
        self.optimize_conformers()
        self.convert_to_rdkit()
    
    def GeneratewithOpenMM(self):
        self.GeneratewithRDKit()
        self.initializeff()
        self.conformer_energies = self.generate_conformers()


   





    

    




