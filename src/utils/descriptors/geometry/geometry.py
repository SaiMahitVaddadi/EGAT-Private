import subprocess,sys,os,traceback,Auto3D
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign,rdDistGeom,rdMolAlign,rdMolDescriptors,Descriptors3D,rdFreeSASA,MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize

from scipy.spatial import ConvexHull

from openff.toolkit.topology import Molecule
from openff.toolkit.utils import RDKitToolkitWrapper
from openff.units import unit 
from molvs.tautomer import TautomerEnumerator

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



class ConformerGenerator:
    def __init__(self,egatecule:MolMatDesc,nconfs=1,method=None,seed=1,verbose=False,theory='UFF'):
        
        self.nconfs = nconfs
        self.seed = seed
        self.method = method
        self.theory = theory
        self.methodlist = ['EDG','ETKDG','ETKDGv2','ETKDGv3','KDG','srETKDGv3']
        self.egatecule = egatecule
        self.Initialize(method,seed,verbose)

        if nconfs == 1:
            self._describe()
            self.conf_ids = 1

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
        failed = rdDistGeom.EmbedMolecule(self.egatecule.mol,maxAttempts=attempts,randomSeed=seed)
        if failed == -1:
            failed = rdDistGeom.EmbedMolecule(self.egatecule.mol,maxAttempts=attempts,randomSeed=seed,useRandomCoords=True)
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
        

    def ConformerCheck(self):
        self.num_conf = self.egatecule.new_mol.GetNumConformers()
        try:
            self.threedcheck = self.egatecule.new_mol.GetConformer().Is3D()
        except:
            self._initialize(seed=self.seed+1)
            self.num_conf = self.egatecule.new_mol.GetNumConformers()
            self.threedcheck = self.egatecule.new_mol.GetConformer().Is3D()
            
    def AlignConformerstoReferenc(self):
        rms_list = []
        AllChem.AlignMolConformers(self.egatecule.new_mol, RMSlist=rms_list)
        return rms_list

    def OptimizeConformer(self):
        function = getattr(AllChem,f'{self.theory}OptimizeMolecule',None)
        if self.conf_ids != 1:
            for id in self.conf_ids:
                function(self.egatecule.new_mol,confId=id)
        else:
            function(self.egatecule.new_mol)


    def GeneratewithRDKit(self,seed=1):
        self.conf_ids = rdDistGeom.EmbedMultipleConfs(self.egatecule.new_mol,self.nconfs,randomSeed=seed)
        self.ConformerCheck()
        self.OptimizeConformer()
        self.AlignConformerstoReference()


    
    def LoadasMolObject(self,outfile):
        mol = Chem.SDMolSupplier(outfile, removeHs=False)[0]
        for mol in Chem.SDMolSupplier(outfile, removeHs=False):
            if mol is not None:
                mol.AddConformer(mol.GetConformer(), assignId=True)
        self.egatecule.new_mol = mol
        self.conf_ids = [conf.GetId() for conf in self.egatecule.new_mol.GetConformers()]

    def GeneratewithAuto3D(self,tmp='tmp',confs=1,window=None,engine = 'ANI2x',gpu=False,enumerate_tautomer=True, tauto_engine="rdkit",pKaNorm=True,enumerate_isomer=True,max_confs=None,patience=1000,opt_steps=5000,convergence_threshold=.003,threshold=.3,genmode='confgen',geomopt=False,thermo=False,tauto_k=None,tauto_window=None,opt_tol=.0002):
        
        infile = self.ConverttoSDF()
        outfolder = tempfile.mkdtemp()


        if window == None:
            args = options(infile, k=confs, job_name=outfolder,enumerate_tautomer=enumerate_tautomer, tauto_engine=tauto_engine,pKaNorm=pKaNorm,enumerate_isomer=enumerate_isomer,max_confs=max_confs,optimizing_engine=engine, use_gpu=gpu,patience=patience,opt_steps=opt_steps,convergence_threshold=convergence_threshold,threshold=threshold)
        else:
            args = options(infile, window=window, job_name=outfolder,enumerate_tautomer=enumerate_tautomer, tauto_engine=tauto_engine,pKaNorm=pKaNorm,enumerate_isomer=enumerate_isomer,max_confs=max_confs,optimizing_engine=engine, use_gpu=gpu,patience=patience,opt_steps=opt_steps,convergence_threshold=convergence_threshold,threshold=threshold)
        
        if genmode == 'confgen':
            outfile = main(args)
        elif genmode == 'tautomer':
            outfile = get_stable_tautomers(args, tauto_k=3)

        if geomopt == True:
            outfile = opt_geometry(outfile, model_name=engine, opt_tol=opt_tol)
        
        def custom_func(mol):
            '''The mol object is an RDKit Molecule object.'''
            id = mol.GetProp("_Name")
            t = 273
            return (id, t)

        if thermo == True:
            self.thermo = calc_thermo(outfile, engine, get_mol_idx_t=custom_func, opt_tol=opt_tol)

        self.LoadasMolObject(outfile)
        self.AlignConformerstoReference()


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


    def Gen3DOpenBabel(self,outfile='out.sdf',speed='fast'):
        infile = self.ConverttoSDF()
        command = ['obabel', '-isdf',self.smiles, '-osdf', '--gen3d',speed]
        result = subprocess.run(command)
        self.egatecule.new_mol = Chem.MolFromMolBlock(result.stdout,sanitize=False)
    

    def CreateFilesOpenBabel(self):
        infile = self.ConverttoSDF()
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = os.path.join(tmpdirname, 'new_outfile.sdf')
        return infile,outfile

    def OpenBabelGenerationByAlgorithm(self,children=5,mutability=5,converge=None,score=None):
        infile,outfile = self.CreateFilesOpenBabel()
        command = ['obabel', infile + '"', '-O', outfile]
        command += ['--conformer', '--children',str(children),'--mutability',str(mutability),'--nconf',str(self.num_conformers)]
        if converge:
            command += ['--converge',str(converge)]
        if score: command += ['--score',score]
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
        self.AlignConformerstoReferenc()
    


    def CreateCRESTInput(self):
        infile,_ = self.CreateFilesOpenBabel()
        outfolder = tempfile.mkdtemp()
        return infile,outfolder
        
    def GeneratewithCREST(self,infile,tmp='tmp',theory='gfn2',charge = 0, uhf = 0,solvation = None,optlev=None,sampling=None,sites=None,mdlen=None,shake=None,tstep=None,mddump=None,vbdump=None,
              zsort=False,genzsort=True,norotmd=False,tnmd=None,mrest=None,hflip=True,maxflip=None,gcspeed=None,props=None,origin=True,keepdir=False,noreftopo=False,noopt=False,wall=None,
              scthr=None,ssthr=None,trange=None,ptot=None,fscal =None,sthr=None,ithr=None,cinp=None,cbonds=None,cheavy=None,clight=None,fc=None,mdopt=None,screen=None,rrhoav=None,
              thermo=None,nanoreactor=None,solvent=None,testtopo=None,inputfile=None):
        """
        Generate a set of conformers using CREST.
        """
        infile,outfolder = self.CreateCRESTInput(self)

        

        command = f"crest {infile} --{theory} --scratch {outfolder} "

        if optlev != None: command += f"--opt {optlev} "
        if solvation != None: command += f"--{solvation} {solvent}"
        if sampling != None:
            samplingdict = dict()
            samplingdict['MF-MD-GC'] = 'v1'
            samplingdict['MTD-GC'] = 'v2i'
            samplingdict['iMTD-GC'] = 'v3'
            samplingdict['iMTD-sMTD'] = 'v4'
            samplingdict['Entropy'] = 'entropy'
            
            command += f"--{samplingdict[sampling]} "
        
        if sites != None: command += f"--{sites} "
        if mdlen != None: command += f"--len x={mdlen} "
        if shake != None: command += f"--shake {shake} "
        if tstep != None: command += f"--tstep {tstep} "
        if mddump != None: command += f"--mddump {mddump} "
        if vbdump != None: command += f"--vbdump {vbdump} "

        if zsort: command += '--zs '
        if not genzsort: command += '--nocross '
        if norotmd: command += '--norotmd '
        if tnmd: command += f'--tnmd {tnmd} '
        if mrest: command += f'--mrest {mrest} '
        if not hflip: command += f'--noflip '
        if maxflip: command += f'--maxflip {maxflip} '
        if gcspeed: command += f'--{gcspeed} '

        if props: command += f'--prop {props}'
        if keepdir: command += f'--keepdir '
        if noreftopo: command += f'--noreftopo '
        if noopt: command += f'--noopt '
        if wall != None: command += f'--{wall} '
        if scthr != None: command += f'--scthr {scthr} '
        if ssthr != None: command += f'--ssthr {ssthr} '
        if trange != None: command += f'--trange {trange[0]} {trange[1]} {trange[2]} '
        if ptot != None: command += f'--ptot {ptot}'
        if fscal != None: command += f'--fscal {fscal} '
        if sthr != None: command += f'--sthr {sthr} '
        if ithr != None: command += f'--ithr {ithr} '

        if cinp != None: command += f'--cinp {cinp} '
        if cbonds != None: 
            if cbonds == False:
                command += '--nocbonds '
            else:
                command += f'--cbonds {cbonds} '
        

        if cheavy != None: command += f'--cheavy {cheavy} '
        if clight != None: command += f'--clight {clight} '
        if fc != None: command += f'--fc {fc} '
        if mdopt != None: command += f'--mdopt {mdopt} '
        if screen != None: command += f'--screen {screen} '
        if rrhoav != None: command += f'--rrhoav {rrhoav} '
        if thermo != None: command += f'--thermo {thermo} '
        if testtopo != None: command += f'--testtopo {testtopo} '
        if nanoreactor: command += '--nanoreactor '
        if inputfile != None: command += f'--input {inputfile}'

    
        # Run the CREST command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check if CREST ran successfully
        if result.returncode != 0:
            raise RuntimeError(f"CREST failed with error: {result.stderr}")

        # Parse the CREST output
        self.egatecule.new_mols = self.ParseCRESTOutput()
    
    def parse_crest_output(self):
        """
        Parse the output from CREST to extract conformers.
        """
        # Assuming CREST outputs a file named 'crest_conformers.xyz'
        crest_output_file = 'crest_conformers.xyz'

        if not os.path.exists(crest_output_file):
            raise FileNotFoundError(f"CREST output file {crest_output_file} not found")

        # Parse the XYZ file to extract conformers
        conformers = read(crest_output_file, index=':')
        return conformers

    def GeneratewithASE(self):
        # Convert RDKit mol object to ASE Atoms object
        mol_block = Chem.MolToMolBlock(self.egatecule.new_mol)
        ase_molecule = rdmolfiles.MolFromMolBlock(mol_block, sanitize=False)

        # Set up ASE Atoms object
        symbols = [atom.GetSymbol() for atom in self.egatecule.new_mol.GetAtoms()]
        positions = self.egatecule.new_mol.GetConformer().GetPositions()
        ase_atoms = Atoms(symbols=symbols, positions=positions)

        # Set calculator
        ase_atoms.set_calculator(EMT())

        # Optimize geometry
        dyn = BFGS(ase_atoms)
        dyn.run(fmax=0.05)

        # Generate conformers
        conformers = []
        for i in range(self.nconfs):
            ase_atoms.rattle(stdev=0.1)
            dyn.run(fmax=0.05)
            conformers.append(ase_atoms.copy())

        self.ase_conformers = conformers
        # Convert ASE Atoms objects back to RDKit mol object
        for i, ase_conformer in enumerate(self.ase_conformers):
            positions = ase_conformer.get_positions()
            conf = Chem.Conformer(self.egatecule.new_mol.GetNumAtoms())
            for j, pos in enumerate(positions):
                conf.SetAtomPosition(j, pos)
                conf.SetId(i)
                self.egatecule.new_mol.AddConformer(conf, assignId=True)

        pass

    def GeneratewithQCG(self, infile, tmp='tmp', solvent='h2o', run='grow', nsolv=None, nopreopt=False, keepdir=False, gfn1=False, gfn2=False, gfnff=False, samerand=False, 
                 chrg=None, uhf=None, wscal=None, fixsolute=False, nofix=False, xtbiff=False, normdock=False, directed=None, qcgmtd=False, ncimtd=False, mtd=False, 
                 md=False, enslvl=None, mdlen=None, mddump=None, tstep=None, vbdump=None, norotmd=False, tnmd=None, mreset=None, fin_opt_gfn2=False, nocff=False, 
                 esolv=False, nclus=None, freqlvl=None, freqscal=None):
        infile,outfolder = self.CreateCRESTInput(self)
        command = f"crest {infile} --scratch {os.path.join(tmp,'crest-qcg')} -qcg {solvent} --{run} "

        if run == 'grow':
            if nsolv: command += f'--nsolv {nsolv} '
            if nopreopt: command += '--nopreopt '
            if keepdir: command += '--keepdir '
            if gfn1: command += '--gfn1 '
            if gfn2: command += '--gfn2 '
            if gfnff: command += '--gfnff '
            if samerand: command += '--samerand '
            if chrg: command += f'--chrg {chrg} '
            if uhf: command += f'--uhf {uhf} '
            if wscal: command += f'--wscal {wscal} '
            if fixsolute: command += '--fixsolute '
            if nofix: command += '--nofix '
            if xtbiff: command += '--xtbiff '
            if normdock: command += '--normdock '
            if directed: command += f'--directed {directed} '
            print('Under construction')
        elif run == 'ensemble':
            if qcgmtd: command += '--qcgmtd '
            if ncimtd: command += '--ncimtd '
            if mtd: command += '--mtd '
            if md: command += '--md '
            if enslvl: command += f'--enslvl {enslvl} '
            if mdlen: command += f'--mdlen {mdlen} '
            if mddump: command += f'--mddump {mddump} '
            if tstep: command += f'--tstep {tstep} '
            if vbdump: command += f'--vbdump {vbdump} '
            if norotmd: command += '--norotmd '
            if tnmd: command += f'--tnmd {tnmd} '
            if mreset: command += f'--mreset {mreset} '
            if fin_opt_gfn2: command += '--fin_opt_gfn2 '
            print('Under construction')
        elif run == 'solvation':
            if nocff: command += '--nocff '
            if esolv: command += '--esolv '
            if nclus: command += f'--nclus {nclus} '
            if freqlvl: command += f'--freqlvl {freqlvl} '
            if freqscal: command += f'--freqscal {freqscal} '
            print('Under construction')

        self.RunCRESTcommand()

        # Parse the CREST output
        self.egatecule.new_mols = self.ParseCRESTOutput()

    def GeneratewithCREGEN(self, ewin=None, rthr=None, ethr=None, bthr=None, pthr=None, nmr=False, eqv=False, athr=None, temp=None, esort=False, nowr=False, subrmsd=False, notopo=None, cluster=None, pccap=None, nopcmin=False, pcaex=None):

        infile, outfile = self.CreateFilesOpenBabel(self)
        command = f"crest {infile} --cregen {outfile} "
        #crest struc.xyz --cregen input-ensemble.xyz

        if ewin: command += f'--ewin {ewin} '
        if rthr: command += f'--rthr {rthr} '
        if ethr: command += f'--ethr {ethr} '
        if bthr: command += f'--bthr {bthr} '
        if pthr: command += f'--pthr {pthr} '
        if nmr: command += '--nmr '
        if eqv: command += '--eqv '
        if athr: command += f'--athr {athr} '
        if temp: command += f'--temp {temp} '
        if esort: command += '--esort '
        if nowr: command += '--nowr '
        if subrmsd: command += '--subrmsd '
        if notopo: command += f'--notopo {notopo} '
        if cluster: command += f'--cluster {cluster} '
        if pccap: command += f'--pccap {pccap} '
        if nopcmin: command += '--nopcmin '
        if pcaex: command += f'--pcaex {pcaex} '
        self.RunCRESTcommand()
        
        pass

    # Run the CREST command
    def RunCRESTcommand(self,command):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check if CREST ran successfully
        if result.returncode != 0:
            raise RuntimeError(f"CREST failed with error: {result.stderr}")
    
    def openff(self): 
        # Convert RDKit mol object to OpenFF Molecule
        off_molecule = Molecule.from_rdkit(self.egatecule.new_mol, allow_undefined_stereo=True)
        
        # Generate conformers
        off_molecule.generate_conformers(n_conformers=self.nconfs, rms_cutoff=1.0 * unit.angstrom)
        
        # Convert back to RDKit mol object
        self.egatecule.new_mol = off_molecule.to_rdkit()

    





    

    




