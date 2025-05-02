"""Compute various scores with RDKit"""
import numpy as np
from rdkit.Chem import AllChem as Chem, Crippen, Descriptors, Lipinski
import math
import os.path as op
import pickle
from dockstring import load_target
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from .....addons.scscore import SCScorer
from .....addons.HamDiv.diversity import diversity_all, HamDiv
from .....addons.molcomplexity.molecular_complexity.complexity import molecular_complexity


_fscores = None
class RDKitDescriptors:
    def __init__(self, sandp=False,grid_spacing = .2,box_margin=.2,method='default',fplen=2048,protein=None):
        self.descriptors = {
    "Qed": Descriptors.qed,
    "MolecularWeight": Descriptors.MolWt,
    "GraphLength": graph_length,
    "NumAtomStereoCenters": Chem.CalcNumAtomStereoCenters,
    "HBondAcceptors": Lipinski.NumHAcceptors,
    "HBondDonors": Lipinski.NumHDonors,
    "NumRotBond": Lipinski.NumRotatableBonds,
    "Csp3": Lipinski.FractionCSP3,
    "numsp": num_sp,
    "numsp2": num_sp2,
    "numsp3": num_sp3,
    "NumHeavyAtoms": Lipinski.HeavyAtomCount,
    "NumHeteroAtoms": Lipinski.NumHeteroatoms,
    "NumRings": Lipinski.RingCount,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "SlogP": Crippen.MolLogP}
        self.sandp = sandp
        self.grid_spacing = grid_spacing
        self.box_margin = box_margin
        self.method = method
        self.fplen = fplen
        self.protein = protein

    def _computefunc(self, mol: Chem.Mol, descriptor):
        if descriptor == 'tpsa':
            return self.tpsa(mol)
        elif descriptor == 'volume':
            return self.volume(mol)
        elif descriptor == 'pmi':
            return self.pmi(mol)
        elif descriptor == 'sascore':
            return self.sascore(mol)
        elif descriptor == 'scscore':
            return self.scscore(mol)
        elif descriptor == 'dockstring':
            return self.dockstring(mol)
        elif descriptor == 'molcomplexity':
            return self.molcomplexity(mol)
        elif descriptor in ['Richness','FG','RS','BM','IntDiv','HamDiv','Diam','SumDiam','SumDiv','Bot','SumBot','DPP','NCircles']:
            return self.hamdiv(mol,descriptor)
        else:
            return self.descriptors[descriptor](mol)

    def compute(self, smiles,descriptor) -> list:
        mol = Chem.MolFromSmiles(smiles)
        if isinstance(descriptor, str):
            return self._computefunc(mol, descriptor)
        elif isinstance(descriptor, list):
            return [self._computefunc(mol, descriptor) for desc in descriptor]
        
    def tpsa(self, mol: Chem.Mol) -> float:
        return Descriptors.TPSA(mol, includeSandP=self.sandp)
    
    def volume(self, mol: Chem.Mol,) -> float:
        mol3d = Chem.AddHs(mol)  # will not consider protonation state
        Chem.EmbedMolecule(mol3d)
        volume = Chem.ComputeMolVolume(
            mol3d, gridSpacing=self.grid_spacing, boxMargin=self.box_margin
        )
        return volume
    
    def pmi(self, mol: Chem.Mol):
        mol3d = Chem.AddHs(mol)
        embed_result = Chem.EmbedMolecule(mol3d)
        
        if embed_result == -1:  # embedding failed
            npr1 = -1
            npr2 = -1
        else:
            npr1 = Chem.CalcNPR1(mol3d)
            npr2 = Chem.CalcNPR2(mol3d)

        return npr1, npr2
        
    def sascore(self, mol: Chem.Mol) -> float:
        return calculateScore(mol) 

    def scscore(self,smi):
        project_root =  f'../../../../addons/scscore/models/full_reaxys_model_{self.fplen}bool/model.ckpt-10654.as_numpy.pickle'
        model = SCScorer()
        model.restore(project_root)
        (smi, sco) = model.get_score_from_smi(smi)
        return sco

    def dockstring(self,smi):
        if isinstance(self.protein, str):
            target = load_target(self.protein)
            target = load_target(self.protein)
            score, _ = target.dock(smi)
            return score
        elif isinstance(self.protein, list):
            scores = [] 
            for p in self.protein:
                target = load_target(p)
                score, _ = target.dock(smi)
                scores.append(score)
            return scores

    def molcomplexity(self,smi):
        cm, cm_star, cse = molecular_complexity(smi)
        return [cm, cm_star, cse]

    def hamdiv(self,smi,descriptor):
        if descriptor =='HamDiv':
            return HamDiv(smi,method=self.method)
        else:
            return diversity_all(smi,descriptor)



def num_sp(mol: Chem.Mol) -> int:
    num_sp_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP]
    )

    return num_sp_atoms

def num_sp2(mol: Chem.Mol) -> int:
    num_sp2_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2]
    )

    return num_sp2_atoms

def num_sp3(mol: Chem.Mol) -> int:
    num_sp3_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3]
    )
    return num_sp3_atoms

def graph_length(mol: Chem.Mol) -> int:
    return int(np.max(Chem.GetDistanceMatrix(mol)))




#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#





def readFragmentScores(name="fpscores"):
    import gzip

    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open("%s.pkl.gz" % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2
    )  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore


#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

'''
if self.feature == 'SASA':
                    gen.calculate_sasa(cp)
                    row.append(gen.sasa)                
                elif self.feature in ['SA','Volume','SA-V Ratio']:
                    if self.volmethod == 'Spherical':
                        gen.calculate_volume_spherical(cp)
                    elif self.volmethod == 'Cubic':
                        gen.featurize_double_cubic_lattice_volume(cp)
                    elif self.volmethod == 'Connolly':
                        gen.featurize_connolly(cp)


                    if self.feature == 'SA':
                        row.append(gen.surface_area)
                    elif self.feature == 'Volume':
                        row.append(gen.volume)
                    else:
                        row.append(gen.sa_to_vol_ratio)
                
                elif self.feature in ['eccentricity','shape_factor','si','asphericity']:
                    gen.featurize_rdkit(cp)

                    
def scoreconfs(self,mode='crippen'):
        function = getattr(self,f'_{mode}',None)
        function()
        try:
            scores = []
            base = Chem.Mol(self.mol,confId=self.conf_ids[0])
            for _ in self.conf_ids:
                if _ > self.conf_ids[0]:
                    cp = Chem.Mol(self.mol,confId=_)
                    if mode == 'crippen':
                        o3a = rdMolAlign.GetCrippenO3A(base,cp,self.prob_contrib[_],self.ref_contrib,_,0)
                    elif mode == 'mmff':
                        o3a = rdMolAlign.GetCrippenO3A(base,cp,self.prob_contrib[_],self.ref_contrib,_,0)
                    
                    o3a.Align()
                    scores.append(o3a.Score())
            self.scores = scores 
        except:
            self.scores = None


    def featurize_connolly(self,mol):
        # Calculate the Connolly surface area
        radii = [rdMolDescriptors.CalcAtomGaussTypeRadius(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
        surface = Chem.MolSurf.pyMolSurf(mol, radii, 1.4)

        # Extract surface area and volume
        self.surface_area = surface.GetSurfaceArea()
        self.volume = surface.GetVolume()
        # Calculate surface area to volume ratio
        self.sa_to_vol_ratio = self.surface_area / self.volume if self.volume != 0 else float('inf')

    def featurize_sphere_sphere_intersection(self, mol):
        # Calculate the sphere-sphere intersection surface area and volume
        radii = [rdMolDescriptors.CalcAtomGaussTypeRadius(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()
        
        # Calculate pairwise distances
        dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
        
        # Calculate intersection areas and volumes
        intersection_areas = []
        intersection_volumes = []
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                r1, r2 = radii[i], radii[j]
                d = dist_matrix[i, j]
                if d < r1 + r2:
                    intersection_area = self._sphere_sphere_intersection_area(r1, r2, d)
                    intersection_areas.append(intersection_area)
                    
                    # Calculate intersection volume
                    intersection_volume = self._sphere_sphere_intersection_volume(r1, r2, d)
                    intersection_volumes.append(intersection_volume)
        
        self.surface_area = sum(intersection_areas)
        self.num_intersections = len(intersection_areas)
        self.volume = sum(intersection_volumes)
        # Calculate surface area to volume ratio
        self.sa_to_vol_ratio = self.surface_area / self.volume if self.volume != 0 else float('inf')
        
    def _sphere_sphere_intersection_volume(self, r1, r2, d):
        # Calculate the volume of intersection between two spheres
        if d >= r1 + r2:
            return 0
        elif d <= abs(r1 - r2):
            return 4/3 * np.pi * min(r1, r2)**3
        else:
            part1 = (np.pi * (r1 + r2 - d)**2 * (d**2 + 2*d*r1 - 3*r1**2 + 2*d*r2 + 6*r1*r2 - 3*r2**2 + 2*d*r1*r2)) / (12 * d)
            return part1
    def _sphere_sphere_intersection_area(self, r1, r2, d):
        # Calculate the area of intersection between two spheres
        if d >= r1 + r2:
            return 0
        elif d <= abs(r1 - r2):
            return 4 * np.pi * min(r1, r2)**2
        else:
            part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
            return part1 + part2 - part3

    
    def featurize_rdkit(self,mol):
        self.eccentricity = Descriptors3D.Eccentricity(mol)
        self.shapefactor = Descriptors3D.InertialShapeFactor(mol)
        self.pbf = Descriptors3D.PBF(mol)
        self.pmi1 = Descriptors3D.PMI1(mol)
        self.pmi2 = Descriptors3D.PMI2(mol)
        self.pmi3 = Descriptors3D.PMI3(mol)
        self.rg = Descriptors3D.RadiusOfGyration(mol)
        self.si = Descriptors3D.SpherocityIndex(mol)
        self.asphericity = Descriptors3D.Asphericity(mol)
        
    

    def calculate_convex_hull(self, mol):
        # Calculate the convex hull volume and surface area

        coords = mol.GetConformer().GetPositions()
        hull = ConvexHull(coords)

        self.volume = hull.volume
        self.surface_area = hull.area
        # Calculate surface area to volume ratio
        self.sa_to_vol_ratio = self.surface_area / self.volume if self.volume != 0 else float('inf')


    def calculate_sasa(self, mol):
        # Calculate the Solvent Accessible Surface Area (SASA)
        radii = rdFreeSASA.classifyAtoms(mol)
        sasa = rdFreeSASA.CalcSASA(mol, radii)
        self.sasa = sasa


'''