# src/kallisto/methods.py
from typing import Tuple
import numpy as np
from ..base import BaseFeaturizer
from rdkit import Chem
from .data.symbols import covalent_radius as rcov
from .data.symbols import pauling_en,eeq_alp, eeq_cnfak, eeq_en, eeq_gamm,chemical_symbols
from scipy import special
from .data.vdw import rahm, truhlar
from numpy import linalg as LA
from .data.alpha import refx, refh, hcount, ascale, refn,refcn, refsys, alphaiw, zeff,sscale, seciw, gam
from .data.scaling import zeta, cngw
from dataclasses import dataclass





@dataclass
class KallistoParams:
    threshold: float = 10.0
    covalentnumber: str = "exp"
    ps_size: Tuple[float, float] = (1.0, 1.5)
    kallisto_thresholdCN: float = 10.0
    kallisto_thresholdBond: float = 0.5
    vdw_scale: float = 1.0
    vdwtype: str = "truhlar"
    getCN: bool = True
    getPS: bool = True
    getAPC: bool = True
    getvdwkallisto: bool = True
    getAP: bool = True


class KallistoBase(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def GrabConformerKallisto(self,id=1):
        if id == 1: return self.matrixdescriptors.new_mol
        else:
            return self.matrixdescriptors.new_mol.GetConformer(conf_id=id)
    
    def GrabGeometryKallisto(self,id=1):
        mol = self.GrabConformerKallisto(id)
        self.positions = {}
        self.at = {}
        self.positions_list = []
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            atom_map_num = atom.GetAtomMapNum()
            ind = atom.GetIdx()
            self.positions[ind] = [pos.x, pos.y, pos.z]
            self.at[ind] = atom.GetAtomicNum()

        # Sort the positions dictionary by key
        self.positions = dict(sorted(self.positions.items()))
        self.at = dict(sorted(self.at.items()))
        
        # Extract all values for positions and at and save them as lists
        self.positions_list = [value for value in self.positions.values()]
        self.at_list = [value for value in self.at.values()]
        self.nat = len(self.at_list)
        self.cns = np.zeros(shape=(self.nat,), dtype=np.float64)

        self.prox1 = np.zeros(shape=(self.nat,), dtype=np.float64)
        self.prox2 = np.zeros(shape=(self.nat,), dtype=np.float64)


class CN(KallistoBase):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def expfunc(self):
        k1 = 16.0
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            for j in range(self.nat):
                if i is j:
                    continue
                dx = self.positions_list[j][0] - self.positions_list[i][0]
                dy = self.positions_list[j][1] - self.positions_list[i][1]
                dz = self.positions_list[j][2] - self.positions_list[i][2]
                rSquared = dx * dx + dy * dy + dz * dz
                if rSquared > self.params.threshold:
                    continue
                ja = self.at_list[j] - 1
                r = np.sqrt(rSquared)
                rco = rcov[ia] + rcov[ja]
                rr = rco / r
                k = -k1 * (rr - 1.0)
                damp = 1.0 / (1.0 + np.exp(k))
                self.cns[i] += damp


    def erffunc(self):
        kn = 7.50
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            for j in range(self.nat):
                if i is j:
                    continue
                ja = self.at_list[j] - 1
                dx = self.positions_list[j][0] - self.positions_list[i][0]
                dy = self.positions_list[j][1] - self.positions_list[i][1]
                dz = self.positions_list[j][2] - self.positions_list[i][2]
                rSquared = dx * dx + dy * dy + dz * dz
                if rSquared > self.params.threshold:
                    continue
                r = np.sqrt(rSquared)
                rco = rcov[ia] + rcov[ja]
                damp = 0.5 * (1.0 + special.erf(-kn * (r - rco) / rco))
                self.cns[i] += damp


    def covfunc(self):
        # Fitted to match Wiberg bond orders of diatomic molecules
        k4 = 4.10451
        k5 = 19.08857
        k6 = 2 * 11.28174**2

        kn = 7.50
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            for j in range(self.nat):
                if i is j:
                    continue
                ja = self.at_list[j] - 1
                dx = self.positions_list[j][0] - self.positions_list[i][0]
                dy = self.positions_list[j][1] - self.positions_list[i][1]
                dz = self.positions_list[j][2] - self.positions_list[i][2]
                rSquared = dx * dx + dy * dy + dz * dz
                if rSquared > self.params.threshold:
                    continue
                r = np.sqrt(rSquared)
                rco = rcov[ia] + rcov[ja]
                eni = pauling_en[ia]
                enj = pauling_en[ja]
                den = k4 * np.exp(-((np.abs(eni - enj) + k5) ** 2) / k6)
                damp = den * 0.5 * (1 + special.erf(-kn * (r - rco) / rco))
                self.cns[i] += damp


    def ObtainCN(self,id=1):
        """A method to compute atomic covalent numbers (cns).

        CN values are calculated for a given structure and are returned as an
        array."""

        self.GrabGeometryKallisto(id)
        
        # Get covalent number
        if self.params.covalentnumber == "exp":
            self.expfunc()
        elif self.params.covalentnumber == "erf":
            self.erffunc()
        elif self.params.covalentnumber == "cov":
            self.covfunc()


class ProximityShell(KallistoBase):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def ObtainProximityShell(self,id=1):
        """A method to compute atomic proximity shells (prox).

        Prox values are calculated for a given structure and are returned as an
        array."""

        self.GrabGeometryKallisto(id)
        # Fitted to match Wiberg bond orders of diatomic molecules
        k4 = 4.10451
        k5 = 19.08857
        k6 = 2 * 11.28174**2

        kn = 7.50
        threshold = 800.0

        # unpack tuple in sizes
        scale1, scale2 = self.params.ps_size

        for i in range(self.nat):
            ia = self.at_list[i] - 1
            for j in range(self.nat):
                if i is j:
                    continue
                ja = self.at_list[j] - 1
                dx = self.positions_list[j][0] - self.positions_list[i][0]
                dy = self.positions_list[j][1] - self.positions_list[i][1]
                dz = self.positions_list[j][2] - self.positions_list[i][2]
                rSquared = dx * dx + dy * dy + dz * dz
                if rSquared > threshold:
                    continue
                r = np.sqrt(rSquared)
                eni = pauling_en[ia]
                enj = pauling_en[ja]
                den = k4 * np.exp(-((np.abs(eni - enj) + k5) ** 2) / k6)

                # smaller border
                rco = scale1 * (rcov[ia] + rcov[ja])
                damp = den * 0.5 * (1 + special.erf(-kn * (r - rco) / rco))
                self.prox1[i] += damp
                # larger border
                rco = scale2 * (rcov[ia] + rcov[ja])
                damp = den * 0.5 * (1 + special.erf(-kn * (r - rco) / rco))
                self.prox2[i] += damp

        self.ps = self.prox2 - self.prox1


class APC(CN):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def SetupAPC(self,id=1):
        # parameter
        self.sqrt2pi = np.sqrt(2.0 / np.pi)
        
        # Lagragian space is +1 in dimensionality
        self.m = self.nat + 1

        # convert lists to numpy arrays for vectorization
        self.eeq_alp = np.array(eeq_alp)
        self.eeq_cnfak = np.array(eeq_cnfak)
        self.eeq_en = np.array(eeq_en)
        self.eeq_gamm = np.array(eeq_gamm)

        # setup parameter arrays
        self.z = np.array(self.at_list) - 1
        self.xi = []
        self.gam = []
        self.kappa = []
        self.alpha = []
        for z in self.z:
            self.xi.append(self.eeq_en[z])
            self.gam.append(self.eeq_gamm[z])
            self.kappa.append(self.eeq_cnfak[z])
            self.alpha.append(np.power(self.eeq_alp[z], 2))
        
        # Convert xi, gam, kappa, and alpha into numpy arrays
        self.xi = np.array(self.xi, dtype=np.float64)
        self.gam = np.array(self.gam, dtype=np.float64)
        self.kappa = np.array(self.kappa, dtype=np.float64)
        self.alpha = np.array(self.alpha, dtype=np.float64)

        """Set up A matrix and X vector

                αi -> alpha(i), ENi -> xi(i), κi -> kappa(i), Jii -> gam(i)
                γij = 1/√(αi+αj)
                Xi  = -ENi + κi·√CNi
                Aii = Jii + 2/√π·γii
                Aij = erf(γij·Rij)/Rij = 2/√π·F0(γ²ij·R²ij)"""

        # A matrix
        self.A = np.zeros(shape=(self.m, self.m), dtype=np.float64)
        self.ObtainCN(id)

    def ObtainAPC(self,id=1):
        """A method to compute atomic partial charges (apc).

        APC values are calculated for a given structure and are returned as an
        array."""

        self.GrabGeometryKallisto(id)
        self.SetupAPC(id)
        for i in range(self.nat):
            xyzi = self.positions_list[i]
            self.A[i][i] = self.gam[i] + self.sqrt2pi / np.sqrt(self.alpha[i])
            for j in range(self.nat):
                if i == j:
                    continue
                xyzj = self.positions_list[j]
                r = LA.norm(np.array(xyzj) - np.array(xyzi))
                gamij = 1.0 / np.sqrt(self.alpha[i] + self.alpha[j])
                self.A[j][i] = special.erf(gamij * r) / r
                self.A[i][j] = self.A[j][i]

        # X vector
        X = np.zeros(shape=(self.m,), dtype=np.float64)
        X[:self.nat] = -self.xi + self.kappa * np.sqrt(self.cns)

        # setup Lagragian constraints
        self.A[:, self.nat] = 1.0
        self.A[self.nat, :] = 1.0
        self.A[self.nat, self.nat] = 0.0
        # Calculate the formal charge of the molecule
        formal_charge = sum([atom.GetFormalCharge() for atom in self.matrixdescriptors.new_mol.GetAtoms()])
        X[self.nat] = formal_charge

        # get eeq charges
        self.qs = np.linalg.solve(self.A, X)

        self.qs = self.qs[:-1]


class CovalentPartner(KallistoBase):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()



    def ObtainCovalentPartner(self):
        """A method to compute covalent bonding partners (covalentPartner).

        Covalent bonding partners are calculated for a given structure and are
        returned as an array."""

        # parameter
        k1 = 16.0
        btbl = np.zeros(shape=(self.nat, self.nat), dtype=np.int32)

        

        self.GrabGeometryKallisto()
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            for j in range(self.nat):
                if i is j:
                    continue
                dx = self.positions_list[j][0] - self.positions_list[i][0]
                dy = self.positions_list[j][1] - self.positions_list[i][1]
                dz = self.positions_list[j][2] - self.positions_list[i][2]
                rSquared = dx * dx + dy * dy + dz * dz
                if rSquared > self.params.kallisto_thresholdCN:
                    continue
                ja = self.at_list[j] - 1
                r = np.sqrt(rSquared)
                rco = rcov[ia] + rcov[ja]
                rr = rco / r
                alpha = -k1 * (rr - 1.0)
                damp = 1.0 / (1.0 + np.exp(alpha))
                if damp > self.params.kallisto_thresholdBond:
                    btbl[i][j] = 1       
        
        covalentList = []
        for i in range(self.nat):
            covalentPartner = []
            k = 0
            for elem in btbl[i][:]:
                if elem == 1:
                    covalentPartner.append(k)
                k += 1
            covalentList.append(covalentPartner)
        return covalentList


class AP(APC):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()


    def SetupAP(self):
        self.ObtainAPC()
        # parameter
        self.g_a = 3.0
        self.g_c = 2.0

        # get dimensionality
        self.ndim = 0
        for i in range(self.nat):
            self.ndim += refn[self.at_list[i] - 1]

        # index table
        self.itbl = np.zeros(shape=(7, self.nat), dtype=np.int64)
        self.k = 0
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            for ii in range(refn[ia]):
                self.itbl[ii][i] = self.k
                self.k += 1

        # setup ncount and charge scale polarizabilities
        self.ncount = np.zeros(shape=(7, 86), dtype=np.int64)
        self.alpha = np.zeros(shape=(23,), dtype=np.float64)
        self.alphar = np.zeros(shape=(23, 7, 86), dtype=np.float64)

    def covfunc(self):
        self.covcn = np.zeros(shape=(self.nat,), dtype=np.float64)
        # Fitted to match Wiberg bond orders of diatomic molecules
        k4 = 4.10451
        k5 = 19.08857
        k6 = 2 * 11.28174**2

        kn = 7.50
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            for j in range(self.nat):
                if i is j:
                    continue
                ja = self.at_list[j] - 1
                dx = self.positions_list[j][0] - self.positions_list[i][0]
                dy = self.positions_list[j][1] - self.positions_list[i][1]
                dz = self.positions_list[j][2] - self.positions_list[i][2]
                rSquared = dx * dx + dy * dy + dz * dz
                if rSquared > self.params.threshold:
                    continue
                r = np.sqrt(rSquared)
                rco = rcov[ia] + rcov[ja]
                eni = pauling_en[ia]
                enj = pauling_en[ja]
                den = k4 * np.exp(-((np.abs(eni - enj) + k5) ** 2) / k6)
                damp = den * 0.5 * (1 + special.erf(-kn * (r - rco) / rco))
                self.covcn[i] += damp

    def ObtainAP(self):
        self.SetupAP()
        self.covfunc()
        for i in range(self.nat):
            cncount = np.zeros(shape=(18,), dtype=int)
            cncount[0] = 1
            ia = self.at_list[i] - 1
            for j in range(refn[ia]):
                refis = refsys[j][ia] - 1
                refiz = zeff[refis]
                for jj in range(23):
                    self.alpha[jj] = (
                        sscale[refis]
                        * seciw[jj][refis]
                        * zeta(self.g_a, self.g_c * gam[refis], refiz, refh[j][ia] + refiz)
                    )
                icn = np.rint(refcn[j][ia])
                cncount[int(icn)] += 1
                for jj in range(23):
                    self.alphar[jj][j][ia] = np.maximum(
                        ascale[j][ia] * (alphaiw[jj][j][ia] - hcount[j][ia] * self.alpha[jj]),
                        0,
                    )
            for j in range(refn[ia]):
                icn = cncount[int(np.rint(refcn[j][ia]))]
                self.ncount[j][ia] = icn * (icn + 1) / 2

        # weigths
        gw = np.zeros(shape=(self.ndim,), dtype=np.float64)
        wf = 6.0
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            norm = 0.0
            for ii in range(refn[ia]):
                for iii in range(self.ncount[ii][ia]):
                    twf = (iii + 1) * wf
                    norm = norm + cngw(twf, self.covcn[i], refcn[ii][ia])
            norm = 1.0 / norm
            for ii in range(refn[ia]):
                k = self.itbl[ii][i]
                for iii in range(self.ncount[ii][ia]):
                    twf = (iii + 1) * wf
                    gw[k] += cngw(twf, self.covcn[i], refcn[ii][ia]) * norm

        # polarizabilities
        zetvec = np.zeros(shape=(self.ndim), dtype=np.float64)
        aw = np.zeros(shape=(23, self.nat), dtype=np.float64)
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            iz = zeff[ia]
            for ii in range(refn[ia]):
                k = self.itbl[ii][i]
                zetvec[k] = gw[k] * zeta(self.g_a, self.g_c * gam[ia], refx[ii][ia] + iz, self.qs[i] + iz)
                for iii in range(23):
                    aw[iii][i] += zetvec[k] * self.alphar[iii][ii][ia]

        atomicAiw = np.zeros(shape=(self.nat,), dtype=np.float64)
        for i in range(self.nat):
            atomicAiw[i] = aw[0][i]

        self.ap = atomicAiw


    
class VdW(KallistoBase):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def SetupVdW(self):
        # parameter
        self.osev = 1.0 / 7.0
        self.theta_a = 2.54

        # unpack tuple in sizes
        self.scale = self.params.vdw_scale
        self.vdw = np.zeros(shape=(self.nat,), dtype=np.float64)

    def truhlar(self):
        # Truhlar: theta_b fitted to match radii from DOI:
        # 10.1021/jp8111556
        for i in range(self.nat):
            ia = self.at_list[i]
            isym = chemical_symbols[ia]
            theta_b = truhlar[isym]
            self.vdw[i] = self.scale * theta_b * self.theta_a * np.power(self.aw[i], self.osev)

    def rahm(self):
        # Rahm: theta_b fitted to match radii from DOI:
        # 10.1002/chem.201700610
        for i in range(self.nat):
            ia = self.at_list[i]
            isym = chemical_symbols[ia]
            theta_b = rahm[isym]
            self.vdw[i] = self.scale * theta_b * self.theta_a * np.power(self.aw[i], self.osev)

    def ObtainRadii(self):
        """A method to compute van der Waals radii (vdw).

        VDW values are calculated for a given structure and are returned as an
        array."""

        self.GrabGeometryKallisto()
        self.SetupVdW()

        # get atomic polarizabilities
        self.aw = np.zeros(shape=(self.nat,), dtype=np.float64)
        for i in range(self.nat):
            ia = self.at_list[i] - 1
            self.aw[i] = eeq_alp[ia]

        if self.params.vdwtype == "truhlar":
            self.truhlar()
        elif self.params.vdwtype == "rahm":
            self.rahm()

class KallistoInformation(ProximityShell,AP,VdW,CovalentPartner):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()
    

    def RunKallistoBases(self):
        self.ObtainCN()
        self.ObtainProximityShell()
        self.ObtainAPC()
        self.ObtainAP()
        self.ObtainRadii()

    def GetCN(self,ind):
        if self.params.getCN:
            return [self.cns[ind]]
        else:
            return []
    
    def GetPS(self,ind):
        if self.params.getPS:
            return [self.ps[ind]]
        else:
            return []
    
    def GetAPC(self,ind):
        if self.params.getAPC:
            return [self.qs[ind]]
        else:
            return []
    
    def GetVdW(self,ind):
        if self.params.getvdwkallisto:
            return [self.vdw[ind]]
        else:
            return []   
        
    def GetAP(self,ind):
        if self.params.getAP:
            return [self.ap[ind]]
        else:
            return [] 
    
    
    




    