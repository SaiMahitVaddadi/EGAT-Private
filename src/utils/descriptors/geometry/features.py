
class GeometryFeatures:
    class __init__(self):
        pass


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

    def featurize_double_cubic_lattice_volume(self, mol):
        # Calculate the Double Cubic Lattice Volume surface area
        lattice = rdMolDescriptors.DoubleCubicLatticeVolume(mol)
        self.surface_area = lattice.GetSurfaceArea()
        self.volume = lattice.GetVolume()
        self.vdw = lattice.GetVDWVolume()
        self.sa_to_vol_ratio = self.surface_area / self.volume if self.volume != 0 else float('inf')