import numpy as np
from rdkit.Chem import DataStructs
from scipy.spatial import distance
from rdkit.DataStructs import ConvertToNumpyArray
import traceback 

def convert_to_numpy(fp):
    if not isinstance(fp,np.ndarray):
        arr = np.zeros((0,))
        ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return fp

def ensure_1d(array):
    np_array = np.array(array)  # Convert to NumPy array
    if np_array.ndim != 1:
        print(f"Array is not 1-D: {np_array}")
    return np.ravel(np_array)  # Convert to 1-D array if not already


class Similarity:

    def calculate_tanimoto_batch(self, fp, fps) -> np.array:
        res = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))
        return res
    
    def calculate_tanimoto(self, query_fps, ref_fingerprints) -> np.array:
        return np.array([np.max(DataStructs.BulkTanimotoSimilarity(fp, ref_fingerprints)) for fp in query_fps])

    def calculate_jaccard_distance(self, query_fps, ref_fingerprints) -> np.array:
        tanimoto = self.calculate_tanimoto(query_fps, ref_fingerprints)
        jaccard = 1 - tanimoto
        return jaccard

    def calculate_dice(self, query_fps, ref_fingerprints) -> np.array:
        return np.array(DataStructs.BulkDiceSimilarity(query_fps, ref_fingerprints))

    def calculate_cosine(self, query_fps, ref_fingerprints) -> np.array:
        return np.array(DataStructs.BulkCosineSimilarity(query_fps, ref_fingerprints))

    def calculate_russel_rao(self, query_fps, ref_fingerprints) -> np.array:
        return np.array(DataStructs.BulkRussellSimilarity(query_fps, ref_fingerprints))

    def calculate_kulczynski(self, query_fps, ref_fingerprints) -> np.array:
        return np.array(DataStructs.BulkKulczynskiSimilarity(query_fps, ref_fingerprints))
        
    def calculate_mcconnaughey(self, query_fps, ref_fingerprints) -> np.array:
        return np.array(DataStructs.BulkMcConnaugheySimilarity(query_fps, ref_fingerprints))
    
    def calculate_inverse_dice(self, query_fps, ref_fingerprints) -> np.array:
        return 1 - np.array(DataStructs.BulkDiceSimilarity(query_fps, ref_fingerprints))
    
    def calculate_inverse_tanimoto(self, query_fps, ref_fingerprints) -> np.array:
        return 1 - np.array(DataStructs.BulkTanimotoSimilarity(query_fps, ref_fingerprints))

    def calculate_inverse_cosine(self, query_fps, ref_fingerprints) -> np.array:
        return 1 - np.array(DataStructs.BulkCosineSimilarity(query_fps, ref_fingerprints))

    def calculate_inverse_russel_rao(self, query_fps, ref_fingerprints) -> np.array:
        return 1 - np.array(DataStructs.BulkRussellSimilarity(query_fps, ref_fingerprints))

    def calculate_inverse_kulczynski(self, query_fps, ref_fingerprints) -> np.array:
        return 1 - np.array(DataStructs.BulkKulczynskiSimilarity(query_fps, ref_fingerprints))
        
    def calculate_inverse_mcconnaughey(self, query_fps, ref_fingerprints) -> np.array:
        return 1 - np.array(DataStructs.BulkMcConnaugheySimilarity(query_fps, ref_fingerprints))


    
    def calculate_euclidean(self,fp, ref_fingerprints) -> np.array:
        res = [] 

        if not isinstance(fp,np.ndarray):
            fp =  convert_to_numpy(fp)
        if fp.shape == 2:
            if fp.shape[0] == 1:
                fp = fp[0]
            else:
                fp =  ensure_1d(fp) 
        
        for ref_fp in ref_fingerprints:
            if not isinstance(ref_fp,np.ndarray):
                ref_fp =  convert_to_numpy(ref_fp)
            if ref_fp.shape == 2:
                if ref_fp.shape[0] == 1:
                    ref_fp = ref_fp[0]
                else:
                    ref_fp =  ensure_1d(ref_fp) 
            if len(np.where(np.isnan(fp) | np.isinf(fp))) == 0 and np.where(np.isnan(ref_fp) | np.isinf(ref_fp)) == 0:
                
                s = distance.euclidean(fp,ref_fp)
            else:
                if len(fp) == len(ref_fp):
                    fp_badlocations = np.where(np.isnan(fp) | np.isinf(fp))
                    ref_badlocations = np.where(np.isnan(ref_fp) | np.isinf(ref_fp))
                    badlocations = np.union1d(fp_badlocations, ref_badlocations)
                    tmpfp = fp
                    tmpref_fp = ref_fp
                else:
                    if len(fp) > len(ref_fp):
                        tmpfp = fp[:len(ref_fp)]
                    else:
                        tmpref_fp = ref_fp[:len(fp)]
                    
                    fp_badlocations = np.where(np.isnan(tmpfp) | np.isinf(tmpfp))
                    ref_badlocations = np.where(np.isnan(tmpref_fp) | np.isinf(tmpref_fp))
                    badlocations = np.union1d(fp_badlocations, ref_badlocations)
                try:
                    tmpfp = np.delete(tmpfp, badlocations)
                    tmpref_fp = np.delete(tmpref_fp, badlocations)

                    if not isinstance(tmpfp,np.ndarray):
                        tmpfp =  convert_to_numpy(tmpfp)
                    if tmpfp.shape == 2:
                        if tmpfp.shape[0] == 1:
                            tmpfp = tmpfp[0]
                        else:
                            tmpfp =  ensure_1d(tmpfp) 

                    if tmpref_fp.shape == 2:
                        if tmpref_fp.shape[0] == 1:
                            tmpref_fp = tmpref_fp[0]
                        else:
                            tmpref_fp =  ensure_1d(tmpref_fp) 
                    s = distance.euclidean(tmpfp, tmpref_fp)
                except:
                    traceback.print_exc()
                    s = 10**6
            res += [s]
        res = np.array(res)
        return res 
    def calculate_manhattan(self,fp, ref_fingerprints) -> np.array:
        res = [] 
        for ref_fp in ref_fingerprints:
            try:
                s = distance.euclidean(convert_to_numpy(fp),convert_to_numpy(ref_fp))
            except:
                print('Problem: \n', 'Molecule:\n---------------------------------------\n', fp, '\n---------------------------------------\nMolecule:\n---------------------------------------\n',ref_fp)
                s = 1000
            res += [s]
        res = np.array(res)
        return res 

    def calculate_chebyshev(self, fp, ref_fingerprints) -> np.array:
        res = [] 
        for ref_fp in ref_fingerprints:
            try:
                s = distance.euclidean(convert_to_numpy(fp),convert_to_numpy(ref_fp))
            except:
                print('Problem: \n', 'Molecule:\n---------------------------------------\n', fp, '\n---------------------------------------\nMolecule:\n---------------------------------------\n',ref_fp)
                s = 1000
            res += [s]
        res = np.array(res)
        return res 
    
    def calculate_minkowski(self, fp, ref_fingerprints, p=3) -> np.array:
        res = [] 
        for ref_fp in ref_fingerprints:
            try:
                s = distance.euclidean(convert_to_numpy(fp),convert_to_numpy(ref_fp))
            except:
                print('Problem: \n', 'Molecule:\n---------------------------------------\n', fp, '\n---------------------------------------\nMolecule:\n---------------------------------------\n',ref_fp)
                s = 1000
            res += [s]
        res = np.array(res)
        return res 

    def calculate_canberra(self, fp, ref_fingerprints) -> np.array:
        res = [] 
        for ref_fp in ref_fingerprints:
            try:
                s = distance.euclidean(convert_to_numpy(fp),convert_to_numpy(ref_fp))
            except:
                print('Problem: \n', 'Molecule:\n---------------------------------------\n', fp, '\n---------------------------------------\nMolecule:\n---------------------------------------\n',ref_fp)
                s = 1000
            res += [s]
        res = np.array(res)
        return res 
    def calculate_bray_curtis(self, fp, ref_fingerprints) -> np.array:
        res = [] 
        for ref_fp in ref_fingerprints:
            try:
                s = distance.euclidean(convert_to_numpy(fp),convert_to_numpy(ref_fp))
            except:
                print('Problem: \n', 'Molecule:\n---------------------------------------\n', fp, '\n---------------------------------------\nMolecule:\n---------------------------------------\n',ref_fp)
                s = 1000
            res += [s]
        res = np.array(res)
        return res 
    def calculate_correlation(self, fp, ref_fingerprints) -> np.array:
        res = [] 
        for ref_fp in ref_fingerprints:
            try:
                s = distance.euclidean(convert_to_numpy(fp),convert_to_numpy(ref_fp))
            except:
                print('Problem: \n', 'Molecule:\n---------------------------------------\n', fp, '\n---------------------------------------\nMolecule:\n---------------------------------------\n',ref_fp)
                s = 1000
            res += [s]
        res = np.array(res)
        return res 
    def calculate_hamming(self, fp, ref_fingerprints) -> np.array:
        res = [] 
        for ref_fp in ref_fingerprints:
            try:
                s = distance.euclidean(convert_to_numpy(fp),convert_to_numpy(ref_fp))
            except:
                print('Problem: \n', 'Molecule:\n---------------------------------------\n', fp, '\n---------------------------------------\nMolecule:\n---------------------------------------\n',ref_fp)
                s = 1000
            res += [s]
        res = np.array(res)
        return res 

    