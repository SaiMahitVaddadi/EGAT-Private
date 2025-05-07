import pandas as pd
import re

# Sample texts
output1 = """Function EncodeElement executed successfully for atom index 0, with result: [6, 12.011] and vector length 2
Function GetNeighbors executed successfully for atom index 0, with result: [1, 2, 0, 0] and vector length 4
Function RingCheck executed successfully for atom index 0, with result: [1] and vector length 1
... (truncated for brevity in this explanation) ...
Function AtomResonance executed successfully for atom index 0, with result: [1, 0] and vector length 2"""

output2 = """Function EncodeElement executed successfully for atom index 12, with result: [1, 1.00794] and vector length 2
Function GetNeighbors executed successfully for atom index 12, with result: [0, 0, 0, 1] and vector length 4
Function RingCheck executed successfully for atom index 12, with result: [0] and vector length 1
... (truncated for brevity in this explanation) ...
Function RandicAtom executed successfully for atom index 12, with result: [1.0, 1"""

# Since input is truncated, load full data instead (we assume actual data is available in the original request)
# Use regex to extract Function and vector length
def extract_functions_and_lengths(text):
    pattern = r"Function (.*?) executed successfully.*?vector length (\d+)"
    return re.findall(pattern, text)

# Full outputs provided in user's original message
full_output1 = """Function EncodeElement executed successfully for atom index 0, with result: [6, 12.011] and vector length 2
Function GetNeighbors executed successfully for atom index 0, with result: [1, 2, 0, 0] and vector length 4
Function RingCheck executed successfully for atom index 0, with result: [1] and vector length 1
Function FormalCharge executed successfully for atom index 0, with result: [0] and vector length 1
Function AromaticityCheck executed successfully for atom index 0, with result: [1] and vector length 1
Function HybridizationCheck executed successfully for atom index 0, with result: [0, 1, 0, 0, 0, 0, 0, 0, 0] and vector length 9
Function ChiralityCheck executed successfully for atom index 0, with result: [0, 0, 1] and vector length 3
Function RadicalCheck executed successfully for atom index 0, with result: [0, 0.0] and vector length 2
Function SpiroCheck executed successfully for atom index 0, with result: [1, 0] and vector length 2
Function BridgeHeadCheck executed successfully for atom index 0, with result: [1, 0] and vector length 2
Function ElectronegativityCheck executed successfully for atom index 0, with result: [2.55] and vector length 1
Function ChargeCheck executed successfully for atom index 0, with result: [-0.06199998663933999] and vector length 1
Function AcidBaseCheck executed successfully for atom index 0, with result: [1, 0] and vector length 2
Function IsPartOfBRICSBond executed successfully for atom index 0, with result: [0] and vector length 1
Function AtominFusedRing executed successfully for atom index 0, with result: [1, 0] and vector length 2
Function AtominXRings executed successfully for atom index 0, with result: [1] and vector length 1
Function PiElectrons executed successfully for atom index 0, with result: [2] and vector length 1
Function SigmaElectrons executed successfully for atom index 0, with result: [3] and vector length 1
Function CoreElectrons executed successfully for atom index 0, with result: [0.5] and vector length 1
Function RamificationNumber executed successfully for atom index 0, with result: [2.0, 3.0] and vector length 2
Function IonizationPotential executed successfully for atom index 0, with result: [11.2603] and vector length 1
Function SurroundingIPFeatures executed successfully for atom index 0, with result: [12.039681000000002, 11.2603, 11.2603, 13.598443, 1.1022111804559045] and vector length 5
Function IntrinsicState executed successfully for atom index 0, with result: [2.0] and vector length 1
Function EtaBeta executed successfully for atom index 0, with result: [0.0, 4.0, 1.0, 0.1, 0.7] and vector length 5
Function LocantCountForAtom executed successfully for atom index 0, with result: [] and vector length 0
Function DeltaInteraction executed successfully for atom index 0, with result: [-12.274879199150265, 0, 0, 1] and vector length 4
Function FreeEnergies executed successfully for atom index 0, with result: [0, -0.0, -0.0] and vector length 3
Function GetAP executed successfully for atom index 0, with result: [7.3469847886478625] and vector length 1
Function GetAPC executed successfully for atom index 0, with result: [-0.15049551531799799] and vector length 1
Function GetCN executed successfully for atom index 0, with result: [7.966475135431284] and vector length 1
Function GetPS executed successfully for atom index 0, with result: [2.6716237721071074] and vector length 1
Function GetVdW executed successfully for atom index 0, with result: [2.4755589180129185] and vector length 1
Function BaryszAtom executed successfully for atom index 0, with result: [0.0] and vector length 1
Function ATSAtom executed successfully for atom index 0, with result: [347.1442370883433] and vector length 1
Function PropsRelativetoCarbon executed successfully for atom index 0, with result: [] and vector length 0
Function VertexDistanceDegreeAtom executed successfully for atom index 0, with result: [41.0] and vector length 1
Function SuperdenticIndex executed successfully for atom index 0, with result: [1728.0] and vector length 1
Function Eccentricity executed successfully for atom index 0, with result: [6.0, 0.017751479289940836, 24.0] and vector length 3
Function Schultz executed successfully for atom index 0, with result: [64.0] and vector length 1
Function Xui executed successfully for atom index 0, with result: [8, 32] and vector length 2
Function CoreCount executed successfully for atom index 0, with result: [0.5] and vector length 1
Function VEMAtom executed successfully for atom index 0, with result: [0.5, 2.0, 5.0, -3.0, 0.0] and vector length 5
Function EtaPsi executed successfully for atom index 0, with result: [0.7142857142857143] and vector length 1
Function ZagrebAtom executed successfully for atom index 0, with result: [4] and vector length 1
Function HarmonicAtom executed successfully for atom index 0, with result: [0.5, 0.25] and vector length 2
Function SomborAtom executed successfully for atom index 0, with result: [2.0] and vector length 1
Function RandicAtom executed successfully for atom index 0, with result: [0.7071067811865475, 0.5] and vector length 2
Function NirmalaAtom executed successfully for atom index 0, with result: [4.1132503787829275] and vector length 1
Function ESOSAtom executed successfully for atom index 0, with result: [4.0] and vector length 1
Function AugmentedGraphAttributeAtom executed successfully for atom index 0, with result: [0.7071067811865476, 1.414213562373095] and vector length 2
Function HyperbolicAtom executed successfully for atom index 0, with result: [2.718281828459045] and vector length 1
Function AugZagrebAtom executed successfully for atom index 0, with result: [inf] and vector length 1
Function KleinAtom executed successfully for atom index 0, with result: [3.0] and vector length 1
Function HyperDegreeAtom executed successfully for atom index 0, with result: [0.5] and vector length 1
Function VEWIAtom executed successfully for atom index 0, with result: [19] and vector length 1
Function VEWIAtomByOrder executed successfully for atom index 0, with result: [0.15789473684210525, 0.21052631578947367, 0.21052631578947367] and vector length 3
Function HDSA executed successfully for atom index 0, with result: [-0.0033723546912425117] and vector length 1
Function ETSAtom executed successfully for atom index 0, with result: [2.5] and vector length 1
Function InformationContent executed successfully for atom index 0, with result: [0.1, -0.33219280948873625, -4.318506523353571, -0.08977117175026231, -0.08304820237218406] and vector length 5
Function WeightedInformationContent executed successfully for atom index 0, with result: [0.011223974409338349, -0.07270074151237689, -0.9451096396608996, -0.01964651421180236, -0.018175185378094223] and vector length 5
Function VertexAdjacency executed successfully for atom index 0, with result: [2.0] and vector length 1
Function TPSA executed successfully for atom index 0, with result: [0.0] and vector length 1
Function LabuteASA executed successfully for atom index 0, with result: [6.042418707663295] and vector length 1
Function LogS executed successfully for atom index 0, with result: [-0.3757] and vector length 1
Function EState executed successfully for atom index 0, with result: [-0.48999999999999955, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] and vector length 36
Function TopoChargeAtom executed successfully for atom index 0, with result: [0, 0, -0.04351348522149149, 0.00957195219945653] and vector length 4
Function SMRandSLogP executed successfully for atom index 0, with result: [0.1581, 3.35, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] and vector length 12
Function ChiAtom executed successfully for atom index 0, with result: [0.7071067811865475, 0.7071067811865475] and vector length 2
Function ChiAtomValence executed successfully for atom index 0, with result: [0.7071067811865475, 0, 0.7071067811865475] and vector length 3
Function AtomResonance executed successfully for atom index 0, with result: [1, 0] and vector length 2
"""  # Replace with the actual full text from the user's message
full_output2 = """Function EncodeElement executed successfully for atom index 12, with result: [1, 1.00794] and vector length 2
Function GetNeighbors executed successfully for atom index 12, with result: [0, 0, 0, 1] and vector length 4
Function RingCheck executed successfully for atom index 12, with result: [0] and vector length 1
Function FormalCharge executed successfully for atom index 12, with result: [0] and vector length 1
Function AromaticityCheck executed successfully for atom index 12, with result: [0] and vector length 1
Function HybridizationCheck executed successfully for atom index 12, with result: [0, 0, 0, 1] and vector length 4
Function ChiralityCheck executed successfully for atom index 12, with result: [0, 0, 1] and vector length 3
Function RadicalCheck executed successfully for atom index 12, with result: [0, 0.0] and vector length 2
Function SpiroCheck executed successfully for atom index 12, with result: [1, 0] and vector length 2
Function BridgeHeadCheck executed successfully for atom index 12, with result: [1, 0] and vector length 2
Function ElectronegativityCheck executed successfully for atom index 12, with result: [2.2] and vector length 1
Function ChargeCheck executed successfully for atom index 12, with result: [0.2930920665845635] and vector length 1
Function AcidBaseCheck executed successfully for atom index 12, with result: [0, 0] and vector length 2
Function IsPartOfBRICSBond executed successfully for atom index 12, with result: [0] and vector length 1
Function AtominFusedRing executed successfully for atom index 12, with result: [1, 0] and vector length 2
Function AtominXRings executed successfully for atom index 12, with result: [0] and vector length 1
Function PiElectrons executed successfully for atom index 12, with result: [0] and vector length 1
Function SigmaElectrons executed successfully for atom index 12, with result: [1] and vector length 1
Function CoreElectrons executed successfully for atom index 12, with result: [0.0] and vector length 1
Function RamificationNumber executed successfully for atom index 12, with result: [-0.5, 0.5] and vector length 2
Function IonizationPotential executed successfully for atom index 12, with result: [13.598443] and vector length 1
Function SurroundingIPFeatures executed successfully for atom index 12, with result: [13.61805, 13.61805, 13.61805, 13.61805, 0.0] and vector length 5
Function IntrinsicState executed successfully for atom index 12, with result: [1.0] and vector length 1
Function EtaBeta executed successfully for atom index 12, with result: [0.0, 0.0, 0.75, 0.0, 0.3] and vector length 5
Function LocantCountForAtom executed successfully for atom index 12, with result: [] and vector length 0
Function DeltaInteraction executed successfully for atom index 12, with result: [16.46595644884819, 1, 0, 0] and vector length 4
Function FreeEnergies executed successfully for atom index 12, with result: [0.0, -0.0, 0.0] and vector length 3
Function GetAP executed successfully for atom index 12, with result: [0.8794291335937624] and vector length 1
Function GetAPC executed successfully for atom index 12, with result: [0.5106834160267846] and vector length 1
Function GetCN executed successfully for atom index 12, with result: [2.316333518611657] and vector length 1
Function GetPS executed successfully for atom index 12, with result: [2.1405089033764493] and vector length 1
Function GetVdW executed successfully for atom index 12, with result: [1.5164749179772914] and vector length 1
Function BaryszAtom executed successfully for atom index 12, with result: [-5.0] and vector length 1
Function ATSAtom executed successfully for atom index 12, with result: [205.8626958885928] and vector length 1
Function PropsRelativetoCarbon executed successfully for atom index 12, with result: [] and vector length 0
Function VertexDistanceDegreeAtom executed successfully for atom index 12, with result: [58.0] and vector length 1
Function SuperdenticIndex executed successfully for atom index 12, with result: [6300.0] and vector length 1
Function Eccentricity executed successfully for atom index 12, with result: [7.0, 0.09467455621301776, 7.0] and vector length 3
Function Schultz executed successfully for atom index 12, with result: [89.0] and vector length 1
Function Xui executed successfully for atom index 12, with result: [1, 1] and vector length 2
Function CoreCount executed successfully for atom index 12, with result: [0.0] and vector length 1
Function VEMAtom executed successfully for atom index 12, with result: [0.375, 0.0, 0.75, 0.75, 0.0] and vector length 5
Function EtaPsi executed successfully for atom index 12, with result: [0.0] and vector length 1
Function ZagrebAtom executed successfully for atom index 12, with result: [1] and vector length 1
Function HarmonicAtom executed successfully for atom index 12, with result: [2.0, 1.0] and vector length 2
Function SomborAtom executed successfully for atom index 12, with result: [1.0] and vector length 1
Function RandicAtom executed successfully for atom index 12, with result: [1.0, 1.0] and vector length 2
Function NirmalaAtom executed successfully for atom index 12, with result: [2.718281828459045] and vector length 1
Function ESOSAtom executed successfully for atom index 12, with result: [1.0] and vector length 1
Function AugmentedGraphAttributeAtom executed successfully for atom index 12, with result: [1.0, 1.0] and vector length 2
Function HyperbolicAtom executed successfully for atom index 12, with result: [2.718281828459045] and vector length 1
Function AugZagrebAtom executed successfully for atom index 12, with result: [-1.0] and vector length 1
Function KleinAtom executed successfully for atom index 12, with result: [1.0] and vector length 1
Function HyperDegreeAtom executed successfully for atom index 12, with result: [0.5] and vector length 1
Function VEWIAtom executed successfully for atom index 12, with result: [38] and vector length 1
Function VEWIAtomByOrder executed successfully for atom index 12, with result: [0.02631578947368421, 0.02631578947368421, 0.05263157894736842] and vector length 3
Function HDSA executed successfully for atom index 12, with result: [0.02254554358342796] and vector length 1
Function ETSAtom executed successfully for atom index 12, with result: [3.5] and vector length 1
Function InformationContent executed successfully for atom index 12, with result: [0.05, -0.21609640474436814, -2.809253261676786, -0.058397493596497145, -0.054024101186092036] and vector length 5
Function WeightedInformationContent executed successfully for atom index 12, with result: [0.0056119872046691745, -0.04196235796085762, -0.5455106534911491, -0.011339830170760712, -0.010490589490214405] and vector length 5
Function VertexAdjacency executed successfully for atom index 12, with result: [-inf] and vector length 1
Function TPSA executed successfully for atom index 12, with result: [0.0] and vector length 1
Function LabuteASA executed successfully for atom index 12, with result: [1.4311996572326342] and vector length 1
Function LogS executed successfully for atom index 12, with result: [0.0] and vector length 1
Function EState executed successfully for atom index 12, with result: [6.521296296296296, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] and vector length 36
Function TopoChargeAtom executed successfully for atom index 12, with result: [0, 0, 0.13986677253992721, -0.1630114456762305] and vector length 4
Function SMRandSLogP executed successfully for atom index 12, with result: [-0.2677, 1.395, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] and vector length 12
Function ChiAtom executed successfully for atom index 12, with result: [1.0, 1.0] and vector length 2
Function ChiAtomValence executed successfully for atom index 12, with result: [1.0, 0, 1.0] and vector length 3
Function AtomResonance executed successfully for atom index 12, with result: [1, 0] and vector length 2
"""  # Replace with the actual full text from the user's message

# Extract data
functions_lengths_1 = extract_functions_and_lengths(full_output1)
functions_lengths_2 = extract_functions_and_lengths(full_output2)

# Convert to DataFrame
df1 = pd.DataFrame(functions_lengths_1, columns=["Function", "VectorLength_Atom0"])
df2 = pd.DataFrame(functions_lengths_2, columns=["Function", "VectorLength_Atom12"])

# Merge the two DataFrames
merged_df = pd.merge(df1, df2, on="Function", how="outer")

# Convert lengths to integers for consistency
merged_df["VectorLength_Atom0"] = merged_df["VectorLength_Atom0"].astype("Int64")
merged_df["VectorLength_Atom12"] = merged_df["VectorLength_Atom12"].astype("Int64")

merged_df.to_csv("output.csv", index=False)