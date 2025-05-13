
## Possible Sprints - Atom/Bond Featurization:
- Recap: [1 day]
    - what bonds are broken?
    - What atoms are near to them? 
    - How far away are the bonds from said bond breaks? 
    - How far away are the atoms? 

- Conjugated Pi bonds: [1 day]
    - https://bertiewooster.github.io/2024/10/15/Color-from-Conjugation.html 

- Mendeleev Features [1 week]

- Reaction Only: [1 week]
    - Change in Atomic Fingerprint between Reactant and Product.
    - Get the Delta of any MolecularFeaturizer Atom_feature Matrix and Load it as a DeltaVector.

- RDF [1 month]:
    - Full RDF 
    - Element-Element RDF
    - Index RDF (Full)
    - Index RDF (Element by Element)
    - NOTE: Factor in CS for better results.

- QM: [2 weeks]
    - Charges
    - Fukui
    - Atomic Features 

- Other Featurizers: [1 year]
    - MolGraphConvFeaturizer 
    - Chemprop (v1 and v2)
    - EquivariantGraphFeaturizer
    - PagtnMolGraphFeaturizer
    - MXMNetFeaturizer




## Possible Sprints - Molecular Addons:

- 3D Addons [1 week]
    - Volumes
    - Surface Areas
    - Ratios
    - etc. 

- Interaction Descriptors: [2 weeks]
    - https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors
    - Chem-Protein and Chem-DNA 

- Medchem descriptors. [1 day]
    - CDK Fingerprints [1 week]:
        - use scijava to load cdk files. 
        - References: https://github.com/sebotic/cdk_pywrapper/blob/master/cdk_pywrapper/cdk_pywrapper.py#L452 and https://github.com/hcji/PyFingerprint/blob/master/PyFingerprint/cdk.py
        - Guide: https://cdk.github.io/cdk/latest/docs/api/org/openscience/cdk/aromaticity/package-summary.html

- QM:[2 weeks]
    - Thermodynamics


- Point Cloud Similarities: [1 month]
    - Write them
    - ShapeLinker

## Possible Sprints - EGAT Feature Blocks:
- v1 to v2 adapter: 
    - Port in all of the old egat models. 



- LLM Block:[1 month]
    - Write up an adapter for molfeat and huggingface and the IBM model. 
    - Add in the past finetuning codes from ChemLLM. 
    - Write the LLM Model.
    - Write the GNN+LLM model. 


- Biological LLM Block: [2 weeks]
    - Write up an adapter for huggingface.
    - Write up an adapter for ProteinMPNN. 
    - Write up and adapter from pdb ID or pdb file to Protein Sequence. 

- Biological Feature Block: [1 week]
    - Protein Features: https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors
    - DNA Features: https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors
    - Chem-Bio Interaction Descriptors: https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors

- Reaction + Molecular. [2 months]
    - Write a new dataset and dataloader. 

- Molecule + Protein. [1 month]
    - Write a new dataset and dataloader. 

- Unsupervised Feature Block [2 weeks]
    - gensim
    - PySGL
    - karateclub
    - StellarGraph
    - Ampligraph
    - CogDL
    - GRaKeL
    
## Possible Sprints - New Types of Learning:
- Continual Learning: [1 year]
    - setup scripts that use avalanche. 


- Imbalanced Learning: [1 year]
    - Basic Process: 
        1) Create Four Datasets: Train, Imb-Train, Test, Val.
        2) Load them into a dataloader. 
        3) Train on Train, Test, Val for x epochs. 
    - Write out skeleton code for undersampling by masking the dataset and creating a new dataloader from it. 
    - Write out skeleton code for oversampling 
    - Write out skeleton code for data augmentation.


- Few-shot learning: [1 year]
    - Write out the skeleton code that does so.
    - Use 2D and 3D molecular similarity. Clean up the REINVENT code that does so. 
    - Use the MCS tool to see the level of similarity between the support and query. 

- Symbolic Regression: [1 year]
    - Write out the skeleton code. 

- Active Learning: [1 year]
    - Write out skeleton code. 

## Possible Sprints - New Layers: 
- Mixing Layers for Multi-Component Runs: [1 week]
    - https://www.sciencedirect.com/science/article/pii/S138589472503058X?via%3Dihub 
    - Digital Discovery,2023,2,138
    - https://www.sciencedirect.com/science/article/pii/S0016236124023676?via%3Dihub 

- Quadratic NN: [1 year]
    - Write out skeleton code. 

- Error Analysis: [1 year]
    - Write out Polished Analysis code on results
    - Write out code on steps.
    - Write out code on ensemble models. 
    - Write out code on CV models. 
    - Write out Outlier Detection tools.
    - Write out Euclidian Distance Tools. 

- Cluster Analysis: [1 year]
    - Write out UMAP and other algorithms. 
    - Write out Graph Outlier Detection Algorithms. 

- Explainability: [1 year]
    - FP Model --> use SHAP scores. 
    - GNN --> write adapter for existing codes. 
    - LLM --> see what tools are out there. 
    - Write a new analyze command. 


- Other Predictors: [1 year]
    - ProgressiveMultitaskRegressor
    - RobustMultitaskRegressor


- Atom-Bond Property Prediction:
    - BPNN
        - SingleNN [https://pubs.acs.org/doi/10.1021/acs.jpcc.0c04225]
    - Guan Paper
    - GAT Deconvolution/MLP. 
    - https://pubs.acs.org/doi/10.1021/acs.jcim.9b00994


### Testing the Graphs:
https://github.com/AntixK/PyTorch-VAE
https://github.com/eriklindernoren/PyTorch-GAN
https://github.com/RobinMagnet/pyFM
https://github.com/asperti/We_love_latent_space
https://github.com/ctom2/latent-space-transform

https://github.com/ldeecke/gmm-torch
https://github.com/tianrui-qi/QuadraticNeurons
https://github.com/phylyd/QBNN/tree/master
https://github.com/NasrinR791/Boolean-functions-NN-implementation/blob/main/AND_OR_Perceptron.py
https://thegrigorian.medium.com/capturing-non-linear-relationships-with-quadratic-layers-55e9f0f5d006
https://github.com/gd-zhang/noisy-quadratic-model/blob/master/nqm.ipynb
https://locuslab.github.io/qpth/
https://github.com/miniHuiHui/QuadraNet/blob/main/QuadraNet.py
https://github.com/yuweien1120/CCQNet
https://github.com/zarekxu/QuadraLib

