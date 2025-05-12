
## Possible Sprints:
- Continual Learning: 
    - setup scripts that use avalanche. 

- 3D Addons
    - Volumes
    - Surface Areas
    - Ratios
    - etc. 

- More Molecular Features: 
    - Recap: [1 day]
        - what bonds are broken?
        - What atoms are near to them? 
        - How far away are the bonds from said bond breaks? 
        - How far away are the atoms? 
    - Conjugated Pi bonds: [1 day]
        - https://bertiewooster.github.io/2024/10/15/Color-from-Conjugation.html 
    - RDF [1 month]:
        - Full RDF 
        - Element-Element RDF
        - Index RDF (Full)
        - Index RDF (Element by Element)
        - NOTE: Factor in CS for better results. 
    - Interaction Descriptors: [2 weeks]
        - https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors
        - Chem-Protein and Chem-DNA 
    - Medchem descriptors. [1 day]
    - CDK Fingerprints [1 week]:
        - use scijava to load cdk files. 
        - References: https://github.com/sebotic/cdk_pywrapper/blob/master/cdk_pywrapper/cdk_pywrapper.py#L452 and https://github.com/hcji/PyFingerprint/blob/master/PyFingerprint/cdk.py
        - Guide: https://cdk.github.io/cdk/latest/docs/api/org/openscience/cdk/aromaticity/package-summary.html
    - Quantum Mechanics: [1 week]
        - use ASE and pySCF to run optimizations and then calculate values. 
    - Mendeleev Features


- Feature Blocks:
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

- Multicomponent: Can do via Code but you need special Datasets and DataLoaders for now. [2 months]
    - Reaction + Molecule runs.
    - Any runs. (Molecule + Protein) 


- Mixing Layers for MC Runs:
    - https://www.sciencedirect.com/science/article/pii/S138589472503058X?via%3Dihub 
    - Digital Discovery,2023,2,138
    - https://www.sciencedirect.com/science/article/pii/S0016236124023676?via%3Dihub 

- Imbalanced Learning: 
    - Basic Process: 
        1) Create Four Datasets: Train, Imb-Train, Test, Val.
        2) Load them into a dataloader. 
        3) Train on Train, Test, Val for x epochs. 
    - Write out skeleton code for undersampling by masking the dataset and creating a new dataloader from it. 
    - Write out skeleton code for oversampling 
    - Write out skeleton code for data augmentation.

- Few-shot learning:
    - Write out the skeleton code that does so.
    - Use 2D and 3D molecular similarity. Clean up the REINVENT code that does so. 
    - Use the MCS tool to see the level of similarity between the support and query. 

- Symbolic Regression:
    - Write out the skeleton code. 

- Active Learning: 
    - Write out skeleton code. 

- Quadratic NN:
    - Write out skeleton code. 

- Error Analysis:
    - Write out Polished Analysis code on results
    - Write out code on steps.
    - Write out code on ensemble models. 
    - Write out code on CV models. 
    - Write out Outlier Detection tools.
    - Write out Euclidian Distance Tools. 

- Cluster Analysis:
    - Write out UMAP and other algorithms. 
    - Write out Graph Outlier Detection Algorithms. 

- Explainability:
    - FP Model --> use SHAP scores. 
    - GNN --> write adapter for existing codes. 
    - LLM --> see what tools are out there. 
    - Write a new analyze command. 

- Other Featurizers: 
    - MolGraphConvFeaturizer 
    - Chemprop (v1 and v2)
    - EquivariantGraphFeaturizer
    - PagtnMolGraphFeaturizer
    - MXMNetFeaturizer

- Other Predictors: 
    - ProgressiveMultitaskRegressor
    - RobustMultitaskRegressor

- Point Cloud Similarities:
    - Write them


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

