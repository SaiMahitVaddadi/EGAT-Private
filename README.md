# ChemEGAT: Edge-Featured Graph Attention Networks for Reaction and Molecular Property Prediction

ChemEGAT is a repository that adapts Edge-Featured Graph Attention Networks (EGAT) for applications in chemical property prediction. It was first described in the paper published on ArXiv (https://chemrxiv.org/engage/chemrxiv/article-details/65410dc248dad23120c6e954) that is now published in the Journal of Physical Chemistry A. More papers are on the way, and those will be highlighted in the READ-ME in the future. 

## Use Cases:
 - Reaction Property Prediction: Performance on RGD1 [https://chemrxiv.org/engage/chemrxiv/article-details/65410dc248dad23120c6e954]
 - Molecular Property Prediction: Coming Soon
 - Large Property Model: Coming Soon
 - Molecular Classification: under construction
 - Reaction Classification: under construction
 - Few-shot Learning: under construction
 - Many-shot Learning: under construction
 - Imbalanced Learning: under construction
 - Multi-Component Modeling: under construction
 - Spectra Prediction: under construction

## Applications:   
 - Reaction Prediction: ChemArXiv  
 - Molecular Property Prediction: Coming Soon
 - Large Property Model: Coming Soon
 
## Features:  

| Feature               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Node Features         | Includes atomic properties such as atomic number, hybridization, and valence.  |
| Edge Features         | Captures bond properties like bond type, bond length, and aromaticity.         |
| Graph-Level Features  | Represents global properties such as molecular weight and total charge.        |
| Custom Features       | Allows users to define domain-specific features for specialized applications.   |


## Graph Information:
| **Feature** | **Molecule (2D)** | **Molecule (3D)** | **Molecule (2D,Global)** | **Molecule (3D,Global)** | **Reaction (2D)** | **Reaction (3D)** | **Reaction (2D,Global)** | **Reaction (3D,Global)** | 
|-------------|-------------------|-------------------|--------------------------|--------------------------|-------------------|-------------------|--------------------------|--------------------------|
| Acid-Base Site<sup>1</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Aromaticity<sup>1</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bond Order<sup>2</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Distance to Center of Mass | X | ✅ | X | ✅ | X | ✅ | X | ✅ |
| Steric Hindrance | X | ✅ | X | ✅ | X | ✅ | X | ✅ |
| VdW Strain | X | ✅ | X | ✅ | X | ✅ | X | ✅ |
| Distance to Center of Mass | X | ✅ | X | ✅ | X | ✅ | X | ✅ |



**Documentation:** 

#### 1. Creating a Config File:

The first step in creating EGAT is to make a configuration file. You can do this in two ways: 1) by modifying the current config files available in the config file folder, or using a CLI based solution which can be done with Config.py. To see what kinds of arguements are there, please use the command below:

python Config.py --help

#### 2. Generating Graphs:

To generate the graphs, please use the command below:

python Generate.py --config 

#### 3. Training:

To train the model, please use the command below:

python Train.py --config 

#### 4. Model Prediction:

To predict the model using another model, please use the command below:

python Predict.py --config 

**Tutorial:** Sides. 

**License:** 

# Hardware Reqirements 

- NVIDIA GPU
- Python 3.8

# Installing conda environment 
You can use the yaml file and create the conda environment. 

conda env create -f ENV_Gilbreth_Purdue.yaml


# How to Download EGAT to your Home Computer/Cluster

1. Clone the repository:

git clone https://github.com/SaiMahitVaddadi/EGAT-Private.git --branch v2












