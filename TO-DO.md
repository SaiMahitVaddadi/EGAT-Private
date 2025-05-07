# TO-DO - Tomorrow:

## To Test and Fix: 
- Dataloader.
- ML Model Setups. 

### Codes to Modify: 
- Cleanup the Training and Prediction codes.
- Write a v1 to v2 Converter. 
- Write a pre-trained model user (Supersede GraphParams for Pre-trained model.)
- Expand for the multi geometry and multi bond matrix case.  
- Expand for the multi component case. 
- Clean up the params class.
- Write a new generate command. 
- Write a new train command. 
- Write a new predict command.
- Write a new fingerprint command. 

## Possible Sprints:
- More Molecular Features:
    - Recap: what bonds are broken? What atoms are near to them? How far away are the bonds from said bond breaks? How far away are the atoms? 
    - Conjugated Pi bonds: https://bertiewooster.github.io/2024/10/15/Color-from-Conjugation.html

- More Add-ons:
    - Chem-Chem Interaction Descriptors: https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors
    - Medchem descriptors. 

- Mixing Layers for MC Runs:
    - https://www.sciencedirect.com/science/article/pii/S138589472503058X?via%3Dihub
    - Digital Discovery,2023,2,138
    - https://www.sciencedirect.com/science/article/pii/S0016236124023676?via%3Dihub 


- LLM Block:
    - Write up an adapter for molfeat and huggingface and the IBM model. 
    - Add in the past finetuning codes from ChemLLM. 
    - Write the LLM Model.
    - Write the GNN+LLM model. 

- Biological Feature Block:
    - Protein Features: https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors
    - DNA Features: https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors
    - Chem-Bio Interaction Descriptors: https://pybiomed.readthedocs.io/en/latest/User_guide.html#calculating-molecular-descriptors

- Biological LLM Block:
    - Write up an adapter for huggingface.
    - Write up an adapter for ProteinMPNN. 
    - Write up and adapter from pdb ID or pdb file to Protein Sequence. 

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

- Pooling for Coarse Graining: 
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

- Point Cloud Similarities:

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





