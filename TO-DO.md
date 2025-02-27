# TO-DO - Immediate:
- Params:
    - Cleanup the params class. 
    - Create a params class that caters to hyperparameter tuning
    - write a function that uses a cli to write a config file 
    - write a function that uses a cli to input the parameters

- CLI execution
    - write a cli for training 
    - write a cli for testing 
    - write a cli for fingerprinting
    - write a cli for few shot learning 
    - write a cli for hyperparameter tuning via grid search
    - write a cli that combines two yamls into one, any overlap is based on the last yaml file given, 

- HT Tuning: 
    - Set up a beam search protocol

- Build it into a pip install 
- Build it into a conda install 
- Build a Read the docs based on Sphinx


# TO-DO - Long-Term:

- GUI:
    - streamlit app interface. 
    - webhosting??????


- Types of Prediction:
    - Atom/Bond Property Prediction
    - Multi-component properties (mixtures, etc.)

- Other Featurizers: 
    - Chemprop
    - Deepchem

- Datasets: 
    - Multi-component dataset and dataloader 
    - write out the workaround that for x conformers for a 3D based model, you load x molecules into the dataset and dataloader.

- DataLoader:
    - write out the dataloader from a JSON folder
    - write out the dataloader from a SDF folder
    - write out the dataloader from a TOML folder
    - write out the dataloader from a CSV folder
    - write out the dataloader from an SQL
    - write out the dataloader from a Parquet folder

- EGAT Models: 
    - Fully custom activation functions 
    - Fully custom end convolution layers for spectra
    - Fully custom softmax style layers 

- Pytorch Lightning Support:
    - FP Model 
    - EGAT Model
    - General Model 

- PyG support: 
    - EGAT Model
    - General Model

- Additional Layers:
    - EGT
    - TGT
    - GRU
    - LSTM
    - HiMol
    - MFGNN

- Featurizers: 
    - Change in Atom and Bond Properties (Geometric)
    - Chenge in Atom and Bond Properties (QM based)

- Conformer Generation:
    - write out the params for it
    - write out the things for Auto3D and CREST
    - write out the things for RDKit
    - write out the things for OpenFF and OpenEye OMEGA
    - write out things for CgenFF
    - write out things that calculate energies using pySCF RDKit.
    - write out things that calculate energies using pySCF OpenFF.
    - write out things that calculate energies using pySCF OpenEye.
    - write out things that calculate energies using pySCF ASE.
    - write out the calculated properties as dummy targets.

- Active Learning: 
    - set up protocol

- Atom mapping:
    - set up other codes for this. 

- Autoencoder:
    - Set up framework 

- Reinforcement Learning: 
    - Set up framework 