# TO-DO:
- Training Loop: 
    - Setup ways to save the additional loss functions. 


- Params:
    - Cleanup the params class. 
    - write a function that uses a cli to write a config file 
    - write a function that uses a cli to input the parameters

- CLI execution
    - write a cli for training 
    - write a cli for testing 
    - write a cli for fingerprinting
    - write a cli for few shot learning 
    - write a cli for hyperparameter tuning via grid search
    - write a cli for active learning 
    

- EGAT Models:
    - Set up Pytorch Lightning support for FP Model
    - Fully custom activation functions
    - Fully custom end convolution layers
    - Fully custom softmax layers   
    - Set up the ablation models. 
    - make them usable in both DGL and PyG.
    - make them malleable to both any layer in PyG and DGL.  
    - Create an EGAT version usable for atom/bond property prediction. 


- Tuning:
    - Set up a hyperopt protocol
    - Set up an optuna protocol
    - Set up a grid search protocol

- Parallelization
    - Setup ensemble models in parallel or series
    - Setup CV in parallel or series. 

- Featurizers: 
    - Reaction Geometry Features for Atoms:
        - Change in Atom properties
        - Change in Bond Properties
    
- Additional Models:
    - add in the EGT and TGT
    - add in the HiMol and MFGNN
    


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
    - write out the workaround that for x conformers for a 3D based model, you load x molecules into the dataset and dataloader.

- DataLoading:
    - write out the dataloader from a JSON folder
    - write out the dataloader from a SDF folder
    - write out the dataloader from a TOML folder
    - write out the dataloader from a CSV folder
    - write out the dataloader from an SQL
    - write out the dataloader from a Parquet folder



- Other Featurizers: 
    - Chemprop
    - Deepchem

- Other types of predictions:
    - Atom property prediction
    - Bond property prediction
    - Atom-bond property prediction
    - Mixture property prediction
    - Spectra property prediction

- Active Learning: 
    - set up protocol


- Atom mapping:
    - set up other codes for this. 

- JEPA:
    - set up way to modify embeddings using LHS or MC sampling see if doing that can affect the decoding ability. 

- Autoencoder:
    - Set up framework 