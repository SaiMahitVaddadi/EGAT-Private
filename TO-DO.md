# TO-DO:
- Training Loop: 
    - Be able to normalize add ons. calculate all add-ons first and then use that. 
    - Setup ways to save the additional loss functions. 

- EGAT:
    - molecular/model.py will have both the 1MLP and 3MLP model
    - reaction/model.py will have both the 1MLP and 3MLP model
    - Write in a Fingerprint Based NN model in Pytorch 
    - Set up Pytorch Lightning support.    
    - make them usable in both DGL and PyG
    - make them malleable to both any layer in PyG and DGL.  

- Slurm or QSUB or other types of automation:
    - Write the job writer script for those
    - Write the grid search script for those

- Tuning:
    - Set up a hyperopt protocol
    - Set up an optuna protocol
    - Set up a grid search protocol

- Parallelization
    - Setup ensemble models in parallel
    - Setup multiple trainings in parallel
    - Setup cv in parallel. 


- Additional Models:
    - add in the EGT and TGT
    - add in the HiMol and MFGNN
    
- Addons:
    - Add Cheminformatic Features
    - Add FSSscore
    - Add Mordred Library
    - Add Descriptasourus 
    - Add Quantum Mechanical Features 
    - Add Molskill, QEPPI, syba, molecular_complexity

- Featurizers: 
    - Reorganize Featurizers
    

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
    - set up method
    - set up way to modify embeddings using LHS or MC sampling see if doing that can affect the decoding ability. 

- Autoencoder:
    - Set up framework 