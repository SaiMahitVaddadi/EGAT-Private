# TO-DO - Immediate:

- Testing:
    - Test the base dataset
    - Test the dataloader
    - Test the model training

- Graphs:
    - Add the Sterimol Properties to the Molecular Graph with Geometry
    - Add the Kallisto Properties to the Molecular Graph with Geometry
    - Add the Jazzy Properties to the Molecular Graph with Geometry
    
- Normalization:
    - Write an object that takes the .csv file, gets the columns needed and uses the scalers needed on either all the data or on some portion of it that will be used. 
    - Make sure that is imported into the training scheme. 

- Loading: 
    - write out the workaround that for x conformers for a 3D based model, you load x molecules into the dataset and dataloader.
    - for the given workaround, calculate the QM properties using pySCF or something basic as a transfer learning protocol. 

- Model:
    - Add All Aggregators to the model. 
    - Add all imbalanced Learning tools to the model.
    - Write out a decoder model for EGAT. 
    - Write out a Multi-component model for EGAT. (Delta-Learning, etc.)
    - Finish out the atom-bond property prediction model. 



# TO-DO - Long-Term:
- Params:
    - Create a params class that caters to hyperparameter tuning
    
- CLI execution
    - write a cli for few shot learning 
    - write a cli for hyperparameter tuning via grid search
    

- YARP:
    - Write the open shell functions. 
    - Write in the automated EGAT model with RGD1 in it. 

- QM:
    - Write code that calculates Atom-bond level QM properties. 
    - Write code that calculates molecular level QM properties. 

- GUI:
    - streamlit app interface. 
    - webhosting??????

- Other Featurizers: 
    - Chemprop
    - Deepchem
    
- EGAT Models: 
    - Fully custom activation functions 
    - Fully custom end convolution layers for spectra
    - Fully custom softmax style layers 
    - Fully custom loss functions 

- Additional Layers to Add using DGL:
    - EGT
    - TGT
    - GRU
    - LSTM
    - HiMol
    - MFGNN
    - D-MPNN
    - GAT
    - GCN
    - GIN
    - whatever PyG has 

- Aggregators
    - Add in the other aggregators. 

- Reaction Featurizers: 
    - Change in Atom and Bond Properties (Geometric)
    - Chenge in Atom and Bond Properties (QM based)

- Conformer Generation:
    - write out things for CgenFF
    
- Active Learning: 
    - set up protocol

- Atom mapping:
    - set up other codes for this. 

- Meta-Learning:
    - set up codes for this. 

- Imbalanced Learning: 
    - use the ChemLLM imbalanced stuff to write those. 

