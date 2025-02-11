# TO-DO:

- Slurm or QSUB or other types of automation:
    - Write the job writer script for those
    - Write the grid search script for those

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

- EGAT Featurizers:
    - Write out the Shortest Path Distance
    - Write out the Shortest Path Distance with weights and periodic boundary conditions
    - write out the random walk commute time
    - write out the number of shortest paths
    - write out the effective resistance
    - write out the common neighbors
    - Write out the Jaccard index
    - Write out the Adamic-Adar index
    - Write out the Preferential Attachment index
    - Write out the Shortest Path Distance with weights


    Katz Centrality Similarity – Measures how well-connected two nodes are via their entire network, using a decaying factor on longer paths.
    Eigenvector Centrality Difference – Difference in influence scores of the two nodes based on their eigenvector centralities.
    Betweenness Centrality Correlation – Compares how often the two nodes appear on shortest paths in the graph.
    Graph Cut and Flow-Based Features
    Minimum Cut Value – The minimum number of edges that need to be removed to disconnect the nodes.
    Maximum Flow – The maximum amount of "flow" (information, traffic, etc.) that can travel between the two nodes using the edges as capacities.
    Spectral Features
    Laplacian Eigenvector Similarity – Measures how similar two nodes are based on their positions in the spectral decomposition of the graph Laplacian.

    Fiedler Vector Similarity – Measures how closely the two atoms are connected via the graph’s spectral properties.
    Laplacian Eigenvector Centrality – Describes global connectivity patterns between the atoms.
    Graph Distance Weighted by Bond Order – Incorporates bond orders (single, double, etc.) in shortest paths.
    Chemical Graph Descriptors
    Betweenness Centrality of Pathways – How often the shortest path between two atoms passes through critical atoms.
    Rings in Shared Path – Number and size of rings that the shortest path passes through (e.g., aromatic rings).
    Local Atomic Environment Similarity – Descriptor-based similarity of atomic environments using fingerprints like Morgan or ECFP.

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