# TO-DO - Tomorrow:

### Testing:
- Test the Aggregators.
- Test all Features for the Graphs. 
- Check on the Datasets. 
- Check on the Dataloader.
- Check on the Training.
- Check on the Prediction 



### Code:
- Normalization -  Write an object that takes the .csv file, gets the columns needed and uses the scalers needed on either all the data or on some portion of it that will be used. Make sure that is imported into the training scheme. 
- Loading Multiple Geometries: Write out the workaround that for x conformers for a 3D based model, you load x molecules into the dataset and dataloader.


# Future Versions:
- Imbalanced Learning: 
    - Add codes that do the Density Smoothing for the Features and Labels. 
    - Add codes that do the Augmentation. 

- Active Learning: 
    - Add codes that sample the uncertainity and pick new training points. 

- Meta-Learning: 
    - Add codes that perform meta-learning tasks.
    - https://github.com/sicara/easy-few-shot-learning
    - https://github.com/uiuc-iml/few-shot-regression/tree/main


- LPM Finetuning:
    - Add codes that finetune parts of the LPM. 

- Multi-Component Models:
    - Add codes that take multiple SMILES columns. 
    - if multiple. Use the EGAT-MC model. The EGAT model is the same as the reaction model. But with Differences in the graph and dataloader. 

- Atom-Bond Property Prediction Model:
    - Write out using either a deconvolution or a set of NN tasks. 
    - Write out a a vector or matrix stack that is being predicted. 

- QM:
    - Write code that calculates Atom-bond level QM properties. 
    - Write code that calculates molecular level QM properties. 

- Conf Sampling:
    - Calculate the conformer energy and use it as a y feature to calculate results for. 

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


- Conformer Generation:
    - write out things for CgenFF
    
- Atom mapping:
    - set up other codes for this. 


- Bayesian Modeling:
    - https://helda.helsinki.fi/server/api/core/bitstreams/5760b1a9-991b-4967-864e-f4404c0dba2d/content
    - https://arxiv.org/pdf/2006.04064
    - https://ieeexplore.ieee.org/abstract/document/10845188?casa_token=MJeJKVW2_jMAAAAA:OUvQi4WciSHjzYgYWtHgJcytBsIimpeOse_NeMV16UU-BkCrNNgzsRQXIIMzYJY3O4x8EGrt
    - https://www.sciencedirect.com/science/article/abs/pii/S0925231224000316
    - https://arxiv.org/abs/2104.08438
    - https://ieeexplore.ieee.org/document/9555949
    - https://arxiv.org/pdf/2501.15573
    - https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00154k
    - https://arxiv.org/pdf/2501.18665
    - https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c01792?casa_token=6DFS3M-_DyEAAAAA%3A7MkeF2KOPbmK7uvF5HLjSGEh6x2Q_pF9NG2ua5CK_8BHk8g0mRrkvVwmKXKbMVqGzrgcFk20WCZmd6Q
    - https://arxiv.org/pdf/2205.12934
    - BFN
    - PFN
- Uncertainity Quanitification 
    - https://arxiv.org/pdf/2504.12627
    - https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/uncertainty/index.html

- Explainability:
    - MCTS
    - https://www.sciencedirect.com/science/article/pii/S2589004222013153
    - GNNExplainer
    - Visualize Attention Scores
    - https://arxiv.org/pdf/2502.02719
    - https://ieeexplore.ieee.org/abstract/document/10925220/authors#authors
    - https://dl.acm.org/doi/full/10.1145/3696444
    - https://dl.acm.org/doi/full/10.1145/3711122?casa_token=dG6Fg0-8ULkAAAAA%3AATQl0Ckx9smkWJZHh0ofYjfEypz8Ta0a-ZxckY4a3LhVAS_AIrTd-ABfG2lrFo13MID18Dyd1q__
    - https://dl.acm.org/doi/full/10.1145/3685678
    - https://arxiv.org/pdf/2401.04829
    - Beyond Topological Self-Explainable GNNs:A Formal Explainability Perspective

- splitting 
    - https://exscientia.github.io/molflux/pages/splits/basic_usage.html
    - astartes

- long short range
    - https://arxiv.org/pdf/2304.13542
    - https://arxiv.org/pdf/2412.08541
    - https://arxiv.org/pdf/2409.17622
    - https://ojs.aaai.org/index.php/AAAI/article/view/34202
    - https://arxiv.org/pdf/2502.02748
    - https://openreview.net/pdf?id=hySoEBuTuM
    - Ewald MP 
    - https://arxiv.org/pdf/2501.19179
    - https://proceedings.neurips.cc/paper_files/paper/2024/file/580c4ec4738ff61d5862a122cdf139b6-Paper-Conference.pdf
    - https://arxiv.org/pdf/2003.03123
    - https://arxiv.org/pdf/2011.14115
    - https://arxiv.org/pdf/2402.04538
    - https://arxiv.org/pdf/2312.05611
    - https://arxiv.org/pdf/2402.10793v2
    - https://arxiv.org/pdf/2007.08026

- Pooling
    - https://arxiv.org/pdf/2102.11533
    - https://arxiv.org/pdf/2112.09990
    - https://proceedings.neurips.cc/paper_files/paper/2020/file/1764183ef03fc7324eb58c3842bd9a57-Paper.pdf
    - https://arxiv.org/pdf/2110.05292
    - https://arxiv.org/pdf/2501.09821
    - https://dl.acm.org/doi/pdf/10.1145/3292500.3330982
    - https://www.ijcai.org/proceedings/2023/0244.pdf
    - https://arxiv.org/pdf/2204.07321
    - https://www.sciencedirect.com/science/article/abs/pii/S0893608021002999
    - https://proceedings.neurips.cc/paper/2020/hash/a26398dca6f47b49876cbaffbc9954f9-Abstract.html
    - https://link.springer.com/chapter/10.1007/978-3-031-26390-3_21?fromPaywallRec=false
    - https://link.springer.com/chapter/10.1007/978-3-030-47436-2_43?fromPaywallRec=false#Sec3
    - https://link.springer.com/article/10.1007/s10462-024-10949-2#Sec2
    - K-hop Hypergraph Neural Network: A Comprehensive Aggregation Approach

- Transfer Learning:
    - Latent Functional Maps
    - Optimizing the Latent Space of Generative Networks
    - On the Direct Alignment of Latent Spaces
    - Domain Translation via Latent Space Mapping

- Few Shot Learning:
    - https://pubs.acs.org/doi/10.1021/acs.jcim.4c00485#:~:text=We%20describe%20Few%2DShot%20Compound,compounds%20against%20the%20same%20assay.


- Analysis
    - Dimensionality Reduction
    - Clustering
    - Euclidian Mapping
    - Outlier Detection
    


# TO-DO - Long-Term:
- Params:
    - Create a params class that caters to hyperparameter tuning
    
    - YARP:
    - Write the open shell functions. 
    - Write in the automated EGAT model with RGD1 in it. 

- Aggregators
    - Add in the other aggregators. 


