# TO-DO - Tomorrow:
### Testing Torch Setups: 
- Cleanup the Training and Prediction codes.
- Expand for the multi geometry and multi bond matrix case.  
- Expand for the multi component case. 

- LLM Block:
    - Write it up for molfeat. 
    - Write it up for huggingface. 
    - Add in the past finetuning codes. 
    - Write the LLM Model.
    - Write the GNN+LLM model. 

- Training and Prediction:
    - Write out the skeleton code for the imbalanced learning.
    - Write out the skeleton code for the data augmentation. 
    - Write out the skeleton code for the few-shot learning. 
    - Write out the skeleton code for the symbolic regressor. 

- 2D and 3D Similarity: Write out the skeleton code that does so. 
    
- Write a new Generate.py command.

### Testing the Graphs:
- Reaction
- Reaction Geometry
- Reaction Global
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







daBoost Regression: While AdaBoost is widely known for its application in classification problems, it can be adapted for regression by modifying the algorithm’s loss function and the way weak models are combined. It can capture non-linear relationships between the input features and the target variable by leveraging the capabilities of the weak regression models. It has been used in various regression tasks, such as predicting housing prices, stock market prices, and demand forecasting.
Extra Trees Regression: short for Extremely Randomized Trees Regression, is an ensemble learning method used for regression tasks. It is a variation of the Random Forest algorithm that introduces additional randomness during the construction of individual decision trees. In Extra Trees Regression, multiple decision trees are trained on different random subsets of the training data and random subsets of features. During the tree construction process, instead of finding the best-split point based on a criterion like Gini impurity or information gain, Extra Trees randomly selects split points without considering the optimal threshold. This randomization helps to reduce overfitting and increase the diversity among the trees.

