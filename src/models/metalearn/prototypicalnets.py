from ..base.model import EGATModel
import torch
from torch import nn 
from torch.nn import functional

# Write a training protocol that uses the Prototypical Networks approach for few-shot learning. This means that the loss computes the NLL for 

class PrototypicalEGATRegressorNets(EGATModel):
    def __init__(self, cfg, num_node_feats = 17, num_edge_feats=14,addononlength=None,activation=None,endconvolution=None,softmax=None):
        super(EGATModel).__init__(cfg, num_node_feats = 17, num_edge_feats=14,addononlength=None,activation=None,endconvolution=None,softmax=None)
    

    def EGATBlock(self,graphR, graphP,Hr=None):
        ##################################### 
        ############# layer one ############# 
        ##################################### 
        Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats = self.LayerOne(graphR, graphP)

        ##################################### 
        ############# layer two ############# 
        ##################################### 
        if self.MessagePassing:
            Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats,R_combined_matrix,P_combined_matrix = self.LayerTwoMP(graphR, graphP)
        else:
            Rnode_feats, Redge_feats, Pnode_feats, Pedge_feats = self.LayerTwoNoMP(graphR, graphP)
            R_combined_matrix,P_combined_matrix = None,None

        G_features = self.Aggregation(Pnode_feats,Rnode_feats,Pedge_feats,Redge_feats,graphR)
        return G_features,R_combined_matrix,P_combined_matrix
    

    def GetPrototype(self,features,labels=None):
        if labels is not None:
            n_way = len(torch.unique(labels))
            # Prototype i is the mean of all instances of features corresponding to labels == i
            self.prototypes =  torch.cat(
                [
                    features[torch.nonzero(labels == label)].mean(0)
                    for label in range(n_way)
                ]
            )
        else:
            self.prototypes = features.mean(0,keepdim=True)
    def DistanceToPrototype(self, embeddings):
        if self.params.prototype_distance == 'euclidean':
            return -torch.cdist(embeddings, self.prototypes)
        elif self.params.prototype_distance == 'cosine':
            return (
                nn.functional.normalize(embeddings, dim=1)
                @ nn.functional.normalize(self.prototypes, dim=1).T
            )

    def UseSoftmax(self, embeddings, temperature: float = 1.0):
        """
        If the option is chosen when the classifier is initialized, we perform a softmax on the
        output in order to return soft probabilities.
        Args:
            output: output of the forward method of shape (n_query, n_classes)
            temperature: temperature of the softmax
        Returns:
            output as it was, or output as soft probabilities, of shape (n_query, n_classes)
        """
        return (temperature * embeddings).softmax(-1) if self.params.prototype_use_softmax else embeddings


    def forward(self, graphR, graphP,Hr=None,label=None):
        
        G_features,R_combined_matrix,P_combined_matrix = self.EGATBlock(graphR, graphP,Hr)    
        self.GetPrototype(G_features,label)
        distance = self.DistanceToPrototype(G_features)
        if self.params.prototype_use_softmax: 
            distance = self.UseSoftmax(distance, self.params.prototype_temperature)

        G_features = torch.cat((G_features,distance),axis=1)
        #################################### 
        ########## merge features ##########
        #################################### 
        # MLP
        if 'Hr2' in self.params.model_type: G_features= torch.cat((G_features,Hr),axis=1)
        if '1MLP' in self.params.model: 
            x = self.OneMLP(G_features)
        elif '3MLP' in self.params.model:
            x = self.ThreeMLP(G_features,Hr)

        if self.smax == 'smax':
            x = self.smax(x)
            
        
        if self.getembeddings == 0:
            if self.getattentionmaps:
                return x,R_combined_matrix,P_combined_matrix,distance
            else:
                return x,distance
        elif self.getembeddings == 1:
            if self.getattentionmaps:
                return x,G_features,R_combined_matrix,P_combined_matrix,distance
            else:
                return x,G_features,distance
        elif self.getembeddings == 2:
            if self.getattentionmaps:
                return G_features,R_combined_matrix,P_combined_matrix,distance
            else:
                return G_features,distance
