# Only Tune the Pooling Part of the Model
# Only tune the Last EGAT layer 


class PoolingFineTune(nn.Module):
    def __init__(self, cfg, egat_model):


        # Copy EGAT layers from egat_model
        self.egat1 = egat_model.egat1
        self.egat2 = egat_model.egat2

        # Copy NN layers from nn_model
        self.agg_N_feats = egat_model.agg_N_feats
        self.agg_E_feats = egat_model.agg_E_feats
        self.mlp1 = nn_model.mlp1
        self.mlp2 = nn_model.mlp2
        self.mlp3 = nn_model.mlp3




class MessagePassingTune(nn.Module):
    def __init__(self, cfg, egat_model):


        # Copy EGAT layers from egat_model
        self.egat1 = egat_model.egat1
        self.egat2 = egat_model.egat2

        # Copy NN layers from nn_model
        self.agg_N_feats = egat_model.agg_N_feats
        self.agg_E_feats = egat_model.agg_E_feats
        self.mlp1 = nn_model.mlp1
        self.mlp2 = nn_model.mlp2
        self.mlp3 = nn_model.mlp3


