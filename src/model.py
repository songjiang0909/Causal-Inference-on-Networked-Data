import argparse
import torch
import pickle as pkl
import torch.nn as nn
import utils as utils
import numpy as np
from modules import GCN, NN, Predictor,Discriminator

class NetEsimator(nn.Module):

    def __init__(self,Xshape,hidden,dropout):

        super(NetEsimator, self).__init__()
        self.encoder = GCN(nfeat=Xshape, nclass=hidden, dropout=dropout)
        self.predictor = Predictor(input_size=hidden + 2, hidden_size1=hidden, hidden_size2=hidden,output_size=1)
        self.discriminator = Discriminator(input_size=hidden,hidden_size1=hidden,hidden_size2=hidden,output_size=1)
        self.discriminator_z = Discriminator(input_size=hidden+1,hidden_size1=hidden,hidden_size2=hidden,output_size=1)
    

    def forward(self,A,X,T,Z=None):
    
        embeddings = self.encoder(X, A)
        pred_treatment = self.discriminator(embeddings)
        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z
        embed_treatment = torch.cat((embeddings, T.reshape(-1, 1)), 1) 
        pred_z = self.discriminator_z(embed_treatment)
        embed_treatment_avgT = torch.cat((embed_treatment, neighborAverageT.reshape(-1, 1)), 1)
        pred_outcome0 = self.predictor(embed_treatment_avgT).view(-1)

        return pred_treatment,pred_z,pred_outcome0,embeddings, neighborAverageT