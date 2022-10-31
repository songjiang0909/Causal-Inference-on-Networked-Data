import argparse
import torch
import time
import utils as utils
import numpy as np
from model import NetEsimator
from baselineModels import GCN_DECONF,CFR,GCN_DECONF_INTERFERENCE,CFR_INTERFERENCE
from experiment import Experiment


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=7,help='Use CUDA training.')
parser.add_argument('--seed', type=int, default=24, help='Random seed. RIP KOBE')
parser.add_argument('--dataset', type=str, default='BC')#["BC","Flickr"]
parser.add_argument('--expID', type=int, default=4)
parser.add_argument('--flipRate', type=float, default=1)
parser.add_argument('--alpha', type=float, default=0.5,help='trade-off of p(t|x).')
parser.add_argument('--gamma', type=float, default=0.5,help='trade-off of p(z|x,t).')
parser.add_argument('--epochs', type=int, default=300,help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,help='Initial learning rate.')
parser.add_argument('--lrD', type=float, default=1e-3,help='Initial learning rate of Discriminator.')
parser.add_argument('--lrD_z', type=float, default=1e-3,help='Initial learning rate of Discriminator_z.')
parser.add_argument('--weight_decay', type=float, default=1e-5,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dstep', type=int, default=50,help='epoch of training discriminator')
parser.add_argument('--d_zstep', type=int, default=50,help='epoch of training discriminator_z')
parser.add_argument('--pstep', type=int, default=1,help='epoch of training')
parser.add_argument('--normy', type=int, default=1)
parser.add_argument('--hidden', type=int, default=32,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,help='Dropout rate (1 - keep probability).')
parser.add_argument('--save_intermediate', type=int, default=1,help='Save training curve and imtermediate embeddings')


parser.add_argument('--model', type=str, default='NetEsimator',help='Models or baselines')
#["NetEsimator","ND","TARNet","TARNet_INTERFERENCE","CFR","ND_INTERFERENCE","CFR_INTERFERENCE"]
parser.add_argument('--alpha_base', type=float, default=0.5,help='trade-off of balance for baselines.')
parser.add_argument('--printDisc', type=int, default=0,help='Print discriminator result for debug usage')
parser.add_argument('--printDisc_z', type=int, default=0,help='Print discriminator_z result for debug usage')
parser.add_argument('--printPred', type=int, default=1,help='Print encoder-predictor result for debug usage')



startTime = time.time()

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain,valA, valX, valT,cfValT,POVal,cfPOVal,testA,testX, testT,cfTestT,POTest,cfPOTest,\
    train_t1z1,train_t1z0,train_t0z0,train_t0z7,train_t0z2,val_t1z1,val_t1z0,val_t0z0,val_t0z7,val_t0z2,test_t1z1,test_t1z0,test_t0z0,test_t0z7,test_t0z2 = utils.load_data(args)

if args.model == "NetEsimator":
    model = NetEsimator(Xshape=trainX.shape[1],hidden=args.hidden,dropout=args.dropout)
elif args.model == "ND":
    model = GCN_DECONF(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "TARNet":
    model = GCN_DECONF(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "CFR":
    model = CFR(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)

elif args.model == "CFR_INTERFERENCE":
    model = CFR_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "ND_INTERFERENCE":
    model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "TARNet_INTERFERENCE":
    model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)



exp = Experiment(args,model,trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain,valA, valX, valT,cfValT,POVal,cfPOVal,testA, testX, testT,cfTestT,POTest,cfPOTest,\
    train_t1z1,train_t1z0,train_t0z0,train_t0z7,train_t0z2,val_t1z1,val_t1z0,val_t0z0,val_t0z7,val_t0z2,test_t1z1,test_t1z0,test_t0z0,test_t0z7,test_t0z2)

"""Train the model"""
exp.train()

"""Moel Predicting"""
exp.predict()

if args.model == "NetEsimator" and args.save_intermediate:
    exp.save_curve()
    exp.save_embedding()

print("Time usage:{:.4f} mins".format((time.time()-startTime)/60))
print ("================================Setting again================================")
print ("Model:{} Dataset:{}, expID:{}, filpRate:{}, alpha:{}, gamma:{}".format(args.model,args.dataset,args.expID,args.flipRate,args.alpha,args.gamma))
print ("================================BYE================================")

