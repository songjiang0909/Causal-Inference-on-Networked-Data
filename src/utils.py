import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp

import torch
import torch.nn.functional as F


def sparse_mx_to_torch_sparse_tensor(sparse_mx,cuda=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor


def normalize(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def dataTransform(data,cuda):
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    A, X, T,cfT,PO,cfPO = data['network'],data['features'],data["T"],data["cfT"],data["PO"],data["cfPO"]
    X = Tensor(normalize(X))
    A,T,cfT,PO,cfPO = Tensor(A),Tensor(T),Tensor(cfT),Tensor(PO),Tensor(cfPO)

    return A, X, T,cfT,PO,cfPO


def load_data(args):
    print ("================================Dataset================================")
    print ("Model:{}, Dataset:{}, expID:{}, filpRate:{}, alpha:{}, gamma:{}".format(args.model,args.dataset,args.expID,args.flipRate,args.alpha,args.gamma))
    dataset,expID,flipRate,cuda = args.dataset,args.expID,args.flipRate,args.cuda
    if flipRate == 1:
        flipRate = 1
    
    if dataset == "BC":
        file = "../data/BC/simulation/"+str(dataset)+"_fliprate_"+str(flipRate)+"_expID_"+str(expID)+".pkl"
    if dataset == "Flickr":
        file = "../data/Flickr/simulation/"+str(dataset)+"_fliprate_"+str(flipRate)+"_expID_"+str(expID)+".pkl"

    with open(file,"rb") as f:
        data = pkl.load(f)
    dataTrain,dataVal,dataTest = data["train"],data["val"],data["test"]

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain = dataTransform(dataTrain,cuda)
    valA, valX, valT,cfValT,POVal,cfPOVal = dataTransform(dataVal,cuda)
    testA, testX, testT,cfTestT,POTest,cfPOTest = dataTransform(dataTest,cuda)

    
    train_t1z1=dataTrain["train_t1z1"]
    train_t1z0=dataTrain["train_t1z0"]
    train_t0z0=dataTrain["train_t0z0"]
    train_t0z7=dataTrain["train_t0z7"]
    train_t0z2=dataTrain["train_t0z2"]
    val_t1z1=dataVal["val_t1z1"]
    val_t1z0=dataVal["val_t1z0"]
    val_t0z0=dataVal["val_t0z0"]
    val_t0z7=dataVal["val_t0z7"]
    val_t0z2=dataVal["val_t0z2"]
    test_t1z1=dataTest["test_t1z1"]
    test_t1z0=dataTest["test_t1z0"]
    test_t0z0=dataTest["test_t0z0"]
    test_t0z7=dataTest["test_t0z7"]
    test_t0z2=dataTest["test_t0z2"]


    return trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain,valA, valX, valT,cfValT, POVal,cfPOVal,testA, testX, testT,cfTestT,POTest,cfPOTest,train_t1z1,train_t1z0,train_t0z0,train_t0z7,train_t0z2,val_t1z1,val_t1z0,val_t0z0,val_t0z7,val_t0z2,test_t1z1,test_t1z0,test_t0z0,test_t0z7,test_t0z2


def PO_normalize(normy,base,PO,cfPO):
    """Normalize PO"""
    if  normy:
        ym, ys = torch.mean(base), torch.std(base)
        YF,YCF = (PO - ym) / ys, (cfPO - ym) / ys
    else:
        YF,YCF =  PO,cfPO
    
    return YF,YCF 


def PO_normalize_recover(normy,base,nPO):
    
    if normy:
        ym, ys = torch.mean(base), torch.std(base)
        pred_PO = (nPO * ys + ym)
    else:
        pred_PO = nPO
    
    return pred_PO




"""
The following codes are originally by Ruocheng Guo for the WSDM'20 paper
@inproceedings{guo2020learning,
  title={Learning Individual Causal Effects from Networked Observational Data},
  author={Guo, Ruocheng and Li, Jundong and Liu, Huan},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={232--240},
  year={2020}
}
https://github.com/rguo12/network-deconfounder-wsdm20
"""

def wasserstein(x,y,p=0.5,lam=10,its=10,sq=False,backpropT=False,cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    
    x = x.squeeze()
    y = y.squeeze()
    
#    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x,y) #distance_matrix(x,y,p=2)
    
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M,10.0/(nx*ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta*(torch.ones(M[0:1,:].shape).cuda())
    col = torch.cat([delta*(torch.ones(M[:,0:1].shape)).cuda(),(torch.zeros((1,1))).cuda()],0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1))/nx,(1-p)*torch.ones((1,1))],0)
    b = torch.cat([(1-p)*torch.ones((ny,1))/ny, p*torch.ones((1,1))],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a

    u = a

    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b/(torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)




def MI(x,y,z,N):
    if x == 0:
        return torch.FloatTensor([0])
    else:
        return (x/N)*torch.log2((N*x)/(y*z))

def NMI(set1,set2,threshold=0.5):
    set1 =  torch.FloatTensor([(set1 >= threshold).sum(),(set1 < threshold).sum()])
    set2 =  torch.FloatTensor([(set2 >= threshold).sum(),(set2 < threshold).sum()])
    set1 = set1.reshape(1,-1)
    set2 = set2.reshape(1,-1)
    res = torch.cat((set1,set2),0).T
    N = torch.sum(torch.sum(res))
    NW = torch.sum(res,1)
    NC  = torch.sum(res,0)
    HC = -((NC[0]/N)*torch.log2(NC[0]/N)+(NC[1]/N)*torch.log2(NC[1]/N))
    HW = -((NW[0]/N)*torch.log2(NW[0]/N)+(NW[1]/N)*torch.log2(NW[1]/N))
    IF = MI(res[0][0],NW[0],NC[0],N)+MI(res[0][1],NW[0],NC[1],N)+MI(res[1][0],NW[1],NC[0],N)+MI(res[1][1],NW[1],NC[1],N)
    return (IF/torch.sqrt(HC*HW)).cuda()

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val**2




