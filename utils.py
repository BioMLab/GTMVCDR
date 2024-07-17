import os
import numpy as np
from math import sqrt

import pandas as pd
import scipy as sp
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

import math
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
import collections
from scipy.sparse.construct import diags



class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt_ge=None, xt_meth=None, xt_mut=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None,saliency_map=False, test_drug_dict = None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt_ge, xt_meth,xt_mut, y, smile_graph, test_drug_dict)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    ## \brief Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # \param XD - chuỗi SMILES, XT: danh sách các đối tượng đã được mã hóa encoded target (categorical or one-hot),
    # \param Y: list of labels (i.e. affinity)
    # \return: PyTorch-Geometric format processed data
    def process(self, xd, xt_ge, xt_meth, xt_mut,  y, smile_graph, test_drug_dict):
        assert (len(xd) == len(xt_ge) and len(xt_ge) == len(y)) and len(y) == len(xt_meth) and len(xt_meth) == len(xt_mut) , "The four lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print(data_len)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target_ge = xt_ge[i]
            target_meth = xt_meth[i]
            target_mut = xt_mut[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            
            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target_ge = torch.tensor([target_ge], dtype=torch.float, requires_grad=True)
                GCNData.target_meth = torch.tensor([target_meth], dtype=torch.float, requires_grad=True)
                GCNData.target_mut = torch.tensor([target_mut], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target_ge = torch.FloatTensor([target_ge])
                GCNData.target_meth = torch.FloatTensor([target_meth])
                GCNData.target_mut = torch.FloatTensor([target_mut])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)


# for xt_meth
        """for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt_meth[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))

            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target = torch.tensor([target], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list_meth.append(GCNData)

        #append data_list_mut and data_list_meth together
        for x in data_list_meth:
            data_list.append(x)
"""
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def mse_cust(y,f):
    mse = ((y - f)**2)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def draw_cust_mse(mse_dict):
    best_mse = []
    best_mse_title = []
    i = 0
    for (key, value) in mse_dict.items():
        if i < 10 or (i > 13 and i < 24):
            best_mse.append(value)
            best_mse_title.append(key)
        i += 1

    plt.bar(best_mse_title, best_mse)
    plt.xticks(rotation=90)
    plt.title('GE & METH')
    plt.ylabel('MSE')
    plt.savefig("Blind drug.png")

def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method

def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    df = np.array([yp.squeeze(), yt.squeeze()])
    df = pd.DataFrame(df.T, columns=['y_pred', 'y_label'])
    #---f1,acc,recall, specificity, precision
    real_score=np.mat(yt)
    predict_score=np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0],recall[0, 0], specificity[0, 0], precision[0, 0],df


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    """ print(adj)
     (0, 0)	1.0
  (0, 16)	0.9944254
  (0, 6688)	0.9918765
  (0, 4960)	0.9852609
    """
    # print(type(adj))#<class 'scipy.sparse.coo.coo_matrix'>
    # Transform the adjacency matrix into a symmetric matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj.tocoo()
    # print(adj.sum(1))#每一行之和
    rowsum = np.array(adj.sum(1))
    # print(rowsum)
    # print(rowsum.shape)(23168, 1)
    d_inv_sqrt = np.power(rowsum, - 0.5).flatten()#开rowsum的-0.5次方
    # print(d_inv_sqrt)[0.2077547 0.2077547 0.2077547 ... 0.3347606 0.3347606 0.3347606]
    # (23168,)# print(d_inv_sqrt.shape)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.sparse.construct.diags(d_inv_sqrt)  # ndarray类型
    # toarray returns an ndarray; todense returns a matrix. If you want a matrix, use todense otherwise, use toarray
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).tocoo()



