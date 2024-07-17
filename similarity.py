import csv

from scipy.sparse import coo_matrix
import torch
import pandas as pd
import numpy as np
from abc import ABC
import torch.nn as nn
import torch.nn.functional as fun
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

from get_Drug_Features import getFeatures
from sklearn import preprocessing


# def 1111ConstructAdjMatrix(original_adj_mat):
#     # self.adj_mat = original_adj_mat.to(device)
#     # self.device = device
#     drug_identity_matrix, cell_identity_matrix = computing_knn()  # 都没加自连接
#     adj_mat = coo_matrix(original_adj_mat).data
#     # cell_identity = torch.diag(torch.diag(torch.ones(n_cell, n_cell, dtype=torch.float, device=self.device)))
#     # drug_identity = torch.diag(torch.diag(torch.ones(n_drug, n_drug, dtype=torch.float, device=self.device)))
#     cell_drug = torch.cat((cell_identity_matrix, adj_mat), dim=1)
#     drug_cell = torch.cat((torch.t(adj_mat), drug_identity_matrix), dim=1)
#     adj_matrix = torch.cat((cell_drug, drug_cell), dim=0)
#     d = torch.diag(torch.pow(torch.sum(adj_matrix, dim=1), -1 / 2))
#     identity = torch.diag(torch.diag(torch.ones(d.shape, dtype=torch.float)))
#     adj_matrix_hat = torch.add(identity, torch.mm(d, torch.mm(adj_matrix, d)))
#     return adj_matrix_hat
from utils import sym_adj


def computing_drug_sim_matrix():#可运行
    Drug_SMILES = "data/Drug/11111.csv"
    reader2 = csv.reader(open(Drug_SMILES, 'r'))  # 使用csv的reader()方法，创建一个reader对象
    next(reader2)
    rows = [item for item in reader2]  # 生成一个列表
    drug_smiles = {item[0]: item[1] for item in
                   rows}  # 定义字典drug_smiles，key为drug_id,values为SMILES,isdigit()判断字符串是否只由数字组成的函数
    drug_sim_matrix = np.zeros((len(drug_smiles), len(drug_smiles)))
    drug_smiles_list = list(drug_smiles.values())

    mi = [Chem.MolFromSmiles(drug_smiles_list[i]) for i in range(len(drug_smiles_list))]
    fps = [AllChem.GetMorganFingerprint(x, 4) for x in mi]
    for i in range(len(drug_smiles_list)):
        drug_sim_matrix[i][i] = 1                 #加了自连接

        for j in range(len(drug_smiles_list)):
            if i != j:
                drug_sim_matrix[i][j] = DataStructs.DiceSimilarity(fps[i], fps[j])


    return drug_sim_matrix,len(drug_smiles)
# drug_sim_matrix,a = computing_drug_sim_matrix()
# print(drug_sim_matrix.shape)


def computing_cell_sim_matrix_minmax():
    Gene_expression_file = './data/TCGA_cell/EXP.csv'
    gexpr_feature1= pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    gexpr_feature1 = gexpr_feature1.sort_index()
    min_max = MinMaxScaler()
    gexpr_feature1 = min_max.fit_transform(gexpr_feature1)

    # dic = gexpr_feature1.T.to_dict('list')
    cell_sim_matrix = np.zeros((gexpr_feature1.shape[0], gexpr_feature1.shape[0]))
    #标准化
    cell_feature = []
    for v in gexpr_feature1:
        cell_feature.append(v)

    for i in range(gexpr_feature1.shape[0]):
        cell_sim_matrix[i][i] = 1  # 加了自连接
        for j in range(gexpr_feature1.shape[0]):
            if i != j:
                cell_sim_matrix[i][j], _ = pearsonr(cell_feature[i], cell_feature[j])
                if cell_sim_matrix[i][j] < 0:
                    cell_sim_matrix[i][j] = 0
    return cell_sim_matrix,gexpr_feature1.shape[0]




def computing_cell_sim_matrix():
    Gene_expression_file = './data/TCGA_cell/EXP_norm - 副本.csv'
    gexpr_feature1= pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])

    dic = gexpr_feature1.T.to_dict('list')
    cell_sim_matrix = np.zeros((len(dic), len(dic)))
    cell_feature = []
    for v in dic.values():
        v = np.array(v)

        v_scale = preprocessing.scale(v)  # 调用sklearn包的方法
        cell_feature.append(v_scale)
    for i in range(len(dic)):
        for j in range(len(dic)):
            if i != j:
                cell_sim_matrix[i][j], _ = pearsonr(cell_feature[i], cell_feature[j])
                if cell_sim_matrix[i][j] < 0:
                    cell_sim_matrix[i][j] = 0
    return cell_sim_matrix,dic


def computing_knn_weight():
    drug_sim_matrix,len_drug_smiles = computing_drug_sim_matrix()
    # print(drug_sim_matrix)
    cell_sim_matrix,cell_num = computing_cell_sim_matrix_minmax()
    # print(cell_sim_matrix)
    cell_sim_matrix_new = np.zeros_like(cell_sim_matrix)
    for u in range(cell_num):#换细胞系相似性方法的时候爱这里需要改
        # print("----")
        # s =cell_sim_matrix[u].argsort()[-6:]
        # print(s)
        # # print("-----------1111111111")
        v = cell_sim_matrix[u].argsort()[-6:]#argsort函数返回的是数组值从小到大的索引值,#加了自连接，所以改为[-6:],原来是[-6:-1]
        # print(v)
        cell_sim_matrix_new[u][v] = cell_sim_matrix[u][v]

    drug_sim_matrix_new = np.zeros_like(drug_sim_matrix)
    for u in range(len_drug_smiles):
        v = drug_sim_matrix[u].argsort()[-6:]#加了自连接，所以改为[-6:],原来是[-6:-1]
        drug_sim_matrix_new[u][v] = drug_sim_matrix[u][v]
    drug_edges = np.argwhere(drug_sim_matrix_new > 0)
    cell_edges = np.argwhere(cell_sim_matrix_new > 0)
    drug_identity_matrix = np.zeros((len_drug_smiles,len_drug_smiles))
    cell_identity_matrix = np.zeros((cell_num,cell_num))
    for item in drug_edges:
        drug_identity_matrix[item[0]][item[1]] = 1
    for num in cell_edges:
        cell_identity_matrix[num[0]][num[1]]= 1

    return drug_sim_matrix_new,cell_sim_matrix_new

# drug_sim_matrix_new,cell_sim_matrix_new = computing_knn_weight()
# print(drug_sim_matrix_new)
# print(cell_sim_matrix_new)


def computing_knn():
    drug_sim_matrix,len_drug_smiles = computing_drug_sim_matrix()
    cell_sim_matrix,cell_num = computing_cell_sim_matrix_minmax()
    cell_sim_matrix_new = np.zeros_like(cell_sim_matrix)
    for u in range(cell_num):#换细胞系相似性方法的时候爱这里需要改
        v = cell_sim_matrix[u].argsort()[-6:-1]#argsort函数返回的是数组值从小到大的索引值
        cell_sim_matrix_new[u][v] = cell_sim_matrix[u][v]
    drug_sim_matrix_new = np.zeros_like(drug_sim_matrix)
    for u in range(len_drug_smiles):
        v = drug_sim_matrix[u].argsort()[-6:-1]
        drug_sim_matrix_new[u][v] = drug_sim_matrix[u][v]
    drug_edges = np.argwhere(drug_sim_matrix_new > 0)
    cell_edges = np.argwhere(cell_sim_matrix_new > 0)
    drug_identity_matrix = np.zeros((len_drug_smiles,len_drug_smiles))
    cell_identity_matrix = np.zeros((cell_num,cell_num))
    for item in drug_edges:
        drug_identity_matrix[item[0]][item[1]] = 1
    for num in cell_edges:
        cell_identity_matrix[num[0]][num[1]]= 1

    return drug_identity_matrix,cell_identity_matrix





def cell_Drug_sim_index_weight(drug_sim_matrix_new,cell_sim_matrix_new):#得到相似性边的索引
    drug_index = []
    drug_sim_weight = []
    for i in range(drug_sim_matrix_new.shape[0]):
        for j in range(drug_sim_matrix_new.shape[1]):
            if drug_sim_matrix_new[i,j] != 0:
                drug_index.append([i,j])
                drug_sim_weight.append(drug_sim_matrix_new[i,j])


    cell_index = []
    cell_sim_weight = []
    for i in range(cell_sim_matrix_new.shape[0]):
        for j in range(cell_sim_matrix_new.shape[1]):
            if cell_sim_matrix_new[i, j] != 0:
                cell_index.append([i, j])
                cell_sim_weight.append(cell_sim_matrix_new[i, j])
    drug_index = torch.tensor(drug_index, dtype=torch.long).t()
    cell_index = torch.tensor(cell_index, dtype=torch.long).t()
    drug_sim_weight = torch.tensor(drug_sim_weight,dtype=torch.float32)
    cell_sim_weight = torch.tensor(cell_sim_weight, dtype=torch.float32)
    return drug_index,cell_index,drug_sim_weight,cell_sim_weight




def get_drug_cell_edgeidx(drug_index,cell_index,drug_sim_weight,cell_sim_weight):
#------------drug
    drug_row = drug_index[0]
    drug_row = drug_row.squeeze(0)
    drug_row = drug_row.type(torch.int64)


    drug_col = drug_index[1]
    drug_col = drug_col.squeeze(0)
    drug_col = drug_col.type(torch.int64)


    drug_edge_idx_re = coo_matrix((drug_sim_weight, (drug_row, drug_col)), shape=(222, 222))
    drug_edge_idx_re = sym_adj(drug_edge_idx_re)
    drug_values = drug_edge_idx_re.data
    drug_index = torch.vstack((torch.LongTensor(drug_edge_idx_re.row), torch.LongTensor(drug_edge_idx_re.col)))
    drug_edge_idx = torch.sparse_coo_tensor(drug_index, drug_values, size=(222, 222))
#-----------------cell
    cell_row = cell_index[0]
    cell_row = cell_row.squeeze(0)
    cell_row = cell_row.type(torch.int64)
    cell_col = cell_index[1]
    cell_col = cell_col.squeeze(0)
    cell_col = cell_col.type(torch.int64)
    cell_edge_idx_re = coo_matrix((cell_sim_weight, (cell_row, cell_col)), shape=(568, 568))
    cell_edge_idx_re = sym_adj(cell_edge_idx_re)

    cell_values = cell_edge_idx_re.data
    cell_index = torch.vstack((torch.LongTensor(cell_edge_idx_re.row), torch.LongTensor(cell_edge_idx_re.col)))
    cell_edge_idx = torch.sparse_coo_tensor(cell_index, cell_values, size=(568, 568))
    return drug_edge_idx,cell_edge_idx
# drug_sim_matrix_new,cell_sim_matrix_new = computing_knn_weight()
# drug_index,cell_index,drug_sim_weight,cell_sim_weight = cell_Drug_sim_index_weight(drug_sim_matrix_new,cell_sim_matrix_new)
# drug_edge_idx,cell_edge_idx = get_drug_cell_edgeidx(drug_index,cell_index,drug_sim_weight,cell_sim_weight)


















# drug_identity_matrix,cell_identity_matrix = computing_knn()
# drug_index,cell_index = cell_Drug_sim_index(drug_identity_matrix,cell_identity_matrix)
# print(drug_index)
# print(cell_index)

"""
药物相似性矩阵:
[[0.         0.90349871 0.78966883 ... 0.737572   0.84246815 0.70966288]
 [0.90349871 0.         0.7802686  ... 0.71310965 0.82455598 0.7198641 ]
 [0.78966883 0.7802686  0.         ... 0.72707442 0.79808759 0.74821174]
 ...
 [0.737572   0.71310965 0.72707442 ... 0.         0.72312245 0.67317755]
 [0.84246815 0.82455598 0.79808759 ... 0.72312245 0.         0.72484816]
 [0.70966288 0.7198641  0.74821174 ... 0.67317755 0.72484816 0.        ]]
 细胞相似度矩阵（未进行归一化）
[[0.         0.15625    0.14345992 ... 0.17508418 0.15322581 0.16593886]
 [0.15625    0.         0.11764706 ... 0.23481781 0.27272727 0.16759777]
 [0.14345992 0.11764706 0.         ... 0.09649123 0.11173184 0.2       ]
 ...
 [0.17508418 0.23481781 0.09649123 ... 0.         0.30125523 0.15454545]
 [0.15322581 0.27272727 0.11173184 ... 0.30125523 0.         0.19883041]
 [0.16593886 0.16759777 0.2        ... 0.15454545 0.19883041 0.        ]]
 数据集排序之后的相似度矩阵（未进行归一化）
 [[0.         0.72855052 0.68869815 ... 0.83243051 0.83732087 0.83980309]
 [0.72855052 0.         0.83054614 ... 0.74919117 0.74819075 0.75587887]
 [0.68869815 0.83054614 0.         ... 0.7095713  0.69842941 0.70877937]
 ...
 [0.83243051 0.74919117 0.7095713  ... 0.         0.87009052 0.87686728]
 [0.83732087 0.74819075 0.69842941 ... 0.87009052 0.         0.93426579]
 [0.83980309 0.75587887 0.70877937 ... 0.87686728 0.93426579 0.        ]]
"""

# cell_sim_matrix,dic = computing_cell_sim_matrix()
# print(cell_sim_matrix)