import argparse

import numpy as np
import pandas as pd
import os
import csv
import scipy
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import random

from base_model.GNN_cell import GNN_cell


def get_genes_graph(genes_path, save_path, method='pearson', thresh=0.95, p_value=False):
    """
    determining adjaceny matrix based on correlation
    :param genes_exp_path:
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)#os.path.join()函数用于路径拼接文件路径，可以传入多个路径
    genes_exp_df = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)

    # calculate correlation matrix
    genes_exp_corr = genes_exp_df.corr(method=method)
    genes_exp_corr = genes_exp_corr.apply(lambda x: abs(x))
    n = genes_exp_df.shape[0]

    # binarize
    if p_value == True:
        dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        thresh = dist.isf(0.05)

    adj = np.where(genes_exp_corr > thresh, 1, 0)
    adj = adj - np.eye(genes_exp_corr.shape[0], dtype=np.int)
    edge_index = np.nonzero(adj)
    np.save(os.path.join(save_path, 'edge_index_{}_{}.npy').format(method, thresh), edge_index)

    return n, edge_index


def ensp_to_hugo_map():
    with open('./data/TCGA_cell/9606.protein.info.v11.0.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensp_map = {row[0]: row[1] for row in csv_reader if row[0] != ""}

    return ensp_map


def hugo_to_ncbi_map():
    with open('./data/TCGA_cell/enterez_NCBI_to_hugo_gene_symbol_march_2019.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        hugo_map = {row[0]: int(row[1]) for row in csv_reader if row[1] != ""}

    return hugo_map


def save_cell_graph(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp = pd.read_csv(('./data/TCGA_cell/EXP_norm.csv'), index_col=0)
    cn = pd.read_csv(('./data/TCGA_cell/CNV_norm.csv'), index_col=0)
    mu = pd.read_csv(('./data/TCGA_cell/MUT_norm.csv'), index_col=0)
    # me = pd.read_csv(os.path.join(genes_path, 'me.csv'), index_col=0)
    # print('Miss values：{}，{}，{}, {}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum(),
    #                                         me.isna().sum().sum()))

    index = exp.index
    columns = exp.columns

    scaler = StandardScaler()
    exp = scaler.fit_transform(exp)
    cn = scaler.fit_transform(cn)
    # me = scaler.fit_transform(me)

    imp_mean = SimpleImputer()
    exp = imp_mean.fit_transform(exp)

    exp = pd.DataFrame(exp, index=index, columns=columns)
    cn = pd.DataFrame(cn, index=index, columns=columns)
    mu = pd.DataFrame(mu, index=index, columns=columns)
    # me = pd.DataFrame(me, index=index, columns=columns)
    cell_names = exp.index
    # print('Miss values：{}，{}，{}, {}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum(),
    #                                         me.isna().sum().sum()))

    cell_dict = {}
    for i in cell_names:
        # cell_dict[i] = Data(x=torch.tensor([exp.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([cn.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([mu.loc[i]], dtype=torch.float).T)
        cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i], me.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = [np.array(exp.loc[i], dtype=np.float32), np.array(cn.loc[i], dtype=np.float32),
        #                 np.array(mu.loc[i], dtype=np.float32)]
    print(cell_dict)
    np.save(os.path.join(save_path, 'cell_feature_all_std222.npy'), cell_dict)
    print("finish saving cell mut&ge&cnv data!")



def save_cell_graph(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp = pd.read_csv(('./data/TCGA_cell/EXP.csv'), index_col=0)
    cn = pd.read_csv(('./data/TCGA_cell/CN.csv'), index_col=0)
    mu = pd.read_csv(('./data/TCGA_cell/MUT.csv'), index_col=0)
    # me = pd.read_csv(os.path.join(genes_path, 'me.csv'), index_col=0)
    # print('Miss values：{}，{}，{}, {}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum(),
    #                                         me.isna().sum().sum()))

    index = exp.index
    columns = exp.columns

    scaler = StandardScaler()
    exp = scaler.fit_transform(exp)
    cn = scaler.fit_transform(cn)
    # me = scaler.fit_transform(me)

    imp_mean = SimpleImputer()
    exp = imp_mean.fit_transform(exp)

    exp = pd.DataFrame(exp, index=index, columns=columns)
    cn = pd.DataFrame(cn, index=index, columns=columns)
    mu = pd.DataFrame(mu, index=index, columns=columns)
    # me = pd.DataFrame(me, index=index, columns=columns)
    cell_names = exp.index
    # print('Miss values：{}，{}，{}, {}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum(),
    #                                         me.isna().sum().sum()))

    cell_dict = {}
    for i in cell_names:
        # cell_dict[i] = Data(x=torch.tensor([exp.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([cn.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([mu.loc[i]], dtype=torch.float).T)
        cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i], me.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = [np.array(exp.loc[i], dtype=np.float32), np.array(cn.loc[i], dtype=np.float32),
        #                 np.array(mu.loc[i], dtype=np.float32)]
    print(cell_dict)
    np.save(os.path.join(save_path, 'cell_feature_all_std222.npy'), cell_dict)
    print("finish saving cell mut&ge&cnv data!")





def get_STRING_graph(genes_path, thresh=0.95):
    save_path = os.path.join(genes_path, 'edge_index_PPI_{}.npy'.format(thresh))

    if not os.path.exists(save_path):
        # gene_list
        exp = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)

        gene_list = exp.columns.to_list()
        gene_list = [int(gene[1:-1]) for gene in gene_list]

        # load STRING
        ensp_map = ensp_to_hugo_map()
        hugo_map = hugo_to_ncbi_map()
        edges = pd.read_csv('./data/TCGA_cell/9606.protein.links.detailed.v11.0.txt', sep=' ')

        # edge_index
        selected_edges = edges['combined_score'] > (thresh * 1000)

        edge_list = edges[selected_edges][["protein1", "protein2"]].values.tolist()

        edge_list = [[ensp_map[edge[0]], ensp_map[edge[1]]] for edge in edge_list if
                     edge[0] in ensp_map.keys() and edge[1] in ensp_map.keys()]

        edge_list = [[hugo_map[edge[0]], hugo_map[edge[1]]] for edge in edge_list if
                     edge[0] in hugo_map.keys() and edge[1] in hugo_map.keys()]
        edge_index = []
        for i in edge_list:
            if (i[0] in gene_list) & (i[1] in gene_list):
                edge_index.append((gene_list.index(i[0]), gene_list.index(i[1])))
                edge_index.append((gene_list.index(i[1]), gene_list.index(i[0])))
        edge_index = list(set(edge_index))
        edge_index = np.array(edge_index, dtype=np.int64).T


        # 保存edge_index
        # print(len(gene_list))
        # print(thresh, len(edge_index[0]) / len(gene_list))
        np.save(
            os.path.join('./data/TCGA_cell', 'edge_index_PPI_{}.npy'.format(thresh)),edge_index)

    else:
        edge_index = np.load(save_path)

    return edge_index


def get_predefine_cluster(edge_index, save_path, thresh, device):
    save_path = os.path.join(save_path, 'cluster_predefine_PPI_{}.npy'.format(thresh))
    if not os.path.exists(save_path):
        g = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), x=torch.zeros(706, 1))
        g = Batch.from_data_list([g])
        cluster_predefine = {}
        for i in range(5):
            cluster = graclus(g.edge_index, None, g.x.size(0))
            print(len(cluster.unique()))
            g = max_pool(cluster, g, transform=None)
            cluster_predefine[i] = cluster
        np.save(save_path, cluster_predefine)
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}
    else:
        cluster_predefine = np.load(save_path, allow_pickle=True).item()
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}

    return cluster_predefine


# if __name__ == '__main__':
#     gene_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
#     save_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
#     # get_genes_graph(gene_path,save_path, thresh=0.53)
#     # save_cell_graph(gene_path, save_path)
#     edge_index,selected_edges2= get_STRING_graph(gene_path, thresh=0.95)
#     print(edge_index)
#
def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def load_graph_data_SA(args):

    cell_name2feature_dict = np.load('./data/TCGA_cell/cell_feature_all_std.npy',
                                        allow_pickle=True).item()
    # print("00000000")
    # print(cell_name2feature_dict)    {'ACH-000001': Data(x=[706, 3]), 'ACH-000002': Data(x=[706, 3]), 'ACH-000006': Data(x=[706, 3]), 'ACH-000007': Data(x=[706, 3]), 'ACH-000008': Data(x=[706, 3]), 'ACH-000009': Data(x=[706, 3]),
    # dict_dir = r'C:\复现\TGSA-master\data\similarity_augment\dict'  # 改过的

    cell_idx2feature_dict = {u: cell_name2feature_dict[v] for u, v in cell_idx2id_dict.items()}
    # print("@@@@@@@@@")
    # print(cell_idx2feature_dict){0: Data(x=[706, 3]), 1: Data(x=[706, 3]), 2: Data(x=[706, 3]), 3: Data(x=[706, 3]), 4: Data(x=[706, 3]), 5: Data(x=[706, 3]), 6: Data(x=[706, 3]), 7: Data(x=[706, 3]), 8: Data(x=[706, 3]), 9: Data(x=[706, 3]),


    cell_graph = [u for _, u in cell_idx2feature_dict.items()]
    # print("4444444444")
    # print(cell_graph)[Data(x=[706, 3]), Data(x=[706, 3]), Data(x=[706, 3]), Data(x=[706, 3]), Data(x=[706, 3]), Data(x=[706, 3]), Data(x=[706, 3]), Data(x=[706, 3]), Data(x=[706, 3]), Data(x=[706, 3]),
    cell_feature_edge_index = np.load(
        './data/CellLines_DepMap/CCLE_580_18281/census_706/edge_index_PPI_{}.npy'.format(args.edge))
    cell_feature_edge_index = torch.tensor(cell_feature_edge_index, dtype=torch.long)
    for u in cell_graph:
        u.edge_index = cell_feature_edge_index
    # set_random_seed(args.seed)


    genes_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'

    edge_index = get_STRING_graph(genes_path, args.edge)
    cluster_predefine = get_predefine_cluster(edge_index, genes_path, args.edge, args.device)
    args.num_feature = 3
    model = TG(cluster_predefine, args).to(args.device)
    # parameter = {'drug_emb':model.drug_emb, 'cell_emb':model.cell_emb, 'regression':model.regression}
    cell_nodes = model.GNN_cell(Batch.from_data_list(cell_graph).to(args.device)).detach()
    # print(cell_nodes.shape)torch.Size([580, 3056])




    return cell_nodes

class TG(nn.Module):
    def __init__(self, cluster_predefine, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_feature = 3
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.dropout_ratio = args.dropout_ratio



        # cell graph branch
        self.GNN_cell = GNN_cell(self.num_feature, self.layer_cell, self.dim_cell, cluster_predefine)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.GNN_cell.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 1)
        )

    def forward(self, cell):

        # forward cell
        x_cell = self.GNN_cell(cell)
        x_cell = self.cell_emb(x_cell)

        return  x_cell
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--knn', type=int, default=5,
                        help='k-nearest-neighbour')
    parser.add_argument('--layer_drug', type=int, default=3, help='layer for drug')
    parser.add_argument('--dim_drug', type=int, default=128, help='hidden dim for drug')
    parser.add_argument('--layer', type=int, default=3, help='number of GNN layer')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dim for cell')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--edge', type=float, default='0.95', help='edge for gene graph')
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--pretrain', type=int, default=1, help='pretrain(0 or 1)')
    parser.add_argument('--weight_path', type=str, default='',
                        help='filepath for pretrained weights')

    return parser.parse_args()


args = arg_parse()

genes_path = './data/TCGA_cell'
save_path ='./data/TCGA_cell'
save_cell_graph(save_path)

# edge_index = get_STRING_graph(genes_path)
# cluster_predefine = get_predefine_cluster(edge_index, genes_path, args.edge, args.device)
# cell_ex = TGDRP(cluster_predefine, args).to(args.device)
# cell_nodes = load_graph_data_SA(args)
# x_cell = cell_ex()




