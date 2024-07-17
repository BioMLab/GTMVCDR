import time
import numpy as np
import csv
import pandas as pd

import torch.utils.data as Data
from scipy.sparse import coo_matrix

from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool

from base_model.LAGCN import GraphConvolution
from torch.nn import Parameter
from similarity import computing_knn, computing_knn_weight, cell_Drug_sim_index_weight, get_drug_cell_edgeidx
from utils import reset, glorot, sym_adj
from Cell_graph import get_STRING_graph, get_predefine_cluster
from base_model.GNN_cell import GNN_cell
from sklearn.model_selection import KFold
from torch import nn
import torch
from model_helper import Encoder_MultipleLayers, Embeddings
import torch.nn.functional as F
import codecs
from subword_nmt.apply_bpe import BPE

from smilegraph import smiles2graph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPS = 1e-15




"""Transformer for drug"""
class DataEncoding:
    def __init__(self, vocab_dir):
        self.vocab_dir = vocab_dir
        self.device = torch.device('cuda:0')
    def _drug2emb_encoder(self, smile):
        vocab_path = "{}/drug_codes_chembl_freq_1500.txt".format(self.vocab_dir)
        sub_csv = pd.read_csv("{}/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))
        bpe_codes_drug = codecs.open(vocab_path)#codecs.open类似于open函数
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values#idx2word_d为index
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(smile).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d
        return i,np.asarray(input_mask)

    def encode(self, smile):
        return self._drug2emb_encoder(smile)
"""GNN for drug"""
class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.dropout_ratio = 0.2
        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)
            self.drug_emb = nn.Sequential(
                nn.Linear(self.dim_drug * self.layer_drug, 256),#改这里来调整药物向量维度
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )

    def forward(self, drug):
        x = drug.x
        edge_index = drug.edge_index
        batch = drug.batch

        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index).to(device)).to(device)
            x = self.bns_drug[i](x).to(device)
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch).to(device)
        x_drug = self.drug_emb(x_drug)
        # print("111111111111111111")
        # print(x_drug)
        # print(x_drug.shape)torch.Size([128, 384]),torch.Size([92, 384])
        return x_drug



class transformer(nn.Sequential):
    def __init__(self):
        super(transformer, self).__init__()
        input_dim_drug = 2586
        transformer_emb_size_drug = 256
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                         transformer_emb_size_drug,
                         50,
                         transformer_dropout_rate)
        self.emb = self.emb.to(device)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
        self.encoder = self.encoder.to(device)
    def forward(self, v):

        """
               print("-----------------")
               print(v)
               [tensor([ 809, 1594,  781,   38,   82,   35,  304,  238,  853,   98,  129,  606,
                688,  187,  168,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                  0,    0], device='cuda:0', dtype=torch.int32), tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0], device='cuda:0', dtype=torch.int32)]

        """
        e = v[0].long().to(device)
        #print(e.size())#50
        e_mask = v[1].long().to(device)
        """print(e_mask)torch.Size([50])
        tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0], device='cuda:0')
        """
        ex_e_mask = e_mask.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
        #print(ex_e_mask.shape)torch.Size([1, 1, 1, 50])
        #print(ex_e_mask.shape)torch.Size([50, 1, 1])
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0
        emb = self.emb(e).to(device)
        emb = emb.unsqueeze(0)
        #print(emb.shape)#torch.Size([1, 50, 128])
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float()).to(device)
        return encoded_layers[:, 0]


def cmask(num, ratio, seed):
    mask = np.ones(num, dtype=bool)
    mask[0:int(ratio * num)] = False
    np.random.seed(seed)
    np.random.shuffle(mask)
    return mask

def augment_sim(data_new):
    drug_identity_matrix, cell_identity_matrix = computing_knn()
    drug_identity_matrix = torch.tensor(drug_identity_matrix)
    cell_identity_matrix = torch.tensor(cell_identity_matrix)
    cellineid = list(set([item[0] for item in data_new]));
    cellineid.sort()  # 生成列表cellineid，对细胞系排序
    pubmedid = list(set([item[1] for item in data_new]));
    pubmedid.sort()  # 生成列表pubmedid，对pubchem ID号排序
    adj_mat_zeros = torch.zeros([len(cellineid), len(pubmedid)])
    adj_mat_T_zeros = torch.t(adj_mat_zeros)
    cell_drug = torch.cat((cell_identity_matrix, adj_mat_zeros), dim=1)
    drug_cell = torch.cat((adj_mat_T_zeros, drug_identity_matrix), dim=1)
    adj_matrix_only_sim = torch.cat((cell_drug, drug_cell), dim=0)
    return adj_matrix_only_sim



def process(cell_graph, data_new, nb_celllines, nb_drugs):
    # -----construct cell line-drug response pairs
    cellineid = list(set([item[0] for item in data_new]));
    cellineid=[str(i) for i in cellineid]
    cellineid.sort()  # 生成列表cellineid，对细胞系排序
    pubmedid = list(set([item[1] for item in data_new]));
    pubmedid = [str(i) for i in pubmedid]
    pubmedid.sort()  # 生成列表pubmedid，对pubchem ID号排序
    cellmap = list(zip(cellineid, list(range(len(cellineid)))))  # len(cellineid)返回cellineid列表长度
    pubmedmap = list(zip(pubmedid, list(range(len(cellineid), len(cellineid) + len(pubmedid)))))
    cellline_num = np.squeeze(
        [[j[1] for j in cellmap if i[0] == j[0]] for i in data_new])  # cellline_num = [560  ... 0]
    pubmed_num = np.squeeze([[int(j[1]) for j in pubmedmap if i[1] == j[0]] for i in
                             data_new])  # pubmed_num=[782 781 780 ... 563 562 561],len(pubmed_num)=100572
    IC_num = np.squeeze([int(i[2]) for i in data_new])  # IC_num = [-1 -1  1 ... -1 -1 -1],len(IC_num) = 100572
    allpairs = np.vstack((cellline_num, pubmed_num, IC_num)).T
    allpairs = allpairs[allpairs[:, 2].argsort()]  # 为了划分训练集、测试集



    #split
    use_independent_testset = False
    if (use_independent_testset == True):
        edge_mask = cmask(len(allpairs), 0.1, 666)  # edge_mask = [True,False,True,False,True,False]
        train = allpairs[edge_mask][:, 0:3]  # 通过true,来选择训练边
        test = allpairs[~edge_mask][:, 0:3]  # 剩下的即为验证边
    else:
        # CV_edgemask = cmask(len(allpairs), 0.1, 666)
        # cross_validation = allpairs[CV_edgemask][:, 0:3]
        # vali_mask = cmask(len(cross_validation), 0.2, 66)
        # train = cross_validation[vali_mask][:, 0:3]
        # test = cross_validation[~vali_mask][:, 0:3]
        #新加的:
        train_temp_list = []
        test_temp_list = []
        kf = KFold(n_splits=5, shuffle=True,random_state=0)
        for train_index_temp, test_index_temp in kf.split(allpairs):  # 调用split方法切分数据
            train_temp_list.append(train_index_temp)
            test_temp_list.append(test_index_temp)
            print('train_index:%s , test_index: %s ' % (train_index_temp, test_index_temp))

        train_index = train_temp_list[3].tolist()
        test_index = test_temp_list[3].tolist()
        train = allpairs[train_index][:, 0:3]
        test = allpairs[test_index][:, 0:3]


    train_shengchengxiangliang = train.copy()
    train[:, 1] -= nb_celllines#为了生成细胞系-药物反应矩阵的，减去了序号
    test[:, 1] -= nb_celllines


    train_mask = coo_matrix((np.ones(train.shape[0], dtype=bool), (train[:, 0], train[:, 1])),
                            shape=(nb_celllines, nb_drugs)).toarray()   # 构建整个细胞系-药物反应矩阵（敏感耐药都包括）

    test_mask = coo_matrix((np.ones(test.shape[0], dtype=bool), (test[:, 0], test[:, 1])),
                           shape=(nb_celllines, nb_drugs)).toarray()  # 验证构建整个细胞系-药物反应矩阵（敏感耐药都包括）
    train_mask = torch.from_numpy(train_mask).view(-1)
    # print(train_mask)tensor([ True, False,  True,  ...,  True,  True,  True])
    test_mask = torch.from_numpy(test_mask).view(-1)
    if (use_independent_testset == True):
        pos_edge = allpairs[allpairs[:, 2] == 1, 0:2]  # 在所有反应边中取出反应值为1的细胞系和药物索引
    else:
        pos_edge = allpairs[allpairs[:, 2] == 1, 0:2]

    pos_edge[:, 1] -= nb_celllines  # 让allpairs中第二列减去细胞系的数目，这样药物索引也是从0开始
    label_pos1 = coo_matrix((np.ones(pos_edge.shape[0]), (pos_edge[:, 0], pos_edge[:, 1])),
                            shape=(nb_celllines, nb_drugs)).toarray()  # 生成细胞系药物反应矩阵，敏感即为1，其余位置为0
    label_pos=label_pos1
    label_pos = torch.from_numpy(label_pos).type(torch.FloatTensor).view(-1)  # array转为torch，view（-1）转为1维
    # 用药物生成细胞系：
    train_shengchengxiangliang = train_shengchengxiangliang[train_shengchengxiangliang[:, 0].argsort()]
    temp1 = -1
    cell_all_list_withDRUG = []#cell_all_list_withDRUG保存了用药物表示所有细胞系的向量
    cell_list_withDRUG=[]
    for item in train_shengchengxiangliang:
        if item[0] !=temp1:
            cell_all_list_withDRUG.append(cell_list_withDRUG)
            temp1 = item[0]
            cell_list_withDRUG = [0 for _ in range(nb_drugs)]
            temp_index =item[1] - nb_celllines
            cell_list_withDRUG[temp_index] = item[2]
        else:
            temp_index = item[1] - nb_celllines
            cell_list_withDRUG[temp_index] = item[2]
    cell_all_list_withDRUG.append(cell_list_withDRUG)
    cell_all_list_withDRUG=cell_all_list_withDRUG[1:]
    cell_new_tensor = torch.tensor(cell_all_list_withDRUG)#得到的用药物表示的细胞系向量
    #用细胞系生成药物：
    train_shengchengxiangliang = train_shengchengxiangliang[train_shengchengxiangliang[:, 1].argsort()]#按照药物序号排序
    temp2 = -1
    drug_all_list_withCELL = []
    drug_list_withCELL = []
    for item in train_shengchengxiangliang:
        if item[1] != temp2:
            drug_all_list_withCELL.append(drug_list_withCELL)
            temp2 = item[1]
            drug_list_withCELL = [0 for _ in range(nb_celllines)]
            drug_list_withCELL[item[0]] = item[2]
        else:
            drug_list_withCELL[item[0]] = item[2]
    drug_all_list_withCELL.append(drug_list_withCELL)
    drug_all_list_withCELL = drug_all_list_withCELL[1:]
    drug_new_tensor = torch.tensor(drug_all_list_withCELL)#得到的用细胞系表示的药物向量
    allpairs_only_sim = []
    adj_matrix_only_sim = augment_sim(data_new)
    for row in range(adj_matrix_only_sim.shape[0]):
        for col in range(adj_matrix_only_sim.shape[1]):
            if adj_matrix_only_sim[row, col] != 0:
                allpairs_only_sim.append([row, col, adj_matrix_only_sim[row, col]])
    allpairs_only_sim = np.array(allpairs_only_sim, dtype=int)  # 得到相似性边


    # # ----cell line_feature_input
    cellineid = list(set([item[0] for item in data_new]))
    cellineid.sort()
    cellmap = list(zip(cellineid, list(range(len(cellineid)))))
    cellid = [item[0] for item in cellmap]

    cell_feature_edge_index = np.load('./data/TCGA_cell/edge_index_PPI_0.95.npy')

    cell_feature_edge_index = torch.tensor(cell_feature_edge_index, dtype=torch.long)
    for u in cell_graph:
        u.edge_index = cell_feature_edge_index
    genes_path = './data/TCGA_cell'
    edge_index = get_STRING_graph(genes_path, 0.95)
    cluster_predefine = get_predefine_cluster(edge_index, genes_path, 0.95, device)

    cell_feature_model = GNN_cell(3, 3, 8, cluster_predefine).to(
        device)  # num_feature组学数, layer_cell图卷积层数, dim_cell隐藏层维度, cluster_predefine
    # print(type(cell_feature))<class 'base_model.GNN_cell.GNN_cell'>
    print(Batch.from_data_list(cell_graph).to(device))
    cell_tensor = cell_feature_model(Batch.from_data_list(cell_graph).to(device)).detach()
    # print(cell_tensor.shape)torch.Size([568, 3056])
    drug_set = Data.DataLoader(dataset= drug_new_tensor,
                               batch_size=nb_drugs,
                               shuffle=False)
    cellline_set = Data.DataLoader(dataset= cell_new_tensor,
                                   batch_size=nb_celllines,
                                   shuffle=False)
    return  cell_tensor, allpairs, allpairs_only_sim,cell_new_tensor,drug_new_tensor,drug_set,cellline_set,label_pos, train_mask, test_mask
class Attention(nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        hidden_size = in_size

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class NodeRepresentation(nn.Module):
    def __init__(self, output,  units_list=[256, 256, 256],
                 use_relu=True, use_bn=True,
                 use_GMP=True, use_mutation=True, use_gexpr=True, use_methylation=True):
        super(NodeRepresentation, self).__init__()
        torch.manual_seed(0)  # 设置CPU生成随机数的种子，方便下次复现实验结果
        self.Transformer = transformer().to(device)
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP
        # 套用GADRP_for  drug
        self.dropout_drug = nn.Dropout(0.2)
        self.relu_drug = nn.ReLU(inplace=True)
        self.gcn_drug = GraphConvolution(568, 568, device=device)
        self.att_drug = Parameter(torch.tensor([1 / 1, 1 / 1, 1 / 1, 1 / 1, 1 / 1]))

        # 套用GADRP_for  cell
        self.dropout_cell = nn.Dropout(0.2)
        self.relu_cell = nn.ReLU(inplace=True)
        self.gcn_cell = GraphConvolution(222, 222, device=device)
        self.att_cell = Parameter(torch.tensor([1 / 1, 1 / 1, 1 / 1, 1 / 1, 1 / 1]))
        #套用TGDRP
        self.dim_cell = 3056
        self.dropout_ratio = 0.2
        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio)
        )
        # #设置全连接层调整细胞系药物向量维度
        # self.FNN_cell = nn.Linear(222,128)
        # self.FNN_drug = nn.Linear(568,128)
        self.batchc_drug = nn.BatchNorm1d(568)
        self.batchc_cell = nn.BatchNorm1d(222)
        self.reset_para()
        self.meta_heads = 6
        self.num_tensors = 2
        self.softmax = nn.Softmax(dim=0)
        self.attention_weights_multi = nn.Parameter(torch.randn(self.meta_heads, self.num_tensors))
        self.softmax_multi = nn.Softmax(dim=-1)
        self.GNN_drug = GNN_drug(layer_drug=3, dim_drug=128)
        self.layer_att_drug = Attention(568)
        self.layer_att_cell = Attention(222)

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return
    def meta_weight_multi(self, *tensors):
        # 归一化注意力权重
        normalized_attention_weights = self.softmax_multi(self.attention_weights_multi)

        # 应用每个头的权重并计算加权和
        weighted_sums = []
        for i in range(self.meta_heads):
            weighted_sum = sum(tensor * normalized_attention_weights[i, idx]
                               for idx, tensor in enumerate(tensors))
            weighted_sums.append(weighted_sum)

        # 合并所有头的结果
        combined_weighted_sum = torch.stack(weighted_sums).mean(dim=0)
        column_means = normalized_attention_weights.mean(dim=0)
        # 重塑tensor以符合目标大小 [6, 1]
        column_means_reshaped = column_means.view(1, -1).expand(6, -1)
        new_attention = column_means_reshaped[0, :]
        new_attention = self.softmax(new_attention)
        return combined_weighted_sum, new_attention



    def forward(self,cell_tensor,cell_new_tensor,drug_new_tensor,drug_smiles):
        # -----drug representation

        drug_feature_graph = {}
        drug_feature_Transformer = {}
        for cid, smile in drug_smiles.items():
            drug_feature_graph[cid] = torch.tensor(self.GNN_drug(smiles2graph(smile))).to(device)

            vocab_dir = './data'
            obj = DataEncoding(vocab_dir=vocab_dir)

            i, enc = obj.encode(smile)
            enc = torch.from_numpy(enc).to(device)
            i = torch.from_numpy(i).to(device)
            enc = [i, enc]
            drug_transformer_initial = self.Transformer(enc)
            drug_feature_Transformer[cid] = torch.tensor((drug_transformer_initial.to(device))).to(device)
        drug_feature_graph = list(drug_feature_graph.values())
        drug_feature_Transformer = list(drug_feature_Transformer.values())

        x_drug_graph = torch.tensor([item.cpu().detach().numpy() for item in drug_feature_graph])
        x_drug_graph = x_drug_graph.to(device)

        x_drug_graph = torch.squeeze(x_drug_graph, dim=1)

        x_drug_Transformer = torch.tensor([item.cpu().detach().numpy() for item in drug_feature_Transformer])
        x_drug_Transformer = x_drug_Transformer.to(device)
        x_drug_Transformer = torch.squeeze(x_drug_Transformer, dim=1)
        x_drug, attention = self.meta_weight_multi(x_drug_Transformer, x_drug_graph)

        #细胞系表示
        x_cell = cell_tensor.to(device)
        x_cell = self.cell_emb(x_cell).to(device)

        #计算Cell特征向量相似性(余选相似度)
        x_cell_sim = x_cell / torch.norm(x_cell, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
        cell_similarity = torch.mm(x_cell_sim, x_cell_sim.T)  # 矩阵乘法
        cell_similarity_np = cell_similarity.detach().cpu().numpy()

        # 计算Drug特征向量相似性(余选相似度)
        x_drug_sim = x_drug / torch.norm(x_drug, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模

        drug_similarity = torch.mm(x_drug_sim, x_drug_sim.T)  # 矩阵乘法

        drug_similarity_np = drug_similarity.detach().cpu().numpy()


        drug_index, cell_index, drug_sim_weight, cell_sim_weight = cell_Drug_sim_index_weight(drug_similarity_np,
                                                                                      cell_similarity_np)
        drug_edge_idx, cell_edge_idx = get_drug_cell_edgeidx(drug_index, cell_index, drug_sim_weight, cell_sim_weight)
        cell_edge_idx =cell_edge_idx.to(device)
        drug_edge_idx =drug_edge_idx.to(device)
        x_drug_list=[]
        x_cell_list=[]

        #GADRP_drug
        x_drug = drug_new_tensor.to(device)
        x_drug = x_drug.type(torch.float32)
        x_drug = self.dropout_drug(x_drug)

        drug_output1 = self.relu_drug(self.gcn_drug(x_drug , x_drug , drug_edge_idx, 0.1))
        drug_output1 = self.dropout_drug(drug_output1)
        x_drug_list.append(drug_output1)

        drug_output2 = self.relu_drug(self.gcn_drug(x_drug , drug_output1, drug_edge_idx, 0.1))
        x_drug_list.append(drug_output2)

        drug_output3 = self.relu_drug(self.gcn_drug(x_drug , drug_output2, drug_edge_idx, 0.1))
        drug_output3 = self.dropout_drug(drug_output3)
        x_drug_list.append(drug_output3)

        drug_output4 = self.relu_drug(self.gcn_drug(x_drug , drug_output3, drug_edge_idx, 0.1))
        x_drug_list.append(drug_output4)


        drug_output5 = self.relu_drug(self.gcn_drug(x_drug , drug_output4, drug_edge_idx, 0.1))
        drug_output5 = self.dropout_drug(drug_output5)
        x_drug_list.append(drug_output5)

        x_drug_features = torch.stack(x_drug_list, dim=1)
        x_drug, _ = self.layer_att_drug(x_drug_features)

        x_drug = self.dropout_drug(x_drug)

        #GADRP_cell
        x_cell = cell_new_tensor.to(device)#新加的
        x_cell = x_cell.type(torch.float32)

        x_cell = self.dropout_cell(x_cell)
        cell_output1 = self.relu_cell(self.gcn_cell(x_cell, x_cell, cell_edge_idx, 0.1))
        cell_output1 = self.dropout_cell(cell_output1)
        x_cell_list.append(cell_output1)
        cell_output2 = self.relu_cell(self.gcn_cell(x_cell, cell_output1, cell_edge_idx, 0.1))
        x_cell_list.append(cell_output2)

        cell_output3 = self.relu_cell(self.gcn_cell(x_cell, cell_output2, cell_edge_idx, 0.1))
        cell_output3 = self.dropout_cell(cell_output3)
        x_cell_list.append(cell_output3)

        cell_output4 = self.relu_cell(self.gcn_cell(x_cell, cell_output3, cell_edge_idx, 0.1))
        x_cell_list.append(cell_output4)

        cell_output5 = self.relu_cell(self.gcn_cell(x_cell, cell_output4, cell_edge_idx, 0.1))
        cell_output5 = self.dropout_cell(cell_output5)
        x_cell_list.append(cell_output5)
        x_cell_features = torch.stack(x_cell_list, dim=1)
        x_cell, _ = self.layer_att_cell(x_cell_features)

        x_cell = self.dropout_cell(x_cell)
        x_cell = self.batchc_cell(x_cell)
        x_drug = self.batchc_drug(x_drug)
        return x_cell,x_drug

class GraphCDR(nn.Module):
    def __init__(self, hidden_channels, feat, index_cell,index_drug):
        super(GraphCDR, self).__init__()
        self.features_before_output = None
        self.hidden_channels = hidden_channels
        self.feat = feat
        self.index_cell = index_cell
        self.index_drug = index_drug
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.act = nn.Sigmoid()
        self.dropout_ratio = 0.2
        # self.fc = nn.Linear(272, 10)
        # self.fd = nn.Linear(272, 10)
        self.reset_parameters()
        self.regression = nn.Sequential(
            nn.Linear(790, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(64, 1)
        )
    def reset_parameters(self):  # 参数初始化
        reset(self.feat)

        glorot(self.weight)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cell_tensor,cell_new_tensor,drug_new_tensor,drug_smiles):

        # ---cell+drug embedding from the CDR graph and the corrupted CDR graph
        start_time2 = time.time()

        cellpos,drugpos = self.feat(cell_tensor,cell_new_tensor,drug_new_tensor,drug_smiles)#药物和细胞系竖着罗列的特征,融合完异构信息的特征表示
        pos_adj = torch.empty(self.index_cell ,self.index_drug).to(device)
        alls=[]
        for i in range(self.index_cell):
            for j in range(self.index_drug):
                all_pos = torch.cat((cellpos[i],drugpos[j]),-1)#all_pos为单个细胞系单个药物向量拼接
                alls.append(all_pos)
        allse=torch.stack(alls[:],dim=0)

        features = allse
        for layer in self.regression[:-1]:
            features = layer(features)

        self.features_before_output = features

        # 然后只通过最后一层来获取logits_FNN
        pos_adj_dange = self.regression[-1](features).squeeze()

        for i in range(self.index_cell):
            for j in range(self.index_drug):
                pos_adj[i][j] = pos_adj_dange[i*self.index_drug+j]
        pos_adj = self.act(pos_adj)

        return pos_adj.view(-1),self.features_before_output

