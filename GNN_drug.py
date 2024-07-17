import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool

from smilegraph import smiles2graph
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

#
# if __name__ == '__main__':
#     graph = smiles2graph(smile)
#     x = GNN_drug(layer_drug=3, dim_drug=128).to(device)
#     x_drug = x(graph)
#     print(x_drug)
#     print(x_drug.shape)
#     # save_drug_graph()

def getGraphFea_TGSA(smile):
    graph = smiles2graph(smile)
    # print(graph)Data(x=[23, 77], edge_index=[2, 54], edge_attr=[54, 4], dtype=torch.float32)
    x= GNN_drug(layer_drug=3, dim_drug=128).to(device)
    drug_graph_x =x(graph)
    return drug_graph_x

# getGraphFea_TGSA('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
# Data(x=[23, 77], edge_index=[2, 54], edge_attr=[54, 4], dtype=torch.float32)