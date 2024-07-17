import pandas as pd
from DrugEndoerFea import getDrugTransformerFea
import torch
import numpy as np
import csv
import torch.utils.data as Data

# class concatenate_drug(nn.Module):
#     def __init__(self, drug_fea2,drug_fea1, output):
#         super(concatenate_drug, self).__init__()
#         self.drug_fea2 = drug_fea2
#         self.drug_fea1 = drug_fea1
#     def process
from GNN_drug import getGraphFea_TGSA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def getFeatures():
    #读取药物数据
    # data = pd.read_csv('data/Drug/drug.csv')
    Drug_SMILES = "data/Drug/11111.csv"
    Drug_info_file = "data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv"
    IC50_threds_file = "data/Drug/drug_threshold.txt"

    reader = csv.reader(open(Drug_info_file, 'r'))  # 使用csv的reader()方法，创建一个reader对象
    rows = [item for item in reader]  # 生成一个列表
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}
    reader2 = csv.reader(open(Drug_SMILES, 'r'))  # 使用csv的reader()方法，创建一个reader对象
    next(reader2)
    rows = [item for item in reader2]  # 生成一个列表
    drug_smiles = {item[0]: item[1] for item in rows}  # 定义字典drug_smiles，key为drug_id,values为SMILES,isdigit()判断字符串是否只由数字组成的函数

    reader2 = csv.reader(open(Drug_SMILES, 'r'))  # 使用csv的reader()方法，创建一个reader对象
    next(reader2)
    rows = [item for item in reader2]  # 生成一个列表
    drug_smiles = {item[0]: item[1] for item in rows}#定义字典drug_feature，key为drug_id,values为SMILES,isdigit()判断字符串是否只由数字组成的函数

    drug2thred = {}  # 存储了pubchem ID对应的IC50
    for line in open(IC50_threds_file).readlines()[1:]:
        drug2thred[str(line.split('\t')[0])] = float(line.strip().split('\t')[1])
    drug_pubchem_id_set = []
    reader2 = csv.reader(open(Drug_SMILES, 'r'))  # 使用csv的reader()方法，创建一个reader对象
    next(reader2)
    for each in reader2:
        drug_pubchem_id_set.append(each[0])
    assert len(drug_pubchem_id_set) == len(drug_smiles.values())
    drug_feature_graph = {}
    drug_feature_Transformer = {}
    #print(drug_pubchem_id_set)
    #drug_pubchem_id_set存储了药物smiles集合，用drug_pubchem_id_set做drug_feature的key
    for cid,smile in drug_smiles.items():
        drug_fea1 = getGraphFea_TGSA(smile).to(device)#改为
        # print("Drug_fea1")
        # print(drug_fea1.shape)#[1,240]
        drug_fea2 = getDrugTransformerFea(smile).to(device)
        # print("Drug_fea2")
        # print(drug_fea2.shape)


        drug_feature_graph[cid] = torch.tensor(drug_fea1).to(device)
        drug_feature_Transformer[cid] = torch.tensor((drug_fea2)).to(device)
        #drug_feature[cid] = torch.cat((drug_fea1.reshape(1,128),drug_fea2),dim=1).to(device)graphdrp用到的
        # drug_feature[cid] = drug_fea2
    drug_feature_graph = list(drug_feature_graph.values())
    drug_feature_Transformer = list(drug_feature_Transformer.values())
    # print(len(drug_data))#222

    return drug_feature_graph,drug_feature_Transformer

# print(drug_smiles.shape){'1401': 'CCN(CC)CCCCNc1ncc2cc(-c3cc(OC)cc(OC)c3)c(NC(=O)NC(C)(C)C)nc2n1', '2375': 'CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O  ',





    # i=0
    # drug_index=[]
    # # print(data)
    # C=[]
    # D=[]
    # for smile ,cid in zip(data.iloc[:,1],data.iloc[:,0]):
    #     print(i)
    #     i+=1
    #     print(smile)
    #     try:
    #         drug_fea1 = getGraphFea(smile)
    #     except:
    #         continue
    #     print("Drug_fea1")
    #     print(drug_fea1.shape)
    #     #print(np.shape(fea1))
    #     drug_fea2 = getDrugTransformerFea(smile)
    #     print("Drug_fea2")
    #     print(drug_fea2.shape)
    #     C.append(torch.cat((drug_fea1.reshape(1,128),drug_fea2),dim=1))
    #     D.append(cid)
    #     #features.append(fea1.reshape(1,-1)+fea2)
    #
    # return torch.stack(C),D

# if __name__ == '__main__':
#     C,D=getFeatures()
#     C=C.reshape(C.shape[0],C.shape[-1])
#     print(C.shape)
#     print(dict(zip(D,C)))
