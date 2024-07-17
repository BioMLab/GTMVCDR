import csv

from torch import nn
import torch
from model_helper import Encoder_MultipleLayers, Embeddings
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DataEncoding:
    def __init__(self, vocab_dir):
        self.vocab_dir = vocab_dir
        self.device = torch.device('cuda:0')
    def _drug2emb_encoder(self, smile):
        vocab_path = "./data/drug_codes_chembl_freq_1500.txt".format(self.vocab_dir)#格式化字段将会被 format() 中的参数替换
        sub_csv = pd.read_csv("./data/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))
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
def getDrugTransformerFea(smile):
    vocab_dir = './data'
    obj = DataEncoding(vocab_dir=vocab_dir)
    encoder = transformer().to(device)
    i,enc=obj.encode(smile)
    enc =torch.from_numpy(enc).to(device)
    i = torch.from_numpy(i).to(device)
    enc = [i,enc]

    return encoder(enc)
# Drug_SMILES = "data/Drug/11111.csv"
# reader2 = csv.reader(open(Drug_SMILES, 'r'))  # 使用csv的reader()方法，创建一个reader对象
# next(reader2)
# rows = [item for item in reader2]  # 生成一个列表
# drug_smiles = {item[0]: item[1] for item in rows}
# for k,v in drug_smiles.items():
#     getDrugTransformerFea(v)
"""
[tensor([ 809, 1594,  781,   38,   82,   35,  304,  238,  853,   98,  129,  606,
         688,  187,  168,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0], device='cuda:0', dtype=torch.int32), tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0], device='cuda:0', dtype=torch.int32)]
"""