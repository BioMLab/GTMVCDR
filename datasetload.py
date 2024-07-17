import csv
import pandas as pd
from utils import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPS = 1e-15

def dataload(Drug_info_file, IC50_threds_file, Cell_line_info_file,
             Cancer_response_exp_file):
    #-----drug_dataload

    reader = csv.reader(open(Drug_info_file,'r'))#使用csv的reader()方法，创建一个reader对象
    rows = [item for item in reader]#生成一个列表
    drug_pubchem_id_set = []
    for line in pd.read_csv('data/Drug/11111.csv').iloc[0:, 0]:
        drug_pubchem_id_set.append(str(line))
    drug_pubchem_id_set.sort()
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}
    drug2thred={}
    for line in open(IC50_threds_file).readlines()[1:]:
        drug2thred[str(line.split('\t')[0])]=float(line.strip().split('\t')[1])

    #-----cell line_dataload

    exp = pd.read_csv(('./data/TCGA_cell/EXP_norm.csv'), index_col=0)
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:#Cell_line_info_file='../data/Celline/Cell_lines_annotations.txt'
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label#定义了一个字典，key为cellline_id，values为TCGA_label
    cell_name2feature_dict = np.load('./data/TCGA_cell/cell_feature_all_std222.npy',
                                     allow_pickle=True).item()
    cell_index = list(exp.index)
    cell_num = list(range(568))
    cell_id_dict = {item: price for item, price in zip(cell_num, cell_index)}
    cell_idx2feature_dict = {u: cell_name2feature_dict[v] for u, v in cell_id_dict.items()}
    cell_graph = [u for _, u in cell_idx2feature_dict.items()]


    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])#Cancer_response_exp_file='../data/Celline/GDSC_IC50.csv'
    # print(experiment_data)
    #-----drug_cell line_pairs dataload
    drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    data_idx = []
    use_thred=True
    genes_path = './data/TCGA_cell'
    save_path = os.path.join(genes_path, 'data_new_npy.npy')
    if not os.path.exists(save_path):
        for each_drug in experiment_data_filtered.index:
            # print('----drug---')
            # print(each_drug)
            for each_cellline in experiment_data_filtered.columns:
                # print('----cellline---')
                # print(each_cellline)
                pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]

                if pubchem_id in drug_pubchem_id_set\
                        and each_cellline in exp.index:

                    if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():

                        ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                        if use_thred:

                            if pubchem_id in drug2thred.keys():
                                binary_IC50 = 1 if ln_IC50 < drug2thred[pubchem_id] else -1
                                data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline]))
                        else:
                            binary_IC50 = 1 if ln_IC50 < -2 else -1
                        data_idx.append((each_cellline, pubchem_id, binary_IC50, cellline2cancertype[each_cellline]))

        #----eliminate ambiguity responses
        data_sort=sorted(data_idx, key=(lambda x: [x[0], x[1], x[2]]), reverse=True)
        data_tmp=[]
        data_new=[]
        data_idx1 = [[i[0],i[1]] for i in data_sort]
        for i,k in zip(data_idx1,data_sort):
            if i not in data_tmp:
                data_tmp.append(i)
                data_new.append(k)

        nb_celllines = len(set([item[0] for item in data_new]))
        nb_drugs = len(set([item[1] for item in data_new]))
        print('All %d pairs across %d cell lines and %d drugs.'%(len(data_new),nb_celllines,nb_drugs))
        data_new_npy = np.array(data_new)

        np.save(
            os.path.join('./data/TCGA_cell', 'data_new_npy.npy'),
            data_new_npy) # 保存为.npy格式
        # # 读取
        # a = np.load('data_new_npy.npy')
        # a = a.tolist()

    else:
        data_new_npy = np.load(save_path)
        # for i in range(len(data_new_npy)):
        # print(data_new_npy[0][0].dtype)<U10
        # print(data_new_npy[0][1].dtype)<U7
        # print(data_new_npy[0][2].dtype)<U1

        data_new = data_new_npy.tolist()


        nb_celllines = len(set([item[0] for item in data_new]))
        nb_drugs = len(set([item[1] for item in data_new]))
        print("# 已经将data_new保存为.npy格式")
        print('All %d pairs across %d cell lines and %d drugs.'%(len(data_new),nb_celllines,nb_drugs))
        #['ACH-000002', '9810884', '1', 'LAML']
    return cell_graph, data_new, nb_celllines,nb_drugs

# Drug_info_file='data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
# IC50_threds_file='data/Drug/drug_threshold.txt'
# Drug_feature_file='data/Drug/11111.csv'
# Cell_line_info_file='data/Celline/Cell_lines_annotations.txt'
# Cancer_response_exp_file='data/Celline/GDSC_IC50.csv'
#
#
# cell_graph, data_new, nb_celllines,nb_drugs = dataload(Drug_info_file, IC50_threds_file, Cell_line_info_file,
#              Cancer_response_exp_file)