
from datasetload import *
import argparse
from utils import *
from GTMVCDR import *

def getdrugFeatures():
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
    return drug_smiles
drug_smiles = getdrugFeatures()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Drug_response_pre')#建立解析对象，描述description：大多数对 ArgumentParser 构造方法的调用都会使用 description= 关键字参数。这个参数简要描述这个程度做什么以及怎么做。在帮助消息中，这个描述会显示在命令行用法字符串和各种参数的帮助消息之间。
parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='')#dest:解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线
parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=256, help='')
parser.add_argument('--output_channels', dest='output_channels', type=int, default=256, help='')
args = parser.parse_args()#ArgumentParser 通过 parse_args() 方法解析参数
start_time = time.time()

#------data files
Drug_info_file='data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
IC50_threds_file='data/Drug/drug_threshold.txt'
Drug_feature_file='data/Drug/11111.csv'
Cell_line_info_file='data/Celline/Cell_lines_annotations.txt'
Cancer_response_exp_file='data/Celline/GDSC_IC50.csv'

#-------bio-feature extraction
cell_graph, data_new, nb_celllines,nb_drugs = dataload(Drug_info_file, IC50_threds_file, Cell_line_info_file,
             Cancer_response_exp_file)
cell_tensor, allpairs, allpairs_only_sim,cell_new_tensor,drug_new_tensor,drug_set,cellline_set,label_pos, train_mask, test_mask =process(cell_graph, data_new, nb_celllines, nb_drugs)

#-------split train and test sets

model = GraphCDR(hidden_channels=args.hidden_channels,
                 feat=NodeRepresentation(args.output_channels),index_cell=nb_celllines,index_drug = nb_drugs)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
myloss = nn.BCELoss().to(device)


def train():
    model.train()
    loss_temp=0
    for batch, (drug,cell) in enumerate(zip(drug_set,cellline_set)):
        optimizer.zero_grad()#清空过往梯度；
        pos_adj,_=model(cell_tensor.cuda(), cell_new_tensor.cuda(),drug_new_tensor.cuda(),drug_smiles)
        pos_loss = myloss(pos_adj[train_mask],label_pos[train_mask].cuda())

        pos_loss .backward()#反向传播，计算当前梯度；
        optimizer.step()#根据梯度更新网络参数
        loss_temp += pos_loss.item()
    print('train loss: ', str(round(loss_temp, 4)))


def test(epoch):
    model.eval()
    # model.to(device)
    with torch.no_grad():
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            # print("------------")
            # print(type((drug)))
            # print("===============")
            # print(type((cell)))
            pre_adj,tezheng=model(cell_tensor.cuda(), cell_new_tensor.cuda(),drug_new_tensor.cuda(),drug_smiles)
            loss_temp = myloss(pre_adj[test_mask],label_pos[test_mask].cuda())
        yp=pre_adj[test_mask].detach().cpu().numpy()
        ytest=label_pos[test_mask].detach().cpu().numpy()
        tezheng_test = tezheng[test_mask].detach().cpu().numpy()
        print(tezheng_test.shape)
        df_tesnor = pd.DataFrame(tezheng_test)
        df_tesnor['y_true'] = ytest
        df_tesnor['y_pred'] = yp
        # df_tesnor.to_csv(f'epoch_tensor/weighted_sums_epoch_{epoch}.csv', index=False)
        # print("保存成功tensor")
        AUC, AUPR, F1, ACC,recall, specificity, precision,df =metrics_graph(ytest,yp)
        print('test loss: ', str(round(loss_temp.item(), 4)))
        print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
              '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4))+'  test recall: ' +  str(round(final_recall, 4))
              +'  test  specificity: ' +str(round(specificity, 4)) + '  test final_precision: ' +str(round(precision, 4))
              )

    return AUC, AUPR, F1, ACC,recall, specificity, precision,df

final_AUC = 0;final_AUPR = 0;final_F1 = 0;final_ACC = 0;final_recall = 0 ;final_specificity = 0 ;final_precision = 0;
AUC_sum = 0;AUPR_sum = 0; F1_sum =0;ACC_sum = 0;recall_sum =0;specificity_sum = 0;precision_sum =0;
Ca = args.epoch
for epoch in range(args.epoch):
    print('\nepoch: ' + str(epoch))
    train()
    AUC, AUPR, F1, ACC,recall, specificity, precision,df = test(epoch)
    AUC_sum += AUC
    AUPR_sum += AUPR
    F1_sum += F1
    ACC_sum +=ACC
    recall_sum += recall
    specificity_sum += specificity
    precision_sum += precision

    if (AUC > final_AUC):
        best_epch = epoch
        final_AUC = AUC;final_AUPR = AUPR;final_F1 = F1;final_ACC = ACC;final_recall =recall;final_specificity =specificity;final_precision = precision;final_df = df
elapsed = time.time() - start_time
AUC_avaerage = AUC_sum/Ca
AUPR_avaerage = AUPR_sum/Ca
ACC_average = ACC_sum/Ca
F1_average = F1_sum/Ca
recall_average = recall_sum/Ca
specificity_avaerage =specificity_sum/Ca
precision_avaerage = precision_sum/Ca
final_df.to_csv(r'实验结果\final_df.csv')
print("success save np")

with open(r'实验结果\result.txt', 'w') as f:
    print('---------------------------------------', file=f)
    print('Final_AUC: ' + str(round(final_AUC, 4)) + '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
          '  Final_F1: ' + str(round(final_F1, 4)) + '  Final_ACC: ' + str(
        round(final_ACC, 4)) + ' Final_recall: ' + str(round(final_recall, 4))
          + ' Final_specificity: ' + str(round(final_specificity, 4)) + ' Final_precision: ' + str(
        round(final_precision, 4))
          , file=f)
    print("Best epoch is ", best_epch, file=f)
    print('---------------------------------------', file=f)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print('---------------------------------------')
print('Elapsed time: ', round(elapsed, 4))
print('AUC_avaerage: ' + str(round(AUC_avaerage, 4)) + '  AUPR_avaerage: ' + str(round(AUPR_avaerage, 4)) +
      '  F1_average: ' + str(round(F1_average, 4)) + '  ACC_average: ' + str(round(ACC_average, 4)) +'  recall_average:  ' +  str(round(recall_average, 4))
+'  specificity_avaerage:  ' +str(round(specificity_avaerage, 4)) + '  precision_avaerage:  ' +str(round(precision_avaerage, 4))
      )
print('---------------------------------------')

