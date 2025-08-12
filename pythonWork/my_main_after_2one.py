#-*- coding: utf-8 -*-

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
from collectMethodVecAfter2one import saveAllDataToRam
from testModels.defect_detection import DefectDetection
import torch.optim as optim
from tqdm import tqdm, trange
from myModels.GAT_Edgepool_graphEmb import graphEmb
from myModels.GAT_Edgepool_bi_lstm import bi_lstm_detect
from sklearn.metrics import recall_score,precision_score,f1_score
import csv
from testModels.manyVecToOne import FlattenAndDense


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--hidden', type=int, default=16)# CFG 32 PDG 16
parser.add_argument('--num_classes', type=int, default=1120)
parser.add_argument('--nheads', type=int, default=16)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--alpha', type=int, default=0.2)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(fold, epoch, val_loss, test_acc))


def split_dict_data(data_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    keys = list(data_dict.keys())
    random.shuffle(keys)

    total_samples = len(keys)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_keys = keys[:train_size]
    val_keys = keys[train_size:train_size + val_size]
    test_keys = keys[train_size + val_size:]

    train_data = [{key: data_dict[key]} for key in train_keys]
    val_data = [{key: data_dict[key]} for key in val_keys]
    test_data = [{key: data_dict[key]} for key in test_keys]

    return train_data, val_data, test_data


def getCodePairDataList(ramData, pathlist):
    datalist = []
    for line in pathlist:
        # googlejam4_src/8/googlejam8.p934.Main.java
        codedata = ramData[line]
        pairdata = [codedata[0], codedata[1],codedata[2],codedata[3],codedata[4],  codedata[5]]
        datalist.append(pairdata)

    return datalist

def getBatch(line_list, batch_size, batch_index, device):
    start_line = batch_size*batch_index
    end_line = start_line+batch_size
    # dataList = getCodePairDataList(ramData,line_list[start_line:end_line])
    dataList = line_list[start_line:end_line]
    return dataList

jsonVecPath = "outPut_cfg/codeJsonVec/camel-camel-1.0.0"
#jsonVecPath = "out/outPut_pdg/codeJsonVec/camel-camel-1.0.0"
bugDataPath = "out/bugData/camel/camel-1.0.csv"

#ramData = saveAllDataToRam(jsonVecPath,bugDataPath)

def data_pre_deal(jsonVecPath,ramData,outPath):
    model = DefectDetection(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha,True).to(device)
    model.train()
    # 保存特征向量
    all_vec = []
    for key,value in ramData.items():
        print(f"{key}类开始向量处理")
        dataMap = value[0]
        lable = value[1]
        # 每个方法单独处理成一个特征向量
        method_vecs = []
        for data in dataMap:
            output = model(data)
            output = output.tolist()
            method_vecs.append(output)
        if len(method_vecs) < 1:
            continue
        # 一个类所有方法的特征向量处理成一个类的特征向量
        # 创建模型实例
        # 将列表转换为张量
        method_vecs = torch.tensor(method_vecs, dtype=torch.float32)
        # 获取输入的维度
        n = method_vecs.size(0)
        m = method_vecs.size(1)
        vecToOneMdel = FlattenAndDense(n * m, m)
        method_vecs = method_vecs.unsqueeze(0)
        method_vec = vecToOneMdel(method_vecs)
        method_vec = method_vec.tolist()[0]
        method_vec.extend(lable)
        all_vec.append(method_vec)


    # 指定CSV文件的路径
    pathArray = jsonVecPath.split("/")
    filename = outPath + pathArray[len(pathArray) - 1] + '-after2one.csv'

    # 打开文件，准备写入
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # 遍历二维数组的每一行
        for row in all_vec:
            writer.writerow(row)

outPath = 'out/featureVec/CFG/'
#data_pre_deal(jsonVecPath,ramData,outPath)


# 70%为训练集，15%为验证集，15%为测试集
#trainlist,validlist,testlist = split_dict_data(ramData)

# print("trainlist",len(trainlist))
# print("validlist",len(validlist))
# print("testlist",len(testlist))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, size_average=None, reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, size_average=None, reduce=None)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


criterion = FocalLoss().to(device)

def graph_emb(data,epoch):
    # 创建模型对象
    model = graphEmb(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha, False).to(device)

    # 加载保存的模型参数
    saveModel = torch.load('./saveModel/epoch' + str(epoch) + '.pkl')

    # 获取当前模型的状态字典
    model_dict = model.state_dict()

    # 筛选出保存的模型参数中与当前模型匹配的参数
    state_dict = {k: v for k, v in saveModel.items() if k in model_dict.keys()}

    # 更新当前模型的状态字典
    model_dict.update(state_dict)

    # 将更新后的状态字典加载到模型中
    model.load_state_dict(model_dict)

    # 将模型设置为评估模式
    model.eval()

    # 解包输入数据
    features, edge_index, edgesAttr, adjacency, node2node_features, label = data

    # 构建用于模型输入的数据元组
    data = features, edge_index, edgesAttr, adjacency, node2node_features

    # 使用模型生成图嵌入
    h = model(data)

    # 返回图嵌入
    return h

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def bi_lstm_detection(data,epoch):
    model = bi_lstm_detect(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha, False).to(device)
    saveModel = torch.load('./saveModel/epoch'+str(epoch)+'.pkl')
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in saveModel.items() if k in model_dict.keys()}
    #print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    #print("loaded "+ 'epoch'+str(epoch)+'.pkl')
    model.eval()
    h1 = data
    out = model(h1)
    return out

def test(testlist, model_index, ramData, batch_size):
    graphEmbDict = {}
    print("save graphEmbDict...")
    for codeID in tqdm(ramData):
        data = ramData[codeID]
        # 对图数据进行嵌入计算
        graphEmbDict[codeID] = graph_emb(data, model_index).tolist()

    notFound = 0
    testCount = 0
    y_preds = []
    y_trues = []
    batches = split_batch(testlist, batch_size)
    Test_data_batches = trange(len(batches), leave=True, desc="Test")
    for i in Test_data_batches:
        h1_batch = []
        label_batch = []
        for codepair in batches[i]:
            try:
                #graphEmbDict[codepair]
                graphEmbDict[list(codepair)[0]]
                testCount += 1
            except:
                notFound += 1
                continue

            #h1 = torch.as_tensor(graphEmbDict[codepair]).to(device)
            h1 = torch.as_tensor(graphEmbDict[list(codepair)[0]]).to(device)
            #label = int(list(ramData[codepair])[5])
            label = int(list(ramData[list(codepair)[0]])[5])

            h1_batch.append(h1)
            label_batch.append(label)

        h1_batch_t = torch.stack(h1_batch, dim=1).squeeze(0)
        # print("h1_batch",h1_batch.shape)
        data = h1_batch_t
        outputs = bi_lstm_detection(data, model_index)
        _, predicted = torch.max(outputs.data, 1)
        y_preds += predicted.tolist()
        y_trues += label_batch

        r_a = recall_score(y_trues, y_preds, average='macro')
        p_a = precision_score(y_trues, y_preds, average='macro')
        f_a = f1_score(y_trues, y_preds, average='macro')

        Test_data_batches.set_description("Test (p_a=%.4g,r_a=%.4g,f_a=%.4g)" % (p_a, r_a, f_a))
    print("testCount", testCount)
    print("notFound", notFound)
    return p_a, r_a, f_a

def train():
    addNum = 0
    model = DefectDetection(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha,
                               True).to(device)
    # 优化器使用Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    random.shuffle(trainlist)
    epochs = trange(args.epochs, leave=True, desc="Epoch")
    iterations = 0
    for epoch in epochs:
        # print(epoch)
        totalloss = 0.0
        main_index = 0.0
        count = 0
        right = 0
        acc = 0
        for batch_index in tqdm(range(int(len(trainlist) / args.batch_size))):
            batch = getBatch(trainlist, args.batch_size, batch_index, device)
            optimizer.zero_grad()
            batchloss = 0
            recoreds = open("recoreds.txt", 'a')

            for data in batch:
                features, edge_index, edgesAttr, adjacency, node2node_features, label = list(data.values())[0]

                data = features, edge_index, edgesAttr, adjacency, node2node_features
                label = torch.Tensor([1]).to(device) if label == 1 else torch.Tensor([0]).to(device)
                output = model(data)
                batchloss = batchloss + criterion(output,label)
                count += 1
                right += torch.sum(torch.eq(output,label))
            # print("batchloss",batchloss)
            acc = right * 1.0 / count
            print("acc:", acc)
            batchloss.backward(retain_graph=True)
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch)
            loss = totalloss / main_index
            print("loss:",loss)
            epochs.set_description("Epoch (Loss=%g) (Acc = %g)" % (round(loss, 5), acc))
            iterations += 1
            recoreds.write(str(iterations + addNum * 14078) + " " + str(acc.item()) + " " + str(loss) + "\n")
            recoreds.close()
        # if(epoch%10==0 and epoch>0):
        torch.save(model.state_dict(), './saveModel/epoch' + str(epoch + addNum) + '.pkl')
        val_recoreds = open("result/val_recoreds_cfg.txt", 'a')
        p, r, f1 = test(validlist, epoch + addNum, ramData, 15000)
        val_recoreds.write(str(epoch + addNum) + " " + str(p) + " " + str(r) + " " + str(f1) + "\n")
        val_recoreds.close()

        test_recoreds = open("result/test_recoreds.txt", 'a')
        p, r, f1 = test(testlist, epoch + addNum, ramData, 15000)
        test_recoreds.write(str(epoch + addNum) + " " + str(p) + " " + str(r) + " " + str(f1) + "\n")
        test_recoreds.close()




