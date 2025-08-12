#-*- coding: utf-8 -*-

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch
import argparse
import random
from collectMethodVec import saveAllDataToRam
from Models.defect_detection import DefectDetection
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--hidden', type=int, default=16) # CFG 32 PDG 16
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

jsonVecPath = "out/outPut_cfg/codeJsonVec/xalan-j_2_6_0"
#jsonVecPath = "out/outPut_pdg/codeJsonVec/camel-camel-1.0.0"
bugDataPath = "out/bugData/xalan/xalan-2.6.csv"

ramData = saveAllDataToRam(jsonVecPath,bugDataPath, 32, torch.float32)

def data_pre_deal(jsonVecPath,ramData,outPath, count=args.num_classes):
    model = DefectDetection(args.num_layers, args.hidden, args.nheads, count, args.dropout, args.alpha,True).to(device)
    model.train()
    # 保存特征向量
    all_vec = []
    for key,data in ramData.items():
        features, edge_index, edgesAttr, adjacency, node2node_features, label = data
        data = features, edge_index, edgesAttr, adjacency, node2node_features
        output = model(data)
        output = output.tolist()
        output.extend(label)
        all_vec.append(output)

    # 指定CSV文件的路径
    pathArray = jsonVecPath.split("/")
    # filename = outPath + pathArray[len(pathArray) - 1]+'-first2one'+str(count)+'.csv'
    filename = outPath + pathArray[len(pathArray) - 1]+'-first2one.csv'

    # 打开文件，准备写入
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # 遍历二维数组的每一行
        for row in all_vec:
            writer.writerow(row)
outPath = 'out/featureVec/CFG/'

#data_pre_deal(jsonVecPath, ramData, outPath, model)

