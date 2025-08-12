#-*- coding: utf-8 -*-

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm
import torch
import os
import json
import pandas as pd


'''
生成邻接矩阵和节点之间的特征。
'''
def get_adj_node2node(h, edge_index, edge_attr):
    indices = edge_index.to('cuda')
    values = torch.ones((len(edge_index[0]))).to('cuda')
    adjacency = torch.sparse.FloatTensor(indices, values, torch.Size((len(h),len(h)))).to_dense()

    node2node_features = torch.zeros(len(h)*len(h),edge_attr.size()[1]).to('cuda')
    for i in range(len(edge_index[0])):
        node2node_features[len(h)*edge_index[0][i]+edge_index[1][i]] = edge_attr[i]

    return adjacency, node2node_features


def saveAllDataToRam(jsonVecPath,bugDataPath, hidden, dtype):
    # 获取bug数标签
    # 读取CSV文件
    df = pd.read_csv(bugDataPath)
    # 以'name'列为key, 'bug'列为value创建字典
    # name_bug = {row['name'].replace('.', '/')+".java": 0 if row['bug'] == 0 else 1 for _, row in df.iterrows()}
    # 以'name'列为key, 20个特征为value[0],'bug'列为value[1]创建字典
    name_bug = {row['name'].replace('.', '/') + ".java": row.iloc[1:21].tolist() + [0 if row['bug'] == 0 else 1] for _, row in df.iterrows()}
    ramData = {}  # save all data to a dict. i.e. {"jsonVecID1":[lines, features, edge_index, edge_attr], "jsonVecID2":[lines, features, edge_index, edge_attr],...}
    count = 0
    for root, dirs, files in os.walk(jsonVecPath):
        for file in tqdm(files):
            codePath = root.replace('\\', '/') + '/' + file[:-len(".json")]
            if "/java/" in codePath:
                codePath = codePath.split("/java/")
                codePath = codePath[1]
            elif "/main/" in codePath:
                codePath = codePath.split("/main/")
                codePath = codePath[1]
            elif "/jEdit32/" in codePath:
                codePath = codePath.split("/jEdit32/")
                codePath = codePath[1]
            elif "/jEdit40/" in codePath:
                codePath = codePath.split("/jEdit40/")
                codePath = codePath[1]
            elif "/jEdit41/" in codePath:
                codePath = codePath.split("/jEdit41/")
                codePath = codePath[1]
            elif "/src/" in codePath:
                codePath = codePath.split("/src/")
                codePath = codePath[1]

            # 根据数据集中标签筛选数据
            if codePath in name_bug:
                #try:
                    jsonPath = root.replace('\\','/') + '/' + file
                    # for codedata
                    data = json.load(open(jsonPath))
                    nodes = []
                    features = []
                    edgeSrc = []
                    edgeTag = []
                    edgesAttr = []
                    max_node_token_num = 0
                    for node in data["jsonNodesVec"]:
                        if len(data["jsonNodesVec"][node]) > max_node_token_num:
                            max_node_token_num = len(data["jsonNodesVec"][node])

                    for i in range(len(data["jsonNodesVec"])):
                        nodes.append(i)
                        node_features = []
                        for list in data["jsonNodesVec"][str(i)]:
                            if list != None:
                                node_features.append(list)
                        if len(node_features) == 0:
                            node_features = [[0 for i in range(hidden)]]
                        if len(node_features) < max_node_token_num:
                            for i in range(max_node_token_num - len(node_features)):
                                node_features.append([0 for i in range(hidden)])
                        features.append(node_features)  # multi vecs offen

                    for edge in data["jsonEdgesVec"]:
                        if data["jsonEdgesVec"][edge][0][0] == 1 and data["jsonEdgesVec"][edge][0][1] == 1 and \
                                data["jsonEdgesVec"][edge][0][3] == 1:
                            edgeSrc.append(int(edge.split("->")[0]))
                            edgeTag.append(int(edge.split("->")[1]))
                            edgesAttr.append([0 for i in range(hidden)])
                        else:
                            edgeSrc.append(int(edge.split("->")[0]))
                            edgeTag.append(int(edge.split("->")[1]))
                            edgesAttr.append(data["jsonEdgesVec"][edge][0])

                    for i in range(len(nodes)):
                        edgeSrc.append(i)
                        edgeTag.append(i)
                        edgesAttr.append([0 for i in range(hidden)])
                    edge_index = [edgeSrc, edgeTag]
                    features = torch.tensor(features, dtype=dtype).to('cuda')
                    edge_index = torch.tensor(edge_index, dtype=torch.long).to('cuda')
                    edgesAttr = torch.tensor(edgesAttr, dtype=dtype).to('cuda')

                    if(len(features) == 0):
                        continue
                    adjacency, node2node_features = get_adj_node2node(features, edge_index, edgesAttr)
                    '''
                    features：节点的特征矩阵
                    edge_index：边的索引矩阵
                    edge_attr：边的特征矩阵
                    adjacency：邻接矩阵
                    node2node_features：节点到节点的特征矩阵
                    '''
                    ramData[codePath] = [features, edge_index, edgesAttr, adjacency, node2node_features,name_bug[codePath]]
                    count += 1

    # 找出两个字典键的不同
    keys1 = set(ramData.keys())
    keys2 = set(name_bug.keys())
    # 在bug_map1中存在但在bug_map2中不存在的键
    diff1 = keys1 - keys2

    # 在bug_map2中存在但在bug_map1中不存在的键
    diff2 = keys2 - keys1

    print(f"Keys in ramData but not in name_bug: {diff1}")
    print(f"Keys in name_bug but not in ramData: {diff2}")
    # 将文件路径保存为DataFrame，并按列存储
    lackClassArray = [[key,name_bug[key][20]] for key in diff2 if key in name_bug]
    df = pd.DataFrame(lackClassArray, columns=['class','label'])

    # 保存为Excel文件
    df.to_excel('E:\pySpace\workData\out\\featureVec\different\\' + jsonVecPath.replace('/', '_') + '.xlsx', index=False)
    return ramData


jsonVecPath = "out/outPut_cfg/codeJsonVec/jEdit32"
bugDataPath = "out/bugData/camel/camel-1.0.csv"

# ramData = saveAllDataToRam(jsonVecPath,bugDataPath, 16, torch.float32)


