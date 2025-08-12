import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from layers.singleNodeAttention import SingleNodeAttentionLayer
from layers.GAT_with_edge import GraphAttentionLayer
from layers.edge_pool_my import EdgePooling
from layers.bi_lstm import LSTMModel
from layers.global_self_att import GlobalSelfAttentionLayer
from torch_geometric.nn import (global_mean_pool, JumpingKnowledge)
from torch_geometric.nn.glob import GlobalAttention

class DefectDetection(nn.Module):
    '''
        num_layers: 层数（在当前实现中未使用）。
        hidden: 隐藏层维度大小。
        nheads: 注意力头的数量。
        nclass: 输出类别数。
        dropout: Dropout 率。
        alpha: leaky ReLU 激活的负斜率。
        training: 布尔值，指示是否为训练模式。
    '''
    def __init__(self, num_layers, hidden, nheads, nclass, dropout, alpha, training):
        super(DefectDetection, self).__init__()
        self.dropout = dropout
        self.num_classes = nclass
        self.training = training
        self.nheads = nheads
        '''
        自定义的单节点注意力层。
        '''
        self.h = SingleNodeAttentionLayer(hidden, hidden, dropout=dropout, alpha=alpha, concat=True)
        '''
        self.global_self_att = [GlobalSelfAttentionLayer(hidden, 2*hidden, dropout=dropout, alpha=alpha) for _ in range(self.nheads)]
        for i, global_self_att in enumerate(self.global_self_att):
            self.add_module('attention_{}'.format(i), global_self_att)
        '''
        self.GAT = [GraphAttentionLayer(hidden, 2*hidden, concat=True, dropout=dropout, alpha=alpha, training = training) for _ in range(self.nheads)]
        for i, attention in enumerate(self.GAT):
            self.add_module('attention_{}'.format(i), attention)
        '''
        自定义的图注意力层，每个注意力头 一个。
        '''
        # 自定义的边池化层。
        self.edge_pool1 = EdgePooling(nheads*2*hidden)
        # 另一个图注意力层。
        self.out_att = GraphAttentionLayer(nheads*2*hidden, 2*hidden, concat=False, dropout=dropout, alpha=alpha, training = training)

        self.edge_pool2 = EdgePooling(2*hidden)
        '''
        跳跃知识层，用于融合多层信息。
        '''
        self.jump = JumpingKnowledge(mode='cat')
        '''
        全连接层。
        '''
        self.lin1 = Linear((1+nheads*2+2)*hidden, hidden)
        '''
        双向 LSTM 模型，用于序列建模。
        '''
        self.bi_lstm = LSTMModel(hidden*(1+nheads*2+2), 128, 2, self.num_classes)
        self.lin2 = Linear(560, 20)
        '''
        mlp_gate1, mlp_gate2, mlp_gate3：多层感知器门控，用于全局注意力机制。
        '''
        self.mlp_gate1=nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool1=GlobalAttention(gate_nn=self.mlp_gate1)

        self.mlp_gate2=nn.Sequential(nn.Linear(hidden*32,1),nn.Sigmoid())
        self.gpool2=GlobalAttention(gate_nn=self.mlp_gate2)

        self.mlp_gate3=nn.Sequential(nn.Linear(hidden*2,1),nn.Sigmoid())
        self.gpool3=GlobalAttention(gate_nn=self.mlp_gate3)

    def reset_parameters(self):
        self.h.reset_parameters()
        self.GAT.reset_parameters()
        self.out_att.reset_parameters()
        self.edge_pool1.reset_parameters()
        self.edge_pool2.reset_parameters()
        self.lin1.reset_parameters()
        self.bi_lstm.reset_parameters()
        
    def forward(self, data):
        features, edge_index, edgesAttr, adjacency, node2node_features = data

        '''
        初始化单节点注意力层
        '''
        h = self.h(features)
        #print()
        #print("input ",h1.shape)
        '''
        batch1 是一个批处理张量，所有值都为0，表示所有节点都属于同一个图。它在全局池化操作中使用。
        '''
        batch = torch.zeros(len(h), dtype=torch.int64, device='cuda')
        '''
        self.gpool1 是 GlobalAttention 层，使用门控网络 self.mlp_gate1。
        它根据节点特征 h1 和批处理信息 batch1 计算全局图特征并存储在 hs1 列表中。
        '''
        hs = [self.gpool1(h, batch)]
        #h1 = F.dropout(h1, self.dropout, training=self.training)
        '''
        self.GAT 包含多个 GraphAttentionLayer，每个注意力头 一个。每个 att 对节点特征 h1 进行处理，
        并考虑边特征 edgesAttr1、邻接矩阵 adjacency1 以及节点到节点的特征 node2node_features1。
        
        GAT1_out_list1 是一个列表，包含每个 GraphAttentionLayer 的输出。每个元素是一个元组，包含节点特征和边特征。
        
        h1 是所有注意力头输出的节点特征的拼接。
        
        edgesAttr1 是所有注意力头输出的边特征的拼接。
        '''
        GAT1_out_list = [att(h, edgesAttr, adjacency, node2node_features) for att in self.GAT]
        h = torch.cat([item[0] for item in GAT1_out_list], dim=1)
        edgesAttr = torch.cat([item[1] for item in GAT1_out_list], dim=1)
        
        #print("layer1 ",h1.shape, edgesAttr1.shape)
        '''
        self.edge_pool1 是 EdgePooling 层，它对节点特征 h1 和边特征 edgesAttr1 进行池化，输出更新的节点特征、边索引、边特征和批处理信息。
        '''
        h, edge_index, edgesAttr, batch, _ = self.edge_pool1(h, edge_index, edgesAttr, batch=batch)
        '''
        更新 batch1，因为节点数可能发生变化。
        '''
        batch = torch.zeros(len(h), dtype=torch.int64, device='cuda')
        #print("h1",h1.shape)
        '''
        使用新的节点特征 h1 和批处理信息 batch1 计算新的全局图特征，并添加到 hs1 列表中。
        '''
        hs += [self.gpool2(h, batch)]
        #h1 = F.dropout(h1, self.dropout, training=self.training)
        '''
        _get_adj_node2node 方法生成新的邻接矩阵 adjacency1 和节点到节点的特征 node2node_features1，用于后续的图注意力层。
        '''
        adjacency, node2node_features = self._get_adj_node2node(h, edge_index, edgesAttr)
        '''
        self.out_att 是 GraphAttentionLayer 层，它处理节点特征 h1 和边特征 edgesAttr1，
        并使用新的邻接矩阵 adjacency1 和节点到节点的特征 node2node_features1 进行计算。
        '''
        h, edgesAttr = self.out_att(h, edgesAttr, adjacency, node2node_features)

        #print("layer2 ",h1.shape, edgesAttr1.shape)
        '''
        self.edge_pool2 是第二个 EdgePooling 层，类似于第一个边池化层的处理。
        '''
        h, edge_index, edgesAttr, batch, _ = self.edge_pool2(h, edge_index, edgesAttr, batch=batch)
        batch = torch.zeros(len(h), dtype=torch.int64, device='cuda')
        '''
        使用新的节点特征 h1 和批处理信息 batch1 计算新的全局图特征，并添加到 hs1 列表中。
        '''
        hs += [self.gpool3(h, batch)]
        
        #print("global_mean_pool(h1, batch1)",global_mean_pool(h1, batch1).shape)
        '''
        global_vec_tensor_list1 = []
        for global_att in self.global_self_att:
            global_vec_tensor1 = global_att(all_token_tensor1)
            global_vec_tensor_list1.append(global_vec_tensor1)
        global_self_att_cat1 = torch.cat(global_vec_tensor_list1, dim=1)

        hs1 += [global_self_att_cat1]
        '''
        '''
        self.jump 是 JumpingKnowledge 层，它将 hs1 列表中的特征进行融合，生成最终的节点特征 h1。
        '''
        h1 = self.jump(hs)


        '''
        将代码片段1的特征 h1 和代码片段2的特征 h2 堆叠在一起，形成一个新的张量 lstm_data，其中包含两个代码片段的特征。
        堆叠后的张量维度为 (batch_size, 2, feature_dim)，第二个维度为2表示两个代码片段
        '''
        lstm_data = torch.stack([h1],dim=1)
        #return lstm_data[0][0]
        #print("lstm_data",lstm_data.shape)
        '''
        self.bi_lstm 是一个双向LSTM层，它处理输入的 lstm_data，并输出每个代码片段对的分类结果。
        
        使用 softmax 函数将输出转换为概率分布，以便进行分类。
        '''
        # #out = F.softmax(self.bi_lstm(lstm_data),dim=-1)
        # # 使用双向LSTM处理 lstm_data，并输出每个代码片段对的分类结果
        lstm_output = self.bi_lstm(lstm_data)
        # 线性连接的对比实验
        # lstm_output = self.lin2(lstm_data.squeeze(1))

        return lstm_output[0]
        #
        # # 使用 sigmoid 函数将输出映射到 [0, 1] 的范围
        # sigmoid_output = torch.sigmoid(lstm_output)
        #
        # # 根据输出的概率分布，以50%的概率将其映射为0或1
        # # 使用 torch.bernoulli 函数，它会以输入的概率返回0或1
        # out = torch.bernoulli(sigmoid_output).squeeze(dim=-1)
        # return out

    def _get_adj_node2node(self, h, edge_index, edge_attr):
        indices = edge_index.to('cuda')
        values = torch.ones((len(edge_index[0]))).to('cuda')
        adjacency = torch.sparse.FloatTensor(indices, values, torch.Size((len(h),len(h)))).to_dense()

        node2node_features = torch.zeros(len(h)*len(h),edge_attr.size()[1]).to('cuda')
        for i in range(len(edge_index[0])):
            node2node_features[len(h)*edge_index[0][i]+edge_index[1][i]] = edge_attr[i]
        # 以上 邻接矩阵 和 node2node_features 在多头注意力机制中是一样的，只计算一次就好，不一样的是 W 和 a

        return adjacency, node2node_features
        
    def __repr__(self):
        return self.__class__.__name__

