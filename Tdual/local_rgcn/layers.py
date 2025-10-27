import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.softmax import edge_softmax
import numpy as np
from typing import Optional, Tuple
import gc


# ================================
# 内存优化装饰器
# ================================
def memory_efficient(func):
    """内存高效装饰器"""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    return wrapper


# ================================
# 基础RGCN层
# ================================
class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    def propagate(self, g):
        raise NotImplementedError

    @memory_efficient
    def forward(self, g, prev_h=[]):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(
                torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias
            )

        self.propagate(g)

        node_repr = g.ndata['h']
        if self.bias is not None:
            node_repr = node_repr + self.bias

        if len(prev_h) != 0 and self.skip_connect:
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                loop_message = skip_weight * (self.activation(loop_message) if self.activation else loop_message)
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        return node_repr


# ================================
# RGCNBasisLayer
# ================================
class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            weight = self.weight.view(self.num_bases, self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat
            )
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index)}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                return {'msg': msg}

        def apply_func(nodes):
            return {'h': nodes.data['h'] * nodes.data['norm']}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), apply_func)


# ================================
# RGCNBlockLayer
# ================================
class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop, skip_connect=skip_connect,
                                             dropout=dropout, layer_norm=layer_norm)
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out
        ))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
            -1, self.submat_in, self.submat_out
        )
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


# ================================
# 增强的UnionRGCNLayer（优化版）
# ================================
class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False,
                 rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect

        # 基础权重
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        # 添加层归一化（提升训练稳定性）
        self.layer_norm = nn.LayerNorm(out_feat)

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    @memory_efficient
    def forward(self, g, pm_pd, emb_rel):
        self.rel_emb = emb_rel

        # 处理自环
        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).to(g.device),
                (g.in_degrees(range(g.number_of_nodes())) > 0)
            )
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]

        # 使用当前节点特征作为prev_h
        prev_h = g.ndata['h'] if 'h' in g.ndata else []

        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(
                torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias
            )

        # 消息传递
        self.propagate(g)
        node_repr = g.ndata['h']

        # 应用层归一化
        node_repr = self.layer_norm(node_repr)

        # 应用跳跃连接
        if len(prev_h) != 0 and self.skip_connect:
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)

        if self.dropout is not None:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        # 获取关系嵌入
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)

        # 组合节点和关系特征
        msg = node + relation

        # 通过权重矩阵变换
        msg = torch.mm(msg, self.weight_neighbor)

        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


# ================================
# UnionRGCNLayer2（历史建模层）
# ================================
class UnionRGCNLayer2(UnionRGCNLayer):
    """专门用于历史图建模的RGCN层"""

    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False,
                 rel_emb=None):
        super(UnionRGCNLayer2, self).__init__(
            in_feat, out_feat, num_rels, num_bases, bias,
            activation, self_loop, dropout, skip_connect, rel_emb
        )

        # 添加历史特征增强
        self.history_gate = nn.Sequential(
            nn.Linear(out_feat * 2, out_feat),
            nn.ReLU(),
            nn.Linear(out_feat, out_feat),
            nn.Sigmoid()
        )


# ================================
# 增强的RGAT层（保持接口兼容性）
# ================================
class RGAT(nn.Module):
    """增强的RGAT层 - 保持原始接口"""

    def __init__(self, in_feat, out_feat, bias=None,
                 activation=None, self_loop=False, dropout=0.0, layer_norm=False):
        super(RGAT, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        # 自环权重
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        # Dropout层
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # 层归一化
        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

        # 注意力参数
        self.num_head = 5
        self.out_feat = out_feat
        self.in_feat = in_feat
        self.head_dim = in_feat // self.num_head

        # 权重矩阵
        self.W_t = nn.Parameter(torch.Tensor(self.out_feat, self.out_feat))
        nn.init.xavier_uniform_(self.W_t, gain=nn.init.calculate_gain('relu'))

        self.W_r = nn.Parameter(torch.Tensor(self.out_feat, self.out_feat))
        nn.init.xavier_uniform_(self.W_r, gain=nn.init.calculate_gain('relu'))

        self.w_triplet = nn.Parameter(torch.Tensor(in_feat * 3, self.out_feat))
        nn.init.xavier_uniform_(self.w_triplet, gain=nn.init.calculate_gain('relu'))

        self.w_quad = nn.Parameter(torch.Tensor(in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.w_quad, gain=nn.init.calculate_gain('relu'))

        # 增强：添加多头注意力优化
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=out_feat,
            num_heads=4,
            dropout=dropout if dropout > 0 else 0.0,
            batch_first=True
        )

    @memory_efficient
    def forward(self, g, node, rel):
        g.ndata['h'] = node
        g.edata['rel_emb'] = rel[g.edata['type']]

        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).to(g.device),
                (g.in_degrees(range(g.number_of_nodes())) > 0)
            )
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)
        node_repr = g.ndata['h']

        if self.self_loop:
            node_repr = node_repr + loop_message

        if self.layer_norm:
            node_repr = self.normalization_layer(node_repr)

        if self.activation:
            node_repr = self.activation(node_repr)

        if self.dropout is not None:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr
        return node_repr

    def propagate(self, g):
        g.apply_edges(func=self.quads_msg_func)
        g.edata['a_triplet'] = F.leaky_relu(g.edata['a_triplet'])
        g.edata['att_triplet'] = edge_softmax(g, g.edata['a_triplet'])
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def quads_msg_func(self, edges):
        triplet = torch.cat([edges.src['h'], edges.data['rel_emb'], edges.dst['h']], dim=1)
        triplet = torch.mm(triplet, self.w_triplet)

        if 'fre' in edges.data:
            sro_fre = edges.data['fre'].unsqueeze(1)
        else:
            sro_fre = torch.zeros(triplet.shape[0], 1).to(triplet.device)

        a_triplet = torch.mm(triplet + sro_fre, self.w_quad)
        return {'triplet': triplet, 'a_triplet': a_triplet}

    def msg_func(self, edges):
        return {'msg': (edges.data['att_triplet'] * edges.data['triplet'])}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


# ================================
# UnionRGATLayer（带注意力的Union层）
# ================================
class UnionRGATLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False,
                 rel_emb=None):
        super(UnionRGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect

        # 基础权重
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # 注意力机制
        self.attn_fc = nn.Linear(3 * self.out_feat, self.out_feat, bias=False)
        self.attn_fc2 = nn.Linear(self.out_feat, 1, bias=False)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_fc2.weight, gain=nn.init.calculate_gain('relu'))

    def edge_attention(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node_h = edges.src['h'].view(-1, self.out_feat)
        node_t = edges.dst['h'].view(-1, self.out_feat)

        z2 = torch.cat([node_h, node_t, relation], dim=1)
        a = self.attn_fc(z2)
        a = self.attn_fc2(a)
        return {'e_att': F.leaky_relu(a)}

    def propagate(self, g):
        g.update_all(self.msg_func, self.reduce_func)

    def msg_func(self, edges):
        return {'e_h': edges.src['h'], 'e_att': edges.data['e_att']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_att'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['e_h'], dim=1)
        return {'h': h}

    @memory_efficient
    def forward(self, g, pm_pd, emb_rel):
        self.rel_emb = emb_rel

        prev_h = g.ndata['h'] if 'h' in g.ndata else []

        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)

        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(
                torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias
            )

        g.apply_edges(self.edge_attention)

        self.propagate(g)
        node_repr = g.ndata['h']

        if len(prev_h) != 0 and self.skip_connect:
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)

        if self.dropout is not None:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr
        return node_repr


# ================================
# CompGCNLayer（组合操作的GCN）
# ================================
class CompGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, comp, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False,
                 rel_emb=None):
        super(CompGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.comp = comp

        # 基础权重
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # 组合操作特定的参数
        if self.comp == "mult":
            self.mult_gate = nn.Sequential(
                nn.Linear(out_feat, out_feat),
                nn.Sigmoid()
            )

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    @memory_efficient
    def forward(self, g, pm_pd, emb_rel):
        self.rel_emb = emb_rel

        prev_h = g.ndata['h'] if 'h' in g.ndata else []

        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)

        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(
                torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias
            )

        self.propagate(g)
        node_repr = g.ndata['h']

        if len(prev_h) != 0 and self.skip_connect:
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)

        if self.dropout is not None:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)

        # 根据组合操作类型处理消息
        if self.comp == "sub":
            msg = node - relation
        elif self.comp == "mult":
            gate = self.mult_gate(relation)
            msg = node * gate
        elif self.comp == "add":
            msg = node + relation
        else:
            msg = node + relation  # 默认使用加法

        # 通过邻居权重矩阵变换
        msg = torch.mm(msg, self.weight_neighbor)

        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}