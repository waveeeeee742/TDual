import torch
import torch.nn as nn


class BaseRGCN(nn.Module):
    """
    通用 RGCN 基类
    职责：
      1. 保存全部超参数（方便子类/层访问）
      2. 调用 build_model() 生成网络层
      3. 提供一个缺省 forward（子类可重写）
    """

    def __init__(self,
                 num_nodes,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=-1,
                 num_basis=-1,
                 num_hidden_layers=1,
                 dropout=0.0,
                 self_loop=False,
                 skip_connect=False,
                 encoder_name='uvrgcn',
                 opn='sub',
                 rel_emb=None,
                 use_cuda=False,
                 analysis=False):
        super(BaseRGCN, self).__init__()

        # ----------- 保存超参数 -----------
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.encoder_name = encoder_name
        self.opn = opn
        self.rel_emb = rel_emb
        self.use_cuda = use_cuda
        self.analysis = analysis

        # 统一设备管理
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # ----------- 构建网络层 -----------
        self.build_model()

        # ----------- 节点原始特征 -----------
        self.features = self.create_features()

    def build_model(self):
        """构建模型层"""
        self.layers = nn.ModuleList()

        # Input to hidden layer
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)

        # Hidden to hidden layers
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

        # Hidden to output layer
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def create_features(self):
        """创建节点特征，默认不使用额外节点特征"""
        return None

    def build_input_layer(self):
        """构建输入层，子类可重写"""
        return None

    def build_hidden_layer(self, idx):
        """构建隐藏层，子类必须实现"""
        raise NotImplementedError("Subclass must implement build_hidden_layer()")

    def build_output_layer(self):
        """构建输出层，子类可重写"""
        return None

    def forward(self, g, *extra_args, **extra_kwargs):
        """
        前向传播
        子类（如 RGCNCell）通常会重写此函数
        这里提供一个最简实现：逐层调用 self.layers

        Args:
            g: DGL graph
            *extra_args: 额外的位置参数
            **extra_kwargs: 额外的关键字参数

        Returns:
            节点隐藏状态
        """
        # 如果有预定义特征，使用它们
        if self.features is not None:
            g.ndata['id'] = self.features

        # 逐层前向传播
        for layer in self.layers:
            # 兼容 layer 需要附加参数的情形
            layer(g, *extra_args, **extra_kwargs)

        # 返回节点隐藏状态
        return g.ndata.pop('h')

    def get_model_info(self):
        """获取模型信息"""
        info = {
            'encoder_name': self.encoder_name,
            'num_nodes': self.num_nodes,
            'h_dim': self.h_dim,
            'out_dim': self.out_dim,
            'num_rels': self.num_rels,
            'num_bases': self.num_bases,
            'num_hidden_layers': self.num_hidden_layers,
            'dropout': self.dropout,
            'self_loop': self.self_loop,
            'skip_connect': self.skip_connect,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return info

    def reset_parameters(self):
        """重置模型参数"""
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def to_device(self, device):
        """将模型移动到指定设备"""
        self.device = device
        return self.to(device)