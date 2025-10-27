import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

# Assuming these import paths are correct
from local_rgcn.layers import (UnionRGCNLayer, RGCNBlockLayer, RGAT,
                               UnionRGCNLayer2, UnionRGATLayer, CompGCNLayer)
from local_rgcn.model import BaseRGCN
from local_rgcn.decoder import ConvTransE, ConvTransR


# ================================
# Helper Classes
# ================================
class MLPLinear(nn.Module):
    """Multi-layer perceptron linear layer"""

    def __init__(self, in_dim: int, out_dim: int):
        super(MLPLinear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))
        return x


class RGCNCell(BaseRGCN):
    """RGCN Cell"""

    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0

        skip_connect = self.skip_connect and (idx != 0) if self.skip_connect else False

        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, self_loop=self.self_loop, dropout=self.dropout,
                                  skip_connect=skip_connect, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, self_loop=self.self_loop, dropout=self.dropout,
                                  skip_connect=skip_connect, rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                                activation=act, self_loop=self.self_loop, dropout=self.dropout,
                                skip_connect=skip_connect, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError(f"Encoder {self.encoder_name} not implemented")

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name in ["uvrgcn", "kbat", "compgcn"]:
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]

            for i, layer in enumerate(self.layers):
                layer(g, [], init_rel_emb[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                g.ndata['id'] = self.features

            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]

            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    """RGCN Cell 2 - for history modeling"""

    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0

        skip_connect = self.skip_connect and (idx != 0) if self.skip_connect else False

        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer2(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                   activation=act, dropout=self.dropout, self_loop=self.self_loop,
                                   skip_connect=skip_connect, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError(f"Encoder {self.encoder_name} not implemented for RGCNCell2")

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]

            for i, layer in enumerate(self.layers):
                layer(g, [], init_rel_emb[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                g.ndata['id'] = self.features

            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]

            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


# ================================
# Simplified Temporal Components
# ================================
class SimplifiedTemporalEncoder(nn.Module):
    """Simplified but effective temporal encoder using learnable Fourier features"""

    def __init__(self, h_dim, num_frequencies=16, max_period=100.0, dropout=0.1):
        super().__init__()
        self.h_dim = h_dim
        self.num_frequencies = num_frequencies

        # Learnable frequency parameters - fewer parameters
        self.register_buffer('base_frequencies',
                             torch.logspace(-2, math.log10(1.0 / max_period), num_frequencies))
        self.freq_scale = nn.Parameter(torch.ones(num_frequencies))
        self.freq_shift = nn.Parameter(torch.zeros(num_frequencies))

        # Simple linear projection
        self.projection = nn.Sequential(
            nn.Linear(num_frequencies * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )

        # Temporal decay factor for importance weighting
        self.temporal_decay = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.freq_scale, mean=1.0, std=0.1)
        nn.init.normal_(self.freq_shift, mean=0.0, std=0.1)
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, t, return_decay_weight=False):
        """
        Args:
            t: Time index (scalar or tensor)
            return_decay_weight: If True, also return temporal decay weight
        Returns:
            Temporal encoding of shape (h_dim,) or (batch_size, h_dim)
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32, device=self.base_frequencies.device)

        t_expanded = t.unsqueeze(-1) if t.dim() == 0 else t.unsqueeze(-1)

        # Scaled frequencies
        frequencies = self.base_frequencies * self.freq_scale

        # Compute Fourier features with phase shift
        angles = 2 * math.pi * frequencies * t_expanded + self.freq_shift
        fourier_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        # Project to hidden dimension
        temporal_emb = self.projection(fourier_features)

        if return_decay_weight:
            # Compute temporal decay weight for this timestep
            decay_weight = torch.exp(-self.temporal_decay * t)
            return temporal_emb, decay_weight

        return temporal_emb


class TemporalGate(nn.Module):
    """Simplified temporal gating mechanism"""

    def __init__(self, h_dim, dropout=0.1):
        super().__init__()
        self.h_dim = h_dim

        # Single gate instead of multiple gates
        self.gate = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

        # Residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.7))

        self._init_weights()

    def _init_weights(self):
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, current_emb, temporal_emb):
        """Apply temporal gating to embeddings"""
        combined = torch.cat([current_emb, temporal_emb], dim=-1)
        gate_value = self.gate(combined)

        # Gated combination with residual
        output = gate_value * temporal_emb + (1 - gate_value) * current_emb
        output = self.residual_weight * current_emb + (1 - self.residual_weight) * output

        return output


# ================================
# Label Smoothing Loss
# ================================
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""

    def __init__(self, smoothing=0.03):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ================================
# Simplified Edge Feature Extractor
# ================================
class EdgeFeatureExtractor(nn.Module):
    """Simplified edge feature extractor"""

    def __init__(self, h_dim, num_rels, dropout=0.1):
        super().__init__()
        self.h_dim = h_dim

        self.edge_type_embed = nn.Embedding(num_rels * 2, h_dim // 4)

        self.edge_transform = nn.Sequential(
            nn.Linear(h_dim * 2 + h_dim // 4, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.edge_type_embed.weight)
        for module in self.edge_transform:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, src_emb, dst_emb, edge_types):
        """Extract edge features"""
        edge_type_emb = self.edge_type_embed(edge_types)
        edge_features = torch.cat([src_emb, dst_emb, edge_type_emb], dim=1)
        return self.edge_transform(edge_features)


# ================================
# Simplified Edge Sampler
# ================================
class EdgeSampler:
    """Simplified edge sampler with fixed sampling ratio"""

    def __init__(self, num_rels, sampling_ratio=0.3):
        self.num_rels = num_rels
        self.sampling_ratio = sampling_ratio

    def compute_edge_importance(self, graph, src, dst, edge_types=None):
        """Compute edge importance scores"""
        num_edges = len(src)
        device = src.device

        if num_edges == 0:
            return torch.ones(0, device=device)

        # Degree-based importance
        in_degrees = graph.in_degrees().float()
        out_degrees = graph.out_degrees().float()

        src_importance = torch.log1p(in_degrees[src] + out_degrees[src])
        dst_importance = torch.log1p(in_degrees[dst] + out_degrees[dst])
        degree_importance = (src_importance + dst_importance) / 2.0

        # Relation rarity
        rel_importance = torch.ones(num_edges, device=device)
        if edge_types is not None:
            unique_rels, rel_counts = torch.unique(edge_types, return_counts=True)
            rel_weights = 1.0 / torch.log1p(rel_counts.float())
            rel_weights = rel_weights / (rel_weights.sum() + 1e-8) * len(rel_weights)

            for rel, weight in zip(unique_rels, rel_weights):
                rel_mask = (edge_types == rel)
                rel_importance[rel_mask] = weight

        # Combined importance
        importance_scores = degree_importance * rel_importance
        importance_scores = importance_scores / (importance_scores.sum() + 1e-8)

        return importance_scores

    def sample_edges(self, graph):
        """Sample edges from graph"""
        src, dst = graph.edges()
        num_edges = len(src)

        if num_edges == 0:
            return src, dst, None, None

        num_sampled = max(1, int(num_edges * self.sampling_ratio))

        # Return all edges if sampling ratio is high
        if num_sampled >= num_edges * 0.9:
            edge_types = graph.edata.get('type', torch.zeros(num_edges, dtype=torch.long, device=src.device))
            return src, dst, edge_types, torch.arange(num_edges, device=src.device)

        # Compute importance and sample
        importance_scores = self.compute_edge_importance(
            graph, src, dst, graph.edata.get('type', None)
        )

        indices = torch.multinomial(importance_scores, num_sampled, replacement=False)

        if 'type' in graph.edata:
            edge_types = graph.edata['type'][indices]
        else:
            edge_types = torch.zeros(num_sampled, dtype=torch.long, device=src.device)

        return src[indices], dst[indices], edge_types, indices


# ================================
# Simplified Dual Stream Processor
# ================================
class DualStreamProcessor(nn.Module):
    """Simplified dual stream processor for node and edge features"""

    def __init__(self, h_dim, num_ents, num_rels, device):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.device = device

        self.edge_feature_extractor = EdgeFeatureExtractor(h_dim, num_rels, dropout=0.1)

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.Sigmoid()
        )

        # Fixed residual weights
        self.register_buffer('node_weight', torch.tensor(0.2))
        self.register_buffer('edge_weight', torch.tensor(0.8))

        self._init_weights()

    def _init_weights(self):
        for module in self.fusion_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, node_emb, edge_emb):
        """Fuse node and edge embeddings"""
        # Gate-controlled fusion
        combined = torch.cat([node_emb, edge_emb], dim=1)
        gate = self.fusion_gate(combined)

        fused = gate * node_emb + (1 - gate) * edge_emb

        # Residual connection with fixed weights
        output = self.node_weight * node_emb + self.edge_weight * fused

        return output

    def create_edge_enhanced_nodes(self, edge_features, src, dst, num_nodes):
        """Aggregate edge features to nodes"""
        device = edge_features.device

        edge_enhanced_nodes = torch.zeros(num_nodes, self.h_dim, device=device)
        edge_enhanced_nodes.index_add_(0, dst, edge_features)

        edge_count = torch.zeros(num_nodes, device=device)
        ones = torch.ones(len(dst), device=device)
        edge_count.index_add_(0, dst, ones)
        edge_count = edge_count.clamp(min=1).unsqueeze(1)

        edge_enhanced_nodes = edge_enhanced_nodes / edge_count

        return edge_enhanced_nodes


# ================================
# Enhanced GRU Cell
# ================================
class EnhancedGRUCell(nn.Module):
    """GRU cell with layer normalization"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.residual_weight = 0.3

    def forward(self, input, hidden):
        output = self.gru(input, hidden)
        output = self.layer_norm(output)
        output = output + self.residual_weight * hidden
        return output


# ================================
# Main Model Class - Optimized RecurrentRGCN
# ================================
class RecurrentRGCN(nn.Module):
    """Optimized RecurrentRGCN with Simplified Enhanced Temporal Embeddings"""

    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn,
                 sequence_len, num_bases=-1, num_basis=-1, num_hidden_layers=1, dropout=0, self_loop=False,
                 skip_connect=False, layer_norm=False, input_dropout=0, hidden_dropout=0, feat_dropout=0,
                 aggregation='cat', weight=1, pre_weight=0.7, discount=0, angle=0, use_static=False,
                 pre_type='short', use_cl=False, temperature=0.007, entity_prediction=False,
                 relation_prediction=False, use_cuda=False, gpu=0, analysis=False,
                 # Backward compatibility parameters
                 use_node_centric_dual_stream=True, edge_sampling_ratio=0.3, label_smoothing=0.03,
                 dropedge_rate=0.08, use_adaptive_loss=True,
                 # Simplified enhanced temporal parameters
                 use_enhanced_temporal=False, num_fourier_frequencies=16,  # Reduced from 32
                 use_temporal_attention=False,  # Disabled by default
                 temporal_fusion_method='gate'):  # Simpler default
        super(RecurrentRGCN, self).__init__()

        print("=" * 80)
        print("Initializing Optimized RecurrentRGCN with Simplified Enhanced Temporal Embeddings")
        print("=" * 80)

        # Basic parameters
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.weight = weight
        self.static_alpha = 1e-5
        self.pre_weight = pre_weight
        self.discount = discount
        self.use_static = use_static
        self.pre_type = pre_type
        self.use_cl = use_cl
        self.temp = temperature
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None

        # Enhanced temporal parameters
        self.use_enhanced_temporal = use_enhanced_temporal
        self.use_temporal_attention = use_temporal_attention  # Simplified - not used by default
        self.temporal_fusion_method = temporal_fusion_method

        print(f"Enhanced Temporal: {use_enhanced_temporal}")
        print(f"Temporal Fusion Method: {temporal_fusion_method}")

        # Device management
        self.device = torch.device(f'cuda:{gpu}' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.gpu = gpu

        # Dual stream configuration
        self.use_dual_stream = use_node_centric_dual_stream
        self.edge_sampling_ratio = edge_sampling_ratio
        self.label_smoothing = label_smoothing
        self.dropedge_rate = dropedge_rate
        self.use_adaptive_loss = use_adaptive_loss

        # Training state tracking
        self.training_step = 0
        self.last_loss = None
        self.warmup_steps = 200

        # Temporal consistency tracking
        self.temporal_consistency_weight = 0.1  # Weight for temporal consistency loss
        self.prev_temporal_embs = []  # Store previous temporal embeddings

        # Loss weights
        self.loss_weights = {
            'entity': 1.0,
            'relation': 0.1,
            'static': 1.0,
            'contrastive': 0.5,
            'temporal_reg': 0.05  # New regularization weight
        }

        # Linear layers
        self.w1 = nn.Linear(self.h_dim * 2, self.h_dim)
        self.w2 = nn.Linear(self.h_dim, self.h_dim)
        self.w4 = nn.Linear(self.h_dim * 2, self.h_dim)
        self.w5 = nn.Linear(self.h_dim, self.h_dim)
        self.w_cl = nn.Linear(self.h_dim * 2, self.h_dim)

        # Simplified Enhanced Temporal Components
        if self.use_enhanced_temporal:
            # Simplified Fourier temporal encoder
            self.temporal_encoder = SimplifiedTemporalEncoder(
                h_dim, num_frequencies=num_fourier_frequencies,
                max_period=sequence_len * 2, dropout=dropout
            )

            # Simplified temporal gating
            self.temporal_gate = TemporalGate(h_dim, dropout=dropout)

            # Direct temporal integration (no complex fusion)
            self.temporal_projection = nn.Linear(h_dim, h_dim)
            nn.init.xavier_uniform_(self.temporal_projection.weight, gain=0.5)
            nn.init.zeros_(self.temporal_projection.bias)

            print("Initialized Simplified Enhanced Temporal Components")
        else:
            # Legacy temporal parameters
            self.weight_t2 = nn.Parameter(torch.randn(1, h_dim))
            self.bias_t2 = nn.Parameter(torch.randn(1, h_dim))
            print("Using Legacy Temporal Embeddings")

        # Embeddings
        self.emb_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim))
        self.dynamic_emb = nn.Parameter(torch.Tensor(num_ents, h_dim))
        self._init_embeddings()

        if self.use_static:
            self.words_emb = nn.Parameter(torch.Tensor(self.num_words, h_dim))
            nn.init.xavier_uniform_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels * 2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False,
                                                    skip_connect=False)
            self.static_loss = nn.MSELoss()

        # Loss functions
        self.loss_e = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.loss_r = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

        # RGCN layers
        self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis,
                             num_hidden_layers, dropout, self_loop, skip_connect,
                             encoder_name, self.opn, self.emb_rel, use_cuda, analysis)

        self.his_rgcn_layer = RGCNCell2(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis,
                                        num_hidden_layers, dropout, self_loop, skip_connect,
                                        encoder_name, self.opn, self.emb_rel, use_cuda, analysis)

        # Dual stream components
        if self.use_dual_stream:
            print("Using Dual-Stream Architecture")
            self.dual_stream_processor = DualStreamProcessor(h_dim, num_ents, num_rels, self.device)
            self.edge_sampler = EdgeSampler(num_rels, edge_sampling_ratio)

            self.node_stream_rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis,
                                             num_hidden_layers, dropout, self_loop, skip_connect,
                                             encoder_name, self.opn, self.emb_rel, use_cuda, analysis)

        self.rgat_layer = RGAT(self.h_dim, self.h_dim, activation=F.rrelu, dropout=dropout, self_loop=True)
        self.projection_model = MLPLinear(self.h_dim, self.h_dim)

        # Gate parameters
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.time_gate_bias)

        # GRU cells
        self.entity_cell = EnhancedGRUCell(self.h_dim, self.h_dim)
        self.relation_cell = EnhancedGRUCell(self.h_dim, self.h_dim)

        # Decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError(f"Decoder {decoder_name} not implemented")

        # Only keep necessary legacy temporal parameters
        if not self.use_enhanced_temporal:
            self.alpha = 0.5
            self.pi = 3.14159265358979323846
            self.alpha_t = nn.Parameter(torch.Tensor(num_ents, self.h_dim))
            self.beta_t = nn.Parameter(torch.Tensor(num_ents, self.h_dim))
            self.temporal_w = nn.Parameter(torch.Tensor(self.h_dim * 2, self.h_dim))
            self.st_static_emb = nn.Parameter(torch.Tensor(num_ents, self.h_dim))
            self._init_temporal_parameters()

        # Gradient clipping
        self.max_grad_norm = 1.0

        print("=" * 80)

    def _init_embeddings(self):
        """Unified embedding initialization"""
        nn.init.xavier_uniform_(self.emb_rel, gain=0.5)
        nn.init.xavier_uniform_(self.dynamic_emb, gain=0.5)

    def _init_temporal_parameters(self):
        """Initialize temporal parameters"""
        if not self.use_enhanced_temporal:
            nn.init.xavier_uniform_(self.alpha_t, gain=0.5)
            nn.init.xavier_uniform_(self.beta_t, gain=0.5)
            nn.init.xavier_uniform_(self.temporal_w)
            nn.init.xavier_uniform_(self.st_static_emb, gain=0.5)

    def clip_gradients(self):
        """Clip gradients to prevent explosion"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

    def apply_temporal_embedding(self, entity_emb, t):
        """Apply temporal embedding to entity embeddings"""
        if not self.use_enhanced_temporal:
            # Legacy method
            return self.get_dynamic_emb(t)

        # Get temporal encoding with decay weight
        temporal_encoding, decay_weight = self.temporal_encoder(t, return_decay_weight=True)

        # Expand for all entities
        if temporal_encoding.dim() == 1:
            temporal_encoding = temporal_encoding.unsqueeze(0).expand(self.num_ents, -1)

        # Project entity embeddings through temporal projection
        temporal_entity = self.temporal_projection(entity_emb)

        # Apply temporal gating
        output = self.temporal_gate(entity_emb, temporal_entity + 0.1 * temporal_encoding)

        # Apply decay weighting to emphasize recent information
        if decay_weight is not None and isinstance(decay_weight, torch.Tensor):
            output = output * (0.7 + 0.3 * decay_weight)  # Scale between 0.7 and 1.0

        return output

    def get_dynamic_emb(self, t):
        """Legacy method - kept for backward compatibility"""
        timevec = self.alpha * self.alpha_t * t + (1 - self.alpha) * torch.cos(2 * self.pi * self.beta_t * t)
        attn = torch.cat([self.st_static_emb, timevec], 1)
        return torch.mm(attn, self.temporal_w)

    def compute_temporal_regularization_loss(self):
        """Compute regularization loss for temporal components"""
        if not self.use_enhanced_temporal:
            return torch.tensor(0.0, device=self.device)

        reg_loss = 0.0

        # Regularize frequency scales to stay close to 1
        if hasattr(self.temporal_encoder, 'freq_scale'):
            reg_loss += 0.01 * torch.mean((self.temporal_encoder.freq_scale - 1.0) ** 2)

        # Regularize frequency shifts to stay small
        if hasattr(self.temporal_encoder, 'freq_shift'):
            reg_loss += 0.01 * torch.mean(self.temporal_encoder.freq_shift ** 2)

        # Regularize temporal decay to reasonable range
        if hasattr(self.temporal_encoder, 'temporal_decay'):
            decay = self.temporal_encoder.temporal_decay
            reg_loss += 0.1 * torch.relu(decay - 0.5) + 0.1 * torch.relu(-decay)  # Keep between 0 and 0.5

        return reg_loss

    def _dual_stream_forward(self, g, init_ent_emb):
        """Dual stream forward pass"""
        # Node stream
        node_emb = self.node_stream_rgcn.forward(g, init_ent_emb, [self.emb_rel, self.emb_rel])
        node_emb = F.normalize(node_emb) if self.layer_norm else node_emb

        # Edge stream
        src_sampled, dst_sampled, edge_types, _ = self.edge_sampler.sample_edges(g)

        if len(src_sampled) > 0:
            src_emb = init_ent_emb[src_sampled]
            dst_emb = init_ent_emb[dst_sampled]

            edge_features = self.dual_stream_processor.edge_feature_extractor(
                src_emb, dst_emb, edge_types
            )

            edge_enhanced_nodes = self.dual_stream_processor.create_edge_enhanced_nodes(
                edge_features, src_sampled, dst_sampled, g.number_of_nodes()
            )
        else:
            edge_enhanced_nodes = torch.zeros_like(node_emb)

        # Fusion
        fused = self.dual_stream_processor(node_emb, edge_enhanced_nodes)

        return fused

    def _process_static_graph(self, static_graph, t):
        """Process static graph with temporal embeddings"""
        static_graph = static_graph.to(self.device)

        if self.use_cl:
            if self.use_enhanced_temporal:
                # Apply temporal embedding to dynamic embeddings
                dynamic_emb = self.apply_temporal_embedding(self.dynamic_emb, t)
            else:
                dynamic_emb = self.get_dynamic_emb(t)
            static_graph.ndata['h'] = torch.cat((dynamic_emb, self.words_emb), dim=0)
        else:
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)

        self.statci_rgcn_layer(static_graph, [])
        static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
        static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
        return static_emb

    def _process_history(self, sub_graph, query_mask):
        """Process historical information"""
        sub_graph = sub_graph.to(self.device)

        if self.use_dual_stream:
            his_emb = self._dual_stream_forward(sub_graph, self.h)
        else:
            his_emb = self.his_rgcn_layer.forward(sub_graph, self.h, [self.emb_rel, self.emb_rel])

        his_emb = F.normalize(his_emb) if self.layer_norm else his_emb

        his_att = F.softmax(self.w5(query_mask + his_emb), dim=1)
        his_emb_weighted = his_att * his_emb
        his_emb_weighted = F.normalize(his_emb_weighted)

        return his_emb, his_emb_weighted

    def _process_timestep(self, g, g_trilist, i, g_list, num_nodes, query_mask, current_time):
        """Process a single time step with simplified temporal embeddings"""
        # Add inverse triples
        inverse_test_triplets = g_trilist[:, [2, 1, 0]]
        inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels
        all_triples = torch.cat((torch.from_numpy(g_trilist), torch.from_numpy(inverse_test_triplets)))

        g = g.to(self.device)

        # Apply temporal embeddings directly to hidden state
        if self.use_enhanced_temporal:
            # Apply temporal embedding to current hidden state
            self.h = self.apply_temporal_embedding(self.h, current_time)
        else:
            # Legacy temporal processing
            t2 = len(g_list) - i + 1
            h_t = torch.cos(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_ents, 1)
            self.h = self.w4(torch.cat([self.h, h_t], dim=1))

        # Process relations
        temp_e = self.h[g.r_to_e]
        x_input = torch.zeros(self.num_rels * 2, self.h_dim, device=self.device)

        for span, r_idx in zip(g.r_len, g.uniq_r):
            x = temp_e[span[0]:span[1], :]
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_input[r_idx] = x_mean

        # Simple relation update without complex temporal processing
        x_input = self.emb_rel + x_input

        # Forward pass
        if self.use_dual_stream:
            current_h = self._dual_stream_forward(g, self.h)
        else:
            current_h = self.rgcn.forward(g, self.h, [self.emb_rel, self.emb_rel])

        current_h = F.normalize(current_h) if self.layer_norm else current_h
        att_e = F.softmax(self.w2(query_mask + current_h), dim=1)

        # Update with GRU
        if i == 0:
            self.h_0 = self.entity_cell(current_h, self.h)
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
        else:
            self.h_0 = self.entity_cell(current_h, self.h_0)
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

        # Temporal gate for relations
        time_weight = F.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
        self.hr = time_weight * x_input + (1 - time_weight) * self.emb_rel
        self.hr = F.normalize(self.hr) if self.layer_norm else self.hr

        self.h = self.h_0
        att_emb = att_e * self.h_0

        return att_emb, self.h_0, self.hr

    def forward(self, sub_graph, T_idx, query_mask, g_list, static_graph, t, input_list, num_nodes, use_cuda):
        """Forward pass with simplified temporal embeddings"""
        # Initialize
        if self.use_static:
            static_emb = self._process_static_graph(static_graph, t)
            self.h = static_emb
        else:
            if self.use_enhanced_temporal:
                # Apply temporal embedding to initial dynamic embeddings
                self.h = self.apply_temporal_embedding(self.dynamic_emb, 0)
            else:
                self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb
            static_emb = None

        # Process history
        self.his_ent, _ = self._process_history(sub_graph, query_mask)
        his_r_emb = F.normalize(self.emb_rel)
        his_att = F.softmax(self.w5(query_mask + self.his_ent), dim=1)
        his_emb = his_att * self.his_ent
        his_emb = F.normalize(his_emb)

        history_embs = []
        att_embs = []
        his_temp_embs = []
        his_rel_embs = []

        # Store temporal embeddings for consistency
        if self.use_enhanced_temporal:
            self.prev_temporal_embs = []

        if self.pre_type == "all":
            for i, g in enumerate(g_list):
                g_trilist = input_list[i]
                current_time = i  # or use actual timestamp if available

                att_emb, h_0, hr = self._process_timestep(
                    g, g_trilist, i, g_list, num_nodes, query_mask, current_time
                )

                history_embs.append(h_0)
                his_rel_embs.append(hr)
                his_temp_embs.append(h_0)
                att_embs.append(att_emb.unsqueeze(0))

                # Store temporal embeddings for consistency loss
                if self.use_enhanced_temporal and self.training:
                    temporal_enc = self.temporal_encoder(current_time)
                    self.prev_temporal_embs.append(temporal_enc)

            att_ent = torch.mean(torch.cat(att_embs, dim=0), dim=0)
            att_ent = F.normalize(att_ent)

            # Simple aggregation without complex attention
            history_emb = att_ent + history_embs[-1]
            history_emb = F.normalize(history_emb) if self.layer_norm else history_emb
        else:
            self.hr = None
            history_emb = None
            his_emb = None
            his_r_emb = None

        # Update training step
        self.training_step += 1

        return history_emb, static_emb, self.hr, his_emb, his_r_emb, his_temp_embs, his_rel_embs, history_embs

    def predict(self, que_pair, tlist, sub_graph, T_id, test_graph, num_rels, static_graph, test_triplets, input_list,
                num_nodes, use_cuda):
        """Prediction method with simplified temporal embeddings"""
        with torch.no_grad():
            all_triples = test_triplets

            # Query processing
            uniq_e = que_pair[0]
            r_len = que_pair[1]
            r_idx = que_pair[2]
            temp_r = self.emb_rel[r_idx]
            e_input = torch.zeros(self.num_ents, self.h_dim, device=self.device)

            for span, e_idx in zip(r_len, uniq_e):
                x = temp_r[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                e_input[e_idx] = x_mean

            # Apply temporal embeddings for query
            if self.use_enhanced_temporal:
                e1_emb = self.apply_temporal_embedding(self.dynamic_emb, tlist[0])
                e1_emb = e1_emb[uniq_e]
            else:
                e1_emb = self.dynamic_emb[uniq_e]

            rel_emb = e_input[uniq_e]
            query_emb = self.w1(torch.cat([e1_emb, rel_emb], dim=1))

            query_mask = torch.zeros(self.num_ents, self.h_dim, device=self.device)
            query_mask[uniq_e] = query_emb

            embedding, _, r_emb, his_emb, his_r_emb, _, _, _ = self.forward(
                sub_graph, T_id, query_mask, test_graph, static_graph, tlist[0],
                input_list, num_nodes, use_cuda
            )

            if self.pre_type == "all":
                scores_ob, _ = self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb,
                                                       self.pre_weight, self.pre_type)
                score_seq = F.softmax(scores_ob, dim=1)

            scores_en = torch.log(score_seq)
            return all_triples, scores_en

    def get_loss(self, que_pair, sub_graph, T_idx, glist, triples, static_graph, tlist, input_list, num_nodes,
                 use_cuda):
        """Loss computation with temporal regularization"""
        # Initialize losses
        loss_ent = torch.zeros(1, device=self.device)
        loss_cl = torch.zeros(1, device=self.device)
        loss_rel = torch.zeros(1, device=self.device)
        loss_static = torch.zeros(1, device=self.device)
        loss_temporal_reg = torch.zeros(1, device=self.device)

        all_triples = triples

        # Query processing
        uniq_e = que_pair[0]
        r_len = que_pair[1]
        r_idx = que_pair[2]
        temp_r = self.emb_rel[r_idx]
        e_input = torch.zeros(self.num_ents, self.h_dim, device=self.device)

        for span, e_idx in zip(r_len, uniq_e):
            x = temp_r[span[0]:span[1], :]
            x_mean = torch.mean(x, dim=0, keepdim=True)
            e_input[e_idx] = x_mean

        query_mask = torch.zeros(self.num_ents, self.h_dim, device=self.device)

        # Apply temporal embeddings to query
        if self.use_enhanced_temporal:
            qe_emb = self.apply_temporal_embedding(self.dynamic_emb, 0)
        else:
            q_t = torch.cos(self.weight_t2 * 0 + self.bias_t2).repeat(self.num_ents, 1)
            qe_emb = self.w4(torch.cat([self.dynamic_emb, q_t], dim=1))

        e1_emb = qe_emb[uniq_e]
        rel_emb = e_input[uniq_e]
        query_emb = self.w1(torch.cat([e1_emb, rel_emb], dim=1))
        query_mask[uniq_e] = query_emb

        embedding, static_emb, r_emb, his_emb, his_r_emb, his_temp_embs, his_rel_embs, history_embs = \
            self.forward(sub_graph, T_idx, query_mask, glist, static_graph, tlist[0], input_list, num_nodes,
                         use_cuda)

        # Entity prediction loss
        if self.pre_type == "all":
            scores_ob, _ = self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb,
                                                   self.pre_weight, self.pre_type)
            loss_ent = self.loss_e(scores_ob, triples[:, 2])

        # Relation prediction loss
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel = self.loss_r(score_rel, all_triples[:, 1])

        # Contrastive learning loss
        if self.use_cl and self.pre_type == "all":
            for id, evolve_emb in enumerate(his_temp_embs):
                query = torch.cat([self.his_ent[all_triples[:, 0]], his_r_emb[all_triples[:, 1]]], dim=1)
                query2 = torch.cat([evolve_emb[all_triples[:, 0]], his_rel_embs[id][all_triples[:, 1]]], dim=1)
                x1 = self.w_cl(query)
                x2 = self.w_cl(query2)
                loss_cl += self.get_loss_conv(x1, x2)

        # Static loss
        if self.use_static and static_emb is not None:
            for time_step, evolve_emb in enumerate(history_embs):
                step = (self.angle * math.pi / 180) * (time_step + 1)
                if self.layer_norm:
                    sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                else:
                    sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                    c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                    sim_matrix = sim_matrix / (c + 1e-6)

                mask = (math.cos(step) - sim_matrix) > 0
                if mask.any():
                    loss_static += self.static_alpha * self.weight * torch.sum(
                        torch.masked_select(math.cos(step) - sim_matrix, mask)
                    )

        # Temporal regularization loss
        if self.use_enhanced_temporal:
            loss_temporal_reg = self.compute_temporal_regularization_loss()

            # Add temporal consistency loss
            if len(self.prev_temporal_embs) > 1:
                for i in range(len(self.prev_temporal_embs) - 1):
                    # Encourage smooth transitions between consecutive temporal embeddings
                    diff = self.prev_temporal_embs[i + 1] - self.prev_temporal_embs[i]
                    loss_temporal_reg += self.temporal_consistency_weight * torch.mean(diff ** 2)

        # Apply loss weights
        loss_ent = loss_ent * self.loss_weights['entity']
        loss_rel = loss_rel * self.loss_weights['relation']
        loss_static = loss_static * self.loss_weights['static']
        loss_cl = loss_cl * self.loss_weights['contrastive']
        loss_temporal_reg = loss_temporal_reg * self.loss_weights['temporal_reg']

        # Update last loss for tracking
        total_loss = loss_ent + loss_rel + loss_static + loss_cl + loss_temporal_reg
        self.last_loss = total_loss.item()

        return loss_ent, loss_rel, loss_static, loss_cl + loss_temporal_reg  # Combine CL and temporal reg

    def get_loss_conv(self, ent1_emb, ent2_emb):
        """Contrastive loss computation"""
        loss_fn = nn.CrossEntropyLoss()
        z1 = self.projection_model(ent1_emb)
        z2 = self.projection_model(ent2_emb)

        pred1 = torch.mm(z1, z2.T)
        pred2 = torch.mm(z2, z1.T)
        pred3 = torch.mm(z1, z1.T)
        pred4 = torch.mm(z2, z2.T)

        labels = torch.arange(pred1.shape[0], device=self.device)
        train_cl_loss = (loss_fn(pred1 / self.temp, labels) +
                         loss_fn(pred2 / self.temp, labels) +
                         loss_fn(pred3 / self.temp, labels) +
                         loss_fn(pred4 / self.temp, labels)) / 4

        return train_cl_loss

    def get_training_status(self):
        """Get comprehensive training status"""
        status = {
            'dual_stream_enabled': self.use_dual_stream,
            'enhanced_temporal_enabled': self.use_enhanced_temporal,
            'temporal_fusion_method': self.temporal_fusion_method,
            'training_step': self.training_step,
            'last_loss': self.last_loss if self.last_loss is not None else 0.0,
            'warmup_completed': self.training_step > self.warmup_steps,
            'device': str(self.device)
        }

        if self.use_enhanced_temporal:
            # Get temporal encoder statistics
            with torch.no_grad():
                status['temporal_stats'] = {
                    'freq_scale_mean': self.temporal_encoder.freq_scale.mean().item(),
                    'freq_shift_mean': self.temporal_encoder.freq_shift.mean().item(),
                    'temporal_decay': self.temporal_encoder.temporal_decay.item(),
                    'gate_residual_weight': self.temporal_gate.residual_weight.item()
                }

        status['loss_weights'] = self.loss_weights.copy()

        return status