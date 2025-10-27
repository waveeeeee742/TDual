import csv
from datetime import datetime
import argparse
import itertools
import os
import sys
import time
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
import gc
from contextlib import contextmanager
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict, deque

sys.path.append(".")
from local_rgcn import utils
from local_rgcn.utils import build_sub_graph, build_graph
from local_rgcn.rrgcn import RecurrentRGCN
import torch.nn.modules.rnn
from local_rgcn.knowledge_graph import _read_triplets_as_list
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ================================
# ID到文本映射器 (新增)
# ================================
class IDMapper:
    """将ID映射回原始文本"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.entity_id_to_name = {}
        self.relation_id_to_name = {}
        self.word_id_to_name = {}

        self.load_mappings()

    def load_mappings(self):
        """加载所有映射文件"""
        dataset_dir = DATA_DIR / self.dataset_name

        # 加载实体映射
        entity_file = dataset_dir / "entity2id.txt"
        if entity_file.exists():
            with open(entity_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        entity_name = ' '.join(parts[:-1])
                        entity_id = int(parts[-1])
                        self.entity_id_to_name[entity_id] = entity_name
            print(f"Loaded {len(self.entity_id_to_name)} entities")

        # 加载关系映射
        relation_file = dataset_dir / "relation2id.txt"
        if relation_file.exists():
            with open(relation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        relation_name = ' '.join(parts[:-1])
                        relation_id = int(parts[-1])
                        self.relation_id_to_name[relation_id] = relation_name
            print(f"Loaded {len(self.relation_id_to_name)} relations")

        # 加载词映射
        word_file = dataset_dir / "word2id.txt"
        if word_file.exists():
            with open(word_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word_name = ' '.join(parts[:-1])
                        word_id = int(parts[-1])
                        self.word_id_to_name[word_id] = word_name
            print(f"Loaded {len(self.word_id_to_name)} words")

    def get_entity_name(self, entity_id: int) -> str:
        """获取实体名称"""
        return self.entity_id_to_name.get(entity_id, f"Entity_{entity_id}")

    def get_relation_name(self, relation_id: int, num_rels: int) -> str:
        """获取关系名称（处理逆关系）"""
        if relation_id >= num_rels:
            # 逆关系
            base_rel_id = relation_id - num_rels
            base_rel_name = self.relation_id_to_name.get(base_rel_id, f"Relation_{base_rel_id}")
            return f"INV_{base_rel_name}"
        else:
            return self.relation_id_to_name.get(relation_id, f"Relation_{relation_id}")


# ================================
# 多跳查询记录器 (新增)
# ================================
class MultiHopQueryRecorder:
    """记录多跳查询的详细信息用于定性分析"""

    def __init__(self, id_mapper: IDMapper, num_rels: int, output_dir: str = "./qualitative_analysis"):
        self.id_mapper = id_mapper
        self.num_rels = num_rels
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.query_records = {
            '1-hop': [],
            '2-hop': [],
            '3-hop': [],
            '4+hop': []
        }

        self.max_samples_per_category = 50  # 每个类别最多记录的样本数

    def add_query_record(self, hop_category: str, query_triple: np.ndarray,
                         predicted_entity: int, true_entity: int,
                         rank: int, score: float, timestamp: int):
        """添加一条查询记录"""
        if hop_category not in self.query_records:
            return

        # 限制每个类别的样本数量
        if len(self.query_records[hop_category]) >= self.max_samples_per_category:
            return

        src, rel, dst = query_triple

        record = {
            'timestamp': timestamp,
            'query': {
                'head_id': int(src),
                'head_name': self.id_mapper.get_entity_name(int(src)),
                'relation_id': int(rel),
                'relation_name': self.id_mapper.get_relation_name(int(rel), self.num_rels),
                'tail_id': int(dst),
                'tail_name': self.id_mapper.get_entity_name(int(dst))
            },
            'prediction': {
                'predicted_entity_id': int(predicted_entity),
                'predicted_entity_name': self.id_mapper.get_entity_name(int(predicted_entity)),
                'rank': int(rank),
                'score': float(score)
            },
            'ground_truth': {
                'true_entity_id': int(true_entity),
                'true_entity_name': self.id_mapper.get_entity_name(int(true_entity))
            },
            'is_correct': int(predicted_entity) == int(true_entity),
            'hop_category': hop_category
        }

        self.query_records[hop_category].append(record)

    def save_records(self, dataset_name: str, model_config: str = ""):
        """保存所有记录到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存JSON格式（机器可读）
        json_file = os.path.join(self.output_dir,
                                 f"{dataset_name}_multihop_queries_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset_name,
                'model_config': model_config,
                'timestamp': timestamp,
                'query_records': self.query_records,
                'statistics': self.get_statistics()
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ JSON records saved to: {json_file}")

        # 保存人类可读的文本格式
        txt_file = os.path.join(self.output_dir,
                                f"{dataset_name}_multihop_queries_{timestamp}.txt")
        self.save_human_readable_format(txt_file)

        print(f"✓ Human-readable records saved to: {txt_file}")

        # 保存CSV格式（便于Excel分析）
        csv_file = os.path.join(self.output_dir,
                                f"{dataset_name}_multihop_queries_{timestamp}.csv")
        self.save_csv_format(csv_file)

        print(f"✓ CSV records saved to: {csv_file}")

    def save_human_readable_format(self, filename: str):
        """保存人类可读的文本格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("MULTI-HOP QUERY QUALITATIVE ANALYSIS\n")
            f.write("=" * 100 + "\n\n")

            for hop_category in ['1-hop', '2-hop', '3-hop', '4+hop']:
                records = self.query_records[hop_category]
                if not records:
                    continue

                f.write(f"\n{'=' * 100}\n")
                f.write(f"{hop_category.upper()} QUERIES (Total: {len(records)})\n")
                f.write(f"{'=' * 100}\n\n")

                for idx, record in enumerate(records, 1):
                    f.write(f"Example {idx}:\n")
                    f.write(f"{'-' * 100}\n")

                    # 查询信息
                    query = record['query']
                    f.write(f"Query at timestamp {record['timestamp']}:\n")
                    f.write(f"  ({query['head_name']}, {query['relation_name']}, ?)\n")
                    f.write(f"  IDs: ({query['head_id']}, {query['relation_id']}, ?)\n\n")

                    # 真实答案
                    gt = record['ground_truth']
                    f.write(f"Ground Truth:\n")
                    f.write(f"  {gt['true_entity_name']} (ID: {gt['true_entity_id']})\n\n")

                    # 预测结果
                    pred = record['prediction']
                    f.write(f"Model Prediction:\n")
                    f.write(f"  Top-1: {pred['predicted_entity_name']} (ID: {pred['predicted_entity_id']})\n")
                    f.write(f"  Rank of true answer: {pred['rank']}\n")
                    f.write(f"  Prediction score: {pred['score']:.6f}\n")
                    f.write(f"  Correct: {'✓ YES' if record['is_correct'] else '✗ NO'}\n")

                    f.write(f"\n")

    def save_csv_format(self, filename: str):
        """保存CSV格式"""
        rows = []
        for hop_category, records in self.query_records.items():
            for record in records:
                row = {
                    'hop_category': hop_category,
                    'timestamp': record['timestamp'],
                    'head_id': record['query']['head_id'],
                    'head_name': record['query']['head_name'],
                    'relation_id': record['query']['relation_id'],
                    'relation_name': record['query']['relation_name'],
                    'tail_id': record['query']['tail_id'],
                    'true_entity_id': record['ground_truth']['true_entity_id'],
                    'true_entity_name': record['ground_truth']['true_entity_name'],
                    'predicted_entity_id': record['prediction']['predicted_entity_id'],
                    'predicted_entity_name': record['prediction']['predicted_entity_name'],
                    'rank': record['prediction']['rank'],
                    'score': record['prediction']['score'],
                    'is_correct': record['is_correct']
                }
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False, encoding='utf-8')

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {}
        for category, records in self.query_records.items():
            if records:
                correct_count = sum(1 for r in records if r['is_correct'])
                stats[category] = {
                    'total': len(records),
                    'correct': correct_count,
                    'accuracy': correct_count / len(records) if records else 0,
                    'avg_rank': np.mean([r['prediction']['rank'] for r in records])
                }
        return stats

    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 80)
        print("QUALITATIVE ANALYSIS SUMMARY")
        print("=" * 80)

        stats = self.get_statistics()
        for category, stat in stats.items():
            print(f"\n{category}:")
            print(f"  Samples collected: {stat['total']}")
            print(f"  Top-1 accuracy: {stat['accuracy']:.2%}")
            print(f"  Average rank: {stat['avg_rank']:.2f}")

        print("=" * 80)


# ================================
# 多跳分析器
# ================================
class HopAnalyzer:
    """多跳推理分析器 - 计算查询的推理深度"""

    def __init__(self, num_nodes: int, num_rels: int):
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.graph_structure = defaultdict(lambda: defaultdict(set))  # {src: {rel: {dst}}}
        self.hop_cache = {}  # 缓存已计算的跳数

    def build_graph_structure(self, history_triples: List[np.ndarray]):
        """从历史三元组构建图结构"""
        print("Building graph structure for hop analysis...")
        self.graph_structure.clear()

        for triples in history_triples:
            for src, rel, dst in triples:
                self.graph_structure[src][rel].add(dst)
                # 添加逆关系
                self.graph_structure[dst][rel + self.num_rels].add(src)

        print(f"Graph structure built: {len(self.graph_structure)} nodes")

    def compute_hop_distance(self, src: int, dst: int, max_hops: int = 5) -> int:
        """
        使用BFS计算从src到dst的最短路径长度（跳数）
        返回: 跳数 (1表示直接连接, 2表示2跳, 等等)
        """
        # 检查缓存
        cache_key = (src, dst)
        if cache_key in self.hop_cache:
            return self.hop_cache[cache_key]

        if src == dst:
            self.hop_cache[cache_key] = 0
            return 0

        # BFS搜索
        queue = deque([(src, 0)])
        visited = {src}

        while queue:
            current_node, current_hop = queue.popleft()

            # 达到最大跳数限制
            if current_hop >= max_hops:
                break

            # 遍历所有邻居
            if current_node in self.graph_structure:
                for rel, neighbors in self.graph_structure[current_node].items():
                    for neighbor in neighbors:
                        if neighbor == dst:
                            hop_distance = current_hop + 1
                            self.hop_cache[cache_key] = hop_distance
                            return hop_distance

                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, current_hop + 1))

        # 未找到路径，标记为最大跳数+1
        hop_distance = max_hops + 1
        self.hop_cache[cache_key] = hop_distance
        return hop_distance

    def classify_queries_by_hops(self, test_triples: np.ndarray) -> Dict[str, List[int]]:
        """
        将查询按跳数分类
        返回: {hop_category: [query_indices]}
        """
        hop_categories = {
            '1-hop': [],
            '2-hop': [],
            '3-hop': [],
            '4+hop': [],
            'unreachable': []
        }

        for idx, (src, rel, dst) in enumerate(test_triples):
            hop_dist = self.compute_hop_distance(int(src), int(dst))

            if hop_dist == 1:
                hop_categories['1-hop'].append(idx)
            elif hop_dist == 2:
                hop_categories['2-hop'].append(idx)
            elif hop_dist == 3:
                hop_categories['3-hop'].append(idx)
            elif hop_dist >= 4 and hop_dist <= 5:
                hop_categories['4+hop'].append(idx)
            else:
                hop_categories['unreachable'].append(idx)

        return hop_categories

    def get_statistics(self, hop_categories: Dict[str, List[int]]) -> Dict:
        """获取跳数统计信息"""
        total = sum(len(v) for v in hop_categories.values())
        stats = {}

        for category, indices in hop_categories.items():
            count = len(indices)
            percentage = (count / total * 100) if total > 0 else 0
            stats[category] = {
                'count': count,
                'percentage': percentage
            }

        return stats


# ================================
# 多跳指标收集器
# ================================
class MultiHopMetricsCollector:
    """收集和管理多跳推理的评估指标"""

    def __init__(self):
        self.metrics = {
            '1-hop': {'ranks_raw': [], 'ranks_filter': []},
            '2-hop': {'ranks_raw': [], 'ranks_filter': []},
            '3-hop': {'ranks_raw': [], 'ranks_filter': []},
            '4+hop': {'ranks_raw': [], 'ranks_filter': []},
            'unreachable': {'ranks_raw': [], 'ranks_filter': []},
            'all': {'ranks_raw': [], 'ranks_filter': []}
        }

    def add_metrics(self, hop_category: str, ranks_raw: List, ranks_filter: List):
        """添加某个跳数类别的指标"""

        # 确保ranks是张量列表格式，处理可能的标量张量
        def ensure_tensor_list(ranks):
            """确保ranks是正确的张量列表格式"""
            if not ranks:
                return []

            # 如果是单个张量，转换为列表
            if torch.is_tensor(ranks):
                if ranks.dim() == 0:  # 标量张量
                    return [ranks.unsqueeze(0)]
                else:
                    return [ranks]

            # 如果是列表，确保每个元素都是1D张量
            processed = []
            for r in ranks:
                if torch.is_tensor(r):
                    if r.dim() == 0:  # 标量张量
                        processed.append(r.unsqueeze(0))
                    else:
                        processed.append(r)
                else:
                    # 如果是python数值，转换为张量
                    processed.append(torch.tensor([r]))
            return processed

        ranks_raw_processed = ensure_tensor_list(ranks_raw)
        ranks_filter_processed = ensure_tensor_list(ranks_filter)

        self.metrics[hop_category]['ranks_raw'].extend(ranks_raw_processed)
        self.metrics[hop_category]['ranks_filter'].extend(ranks_filter_processed)
        self.metrics['all']['ranks_raw'].extend(ranks_raw_processed)
        self.metrics['all']['ranks_filter'].extend(ranks_filter_processed)

    def compute_final_metrics(self) -> Dict:
        """计算所有类别的最终指标"""
        final_metrics = {}

        for category, data in self.metrics.items():
            if len(data['ranks_raw']) > 0:
                try:
                    # 确保所有ranks都是正确的张量格式
                    ranks_raw_tensors = []
                    ranks_filter_tensors = []

                    for r in data['ranks_raw']:
                        if torch.is_tensor(r):
                            if r.dim() == 0:
                                ranks_raw_tensors.append(r.unsqueeze(0))
                            else:
                                ranks_raw_tensors.append(r)
                        else:
                            ranks_raw_tensors.append(torch.tensor([r]))

                    for r in data['ranks_filter']:
                        if torch.is_tensor(r):
                            if r.dim() == 0:
                                ranks_filter_tensors.append(r.unsqueeze(0))
                            else:
                                ranks_filter_tensors.append(r)
                        else:
                            ranks_filter_tensors.append(torch.tensor([r]))

                    # 计算指标
                    mrr_raw, hits_raw = utils.stat_ranks(ranks_raw_tensors, f"{category}_raw")
                    mrr_filter, hits_filter = utils.stat_ranks(ranks_filter_tensors, f"{category}_filter")

                    final_metrics[category] = {
                        'count': len(ranks_raw_tensors),
                        'mrr_raw': float(mrr_raw),
                        'mrr_filter': float(mrr_filter),
                        'hits@1_raw': hits_raw[0],
                        'hits@3_raw': hits_raw[1],
                        'hits@10_raw': hits_raw[2],
                        'hits@1_filter': hits_filter[0],
                        'hits@3_filter': hits_filter[1],
                        'hits@10_filter': hits_filter[2]
                    }
                except Exception as e:
                    print(f"Warning: Error computing metrics for {category}: {e}")
                    final_metrics[category] = {
                        'count': 0,
                        'mrr_raw': 0.0,
                        'mrr_filter': 0.0,
                        'hits@1_raw': 0.0,
                        'hits@3_raw': 0.0,
                        'hits@10_raw': 0.0,
                        'hits@1_filter': 0.0,
                        'hits@3_filter': 0.0,
                        'hits@10_filter': 0.0
                    }
            else:
                final_metrics[category] = {
                    'count': 0,
                    'mrr_raw': 0.0,
                    'mrr_filter': 0.0,
                    'hits@1_raw': 0.0,
                    'hits@3_raw': 0.0,
                    'hits@10_raw': 0.0,
                    'hits@1_filter': 0.0,
                    'hits@3_filter': 0.0,
                    'hits@10_filter': 0.0
                }

        return final_metrics

    def print_detailed_report(self, final_metrics: Dict):
        """打印详细的多跳分析报告"""
        print("\n" + "=" * 80)
        print("MULTI-HOP REASONING ANALYSIS REPORT")
        print("=" * 80)

        categories = ['1-hop', '2-hop', '3-hop', '4+hop']

        for category in categories:
            if category in final_metrics and final_metrics[category]['count'] > 0:
                metrics = final_metrics[category]
                print(f"\n{category.upper()} Queries:")
                print(f"  Count: {metrics['count']}")
                print(f"  MRR (filter): {metrics['mrr_filter']:.6f}")
                print(f"  Hits@1: {metrics['hits@1_filter']:.6f}")
                print(f"  Hits@3: {metrics['hits@3_filter']:.6f}")
                print(f"  Hits@10: {metrics['hits@10_filter']:.6f}")

        # 计算改进幅度（如果有baseline）
        try:
            if ('1-hop' in final_metrics and '3-hop' in final_metrics and
                    final_metrics['1-hop']['count'] > 0 and final_metrics['3-hop']['count'] > 0 and
                    final_metrics['1-hop']['mrr_filter'] > 0):
                improvement = ((final_metrics['3-hop']['mrr_filter'] /
                                final_metrics['1-hop']['mrr_filter']) - 1) * 100
                print(f"\n3-hop vs 1-hop MRR change: {improvement:+.2f}%")

            if ('1-hop' in final_metrics and '2-hop' in final_metrics and
                    final_metrics['1-hop']['count'] > 0 and final_metrics['2-hop']['count'] > 0 and
                    final_metrics['1-hop']['mrr_filter'] > 0):
                improvement_2 = ((final_metrics['2-hop']['mrr_filter'] /
                                  final_metrics['1-hop']['mrr_filter']) - 1) * 100
                print(f"2-hop vs 1-hop MRR change: {improvement_2:+.2f}%")
        except Exception as e:
            print(f"\nNote: Could not compute improvement ratios: {e}")

        print("=" * 80 + "\n")


# ================================
# 内存管理
# ================================
@contextmanager
def memory_manager():
    """内存管理上下文"""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ================================
# 训练监控器
# ================================
class TrainingMonitor:
    """训练过程监控器"""

    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        self.loss_history = []
        self.stability_window = 20

    def log_metrics(self, epoch: int, metrics: Dict):
        """记录指标"""
        for key, value in metrics.items():
            self.metrics[key].append(value)

        if 'loss' in metrics:
            self.loss_history.append(metrics['loss'])
            if len(self.loss_history) > 100:
                self.loss_history = self.loss_history[-100:]

        log_file = os.path.join(self.log_dir, f"metrics_{datetime.now().strftime('%Y%m%d')}.json")
        with open(log_file, 'w') as f:
            json.dump({k: [float(v) if isinstance(v, (int, float, np.number)) else v
                           for v in vals] for k, vals in self.metrics.items()}, f, indent=2)

    def update_best(self, metric_name: str, value: float, epoch: int):
        """更新最佳指标"""
        if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]['value']:
            self.best_metrics[metric_name] = {
                'value': value,
                'epoch': epoch
            }

    def get_training_stability(self) -> float:
        """获取训练稳定性指标"""
        if len(self.loss_history) < self.stability_window:
            return 1.0
        recent_losses = self.loss_history[-self.stability_window:]
        return np.std(recent_losses)

    def is_training_stable(self, threshold=0.01) -> bool:
        """判断训练是否稳定"""
        return self.get_training_stability() < threshold

    def get_summary(self) -> str:
        """获取训练总结"""
        summary = ["\n" + "=" * 50]
        summary.append("Training Summary:")
        for metric, info in self.best_metrics.items():
            summary.append(f"Best {metric}: {info['value']:.4f} at epoch {info['epoch']}")
        summary.append("=" * 50)
        return "\n".join(summary)


# ================================
# 数据增强器
# ================================
class DataAugmenter:
    """数据增强器"""

    def __init__(self, aug_prob=0.05):
        self.aug_prob = aug_prob

    def augment_triples(self, triples: np.ndarray, num_rels: int) -> np.ndarray:
        """增强三元组数据"""
        if random.random() > self.aug_prob:
            return triples

        num_to_reverse = int(len(triples) * 0.05)
        if num_to_reverse > 0:
            indices = np.random.choice(len(triples), num_to_reverse, replace=False)
            reversed_triples = triples[indices].copy()
            reversed_triples[:, [0, 2]] = reversed_triples[:, [2, 0]]
            reversed_triples[:, 1] += num_rels

            augmented = np.concatenate([triples, reversed_triples])
            return augmented

        return triples


# ================================
# 自适应训练调度器
# ================================
class AdaptiveTrainingScheduler:
    """自适应训练调度器"""

    def __init__(self, initial_lr=0.001, patience=10, min_lr=1e-5):
        self.initial_lr = initial_lr
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = 0
        self.wait_count = 0
        self.current_lr = initial_lr
        self.lr_reduction_count = 0

    def should_reduce_lr(self, current_metric):
        """判断是否需要降低学习率"""
        if current_metric <= self.best_metric:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.wait_count = 0
                self.lr_reduction_count += 1
                return True
        else:
            self.best_metric = current_metric
            self.wait_count = 0
        return False

    def get_lr_scale(self):
        """获取学习率缩放因子"""
        return 0.5


# ================================
# 实用函数
# ================================
def update_dict(subg_arr, s_to_sro, sr_to_sro, sro_to_fre, num_rels):
    """更新查询字典"""
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])

    for j, (src, rel, dst) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)


def e2r(triplets, num_rels):
    """实体到关系的映射"""
    src, rel, dst = triplets.transpose()
    uniq_e = np.unique(src)

    e_to_r = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        e_to_r[src].add(rel)

    r_len = []
    r_idx = []
    idx = 0
    for e in uniq_e:
        r_len.append((idx, idx + len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])

    uniq_e = torch.from_numpy(np.array(uniq_e)).long().cuda()
    r_len = torch.from_numpy(np.array(r_len)).long().cuda()
    r_idx = torch.from_numpy(np.array(r_idx)).long().cuda()
    return [uniq_e, r_len, r_idx]


def get_sample_from_history_graph3(subg_arr, sr_to_sro, triples, num_nodes, num_rels, use_cuda, gpu):
    """从历史图中采样"""
    with memory_manager():
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        all_triples = np.concatenate([triples, inverse_triples])

        src_set = set(triples[:, 0])
        dst_set = set(triples[:, 0])

        er_list = list(set([(tri[0], tri[1]) for tri in triples]))
        er_list_inv = list(set([(tri[0], tri[1]) for tri in inverse_triples]))

        inverse_subg = subg_arr[:, [2, 1, 0]]
        inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
        subg_triples = np.concatenate([subg_arr, inverse_subg])

        df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
        subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'freq'})

        keys = list(sr_to_sro.keys())
        values = list(sr_to_sro.values())
        df_dic = pd.DataFrame({'sr': keys, 'dst': values})

        dst_df = df_dic.query('sr in @er_list')
        dst_get = dst_df['dst'].values
        two_ent = set().union(*dst_get) if len(dst_get) > 0 else set()
        all_ent = list(src_set | two_ent)
        result = subg_df.query('src in @all_ent')

        dst_df_inv = df_dic.query('sr in @er_list_inv')
        dst_get_inv = dst_df_inv['dst'].values
        two_ent_inv = set().union(*dst_get_inv) if len(dst_get_inv) > 0 else set()
        all_ent_inv = list(dst_set | two_ent_inv)
        result_inv = subg_df.query('src in @all_ent_inv')

        q_tri = result.to_numpy()
        q_tri_inv = result_inv.to_numpy()

        his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu)
        his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)

        return his_sub, his_sub_inv


# ================================
# 增强的测试函数（支持多跳分析和定性记录）
# ================================
def test(model, history_len, history_list, test_list, num_rels, num_nodes, use_cuda,
         all_ans_list, all_ans_r_list, model_name, static_graph, mode, monitor=None,
         dataset_name="ICEWS18", enable_multihop_analysis=True,
         enable_qualitative_recording=True):
    """
    增强的测试函数，支持多跳推理分析和定性记录

    Args:
        enable_multihop_analysis: 是否启用多跳分析（默认True）
        enable_qualitative_recording: 是否启用定性记录（默认True）
    """

    # 初始化多跳分析器
    hop_analyzer = None
    multihop_collector = None
    query_recorder = None
    id_mapper = None

    if enable_multihop_analysis:
        print("\n" + "=" * 80)
        print("MULTI-HOP ANALYSIS ENABLED")
        print("=" * 80)
        hop_analyzer = HopAnalyzer(num_nodes, num_rels)
        hop_analyzer.build_graph_structure(history_list)
        multihop_collector = MultiHopMetricsCollector()

    # 初始化定性记录器
    if enable_qualitative_recording:
        print("\n" + "=" * 80)
        print("QUALITATIVE RECORDING ENABLED")
        print("=" * 80)
        id_mapper = IDMapper(dataset_name)
        query_recorder = MultiHopQueryRecorder(id_mapper, num_rels)

    # 常规指标列表
    ranks_raw, ranks_filter = [], []
    ranks_raw_inv, ranks_filter_inv = [], []

    if mode == "test":
        print(f"\nLoading model from: {model_name}")
        checkpoint = torch.load(model_name, map_location=torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu'))
        print(f"Using best epoch: {checkpoint['epoch']}")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # 准备数据
    input_list = [snap for snap in history_list[-args.test_history_len:]]
    start_time = len(history_list)
    his_list = history_list[:]
    subg_arr = np.concatenate(his_list)

    sr_to_sro_path = DATA_DIR / dataset_name / 'his_dict' / 'train_s_r.npy'
    sr_to_sro = np.load(str(sr_to_sro_path), allow_pickle=True).item()

    with torch.no_grad():
        with memory_manager():
            for time_idx, test_snap in enumerate(tqdm(test_list, desc="Testing")):
                tc = start_time + time_idx
                tlist = list(range(tc - args.train_history_len, tc))
                tlist = torch.Tensor(tlist).cuda()

                # 多跳分析：分类当前时间步的查询
                hop_categories = None
                if enable_multihop_analysis and hop_analyzer:
                    hop_categories = hop_analyzer.classify_queries_by_hops(test_snap)

                    # 打印统计信息
                    if time_idx == 0:
                        stats = hop_analyzer.get_statistics(hop_categories)
                        print("\nQuery Distribution by Hops:")
                        for category, stat in stats.items():
                            print(f"  {category}: {stat['count']} ({stat['percentage']:.2f}%)")

                # 构建历史图列表
                history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
                                 for g in input_list]

                # 准备查询
                inverse_triples = test_snap[:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                que_pair = e2r(test_snap, num_rels)
                que_pair_inv = e2r(inverse_triples, num_rels)

                # 采样子图
                sub_snap, sub_snap_inv = get_sample_from_history_graph3(
                    subg_arr, sr_to_sro, test_snap, num_nodes, num_rels, use_cuda, args.gpu
                )

                # 预测
                test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
                test_triples_input_inv = torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(
                    inverse_triples)

                test_triples, final_score = model.predict(
                    que_pair, tlist, sub_snap, time_idx, history_glist,
                    num_rels, static_graph, test_triples_input, input_list, num_nodes, use_cuda
                )
                inv_test_triples, inv_final_score = model.predict(
                    que_pair_inv, tlist, sub_snap_inv, time_idx, history_glist,
                    num_rels, static_graph, test_triples_input_inv, input_list, num_nodes, use_cuda
                )

                # 计算指标
                mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
                    test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
                )
                mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(
                    inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
                )

                # 常规收集
                ranks_raw.append(rank_raw)
                ranks_filter.append(rank_filter)
                ranks_raw_inv.append(rank_raw_inv)
                ranks_filter_inv.append(rank_filter_inv)

                # 多跳分析：按类别收集指标 + 定性记录
                if enable_multihop_analysis and hop_categories and multihop_collector:
                    for category, indices in hop_categories.items():
                        if len(indices) > 0:
                            # 安全地提取该类别的ranks
                            try:
                                category_ranks_raw = []
                                category_ranks_filter = []

                                for idx in indices:
                                    if idx < len(rank_raw):
                                        # 确保提取的是张量
                                        r_raw = rank_raw[idx]
                                        r_filter = rank_filter[idx]

                                        # 如果是标量，转换为1D张量
                                        if torch.is_tensor(r_raw):
                                            if r_raw.dim() == 0:
                                                r_raw = r_raw.unsqueeze(0)
                                        else:
                                            r_raw = torch.tensor([r_raw])

                                        if torch.is_tensor(r_filter):
                                            if r_filter.dim() == 0:
                                                r_filter = r_filter.unsqueeze(0)
                                        else:
                                            r_filter = torch.tensor([r_filter])

                                        category_ranks_raw.append(r_raw)
                                        category_ranks_filter.append(r_filter)

                                        # 定性记录：记录查询和预测结果
                                        if enable_qualitative_recording and query_recorder:
                                            query_triple = test_snap[idx]

                                            # 获取预测的实体（top-1）
                                            if idx < len(final_score):
                                                scores = final_score[idx]
                                                if torch.is_tensor(scores):
                                                    predicted_entity = torch.argmax(scores).item()
                                                    prediction_score = scores[predicted_entity].item()
                                                else:
                                                    predicted_entity = 0
                                                    prediction_score = 0.0
                                            else:
                                                predicted_entity = 0
                                                prediction_score = 0.0

                                            true_entity = int(query_triple[2])

                                            # 获取rank
                                            if torch.is_tensor(r_filter):
                                                rank_value = r_filter.item() if r_filter.numel() == 1 else r_filter[
                                                    0].item()
                                            else:
                                                rank_value = int(r_filter)

                                            # 记录查询
                                            query_recorder.add_query_record(
                                                hop_category=category,
                                                query_triple=query_triple,
                                                predicted_entity=predicted_entity,
                                                true_entity=true_entity,
                                                rank=rank_value,
                                                score=prediction_score,
                                                timestamp=tc
                                            )

                                if category_ranks_raw and category_ranks_filter:
                                    multihop_collector.add_metrics(category, category_ranks_raw, category_ranks_filter)
                            except Exception as e:
                                print(f"Warning: Error processing category {category}: {e}")
                                continue

                # 更新输入列表
                if args.multi_step:
                    if not args.relation_evaluation:
                        predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
                    if len(predicted_snap):
                        input_list.pop(0)
                        input_list.append(predicted_snap)
                else:
                    input_list.pop(0)
                    input_list.append(test_snap)

    # 计算最终指标
    mrr_raw, hit_raw = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter, hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_raw_inv, hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_inv")
    mrr_filter_inv, hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")

    all_mrr_raw = (mrr_raw + mrr_raw_inv) / 2
    all_mrr_filter = (mrr_filter + mrr_filter_inv) / 2
    all_hit_raw = [(hit_raw[i] + hit_raw_inv[i]) / 2 for i in range(len(hit_raw))]
    all_hit_filter = [(hit_filter[i] + hit_filter_inv[i]) / 2 for i in range(len(hit_filter))]

    # 打印常规结果
    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS")
    print(f"{'=' * 80}")
    print(f"(raw) MRR: {mrr_raw.item():.6f}, Hits@1: {hit_raw[0]:.6f}, "
          f"Hits@3: {hit_raw[1]:.6f}, Hits@10: {hit_raw[2]:.6f}")
    print(f"(filter) MRR: {mrr_filter.item():.6f}, Hits@1: {hit_filter[0]:.6f}, "
          f"Hits@3: {hit_filter[1]:.6f}, Hits@10: {hit_filter[2]:.6f}")
    print(f"(raw_inv) MRR: {mrr_raw_inv.item():.6f}, Hits@1: {hit_raw_inv[0]:.6f}, "
          f"Hits@3: {hit_raw_inv[1]:.6f}, Hits@10: {hit_raw_inv[2]:.6f}")
    print(f"(filter_inv) MRR: {mrr_filter_inv.item():.6f}, Hits@1: {hit_filter_inv[0]:.6f}, "
          f"Hits@3: {hit_filter_inv[1]:.6f}, Hits@10: {hit_filter_inv[2]:.6f}")
    print(f"(all_raw) MRR: {all_mrr_raw.item():.6f}, Hits@1: {all_hit_raw[0]:.6f}, "
          f"Hits@3: {all_hit_raw[1]:.6f}, Hits@10: {all_hit_raw[2]:.6f}")
    print(f"(all_filter) MRR: {all_mrr_filter.item():.6f}, Hits@1: {all_hit_filter[0]:.6f}, "
          f"Hits@3: {all_hit_filter[1]:.6f}, Hits@10: {all_hit_filter[2]:.6f}")

    # 多跳分析结果
    multihop_metrics = None
    if enable_multihop_analysis and multihop_collector:
        multihop_metrics = multihop_collector.compute_final_metrics()
        multihop_collector.print_detailed_report(multihop_metrics)

    # 保存定性记录
    if enable_qualitative_recording and query_recorder:
        query_recorder.print_summary()
        query_recorder.save_records(dataset_name, model_config=str(vars(args)))
        print("\n✓ Qualitative analysis samples have been saved!")

    # 记录指标
    if monitor:
        monitor_metrics = {
            'mrr_raw': all_mrr_raw.item(),
            'mrr_filter': all_mrr_filter.item(),
            'hits@1': all_hit_filter[0],
            'hits@3': all_hit_filter[1],
            'hits@10': all_hit_filter[2]
        }

        # 添加多跳指标
        if multihop_metrics:
            for category in ['1-hop', '2-hop', '3-hop', '4+hop']:
                if category in multihop_metrics:
                    monitor_metrics[f'{category}_mrr'] = multihop_metrics[category]['mrr_filter']
                    monitor_metrics[f'{category}_hits@10'] = multihop_metrics[category]['hits@10_filter']

        monitor.log_metrics(0, monitor_metrics)

    # 保存结果到CSV（包含多跳指标）
    if mode == "test":
        save_test_results(args, all_mrr_raw, all_mrr_filter, all_hit_raw, all_hit_filter,
                          mrr_filter, hit_filter, mrr_filter_inv, hit_filter_inv,
                          multihop_metrics=multihop_metrics)

    return all_mrr_raw, all_mrr_filter


def save_test_results(args, all_mrr_raw, all_mrr_filter, all_hit_raw, all_hit_filter,
                      mrr_filter, hit_filter, mrr_filter_inv, hit_filter_inv,
                      multihop_metrics=None):
    """
    保存测试结果到CSV（增强版，包含多跳分析）

    Args:
        multihop_metrics: 多跳分析的指标字典
    """
    filename = f'./result/{args.dataset}.csv'
    os.makedirs('./result', exist_ok=True)

    # 准备基础数据
    row_data = {
        'encoder': args.encoder,
        'opn': args.opn,
        'pre_type': args.pre_type,
        'use_static': args.add_static_graph,
        'use_cl': args.use_cl,
        'use_dual_stream': args.use_dual_stream,
        'gpu': args.gpu,
        'datetime': datetime.now(),
        'pre_weight': args.pre_weight,
        'train_len': args.train_history_len,
        'test_len': args.test_history_len,
        'temperature': args.temperature,
        'lr': args.lr,
        'n_hidden': args.n_hidden,
        'label_smoothing': args.label_smoothing,
        'dropedge_rate': args.dropedge_rate,
        'edge_sampling_ratio': args.edge_sampling_ratio,
        'relation_weight': args.relation_weight,
        'filter_MRR': float(mrr_filter),
        'filter_H@1': hit_filter[0],
        'filter_H@3': hit_filter[1],
        'filter_H@10': hit_filter[2],
        'filter_inv_MRR': float(mrr_filter_inv),
        'filter_inv_H@1': hit_filter_inv[0],
        'filter_inv_H@3': hit_filter_inv[1],
        'filter_inv_H@10': hit_filter_inv[2],
        'all_MRR': all_mrr_raw.item(),
        'all_H@1': all_hit_raw[0],
        'all_H@3': all_hit_raw[1],
        'all_H@10': all_hit_raw[2],
        'filter_all_MRR': all_mrr_filter.item(),
        'filter_all_H@1': all_hit_filter[0],
        'filter_all_H@3': all_hit_filter[1],
        'filter_all_H@10': all_hit_filter[2]
    }

    # 添加多跳分析指标
    if multihop_metrics:
        for category in ['1-hop', '2-hop', '3-hop', '4+hop']:
            if category in multihop_metrics:
                metrics = multihop_metrics[category]
                prefix = category.replace('-', '_').replace('+', 'plus_')

                # 安全地添加指标
                try:
                    row_data[f'{prefix}_count'] = metrics.get('count', 0)
                    row_data[f'{prefix}_MRR'] = metrics.get('mrr_filter', 0.0)
                    row_data[f'{prefix}_H@1'] = metrics.get('hits@1_filter', 0.0)
                    row_data[f'{prefix}_H@3'] = metrics.get('hits@3_filter', 0.0)
                    row_data[f'{prefix}_H@10'] = metrics.get('hits@10_filter', 0.0)
                except Exception as e:
                    print(f"Warning: Could not add metrics for {category}: {e}")

        # 计算改进指标（安全版本）
        try:
            if ('1-hop' in multihop_metrics and '3-hop' in multihop_metrics and
                    multihop_metrics['1-hop'].get('count', 0) > 0 and
                    multihop_metrics['3-hop'].get('count', 0) > 0):

                mrr_1hop = multihop_metrics['1-hop'].get('mrr_filter', 0.0)
                mrr_3hop = multihop_metrics['3-hop'].get('mrr_filter', 0.0)

                if mrr_1hop > 0:
                    improvement_3hop = ((mrr_3hop / mrr_1hop) - 1) * 100
                    row_data['3hop_vs_1hop_improvement_percent'] = improvement_3hop

            if ('1-hop' in multihop_metrics and '2-hop' in multihop_metrics and
                    multihop_metrics['1-hop'].get('count', 0) > 0 and
                    multihop_metrics['2-hop'].get('count', 0) > 0):

                mrr_1hop = multihop_metrics['1-hop'].get('mrr_filter', 0.0)
                mrr_2hop = multihop_metrics['2-hop'].get('mrr_filter', 0.0)

                if mrr_1hop > 0:
                    improvement_2hop = ((mrr_2hop / mrr_1hop) - 1) * 100
                    row_data['2hop_vs_1hop_improvement_percent'] = improvement_2hop
        except Exception as e:
            print(f"Warning: Could not compute improvement metrics: {e}")

    # 写入CSV
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # 额外保存详细的多跳分析报告
    if multihop_metrics:
        detailed_filename = f'./result/{args.dataset}_multihop_detailed.json'
        with open(detailed_filename, 'w') as f:
            json.dump({
                'timestamp': str(datetime.now()),
                'configuration': vars(args),
                'multihop_metrics': multihop_metrics
            }, f, indent=2)
        print(f"\nDetailed multi-hop analysis saved to: {detailed_filename}")


# ================================
# 主训练函数
# ================================
def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    """实验运行函数"""

    # 更新配置
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # 加载数据
    print("Loading graph data...")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # 加载答案列表
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    # 模型名称
    model_name = "{}-{}-multihop-qualitative-len{}-gpu{}-lr{}-temp{}-pw{}-cl{}-ds{}-ls{}-de{}-esr{}-rw{}-{}".format(
        args.dataset, args.encoder, args.train_history_len, args.gpu, args.lr,
        args.temperature, args.pre_weight, args.use_cl, args.use_dual_stream,
        args.label_smoothing, args.dropedge_rate, args.edge_sampling_ratio,
        args.relation_weight, str(int(time.time()))
    )

    model_state_file = f'./models/{model_name}.pt'
    os.makedirs('./models', exist_ok=True)

    print(f"Model will be saved to: {model_state_file}")

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # 处理静态图
    if args.add_static_graph:
        static_graph_path = DATA_DIR / args.dataset / "e-w-graph.txt"
        static_triples = np.array(_read_triplets_as_list(str(static_graph_path), {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
        if use_cuda:
            static_node_id = static_node_id.cuda(args.gpu)
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # 创建模型
    model = RecurrentRGCN(
        args.decoder,
        args.encoder,
        num_nodes,
        num_rels,
        num_static_rels,
        num_words,
        args.n_hidden,
        args.opn,
        sequence_len=args.train_history_len,
        num_bases=args.n_bases,
        num_basis=args.n_basis,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        self_loop=args.self_loop,
        skip_connect=args.skip_connect,
        layer_norm=args.layer_norm,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        aggregation=args.aggregation,
        weight=args.weight,
        pre_weight=args.pre_weight,
        discount=args.discount,
        angle=args.angle,
        use_static=args.add_static_graph,
        pre_type=args.pre_type,
        use_cl=args.use_cl,
        temperature=args.temperature,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=args.gpu,
        analysis=args.run_analysis,
        edge_sampling_ratio=args.edge_sampling_ratio,
        label_smoothing=args.label_smoothing,
        dropedge_rate=args.dropedge_rate,
        use_adaptive_loss=args.use_adaptive_loss
    )

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # 优化器配置
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=12, verbose=True, min_lr=1e-6
    )

    # 训练监控器
    monitor = TrainingMonitor(log_dir=f"./logs/{args.dataset}")

    # 数据增强器
    augmenter = DataAugmenter(aug_prob=args.aug_prob) if args.use_augmentation else None

    # 自适应训练调度器
    adaptive_scheduler = AdaptiveTrainingScheduler(initial_lr=args.lr, patience=15)

    # 测试模式
    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter = test(
            model, args.train_history_len, train_list + valid_list, test_list,
            num_rels, num_nodes, use_cuda, all_ans_list_test, all_ans_list_r_test,
            model_state_file, static_graph, "test", monitor, args.dataset,
            enable_multihop_analysis=args.enable_multihop_analysis,
            enable_qualitative_recording=args.enable_qualitative_recording
        )
        return mrr_raw, mrr_filter
    elif args.test and not os.path.exists(model_state_file):
        print(f"Model file {model_state_file} not found! Training from scratch...")

    # ================================
    # 训练循环
    # ================================
    print("Starting training...")

    best_mrr = 0
    patience_counter = 0
    max_patience = args.early_stop_patience

    for epoch in range(args.n_epochs):
        model.train()

        losses = []
        losses_e = []
        losses_r = []
        losses_static = []
        losses_aux = []

        idx = [_ for _ in range(len(train_list))]
        if args.shuffle_train:
            random.shuffle(idx)

        with memory_manager():
            for train_sample_num in tqdm(idx, desc=f"Epoch {epoch}"):
                if train_sample_num == 0:
                    continue

                output = train_list[train_sample_num:train_sample_num + 1]

                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0:train_sample_num]
                    tlist = torch.Tensor(list(range(len(input_list)))).cuda()
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:train_sample_num]
                    tlist = torch.Tensor(
                        list(range(train_sample_num - args.train_history_len, train_sample_num))).cuda()

                subgraph_path = DATA_DIR / args.dataset / 'his_graph_for' / f'train_s_r_{train_sample_num}.npy'
                subgraph_inv_path = DATA_DIR / args.dataset / 'his_graph_inv' / f'train_o_r_{train_sample_num}.npy'

                subgraph_arr = np.load(str(subgraph_path))
                subgraph_arr_inv = np.load(str(subgraph_inv_path))

                if augmenter and epoch > 30:
                    if monitor.is_training_stable(threshold=0.005):
                        output[0] = augmenter.augment_triples(output[0], num_rels)

                subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu)
                subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                que_pair = e2r(output[0], num_rels)
                que_pair_inv = e2r(inverse_triples, num_rels)

                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
                                 for snap in input_list]

                triples = torch.from_numpy(output[0]).long().cuda()
                inverse_triples = torch.from_numpy(inverse_triples).long().cuda()

                for id in range(2):
                    if id % 2 == 0:
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(
                            que_pair, subg_snap, train_sample_num, history_glist,
                            triples, static_graph, tlist, input_list, num_nodes, use_cuda
                        )
                    else:
                        loss_e, loss_r, loss_static, loss_cl = model.get_loss(
                            que_pair_inv, subg_snap_inv, train_sample_num, history_glist,
                            inverse_triples, static_graph, tlist, input_list, num_nodes, use_cuda
                        )

                    loss = loss_e + loss_r * args.relation_weight + loss_static + loss_cl * 0.5

                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())
                    losses_aux.append(loss_cl.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                if train_sample_num % 10 == 0:
                    torch.cuda.empty_cache()

        # 记录epoch指标
        epoch_metrics = {
            'loss': np.mean(losses),
            'loss_entity': np.mean(losses_e),
            'loss_relation': np.mean(losses_r),
            'loss_static': np.mean(losses_static),
            'loss_auxiliary': np.mean(losses_aux),
            'loss_variance': np.std(losses)
        }

        monitor.log_metrics(epoch, epoch_metrics)

        print(f"Epoch {epoch:04d} | Loss: {epoch_metrics['loss']:.4f} | "
              f"E: {epoch_metrics['loss_entity']:.4f} | R: {epoch_metrics['loss_relation']:.4f} | "
              f"S: {epoch_metrics['loss_static']:.4f} | Aux: {epoch_metrics['loss_auxiliary']:.4f} | "
              f"Best MRR: {best_mrr:.4f}")

        # 验证
        if epoch > 0 and epoch % args.evaluate_every == 0:
            mrr_raw, mrr_filter = test(
                model, args.train_history_len, train_list, valid_list,
                num_rels, num_nodes, use_cuda, all_ans_list_valid, all_ans_list_r_valid,
                model_state_file, static_graph, mode="train", monitor=monitor, dataset_name=args.dataset,
                enable_multihop_analysis=False,  # 验证时不启用多跳分析以节省时间
                enable_qualitative_recording=False
            )

            scheduler.step(mrr_filter)

            if adaptive_scheduler.should_reduce_lr(mrr_filter):
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= adaptive_scheduler.get_lr_scale()
                    print(f"  Adaptive LR adjustment: {old_lr:.6f} -> {param_group['lr']:.6f}")

            if mrr_filter < best_mrr:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            else:
                patience_counter = 0
                best_mrr = mrr_filter
                monitor.update_best('valid_mrr', best_mrr, epoch)

                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_mrr': best_mrr,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'training_status': model.get_training_status() if args.use_dual_stream else {},
                    'config': vars(args)
                }, model_state_file)
                print(f"  Model saved with MRR: {best_mrr:.4f}")

        torch.cuda.empty_cache()

    print(monitor.get_summary())

    # 最终测试（启用多跳分析和定性记录）
    print("\nRunning final test with multi-hop analysis and qualitative recording...")
    mrr_raw, mrr_filter = test(
        model, args.train_history_len, train_list + valid_list, test_list,
        num_rels, num_nodes, use_cuda, all_ans_list_test, all_ans_list_r_test,
        model_state_file, static_graph, mode="test", monitor=monitor, dataset_name=args.dataset,
        enable_multihop_analysis=args.enable_multihop_analysis,
        enable_qualitative_recording=args.enable_qualitative_recording
    )

    return mrr_raw, mrr_filter


# ================================
# 主函数
# ================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-hop Analysis with Qualitative Recording')

    # 基础参数
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS18', help="dataset to use")
    parser.add_argument("--test", action='store_true', default= False, help="test mode")
    parser.add_argument("--run-analysis", action='store_true', default=False, help="print log info")
    parser.add_argument("--multi-step", action='store_true', default=False, help="multi-step inference")
    parser.add_argument("--topk", type=int, default=10, help="top k for multi-step")

    # 多跳分析参数
    parser.add_argument("--enable-multihop-analysis", action='store_true', default=True,
                        help="enable multi-hop reasoning analysis")

    # 定性记录参数（新增）
    parser.add_argument("--enable-qualitative-recording", action='store_true', default=True,
                        help="enable qualitative recording of queries and predictions")

    # 图结构参数
    parser.add_argument("--add-static-graph", action='store_true', default=True, help="use static graph")
    parser.add_argument("--pre-type", type=str, default="all", help="prediction type")
    parser.add_argument("--use-cl", action='store_true', default=True, help="use contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.03, help="CL temperature")

    # 双流架构参数
    parser.add_argument("--use-dual-stream", action='store_true', default=True, help="use dual-stream architecture")
    parser.add_argument("--edge-sampling-ratio", type=float, default=1, help="edge sampling ratio")
    parser.add_argument("--label-smoothing", type=float, default=0.03, help="label smoothing")
    parser.add_argument("--dropedge-rate", type=float, default=0.1, help="DropEdge rate")
    parser.add_argument("--use-adaptive-loss", action='store_true', default=False, help="use adaptive loss weights")

    # 训练策略参数
    parser.add_argument("--use-augmentation", action='store_true', default=True, help="data augmentation")
    parser.add_argument("--aug-prob", type=float, default=0.05, help="augmentation probability")
    parser.add_argument("--shuffle-train", action='store_true', default=True, help="shuffle training")
    parser.add_argument("--early-stop-patience", type=int, default=4, help="early stopping patience")
    parser.add_argument("--weight-decay", type=float, default=5e-6, help="weight decay")
    parser.add_argument("--relation-weight", type=float, default=0.1, help="relation loss weight")

    # 编码器参数
    parser.add_argument("--encoder", type=str, default="uvrgcn", help="encoder type")
    parser.add_argument("--opn", type=str, default="sub", help="operation for CompGCN")
    parser.add_argument("--n-hidden", type=int, default=200, help="hidden dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="number of layers")
    parser.add_argument("--n-bases", type=int, default=100, help="number of bases")
    parser.add_argument("--n-basis", type=int, default=100, help="number of basis for CompGCN")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--self-loop", action='store_true', default=True, help="add self loop")
    parser.add_argument("--skip-connect", action='store_true', default=False, help="skip connection")
    parser.add_argument("--layer-norm", action='store_true', default=True, help="layer normalization")

    # 解码器参数
    parser.add_argument("--decoder", type=str, default="convtranse", help="decoder type")
    parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout")
    parser.add_argument("--hidden-dropout", type=float, default=0.2, help="hidden dropout")
    parser.add_argument("--feat-dropout", type=float, default=0.2, help="feature dropout")

    # 损失函数参数
    parser.add_argument("--weight", type=float, default=0.5, help="static constraint weight")
    parser.add_argument("--pre-weight", type=float, default=0.9, help="entity prediction weight")
    parser.add_argument("--discount", type=float, default=1, help="discount factor")
    parser.add_argument("--angle", type=int, default=10, help="evolution angle")
    parser.add_argument("--entity-prediction", action='store_true', default=True, help="entity prediction")
    parser.add_argument("--relation-prediction", action='store_true', default=True, help="relation prediction")
    parser.add_argument("--relation-evaluation", action='store_true', default=False, help="relation evaluation")
    parser.add_argument("--aggregation", type=str, default="none", help="aggregation method")

    # 训练参数
    parser.add_argument("--n-epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--evaluate-every", type=int, default=1, help="evaluation frequency")

    # 序列参数
    parser.add_argument("--train-history-len", type=int, default=7, help="training history length")
    parser.add_argument("--test-history-len", type=int, default=7, help="testing history length")

    args = parser.parse_args()
    args.test_history_len = args.train_history_len

    print("=" * 80)
    print("Multi-Hop Analysis with Qualitative Recording Configuration:")
    print("=" * 80)
    for key, value in vars(args).items():
        print(f"{key:30s}: {value}")
    print("=" * 80)
    print("\nKey Features:")
    print("✓ Multi-hop reasoning analysis enabled")
    print("✓ Automatic hop classification (1-hop, 2-hop, 3-hop, 4+hop)")
    print("✓ Detailed performance metrics per hop category")
    print("✓ Qualitative recording of query-answer pairs")
    print("✓ Entity/Relation ID to text mapping")
    print("✓ Export to JSON, CSV, and human-readable formats")
    print("=" * 80 + "\n")

    # 运行实验
    run_experiment(args)