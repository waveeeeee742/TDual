# Multi-Hop Reasoning Analysis for Temporal Knowledge Graphs

A comprehensive analysis framework for temporal knowledge graph reasoning with multi-hop query classification and qualitative evaluation.

## Features

- **Multi-hop Reasoning Analysis**: Automatically classifies queries by reasoning depth (1-hop, 2-hop, 3-hop, 4+hop)
- **Qualitative Recording**: Records query-prediction pairs with entity/relation name mappings for human-readable analysis
- **Recurrent RGCN**: Temporal knowledge graph reasoning with dual-stream architecture
- **Comprehensive Metrics**: Detailed performance evaluation across different reasoning depths
- **Multiple Export Formats**: Results saved in JSON, CSV, and human-readable text formats

## Usage

### Training

```bash
python main.py --dataset ICEWS18 --gpu 0 --n-epochs 500 --lr 0.001
```

### Testing with Multi-hop Analysis

```bash
python main.py --dataset ICEWS18 --gpu 0 --test \
  --enable-multihop-analysis \
  --enable-qualitative-recording
```

## Key Arguments

**Multi-hop Analysis:**
- `--enable-multihop-analysis`: Enable multi-hop reasoning analysis (default: True)
- `--enable-qualitative-recording`: Enable qualitative recording of queries (default: True)

**Model Configuration:**
- `--encoder`: Encoder type (default: "uvrgcn")
- `--n-hidden`: Hidden dimension (default: 200)
- `--n-layers`: Number of layers (default: 2)
- `--train-history-len`: Training history length (default: 7)

**Training Strategy:**
- `--use-cl`: Use contrastive learning (default: True)
- `--use-dual-stream`: Use dual-stream architecture (default: True)
- `--label-smoothing`: Label smoothing factor (default: 0.03)
- `--dropedge-rate`: DropEdge rate (default: 0.1)

## Output

### Multi-hop Analysis Report
- Performance metrics broken down by hop category
- Comparison of 1-hop vs 2-hop vs 3-hop vs 4+hop queries
- Query distribution statistics

### Qualitative Analysis Files
- `{dataset}_multihop_queries_{timestamp}.json`: Machine-readable query records
- `{dataset}_multihop_queries_{timestamp}.txt`: Human-readable analysis
- `{dataset}_multihop_queries_{timestamp}.csv`: Spreadsheet format for Excel

### Results
- `./result/{dataset}.csv`: Overall performance metrics with multi-hop breakdown
- `./result/{dataset}_multihop_detailed.json`: Detailed multi-hop analysis

## Data Structure

Expected directory structure:
```
data/
  {dataset}/
    entity2id.txt
    relation2id.txt
    word2id.txt
    train.txt
    valid.txt
    test.txt
    his_dict/
    his_graph_for/
    his_graph_inv/
```

## Citation

If you use this code, please cite the original work appropriately.
