# Synthetic Bitcoin Transaction Analysis

> Comparative study of real, synthetic, and random graph structures for Bitcoin transaction networks

**Author:** Mukunthan Sivakumar | IIT Madras  
**Status:** In Progress

## Overview

This project analyzes and compares graph properties across three types of transaction networks to understand what makes Bitcoin transaction graphs unique and how to generate realistic synthetic data.

**Datasets Used:**
- **Elliptic Dataset** - Real Bitcoin transactions with illicit labels
- **IBM AML Dataset** - Synthetic bank transaction network
- **Random Graphs** - Erdős–Rényi and Barabási–Albert baselines

## Experiments

The analysis consists of four key experiments:

1. **Degree Distribution & Clustering** - Analyze node connectivity patterns and local clustering
2. **Motif Analysis** - Identify common subgraph patterns (triangles, chains, etc.)
3. **Temporal Behavior** - Study time-series patterns and transaction burstiness
4. **Label Neighborhood Structure** - Compare graph structure around illicit vs. licit nodes

## Project Structure

```
synthetic-btc-analysis/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned and preprocessed data
├── experiments/
│   ├── exp1_degree_clustering/
│   ├── exp2_motifs/
│   ├── exp3_temporal/
│   └── exp4_label_structure/
├── src/
│   └── etl/              # Data processing scripts
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/mukunthans/synthetic-btc-analysis.git
cd synthetic-btc-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run ETL pipeline
python src/etl/process_data.py

# Run experiments
jupyter notebook experiments/exp1_degree_clustering/analysis.ipynb
```

## Goals

- Build unified ETL pipeline for all datasets
- Compare structural, temporal, and motif properties
- Identify key differences between real and synthetic transaction graphs
- Provide insights for designing realistic synthetic Bitcoin data generators

## Dependencies

- Python 3.8+
- pandas, numpy
- networkx
- matplotlib, seaborn
- jupyter



