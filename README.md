# Synthetic Bitcoin Transaction Analysis

> Comparative study of real Bitcoin transactions and random graph baselines for blockchain network analysis

**Author:** Mukunthan Sivakumar | IIT Madras  
**Repository:** [github.com/mukunthans/synthetic-btc-analysis](https://github.com/mukunthans/synthetic-btc-analysis)

## Overview

This project analyzes and compares graph properties of Bitcoin transaction networks with random graph models to understand structural characteristics and inform the design of realistic synthetic blockchain data generators.

**Datasets Analyzed:**
- **Elliptic Dataset** - Real Bitcoin transactions with illicit/licit labels
- **Erdős–Rényi Model** - Random graph baseline
- **Barabási–Albert Model** - Scale-free network baseline


## Experiments & Results

Four comprehensive experiments comparing real vs. random graph properties:

| Experiment | Description | Results |
|------------|-------------|---------|
| **Exp 1** | Degree Distribution & Clustering | In-degree/Out-degree stats, clustering coefficients |
| **Exp 2** | Motif Analysis | Motif counts and enrichment patterns |
| **Exp 3** | Temporal Behavior | Time-series patterns and burstiness metrics |
| **Exp 4** | Label Neighborhood | Illicit vs. licit node neighborhood structure |

Results are stored in `experiments/experiments/results/` as CSV files.

## Project Structure

```
synthetic-btc-analysis/
├── data/
│   ├── raw/                       # Original datasets (place data here)
│   └── processed/
│       ├── elliptic/              # Cleaned Elliptic dataset
│       │   ├── cleaning_log.txt   # Data cleaning logs
│       │   ├── nodes.csv          # Node features and labels
│       │   ├── edges.csv          # Transaction edges
│       │   ├── graph.gpickle      # NetworkX graph object
│       │   └── metadata.json      # Dataset statistics
│       └── random/                # Generated random graphs
│           ├── barabasi_albert/   # BA model graphs
│           └── erdos_renyi/       # ER model graphs
├── experiments/experiments/
│   ├── experiments.ipynb          # Main analysis notebook
│   ├── plots/                     # Visualization outputs
│   └── results/                   # Experimental results (CSV)
│       ├── exp1_in_degree_stats.csv
│       ├── exp1_out_degree_stats.csv
│       ├── exp2_motif_counts.csv
│       ├── exp2_motif_enrichment.csv
│       ├── exp3_temporal_summary.csv
│       └── exp4_label_summary.csv
├── src/
│   ├── etl/
│   │   ├── clean_elliptic.py           # Process Elliptic dataset
│   │   ├── generate_random_graphs.py   # Generate random baselines
│   │   └── validate_random_graphs.py   # Validate graph generation
│   └── utils/                          # Helper functions
├── .gitignore
├── requirements.txt
└── README.md
```


## Key Findings

The experiments provide insights into:
- How real Bitcoin transaction graphs differ from random networks
- Structural patterns unique to blockchain transactions
- Temporal dynamics of cryptocurrency flows
- Neighborhood characteristics around illicit activities

These findings inform the design of more realistic synthetic Bitcoin data generators for privacy-preserving blockchain research.

## Dependencies

- Python 3.8+
- pandas, numpy - Data manipulation
- networkx - Graph analysis
- matplotlib, seaborn - Visualization
- jupyter - Interactive analysis

