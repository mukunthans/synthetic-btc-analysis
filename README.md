Synthetic Bitcoin Transaction Analysis
Experiments on Real, Synthetic, and Random Graphs (BTC Transaction Graph Study)

Author: Mukunthan Sivakumar
Status: In Progress

📌 Overview

This repository contains code, notebooks, and analysis for four graph-based experiments using:

Elliptic Bitcoin Dataset (real Bitcoin transactions)

IBM AML Synthetic Dataset (bank-style synthetic transactions)

Random Graph Baselines (Erdős–Rényi & Barabási–Albert graphs)

The goal is to compare graph structure, motifs, temporal patterns, and illicit label propagation to ultimately design a state-of-the-art synthetic Bitcoin transaction generator.

📁 Project Structure
synthetic-btc-analysis/
├── data/
│   ├── raw/
│   ├── processed/
├── experiments/
│   ├── exp1_degree_clustering/
│   ├── exp2_motifs/
│   ├── exp3_temporal/
│   ├── exp4_label_structure/
├── notebooks/
├── src/
│   ├── utils/
│   ├── config.py
├── .gitignore
├── README.md
└── requirements.txt

🧪 Experiments

Degree Distribution & Clustering Patterns

Motif Analysis (local transaction patterns)

Temporal Behavior & Burstiness

Illicit Label Neighborhood Structure

⚙️ Setup
Create & activate virtual environment:
python3 -m venv venv
source venv/bin/activate

Install libraries:
pip install pandas numpy networkx matplotlib seaborn tqdm jupyter


📈 Goals

Build a unified ETL pipeline

Run 4 experiments on all datasets

Compare structural + temporal + motif + label properties

Provide insights for building a synthetic, realistic Bitcoin data generator

🚀 Work in Progress