#!/usr/bin/env python3
"""
Elliptic Bitcoin Dataset Cleaning Script

This script cleans and processes the Elliptic Bitcoin dataset for graph analysis.
It loads transaction features, labels, and edges, applies cleaning rules, and
outputs structured files ready for analysis.

Author: Mukunthan Sivakumar
Date: 2025-12-17
"""

import pandas as pd
import networkx as nx
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import sys


# Configuration
RAW_DATA_DIR = Path("data/raw/elliptic_bitcoin_dataset")
PROCESSED_DATA_DIR = Path("data/processed/elliptic")
FEATURES_FILE = RAW_DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = RAW_DATA_DIR / "elliptic_txs_classes.csv"
EDGELIST_FILE = RAW_DATA_DIR / "elliptic_txs_edgelist.csv"

# Output files
NODES_FILE = PROCESSED_DATA_DIR / "nodes.csv"
EDGES_FILE = PROCESSED_DATA_DIR / "edges.csv"
GRAPH_FILE = PROCESSED_DATA_DIR / "graph.gpickle"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"
LOG_FILE = PROCESSED_DATA_DIR / "cleaning_log.txt"


class CleaningStats:
    """Track statistics during data cleaning process."""

    def __init__(self):
        self.initial_features_count = 0
        self.initial_classes_count = 0
        self.initial_edges_count = 0
        self.duplicate_nodes_removed = 0
        self.duplicate_edges_removed = 0
        self.null_features_removed = 0
        self.null_classes_removed = 0
        self.null_edges_removed = 0
        self.invalid_classes_removed = 0
        self.self_loops_removed = 0
        self.invalid_edges_removed = 0
        self.final_node_count = 0
        self.final_edge_count = 0
        self.num_licit = 0
        self.num_illicit = 0
        self.num_unknown = 0
        self.start_time = None
        self.end_time = None
        self.warnings = []

    def add_warning(self, message: str):
        """Add a warning message to the log."""
        self.warnings.append(message)
        logging.warning(message)

    def processing_time(self) -> float:
        """Calculate processing time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


def setup_logging():
    """Set up logging to both file and console."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("=" * 80)
    logging.info("ELLIPTIC BITCOIN DATASET CLEANING SCRIPT")
    logging.info("=" * 80)


def validate_input_files() -> bool:
    """
    Validate that all required input files exist.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    logging.info("Validating input files...")

    files = {
        "Features": FEATURES_FILE,
        "Classes": CLASSES_FILE,
        "Edgelist": EDGELIST_FILE
    }

    all_exist = True
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logging.info(f"  ✓ {name} file found: {path} ({size_mb:.1f} MB)")
        else:
            logging.error(f"  ✗ {name} file NOT FOUND: {path}")
            all_exist = False

    return all_exist


def load_features(stats: CleaningStats) -> pd.DataFrame:
    """
    Load transaction features (only txId and time_step columns).

    Args:
        stats: CleaningStats object to track statistics.

    Returns:
        pd.DataFrame: Cleaned features dataframe with columns [node_id, timestamp].
    """
    logging.info("\n" + "=" * 80)
    logging.info("STEP 1: Loading Features File")
    logging.info("=" * 80)

    # Load only first two columns for memory efficiency
    logging.info(f"Loading {FEATURES_FILE} (columns 0-1 only)...")
    df = pd.read_csv(
        FEATURES_FILE,
        usecols=[0, 1],
        header=None,
        names=['node_id', 'timestamp'],
        dtype={'node_id': str, 'timestamp': int}
    )

    stats.initial_features_count = len(df)
    logging.info(f"  Initial row count: {stats.initial_features_count:,}")

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        stats.null_features_removed = null_counts.sum()
        logging.warning(f"  Found {stats.null_features_removed} null values:")
        logging.warning(f"    - node_id: {null_counts['node_id']}")
        logging.warning(f"    - timestamp: {null_counts['timestamp']}")
        df = df.dropna()
        logging.info(f"  Dropped rows with null values: {stats.null_features_removed}")

    # Check for duplicate transaction IDs
    duplicates = df['node_id'].duplicated()
    if duplicates.any():
        stats.duplicate_nodes_removed = duplicates.sum()
        logging.warning(f"  Found {stats.duplicate_nodes_removed} duplicate node_ids")
        df = df.drop_duplicates(subset=['node_id'], keep='first')
        logging.info(f"  Kept first occurrence, dropped duplicates")

    # Validate data
    logging.info("  Validating data types and ranges...")
    if df['timestamp'].min() < 0:
        stats.add_warning("Found negative timestamp values!")

    logging.info(f"  Final features count: {len(df):,}")
    logging.info(f"  Timestamp range: [{df['timestamp'].min()}, {df['timestamp'].max()}]")

    return df


def load_classes(stats: CleaningStats) -> pd.DataFrame:
    """
    Load transaction classes and map to binary labels.

    Label mapping:
    - '1' (illicit) -> 1
    - '2' (licit) -> 0
    - 'unknown' -> None

    Args:
        stats: CleaningStats object to track statistics.

    Returns:
        pd.DataFrame: Cleaned classes dataframe with columns [node_id, label].
    """
    logging.info("\n" + "=" * 80)
    logging.info("STEP 2: Loading Classes File")
    logging.info("=" * 80)

    logging.info(f"Loading {CLASSES_FILE}...")
    df = pd.read_csv(
        CLASSES_FILE,
        dtype={'txId': str, 'class': str}
    )

    stats.initial_classes_count = len(df)
    logging.info(f"  Initial row count: {stats.initial_classes_count:,}")

    # Rename column
    df = df.rename(columns={'txId': 'node_id'})

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        stats.null_classes_removed = null_counts.sum()
        logging.warning(f"  Found {stats.null_classes_removed} null values")
        df = df.dropna()

    # Count original class distribution
    class_dist = df['class'].value_counts()
    logging.info("  Original class distribution:")
    for cls, count in class_dist.items():
        logging.info(f"    - {cls}: {count:,}")

    # Apply label mapping: 1 -> 1 (illicit), 2 -> 0 (licit), unknown -> None
    def map_label(cls: str) -> Optional[int]:
        if cls == '1':
            return 1  # illicit
        elif cls == '2':
            return 0  # licit
        elif cls == 'unknown':
            return None
        else:
            return -999  # Invalid marker

    df['label'] = df['class'].apply(map_label)

    # Check for invalid classes
    invalid_mask = df['label'] == -999
    if invalid_mask.any():
        stats.invalid_classes_removed = invalid_mask.sum()
        invalid_values = df.loc[invalid_mask, 'class'].unique()
        logging.warning(f"  Found {stats.invalid_classes_removed} invalid class values: {invalid_values}")
        df = df[~invalid_mask]

    # Drop original class column
    df = df.drop(columns=['class'])

    logging.info(f"  Final classes count: {len(df):,}")
    logging.info("  Mapped label distribution:")
    logging.info(f"    - 0 (licit): {(df['label'] == 0).sum():,}")
    logging.info(f"    - 1 (illicit): {(df['label'] == 1).sum():,}")
    logging.info(f"    - None (unknown): {df['label'].isnull().sum():,}")

    return df


def merge_nodes(features_df: pd.DataFrame, classes_df: pd.DataFrame,
                stats: CleaningStats) -> pd.DataFrame:
    """
    Merge features and classes to create final nodes dataframe.

    Args:
        features_df: Features dataframe with [node_id, timestamp].
        classes_df: Classes dataframe with [node_id, label].
        stats: CleaningStats object to track statistics.

    Returns:
        pd.DataFrame: Final nodes dataframe with [node_id, timestamp, label, is_labeled].
    """
    logging.info("\n" + "=" * 80)
    logging.info("STEP 3: Merging Nodes")
    logging.info("=" * 80)

    logging.info("Performing LEFT JOIN on node_id...")
    nodes_df = features_df.merge(classes_df, on='node_id', how='left')

    # Add is_labeled column
    nodes_df['is_labeled'] = nodes_df['label'].notna()

    # Convert label to nullable int (handle None values)
    nodes_df['label'] = nodes_df['label'].astype('Int64')

    stats.final_node_count = len(nodes_df)
    stats.num_licit = (nodes_df['label'] == 0).sum()
    stats.num_illicit = (nodes_df['label'] == 1).sum()
    stats.num_unknown = nodes_df['label'].isna().sum()

    logging.info(f"  Total nodes: {stats.final_node_count:,}")
    logging.info(f"  Labeled nodes: {(stats.num_licit + stats.num_illicit):,}")
    logging.info(f"    - Licit (0): {stats.num_licit:,}")
    logging.info(f"    - Illicit (1): {stats.num_illicit:,}")
    logging.info(f"  Unknown nodes: {stats.num_unknown:,}")

    # Verify no nulls in critical columns
    if nodes_df['node_id'].isnull().any() or nodes_df['timestamp'].isnull().any():
        stats.add_warning("Found null values in node_id or timestamp after merge!")

    return nodes_df


def load_and_clean_edges(nodes_df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """
    Load and clean edge list.

    Args:
        nodes_df: Nodes dataframe with valid node_ids.
        stats: CleaningStats object to track statistics.

    Returns:
        pd.DataFrame: Cleaned edges dataframe with [source, target].
    """
    logging.info("\n" + "=" * 80)
    logging.info("STEP 4: Loading and Cleaning Edges")
    logging.info("=" * 80)

    logging.info(f"Loading {EDGELIST_FILE}...")
    df = pd.read_csv(
        EDGELIST_FILE,
        dtype={'txId1': str, 'txId2': str}
    )

    stats.initial_edges_count = len(df)
    logging.info(f"  Initial edge count: {stats.initial_edges_count:,}")

    # Rename columns
    df = df.rename(columns={'txId1': 'source', 'txId2': 'target'})

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        stats.null_edges_removed = null_counts.sum()
        logging.warning(f"  Found {stats.null_edges_removed} null values:")
        logging.warning(f"    - source: {null_counts['source']}")
        logging.warning(f"    - target: {null_counts['target']}")
        df = df.dropna()
        logging.info(f"  Dropped rows with null values: {stats.null_edges_removed}")

    # Remove duplicate edges
    duplicates = df.duplicated()
    if duplicates.any():
        stats.duplicate_edges_removed = duplicates.sum()
        logging.warning(f"  Found {stats.duplicate_edges_removed} duplicate edges")
        df = df.drop_duplicates(keep='first')
        logging.info(f"  Kept first occurrence, dropped duplicates")

    # Remove self-loops
    self_loops = df['source'] == df['target']
    if self_loops.any():
        stats.self_loops_removed = self_loops.sum()
        logging.warning(f"  Found {stats.self_loops_removed} self-loops")
        df = df[~self_loops]
        logging.info(f"  Removed all self-loops")

    # Validate edges against node list
    logging.info("  Validating edge endpoints against node list...")
    valid_nodes = set(nodes_df['node_id'])

    source_valid = df['source'].isin(valid_nodes)
    target_valid = df['target'].isin(valid_nodes)
    both_valid = source_valid & target_valid

    invalid_count = (~both_valid).sum()
    if invalid_count > 0:
        stats.invalid_edges_removed = invalid_count

        # Log detailed breakdown
        source_invalid = (~source_valid).sum()
        target_invalid = (~target_valid).sum()

        logging.warning(f"  Found {invalid_count} edges with invalid endpoints:")
        logging.warning(f"    - Invalid source only: {source_invalid - (source_invalid - both_valid.sum())}")
        logging.warning(f"    - Invalid target only: {target_invalid - (target_invalid - both_valid.sum())}")
        logging.warning(f"    - Both invalid: {(~source_valid & ~target_valid).sum()}")

        df = df[both_valid]
        logging.info(f"  Removed {invalid_count} invalid edges")
    else:
        logging.info("  ✓ All edges have valid endpoints")

    stats.final_edge_count = len(df)
    logging.info(f"  Final edge count: {stats.final_edge_count:,}")

    return df


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build NetworkX directed graph from nodes and edges.

    Args:
        nodes_df: Nodes dataframe with [node_id, timestamp, label, is_labeled].
        edges_df: Edges dataframe with [source, target].

    Returns:
        nx.DiGraph: Directed graph with node attributes.
    """
    logging.info("\n" + "=" * 80)
    logging.info("STEP 5: Building NetworkX Graph")
    logging.info("=" * 80)

    logging.info("Creating directed graph...")
    G = nx.DiGraph()

    # Add nodes with attributes
    logging.info("  Adding nodes with attributes...")
    for _, row in nodes_df.iterrows():
        G.add_node(
            row['node_id'],
            timestamp=int(row['timestamp']),
            label=int(row['label']) if pd.notna(row['label']) else None,
            is_labeled=bool(row['is_labeled'])
        )

    logging.info(f"  Added {G.number_of_nodes():,} nodes")

    # Add edges
    logging.info("  Adding edges...")
    edges = [(row['source'], row['target']) for _, row in edges_df.iterrows()]
    G.add_edges_from(edges)

    logging.info(f"  Added {G.number_of_edges():,} edges")

    # Verify graph properties
    logging.info("  Verifying graph properties...")
    num_self_loops = nx.number_of_selfloops(G)
    if num_self_loops > 0:
        logging.warning(f"  Graph contains {num_self_loops} self-loops!")
    else:
        logging.info("  ✓ No self-loops")

    logging.info(f"  Graph is directed: {G.is_directed()}")
    logging.info(f"  Graph is multigraph: {G.is_multigraph()}")

    return G


def compute_metadata(G: nx.DiGraph, nodes_df: pd.DataFrame,
                     edges_df: pd.DataFrame, stats: CleaningStats) -> Dict:
    """
    Compute metadata statistics for the processed graph.

    Args:
        G: NetworkX directed graph.
        nodes_df: Nodes dataframe.
        edges_df: Edges dataframe.
        stats: CleaningStats object.

    Returns:
        Dict: Metadata dictionary with graph statistics.
    """
    logging.info("\n" + "=" * 80)
    logging.info("STEP 6: Computing Metadata")
    logging.info("=" * 80)

    logging.info("Calculating graph statistics...")

    # Basic counts
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Label distribution
    num_labeled = stats.num_licit + stats.num_illicit

    # Time range
    time_range = [int(nodes_df['timestamp'].min()), int(nodes_df['timestamp'].max())]

    # Average degree
    if num_nodes > 0:
        avg_degree = (2 * num_edges) / num_nodes  # Undirected degree equivalent
    else:
        avg_degree = 0.0

    # Self-loops
    num_self_loops = nx.number_of_selfloops(G)

    # Isolated nodes
    isolated_nodes = list(nx.isolates(G))
    num_isolated = len(isolated_nodes)

    # Connectivity (for directed graph, use weakly connected)
    logging.info("  Checking connectivity (this may take a moment)...")
    is_weakly_connected = nx.is_weakly_connected(G)
    num_components = nx.number_weakly_connected_components(G)

    metadata = {
        "dataset_name": "elliptic",
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "num_labeled": int(num_labeled),
        "num_licit": int(stats.num_licit),
        "num_illicit": int(stats.num_illicit),
        "num_unknown": int(stats.num_unknown),
        "time_range": [int(time_range[0]), int(time_range[1])],
        "avg_degree": round(float(avg_degree), 2),
        "num_self_loops": int(num_self_loops),
        "num_isolated_nodes": int(num_isolated),
        "is_connected": bool(is_weakly_connected),
        "num_connected_components": int(num_components)
    }

    logging.info("  Metadata summary:")
    for key, value in metadata.items():
        logging.info(f"    - {key}: {value}")

    return metadata


def save_outputs(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                G: nx.DiGraph, metadata: Dict):
    """
    Save all output files.

    Args:
        nodes_df: Nodes dataframe.
        edges_df: Edges dataframe.
        G: NetworkX graph.
        metadata: Metadata dictionary.
    """
    logging.info("\n" + "=" * 80)
    logging.info("STEP 7: Saving Outputs")
    logging.info("=" * 80)

    # Prepare nodes dataframe for output
    nodes_output = nodes_df.copy()
    # Replace pandas NA with empty string for CSV output
    nodes_output['label'] = nodes_output['label'].apply(
        lambda x: int(x) if pd.notna(x) else ''
    )
    nodes_output['is_labeled'] = nodes_output['is_labeled'].apply(
        lambda x: 'true' if x else 'false'
    )

    # Save nodes.csv
    logging.info(f"  Saving {NODES_FILE}...")
    nodes_output.to_csv(NODES_FILE, index=False)
    logging.info(f"    ✓ Saved {len(nodes_output):,} nodes")

    # Save edges.csv
    logging.info(f"  Saving {EDGES_FILE}...")
    edges_df.to_csv(EDGES_FILE, index=False)
    logging.info(f"    ✓ Saved {len(edges_df):,} edges")

    # Save graph.gpickle
    logging.info(f"  Saving {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    logging.info(f"    ✓ Saved NetworkX graph")

    # Save metadata.json
    logging.info(f"  Saving {METADATA_FILE}...")
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"    ✓ Saved metadata")

    logging.info("\n  All outputs saved successfully!")


def print_summary(stats: CleaningStats, metadata: Dict):
    """
    Print a comprehensive summary table.

    Args:
        stats: CleaningStats object with cleaning statistics.
        metadata: Metadata dictionary with final statistics.
    """
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)

    # Initial counts
    print("\nINITIAL COUNTS:")
    print(f"  Features (nodes):  {stats.initial_features_count:>10,}")
    print(f"  Classes:           {stats.initial_classes_count:>10,}")
    print(f"  Edges:             {stats.initial_edges_count:>10,}")

    # Dropped counts
    print("\nDROPPED ITEMS:")
    print(f"  Duplicate nodes:   {stats.duplicate_nodes_removed:>10,}")
    print(f"  Null features:     {stats.null_features_removed:>10,}")
    print(f"  Null classes:      {stats.null_classes_removed:>10,}")
    print(f"  Invalid classes:   {stats.invalid_classes_removed:>10,}")
    print(f"  Duplicate edges:   {stats.duplicate_edges_removed:>10,}")
    print(f"  Null edges:        {stats.null_edges_removed:>10,}")
    print(f"  Self-loops:        {stats.self_loops_removed:>10,}")
    print(f"  Invalid edges:     {stats.invalid_edges_removed:>10,}")

    total_dropped = (stats.duplicate_nodes_removed + stats.null_features_removed +
                    stats.null_classes_removed + stats.invalid_classes_removed +
                    stats.duplicate_edges_removed + stats.null_edges_removed +
                    stats.self_loops_removed + stats.invalid_edges_removed)
    print(f"  {'TOTAL DROPPED:':20} {total_dropped:>10,}")

    # Final counts
    print("\nFINAL COUNTS:")
    print(f"  Nodes:             {metadata['num_nodes']:>10,}")
    print(f"  Edges:             {metadata['num_edges']:>10,}")
    print(f"  Isolated nodes:    {metadata['num_isolated_nodes']:>10,}")
    print(f"  Connected comp.:   {metadata['num_connected_components']:>10,}")

    # Label distribution
    print("\nLABEL DISTRIBUTION:")
    print(f"  Labeled total:     {metadata['num_labeled']:>10,} ({100*metadata['num_labeled']/metadata['num_nodes']:.1f}%)")
    print(f"    - Licit (0):     {metadata['num_licit']:>10,} ({100*metadata['num_licit']/metadata['num_nodes']:.1f}%)")
    print(f"    - Illicit (1):   {metadata['num_illicit']:>10,} ({100*metadata['num_illicit']/metadata['num_nodes']:.1f}%)")
    print(f"  Unknown:           {metadata['num_unknown']:>10,} ({100*metadata['num_unknown']/metadata['num_nodes']:.1f}%)")

    # Graph properties
    print("\nGRAPH PROPERTIES:")
    print(f"  Time range:        {metadata['time_range'][0]} - {metadata['time_range'][1]}")
    print(f"  Average degree:    {metadata['avg_degree']:>10.2f}")
    print(f"  Is connected:      {metadata['is_connected']}")

    # File locations
    print("\nOUTPUT FILES:")
    print(f"  Nodes:             {NODES_FILE}")
    print(f"  Edges:             {EDGES_FILE}")
    print(f"  Graph:             {GRAPH_FILE}")
    print(f"  Metadata:          {METADATA_FILE}")
    print(f"  Log:               {LOG_FILE}")

    # Processing time
    print(f"\nProcessing time:     {stats.processing_time():.2f} seconds")

    # Warnings
    if stats.warnings:
        print(f"\nWARNINGS: {len(stats.warnings)}")
        for warning in stats.warnings:
            print(f"  ⚠ {warning}")
    else:
        print("\n✓ No warnings")

    print("\n" + "=" * 80)
    print("✓ Cleaning completed successfully!")
    print("=" * 80 + "\n")


def main():
    """Main execution function."""
    # Initialize
    setup_logging()
    stats = CleaningStats()
    stats.start_time = datetime.now()

    try:
        # Validate input files
        if not validate_input_files():
            logging.error("Input file validation failed. Exiting.")
            sys.exit(1)

        # Load and clean data
        features_df = load_features(stats)
        classes_df = load_classes(stats)
        nodes_df = merge_nodes(features_df, classes_df, stats)
        edges_df = load_and_clean_edges(nodes_df, stats)

        # Build graph
        G = build_graph(nodes_df, edges_df)

        # Compute metadata
        metadata = compute_metadata(G, nodes_df, edges_df, stats)

        # Save outputs
        save_outputs(nodes_df, edges_df, G, metadata)

        # Record end time
        stats.end_time = datetime.now()

        # Print summary
        print_summary(stats, metadata)

        logging.info("Processing completed successfully!")

    except Exception as e:
        logging.error(f"ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
