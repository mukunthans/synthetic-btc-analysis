"""
Validation Script for Random Graphs
Verifies that generated random graphs match expected schema and properties
"""

import pandas as pd
import networkx as nx
import pickle
import json
from pathlib import Path


def validate_graph(graph_dir, graph_type):
    """
    Validate a random graph directory
    
    Args:
        graph_dir: Path to graph directory
        graph_type: 'erdos_renyi' or 'barabasi_albert'
    
    Returns:
        Boolean indicating if validation passed
    """
    print(f"\n{'=' * 80}")
    print(f"VALIDATING {graph_type.upper()} GRAPH")
    print(f"{'=' * 80}")
    
    graph_dir = Path(graph_dir)
    all_passed = True
    
    # Check 1: All files exist
    print("\n1. Checking file existence...")
    required_files = ['nodes.csv', 'edges.csv', 'graph.gpickle', 'metadata.json', 'generation_log.txt']
    for filename in required_files:
        filepath = graph_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"   ✓ {filename} exists ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ {filename} MISSING")
            all_passed = False
    
    # Check 2: Load and validate nodes.csv
    print("\n2. Validating nodes.csv...")
    try:
        nodes_df = pd.read_csv(graph_dir / 'nodes.csv')
        print(f"   ✓ Loaded: {len(nodes_df):,} rows")
        
        # Check schema
        required_cols = ['node_id', 'timestamp', 'label', 'is_labeled']
        for col in required_cols:
            if col in nodes_df.columns:
                print(f"   ✓ Column '{col}' present")
            else:
                print(f"   ✗ Column '{col}' MISSING")
                all_passed = False
        
        # Check data types
        if nodes_df['node_id'].dtype == 'object':
            print(f"   ✓ node_id is string type")
        else:
            print(f"   ⚠ node_id type: {nodes_df['node_id'].dtype} (expected string)")
        
        if nodes_df['timestamp'].dtype in ['int64', 'int32']:
            print(f"   ✓ timestamp is integer type")
        else:
            print(f"   ⚠ timestamp type: {nodes_df['timestamp'].dtype} (expected int)")
        
        # Check for nulls in critical columns
        null_ids = nodes_df['node_id'].isnull().sum()
        null_timestamps = nodes_df['timestamp'].isnull().sum()
        
        if null_ids == 0:
            print(f"   ✓ No null node_ids")
        else:
            print(f"   ✗ Found {null_ids} null node_ids")
            all_passed = False
        
        if null_timestamps == 0:
            print(f"   ✓ No null timestamps")
        else:
            print(f"   ✗ Found {null_timestamps} null timestamps")
            all_passed = False
        
        # Check label values (should all be None/NaN for random baseline)
        labeled_count = nodes_df['is_labeled'].sum()
        if labeled_count == 0:
            print(f"   ✓ All nodes unlabeled (baseline)")
        else:
            print(f"   ⚠ Found {labeled_count} labeled nodes (expected 0 for baseline)")
        
    except Exception as e:
        print(f"   ✗ Error loading nodes.csv: {e}")
        all_passed = False
    
    # Check 3: Load and validate edges.csv
    print("\n3. Validating edges.csv...")
    try:
        edges_df = pd.read_csv(graph_dir / 'edges.csv')
        print(f"   ✓ Loaded: {len(edges_df):,} rows")
        
        # Check schema
        if 'source' in edges_df.columns and 'target' in edges_df.columns:
            print(f"   ✓ Columns 'source' and 'target' present")
        else:
            print(f"   ✗ Missing source/target columns")
            all_passed = False
        
        # Check for nulls
        null_sources = edges_df['source'].isnull().sum()
        null_targets = edges_df['target'].isnull().sum()
        
        if null_sources == 0 and null_targets == 0:
            print(f"   ✓ No null values in edges")
        else:
            print(f"   ✗ Found {null_sources} null sources, {null_targets} null targets")
            all_passed = False
        
        # Check if all edge endpoints exist in nodes
        node_ids = set(nodes_df['node_id'].astype(str))
        edge_sources = set(edges_df['source'].astype(str))
        edge_targets = set(edges_df['target'].astype(str))
        
        invalid_sources = edge_sources - node_ids
        invalid_targets = edge_targets - node_ids
        
        if len(invalid_sources) == 0 and len(invalid_targets) == 0:
            print(f"   ✓ All edge endpoints exist in nodes")
        else:
            print(f"   ✗ Invalid sources: {len(invalid_sources)}, Invalid targets: {len(invalid_targets)}")
            all_passed = False
        
    except Exception as e:
        print(f"   ✗ Error loading edges.csv: {e}")
        all_passed = False
    
    # Check 4: Load and validate graph.gpickle
    print("\n4. Validating graph.gpickle...")
    try:
        with open(graph_dir / 'graph.gpickle', 'rb') as f:
            G = pickle.load(f)
        
        print(f"   ✓ Loaded NetworkX graph")
        print(f"   ✓ Type: {type(G).__name__}")
        
        if isinstance(G, nx.DiGraph):
            print(f"   ✓ Is directed graph (DiGraph)")
        else:
            print(f"   ⚠ Not a DiGraph: {type(G)}")
        
        print(f"   ✓ Nodes: {G.number_of_nodes():,}")
        print(f"   ✓ Edges: {G.number_of_edges():,}")
        
        # Check node attributes
        sample_node = list(G.nodes())[0]
        attrs = G.nodes[sample_node]
        
        required_attrs = ['node_id', 'timestamp', 'label', 'is_labeled']
        for attr in required_attrs:
            if attr in attrs:
                print(f"   ✓ Node attribute '{attr}' present")
            else:
                print(f"   ✗ Node attribute '{attr}' MISSING")
                all_passed = False
        
        # Compare counts
        if G.number_of_nodes() == len(nodes_df):
            print(f"   ✓ Node count matches CSV")
        else:
            print(f"   ✗ Node count mismatch: graph={G.number_of_nodes()}, csv={len(nodes_df)}")
            all_passed = False
        
        if G.number_of_edges() == len(edges_df):
            print(f"   ✓ Edge count matches CSV")
        else:
            print(f"   ✗ Edge count mismatch: graph={G.number_of_edges()}, csv={len(edges_df)}")
            all_passed = False
        
    except Exception as e:
        print(f"   ✗ Error loading graph.gpickle: {e}")
        all_passed = False
    
    # Check 5: Load and validate metadata.json
    print("\n5. Validating metadata.json...")
    try:
        with open(graph_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"   ✓ Loaded metadata")
        
        required_fields = ['dataset_name', 'graph_type', 'num_nodes', 'num_edges', 
                          'time_range', 'avg_in_degree', 'avg_out_degree']
        for field in required_fields:
            if field in metadata:
                print(f"   ✓ Field '{field}': {metadata[field]}")
            else:
                print(f"   ✗ Field '{field}' MISSING")
                all_passed = False
        
    except Exception as e:
        print(f"   ✗ Error loading metadata.json: {e}")
        all_passed = False
    
    # Final verdict
    print(f"\n{'=' * 80}")
    if all_passed:
        print(f"✅ {graph_type.upper()} VALIDATION PASSED")
    else:
        print(f"❌ {graph_type.upper()} VALIDATION FAILED")
    print(f"{'=' * 80}")
    
    return all_passed


def main():
    """Validate all random graphs"""
    print("\n" + "=" * 80)
    print("RANDOM GRAPH VALIDATION")
    print("=" * 80)
    
    base_dir = Path("data/processed/random")
    
    # Validate Erdős–Rényi
    er_passed = validate_graph(base_dir / "erdos_renyi", "erdos_renyi")
    
    # Validate Barabási–Albert
    ba_passed = validate_graph(base_dir / "barabasi_albert", "barabasi_albert")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Erdős–Rényi:     {'✅ PASSED' if er_passed else '❌ FAILED'}")
    print(f"Barabási–Albert: {'✅ PASSED' if ba_passed else '❌ FAILED'}")
    
    if er_passed and ba_passed:
        print("\n✅ ALL VALIDATIONS PASSED - Ready for experiments!")
    else:
        print("\n❌ SOME VALIDATIONS FAILED - Please check errors above")
    
    print("=" * 80)


if __name__ == "__main__":
    main()