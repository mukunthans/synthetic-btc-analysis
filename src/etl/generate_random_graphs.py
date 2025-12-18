"""
Random Graph Generation for Bitcoin Transaction Analysis
Generates Erdős–Rényi and Barabási–Albert graphs matching Elliptic dataset size
"""

import networkx as nx
import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RandomGraphGenerator:
    """
    Generate random baseline graphs for comparison with Bitcoin transaction graph
    """
    
    def __init__(self, elliptic_metadata_path="data/processed/elliptic/metadata.json",
                 output_base_path="data/processed/random"):
        """
        Initialize random graph generator
        
        Args:
            elliptic_metadata_path: Path to Elliptic metadata for size matching
            output_base_path: Base path for random graph outputs
        """
        self.elliptic_metadata_path = Path(elliptic_metadata_path)
        self.output_base_path = Path(output_base_path)
        
        # Load Elliptic metadata to match size
        self.elliptic_stats = self._load_elliptic_stats()
        self.n_nodes = self.elliptic_stats['num_nodes']
        self.n_edges = self.elliptic_stats['num_edges']
        self.time_range = self.elliptic_stats['time_range']
        
        logger.info(f"Random Graph Generator initialized")
        logger.info(f"Target size: {self.n_nodes:,} nodes, {self.n_edges:,} edges")
    
    def _load_elliptic_stats(self):
        """Load Elliptic dataset statistics"""
        logger.info(f"Loading Elliptic metadata from: {self.elliptic_metadata_path}")
        
        if not self.elliptic_metadata_path.exists():
            raise FileNotFoundError(
                f"Elliptic metadata not found at {self.elliptic_metadata_path}. "
                "Please run clean_elliptic.py first."
            )
        
        with open(self.elliptic_metadata_path, 'r') as f:
            stats = json.load(f)
        
        logger.info(f"✓ Loaded Elliptic stats: {stats['num_nodes']:,} nodes, {stats['num_edges']:,} edges")
        return stats
    
    def generate_erdos_renyi(self):
        """
        Generate Erdős–Rényi random graph
        
        Properties:
        - Random edges with probability p
        - Poisson degree distribution
        - Low clustering coefficient
        - Homogeneous structure (no hubs)
        
        Returns:
            NetworkX DiGraph
        """
        logger.info("=" * 80)
        logger.info("GENERATING ERDŐS–RÉNYI GRAPH")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Calculate edge probability
        # For directed graph: p = m / (n * (n-1))
        n = self.n_nodes
        m = self.n_edges
        p = m / (n * (n - 1))
        
        logger.info(f"Parameters:")
        logger.info(f"  n (nodes) = {n:,}")
        logger.info(f"  m (target edges) = {m:,}")
        logger.info(f"  p (edge probability) = {p:.8f}")
        
        # Generate graph
        logger.info("Generating random graph (this may take a few minutes)...")
        G = nx.gnp_random_graph(n, p, directed=True, seed=42)
        
        generation_time = time.time() - start_time
        actual_edges = G.number_of_edges()
        
        logger.info(f"✓ Graph generated in {generation_time:.2f} seconds")
        logger.info(f"  Nodes: {G.number_of_nodes():,}")
        logger.info(f"  Edges: {actual_edges:,}")
        logger.info(f"  Target edges: {m:,}")
        logger.info(f"  Difference: {abs(actual_edges - m):,} ({abs(actual_edges - m)/m*100:.2f}%)")
        
        return G
    
    def generate_barabasi_albert(self):
        """
        Generate Barabási–Albert preferential attachment graph
        
        Properties:
        - Preferential attachment (rich get richer)
        - Power-law degree distribution
        - Scale-free (hubs exist)
        - Low clustering but higher than ER
        
        Returns:
            NetworkX DiGraph
        """
        logger.info("=" * 80)
        logger.info("GENERATING BARABÁSI–ALBERT GRAPH")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Calculate m (edges per new node)
        # Target: average degree = 2*m_edges / n_nodes
        n = self.n_nodes
        m_total = self.n_edges
        avg_degree = (2 * m_total) / n
        m_per_node = max(1, int(avg_degree / 2))  # Each node adds m edges
        
        logger.info(f"Parameters:")
        logger.info(f"  n (nodes) = {n:,}")
        logger.info(f"  Target average degree = {avg_degree:.2f}")
        logger.info(f"  m (edges per node) = {m_per_node}")
        logger.info(f"  Expected edges ≈ {n * m_per_node:,}")
        
        # Generate graph (undirected first)
        logger.info("Generating preferential attachment graph...")
        G_undirected = nx.barabasi_albert_graph(n, m_per_node, seed=42)
        
        # Convert to directed
        logger.info("Converting to directed graph...")
        G = nx.DiGraph(G_undirected)
        
        generation_time = time.time() - start_time
        actual_edges = G.number_of_edges()
        
        logger.info(f"✓ Graph generated in {generation_time:.2f} seconds")
        logger.info(f"  Nodes: {G.number_of_nodes():,}")
        logger.info(f"  Edges: {actual_edges:,}")
        logger.info(f"  Target edges: {m_total:,}")
        logger.info(f"  Difference: {abs(actual_edges - m_total):,} ({abs(actual_edges - m_total)/m_total*100:.2f}%)")
        
        return G
    
    def add_node_attributes(self, G, graph_type):
        """
        Add node attributes matching Elliptic schema
        
        Args:
            G: NetworkX graph
            graph_type: 'erdos_renyi' or 'barabasi_albert'
        
        Schema:
            node_id: str - "er_node_0" or "ba_node_0"
            timestamp: int - sequential from 1 to time_range[1]
            label: None - all unlabeled for structural baseline
            is_labeled: False - no labels in random graphs
        """
        logger.info(f"\nAdding node attributes...")
        
        # Prefix for node IDs
        prefix = "er" if graph_type == "erdos_renyi" else "ba"
        
        # Get time range from Elliptic
        min_time, max_time = self.time_range
        
        # Assign attributes to all nodes
        for i, node in enumerate(G.nodes()):
            # Sequential timestamp distribution across time range
            timestamp = int(min_time + (i % (max_time - min_time + 1)))
            
            # Create node attributes
            G.nodes[node]['node_id'] = f"{prefix}_node_{node}"
            G.nodes[node]['timestamp'] = timestamp
            G.nodes[node]['label'] = None  # No labels in random baseline
            G.nodes[node]['is_labeled'] = False
        
        logger.info(f"✓ Added attributes to {G.number_of_nodes():,} nodes")
        logger.info(f"  Timestamp range: [{min_time}, {max_time}]")
        logger.info(f"  All nodes unlabeled (baseline comparison)")
    
    def create_dataframes(self, G):
        """
        Create pandas DataFrames for nodes and edges
        
        Args:
            G: NetworkX graph with attributes
        
        Returns:
            nodes_df, edges_df
        """
        logger.info("\nCreating DataFrames...")
        
        # Create nodes DataFrame
        nodes_data = []
        for node in G.nodes():
            attrs = G.nodes[node]
            nodes_data.append({
                'node_id': attrs['node_id'],
                'timestamp': attrs['timestamp'],
                'label': attrs['label'],  # Will be None
                'is_labeled': attrs['is_labeled']  # Will be False
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        
        # Create edges DataFrame
        edges_data = []
        for source, target in G.edges():
            edges_data.append({
                'source': G.nodes[source]['node_id'],
                'target': G.nodes[target]['node_id']
            })
        
        edges_df = pd.DataFrame(edges_data)
        
        logger.info(f"✓ Created DataFrames")
        logger.info(f"  Nodes: {len(nodes_df):,} rows")
        logger.info(f"  Edges: {len(edges_df):,} rows")
        
        return nodes_df, edges_df
    
    def compute_metadata(self, G, graph_type, generation_time):
        """
        Compute graph statistics and metadata
        
        Args:
            G: NetworkX graph
            graph_type: 'erdos_renyi' or 'barabasi_albert'
            generation_time: Time taken to generate (seconds)
        
        Returns:
            Dictionary of metadata
        """
        logger.info("\nComputing graph statistics...")
        
        # Basic stats
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Degree statistics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        # Clustering (sample if too large)
        if n_nodes > 10000:
            logger.info("  Computing clustering on sample (graph too large)...")
            sample_nodes = np.random.choice(list(G.nodes()), size=10000, replace=False)
            clustering = nx.average_clustering(G.subgraph(sample_nodes).to_undirected())
        else:
            clustering = nx.average_clustering(G.to_undirected())
        
        # Connectivity
        is_connected = nx.is_weakly_connected(G)
        n_components = nx.number_weakly_connected_components(G)
        
        # Metadata
        metadata = {
            "dataset_name": f"random_{graph_type}",
            "graph_type": graph_type,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "num_labeled": 0,  # No labels in random baseline
            "num_licit": 0,
            "num_illicit": 0,
            "num_unknown": n_nodes,  # All unlabeled
            "time_range": self.time_range,
            "avg_in_degree": float(np.mean(in_degrees)),
            "avg_out_degree": float(np.mean(out_degrees)),
            "max_in_degree": int(max(in_degrees)),
            "max_out_degree": int(max(out_degrees)),
            "avg_clustering": float(clustering),
            "density": float(nx.density(G)),
            "is_connected": is_connected,
            "num_connected_components": n_components,
            "generation_time_seconds": generation_time,
            "generated_at": datetime.now().isoformat(),
            "seed": 42,
            "target_nodes": self.n_nodes,
            "target_edges": self.n_edges,
            "elliptic_reference": str(self.elliptic_metadata_path)
        }
        
        logger.info(f"✓ Computed metadata")
        logger.info(f"  Average in-degree: {metadata['avg_in_degree']:.2f}")
        logger.info(f"  Average out-degree: {metadata['avg_out_degree']:.2f}")
        logger.info(f"  Clustering coefficient: {metadata['avg_clustering']:.4f}")
        logger.info(f"  Density: {metadata['density']:.6e}")
        logger.info(f"  Connected components: {metadata['num_connected_components']}")
        
        return metadata
    
    def save_outputs(self, G, nodes_df, edges_df, metadata, graph_type):
        """
        Save processed graph data in standard format
        
        Args:
            G: NetworkX graph
            nodes_df: Nodes DataFrame
            edges_df: Edges DataFrame
            metadata: Metadata dictionary
            graph_type: 'erdos_renyi' or 'barabasi_albert'
        """
        # Create output directory
        output_dir = self.output_base_path / graph_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving outputs to: {output_dir}")
        
        # Save nodes CSV
        nodes_path = output_dir / "nodes.csv"
        nodes_df.to_csv(nodes_path, index=False)
        logger.info(f"✓ Saved nodes.csv ({nodes_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save edges CSV
        edges_path = output_dir / "edges.csv"
        edges_df.to_csv(edges_path, index=False)
        logger.info(f"✓ Saved edges.csv ({edges_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save graph pickle
        graph_path = output_dir / "graph.gpickle"
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"✓ Saved graph.gpickle ({graph_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save metadata JSON
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata.json")
        
        # Create generation log
        log_path = output_dir / "generation_log.txt"
        with open(log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"RANDOM GRAPH GENERATION LOG - {graph_type.upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {metadata['generated_at']}\n")
            f.write(f"Graph Type: {graph_type}\n")
            f.write(f"Random Seed: {metadata['seed']}\n\n")
            
            f.write("TARGET (from Elliptic):\n")
            f.write(f"  Nodes: {metadata['target_nodes']:,}\n")
            f.write(f"  Edges: {metadata['target_edges']:,}\n\n")
            
            f.write("GENERATED:\n")
            f.write(f"  Nodes: {metadata['num_nodes']:,}\n")
            f.write(f"  Edges: {metadata['num_edges']:,}\n")
            f.write(f"  Generation Time: {metadata['generation_time_seconds']:.2f} seconds\n\n")
            
            f.write("PROPERTIES:\n")
            f.write(f"  Avg In-Degree: {metadata['avg_in_degree']:.2f}\n")
            f.write(f"  Avg Out-Degree: {metadata['avg_out_degree']:.2f}\n")
            f.write(f"  Max In-Degree: {metadata['max_in_degree']}\n")
            f.write(f"  Max Out-Degree: {metadata['max_out_degree']}\n")
            f.write(f"  Clustering: {metadata['avg_clustering']:.4f}\n")
            f.write(f"  Density: {metadata['density']:.6e}\n")
            f.write(f"  Connected: {metadata['is_connected']}\n")
            f.write(f"  Components: {metadata['num_connected_components']}\n\n")
            
            f.write("OUTPUT FILES:\n")
            f.write(f"  nodes.csv: {len(nodes_df):,} rows\n")
            f.write(f"  edges.csv: {len(edges_df):,} rows\n")
            f.write(f"  graph.gpickle: NetworkX DiGraph object\n")
            f.write(f"  metadata.json: Graph statistics\n")
        
        logger.info(f"✓ Saved generation_log.txt")
        
        return output_dir
    
    def generate_graph(self, graph_type):
        """
        Generate a random graph of specified type
        
        Args:
            graph_type: 'erdos_renyi' or 'barabasi_albert'
        
        Returns:
            Path to output directory
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"GENERATING {graph_type.upper().replace('_', '–')} GRAPH")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Generate graph
        if graph_type == "erdos_renyi":
            G = self.generate_erdos_renyi()
        elif graph_type == "barabasi_albert":
            G = self.generate_barabasi_albert()
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        generation_time = time.time() - start_time
        
        # Add attributes
        self.add_node_attributes(G, graph_type)
        
        # Create DataFrames
        nodes_df, edges_df = self.create_dataframes(G)
        
        # Compute metadata
        metadata = self.compute_metadata(G, graph_type, generation_time)
        
        # Save outputs
        output_dir = self.save_outputs(G, nodes_df, edges_df, metadata, graph_type)
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ {graph_type.upper()} GRAPH COMPLETE in {total_time:.2f} seconds")
        logger.info(f"   Output: {output_dir}")
        
        return output_dir
    
    def generate_all(self):
        """Generate both random graph types"""
        logger.info("\n" + "=" * 80)
        logger.info("RANDOM GRAPH GENERATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Matching Elliptic dataset size:")
        logger.info(f"  Target nodes: {self.n_nodes:,}")
        logger.info(f"  Target edges: {self.n_edges:,}")
        logger.info(f"  Time range: {self.time_range}")
        
        overall_start = time.time()
        
        # Generate Erdős–Rényi
        er_dir = self.generate_graph("erdos_renyi")
        
        # Generate Barabási–Albert
        ba_dir = self.generate_graph("barabasi_albert")
        
        total_time = time.time() - overall_start
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"\nGenerated graphs:")
        logger.info(f"  1. Erdős–Rényi: {er_dir}")
        logger.info(f"  2. Barabási–Albert: {ba_dir}")
        logger.info(f"\n✅ All random graphs ready for experiments!")
        
        return er_dir, ba_dir


def main():
    """Main execution function"""
    try:
        # Initialize generator
        generator = RandomGraphGenerator(
            elliptic_metadata_path="data/processed/elliptic/metadata.json",
            output_base_path="data/processed/random"
        )
        
        # Generate both graphs
        generator.generate_all()
        
        print("\n" + "=" * 80)
        print("SUCCESS! Random graphs generated and ready for experiments.")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Verify outputs in data/processed/random/")
        print("  2. Run validation: python src/etl/validate_random_graphs.py")
        print("  3. Start Experiment 1: degree distribution analysis")
        print()
        
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()