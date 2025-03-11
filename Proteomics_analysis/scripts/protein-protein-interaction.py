#!/usr/bin/env python3
"""
Protein-Protein Interaction (PPI) Network Analysis for Proteomics Data

This script:
1. Takes differentially expressed proteins as input
2. Queries STRING, BioGRID, and IntAct databases for interaction data
3. Constructs PPI networks and identifies functional modules
4. Performs network analysis to find hub proteins and critical nodes
5. Generates interactive network visualizations

Author: Jiyang Jiang
Date: March 11, 2025
"""

import pandas as pd
import numpy as np
import networkx as nx
import requests
import json
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional, Union
import argparse
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import community as community_louvain  # python-louvain package
from matplotlib import cm
import io
import zipfile
from urllib.parse import urlencode
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(snakemake.log[0]),
        logging.StreamHandler()
    ]
)

class PPINetwork:
    """Class for protein-protein interaction network analysis"""
    
    def __init__(self, organism="human", confidence_score=0.7):
        """
        Initialize the PPI network analyzer
        
        Parameters:
        -----------
        organism : str
            Organism name (default: "human")
        confidence_score : float
            Minimum confidence score for interactions (0-1)
        """
        self.organism = organism
        self.confidence_score = confidence_score
        self.string_api_url = "https://string-db.org/api"
        self.string_version = "11.5"
        self.network = nx.Graph()
        self.protein_info = {}
        self.modules = {}
        
        # Map organism to NCBI taxonomy ID
        self.organism_map = {
            "human": "9606",
            "mouse": "10090",
            "rat": "10116",
            "yeast": "4932",
            "zebrafish": "7955",
            "fly": "7227",
            "worm": "6239"
        }
        
        # Initialize for mapping IDs
        self.protein_mapping = {}
    
    def map_protein_ids(self, protein_ids, from_type="UNIPROT", to_type="STRING"):
        """
        Map protein IDs between different identifier systems
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs to map
        from_type : str
            Source ID type (default: "UNIPROT")
        to_type : str
            Target ID type (default: "STRING")
            
        Returns:
        --------
        dict
            Dictionary mapping original IDs to new IDs
        """
        logging.info(f"Mapping {len(protein_ids)} protein IDs from {from_type} to {to_type}")
        
        # Use the STRING API for ID mapping
        params = {
            "identifiers": "\r".join(protein_ids),
            "species": self.organism_map.get(self.organism, "9606"),
            "limit": 1,
            "echo_query": 1,
            "caller_identity": "proteomics_workflow"
        }
        
        request_url = f"{self.string_api_url}/{self.string_version}/get_string_ids"
        
        try:
            response = requests.post(request_url, data=params)
            response.raise_for_status()
            data = response.json()
            
            # Create mapping dictionary
            mapping = {}
            for entry in data:
                if "queryItem" in entry and "stringId" in entry:
                    original_id = entry["queryItem"]
                    string_id = entry["stringId"]
                    mapping[original_id] = string_id
            
            logging.info(f"Successfully mapped {len(mapping)} out of {len(protein_ids)} proteins")
            self.protein_mapping = mapping
            return mapping
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during ID mapping: {str(e)}")
            return {}
    
    def fetch_string_interactions(self, protein_ids, include_indirect=True):
        """
        Fetch protein-protein interactions from the STRING database
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs to query
        include_indirect : bool
            Whether to include indirect interactions (default: True)
            
        Returns:
        --------
        list
            List of interactions with confidence scores
        """
        logging.info(f"Fetching STRING interactions for {len(protein_ids)} proteins")
        
        # Map IDs to STRING IDs if they aren't already
        if not protein_ids[0].startswith("9606."):
            mapped_ids = list(self.map_protein_ids(protein_ids).values())
        else:
            mapped_ids = protein_ids
        
        if not mapped_ids:
            logging.error("No proteins could be mapped to STRING IDs")
            return []
        
        # Prepare parameters
        params = {
            "identifiers": "%0d".join(mapped_ids),
            "species": self.organism_map.get(self.organism, "9606"),
            "network_type": "physical",
            "required_score": int(self.confidence_score * 1000),
            "caller_identity": "proteomics_workflow"
        }
        
        if include_indirect:
            params["add_nodes"] = 20  # Add up to 20 indirect interactors
            
        request_url = f"{self.string_api_url}/{self.string_version}/network"
        
        try:
            response = requests.post(request_url, data=params)
            response.raise_for_status()
            data = response.json()
            
            logging.info(f"Retrieved {len(data)} interactions from STRING")
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching STRING interactions: {str(e)}")
            return []
    
    def fetch_biogrid_interactions(self, protein_ids):
        """
        Fetch protein-protein interactions from the BioGRID database
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs to query
            
        Returns:
        --------
        list
            List of interactions from BioGRID
        """
        logging.info(f"Fetching BioGRID interactions for {len(protein_ids)} proteins")
        
        # BioGRID API endpoint
        biogrid_url = "https://webservice.thebiogrid.org/interactions/"
        
        # Prepare parameters
        params = {
            "searchNames": "true",
            "geneList": "|".join(protein_ids),
            "includeInteractors": "true",
            "includeInteractorInteractions": "false",
            "taxId": self.organism_map.get(self.organism, "9606"),
            "format": "json",
            "max": 10000,
            "accessKey": snakemake.params.get("biogrid_api_key", "")
        }
        
        if not params["accessKey"]:
            logging.warning("No BioGRID API key provided. Skipping BioGRID queries.")
            return []
        
        try:
            response = requests.get(biogrid_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            logging.info(f"Retrieved {len(data)} interactions from BioGRID")
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching BioGRID interactions: {str(e)}")
            return []
    
    def fetch_intact_interactions(self, protein_ids):
        """
        Fetch protein-protein interactions from the IntAct database
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs to query
            
        Returns:
        --------
        list
            List of interactions from IntAct
        """
        logging.info(f"Fetching IntAct interactions for {len(protein_ids)} proteins")
        
        # IntAct PSICQUIC REST API
        intact_url = "https://www.ebi.ac.uk/Tools/webservices/psicquic/intact/webservices/current/search/query"
        
        # Process in batches to avoid URL length limitations
        batch_size = 20
        all_interactions = []
        
        for i in range(0, len(protein_ids), batch_size):
            batch = protein_ids[i:i+batch_size]
            query = " OR ".join([f"identifier:\"{pid}\"" for pid in batch])
            
            params = {
                "query": query,
                "format": "json",
                "maxResults": 10000
            }
            
            try:
                response = requests.get(intact_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "data" in data:
                    all_interactions.extend(data["data"])
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching IntAct interactions (batch {i}): {str(e)}")
        
        logging.info(f"Retrieved {len(all_interactions)} interactions from IntAct")
        return all_interactions
    
    def build_network(self, diff_expr_file, log2fc_cutoff=1.0, pval_cutoff=0.05):
        """
        Build a protein-protein interaction network from differentially expressed proteins
        
        Parameters:
        -----------
        diff_expr_file : str
            Path to differential expression results file
        log2fc_cutoff : float
            Log2 fold change cutoff for significance (default: 1.0)
        pval_cutoff : float
            P-value cutoff for significance (default: 0.05)
            
        Returns:
        --------
        networkx.Graph
            The constructed PPI network
        """
        logging.info(f"Building PPI network from {diff_expr_file}")
        
        # Load differential expression data
        try:
            diff_expr = pd.read_csv(diff_expr_file, sep='\t', index_col=0)
            
            # Check required columns
            required_cols = ['log2FC', 'padj']
            if not all(col in diff_expr.columns for col in required_cols):
                available_cols = list(diff_expr.columns)
                logging.error(f"Missing required columns. Available: {available_cols}")
                raise ValueError(f"Differential expression file must contain: {required_cols}")
            
            # Filter for significant proteins
            sig_proteins = diff_expr[(abs(diff_expr['log2FC']) >= log2fc_cutoff) & 
                                    (diff_expr['padj'] <= pval_cutoff)].copy()
            
            logging.info(f"Found {len(sig_proteins)} significant proteins for network analysis")
            
            if len(sig_proteins) < 5:
                logging.warning("Too few significant proteins for meaningful network analysis")
                return nx.Graph()
            
            # Get protein IDs and expression values
            protein_ids = sig_proteins.index.tolist()
            
            # Store expression data
            for protein_id in protein_ids:
                self.protein_info[protein_id] = {
                    'log2FC': diff_expr.loc[protein_id, 'log2FC'],
                    'padj': diff_expr.loc[protein_id, 'padj']
                }
            
            # Fetch interactions from different databases
            string_interactions = self.fetch_string_interactions(protein_ids)
            biogrid_interactions = self.fetch_biogrid_interactions(protein_ids)
            intact_interactions = self.fetch_intact_interactions(protein_ids)
            
            # Create network
            G = nx.Graph()
            
            # Add nodes (proteins)
            for protein_id in protein_ids:
                G.add_node(protein_id, 
                          log2FC=diff_expr.loc[protein_id, 'log2FC'],
                          padj=diff_expr.loc[protein_id, 'padj'],
                          source='input')
            
            # Add edges from STRING
            for interaction in string_interactions:
                source = interaction.get('preferredName_A', interaction.get('stringId_A'))
                target = interaction.get('preferredName_B', interaction.get('stringId_B'))
                score = float(interaction.get('score', 0)) / 1000.0
                
                if score >= self.confidence_score:
                    # Reverse map to original IDs if available
                    source_original = next((k for k, v in self.protein_mapping.items() if v == source), source)
                    target_original = next((k for k, v in self.protein_mapping.items() if v == target), target)
                    
                    if not G.has_node(source_original):
                        G.add_node(source_original, source='STRING')
                    if not G.has_node(target_original):
                        G.add_node(target_original, source='STRING')
                    
                    G.add_edge(source_original, target_original, 
                              weight=score, 
                              source='STRING')
            
            # Process BioGRID interactions
            for interaction_id, interaction in biogrid_interactions.items():
                source = interaction.get('OFFICIAL_SYMBOL_A')
                target = interaction.get('OFFICIAL_SYMBOL_B')
                
                if source and target:
                    if not G.has_node(source):
                        G.add_node(source, source='BioGRID')
                    if not G.has_node(target):
                        G.add_node(target, source='BioGRID')
                    
                    G.add_edge(source, target, 
                              weight=0.7,  # Assign default weight 
                              source='BioGRID')
            
            # Process IntAct interactions
            for interaction in intact_interactions:
                if len(interaction) >= 2:
                    source = interaction[0].get('identifier', {}).get('db', '')
                    target = interaction[1].get('identifier', {}).get('db', '')
                    
                    if source and target:
                        if not G.has_node(source):
                            G.add_node(source, source='IntAct')
                        if not G.has_node(target):
                            G.add_node(target, source='IntAct')
                        
                        G.add_edge(source, target, 
                                  weight=0.7,  # Assign default weight
                                  source='IntAct')
            
            # Remove isolated nodes
            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)
            
            logging.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            logging.info(f"Removed {len(isolated_nodes)} isolated nodes")
            
            self.network = G
            return G
            
        except Exception as e:
            logging.error(f"Error building network: {str(e)}")
            return nx.Graph()
    
    def identify_modules(self, resolution=1.0):
        """
        Identify functional modules in the network using Louvain community detection
        
        Parameters:
        -----------
        resolution : float
            Resolution parameter for community detection (default: 1.0)
            
        Returns:
        --------
        dict
            Dictionary mapping nodes to community IDs
        """
        logging.info("Identifying functional modules in the network")
        
        if self.network.number_of_nodes() < 3:
            logging.warning("Network too small for module detection")
            return {}
        
        try:
            # Apply Louvain community detection
            partition = community_louvain.best_partition(self.network, resolution=resolution)
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            # Store module information
            self.modules = {i: nodes for i, nodes in communities.items()}
            
            logging.info(f"Identified {len(communities)} modules in the network")
            
            return partition
            
        except Exception as e:
            logging.error(f"Error identifying modules: {str(e)}")
            return {}
    
    def find_hub_proteins(self, top_n=10):
        """
        Identify hub proteins based on network centrality measures
        
        Parameters:
        -----------
        top_n : int
            Number of top hub proteins to return (default: 10)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with hub proteins and their centrality measures
        """
        logging.info("Identifying hub proteins in the network")
        
        if self.network.number_of_nodes() < 3:
            logging.warning("Network too small for hub protein analysis")
            return pd.DataFrame()
        
        try:
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(self.network)
            betweenness_centrality = nx.betweenness_centrality(self.network)
            closeness_centrality = nx.closeness_centrality(self.network)
            
            # Calculate eigenvector centrality if the network is connected
            eigenvector_centrality = {}
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.network, max_iter=1000)
            except:
                # Fall back to power iteration method if regular method fails
                try:
                    eigenvector_centrality = nx.eigenvector_centrality_numpy(self.network)
                except:
                    logging.warning("Could not calculate eigenvector centrality")
            
            # Combine measures into DataFrame
            centrality_df = pd.DataFrame({
                'Degree': pd.Series(degree_centrality),
                'Betweenness': pd.Series(betweenness_centrality),
                'Closeness': pd.Series(closeness_centrality),
                'Eigenvector': pd.Series(eigenvector_centrality)
            })
            
            # Add log2FC and p-value if available
            for protein in centrality_df.index:
                if protein in self.protein_info:
                    centrality_df.loc[protein, 'log2FC'] = self.protein_info[protein]['log2FC']
                    centrality_df.loc[protein, 'padj'] = self.protein_info[protein]['padj']
            
            # Create a composite score (average of normalized centrality measures)
            for col in ['Degree', 'Betweenness', 'Closeness', 'Eigenvector']:
                if not centrality_df[col].empty:
                    max_val = centrality_df[col].max()
                    if max_val > 0:
                        centrality_df[f'{col}_norm'] = centrality_df[col] / max_val
            
            norm_cols = [col for col in centrality_df.columns if col.endswith('_norm')]
            if norm_cols:
                centrality_df['Composite'] = centrality_df[norm_cols].mean(axis=1)
            
                # Sort by composite score
                centrality_df = centrality_df.sort_values('Composite', ascending=False)
            
            logging.info(f"Identified top {min(top_n, len(centrality_df))} hub proteins")
            
            return centrality_df.head(top_n)
            
        except Exception as e:
            logging.error(f"Error identifying hub proteins: {str(e)}")
            return pd.DataFrame()
    
    def visualize_network(self, output_file, color_by='log2FC', 
                         show_modules=True, label_hubs=True, hub_top_n=10):
        """
        Create a visualization of the protein-protein interaction network
        
        Parameters:
        -----------
        output_file : str
            Path to save the visualization
        color_by : str
            Node attribute to use for coloring (default: 'log2FC')
        show_modules : bool
            Whether to color nodes by module (default: True)
        label_hubs : bool
            Whether to label hub proteins (default: True)
        hub_top_n : int
            Number of top hub proteins to label (default: 10)
            
        Returns:
        --------
        None
        """
        logging.info(f"Creating network visualization and saving to {output_file}")
        
        if self.network.number_of_nodes() < 2:
            logging.warning("Network too small for visualization")
            return
        
        try:
            # Create figure
            plt.figure(figsize=(14, 12))
            
            # Use spring layout for node positions
            pos = nx.spring_layout(self.network, k=0.3, iterations=50, seed=42)
            
            # Determine node colors
            node_colors = []
            
            if show_modules and self.modules:
                # Use module assignments for coloring
                modules = self.identify_modules()
                cmap = plt.cm.get_cmap('tab20', len(self.modules))
                
                node_colors = [cmap(modules.get(node, 0)) for node in self.network.nodes()]
                
                # Create legend for modules
                legend_patches = []
                for module_id, nodes in self.modules.items():
                    patch = mpatches.Patch(color=cmap(module_id), 
                                         label=f'Module {module_id} ({len(nodes)} proteins)')
                    legend_patches.append(patch)
                
            elif color_by in nx.get_node_attributes(self.network, color_by):
                # Color by node attribute (e.g., log2FC)
                node_attrs = nx.get_node_attributes(self.network, color_by)
                
                # Define colormap for log2FC (blue-white-red)
                if color_by == 'log2FC':
                    max_abs_val = max([abs(val) for val in node_attrs.values() if val is not None])
                    norm = plt.Normalize(-max_abs_val, max_abs_val)
                    cmap = plt.cm.RdBu_r
                    
                    node_colors = [cmap(norm(node_attrs.get(node, 0))) 
                                  if node in node_attrs else (0.7, 0.7, 0.7, 1)
                                  for node in self.network.nodes()]
                    
                    # Create colorbar
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm)
                    cbar.set_label(color_by, rotation=270, labelpad=15)
                    
                else:
                    # Default coloring
                    node_colors = ['#1f77b4' for _ in self.network.nodes()]
            else:
                # Default coloring
                node_colors = ['#1f77b4' for _ in self.network.nodes()]
            
            # Determine node sizes based on degree
            node_degrees = dict(self.network.degree())
            max_degree = max(node_degrees.values()) if node_degrees else 1
            
            node_sizes = [100 + 500 * (node_degrees.get(node, 0) / max_degree) 
                         for node in self.network.nodes()]
            
            # Draw network
            nx.draw_networkx_nodes(self.network, pos, 
                                  node_size=node_sizes, 
                                  node_color=node_colors, 
                                  alpha=0.8)
            
            nx.draw_networkx_edges(self.network, pos, 
                                  width=0.5, alpha=0.5, 
                                  edge_color='gray')
            
            # Label hub proteins if requested
            if label_hubs:
                hub_proteins = self.find_hub_proteins(top_n=hub_top_n)
                
                if not hub_proteins.empty:
                    hub_labels = {node: node for node in hub_proteins.index if node in self.network.nodes()}
                    
                    nx.draw_networkx_labels(self.network, pos, 
                                          labels=hub_labels, 
                                          font_size=10, 
                                          font_weight='bold')
            
            # Add legend if we have module colors
            if show_modules and self.modules:
                plt.legend(handles=legend_patches, 
                          title="Network Modules", 
                          loc='center left', 
                          bbox_to_anchor=(1, 0.5))
            
            plt.title("Protein-Protein Interaction Network", fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Network visualization saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error visualizing network: {str(e)}")
    
    def export_to_cytoscape(self, output_file):
        """
        Export the network in a format compatible with Cytoscape
        
        Parameters:
        -----------
        output_file : str
            Path to save the exported network file
            
        Returns:
        --------
        None
        """
        logging.info(f"Exporting network for Cytoscape to {output_file}")
        
        if self.network.number_of_nodes() < 2:
            logging.warning("Network too small for export")
            return
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Export node data
            node_data = []
            for node in self.network.nodes():
                attrs = self.network.nodes[node]
                
                node_entry = {'id': node}
                node_entry.update(attrs)
                
                # Add module information if available
                modules = self.identify_modules()
                if modules and node in modules:
                    node_entry['module'] = modules[node]
                
                # Add centrality measures
                if self.network.number_of_nodes() >= 3:
                    try:
                        node_entry['degree_centrality'] = nx.degree_centrality(self.network).get(node, 0)
                        node_entry['betweenness_centrality'] = nx.betweenness_centrality(self.network).get(node, 0)
                    except:
                        pass
                
                node_data.append(node_entry)
            
            # Export edge data
            edge_data = []
            for source, target, attrs in self.network.edges(data=True):
                edge_entry = {
                    'source': source,
                    'target': target
                }
                edge_entry.update(attrs)
                edge_data.append(edge_entry)
            
            # Create export data
            export_data = {
                'nodes': node_data,
                'edges': edge_data
            }
            
            # Save as JSON for Cytoscape import
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logging.info(f"Network exported for Cytoscape to {output_file}")
            
        except Exception as e:
            logging.error(f"Error exporting network: {str(e)}")
    
    def generate_network_report(self, output_file):
        """
        Generate a comprehensive HTML report for the network analysis
        
        Parameters:
        -----------
        output_file : str
            Path to save the HTML report
            
        Returns:
        --------
        None
        """
        logging.info(f"Generating network analysis report to {output_file}")
        
        if self.network.number_of_nodes() < 2:
            logging.warning("Network too small for report generation")
            return
        
        try:
            # Get network statistics
            num_nodes = self.network.number_of_nodes()
            num_edges = self.network.number_of_edges()
            density = nx.density(self.network)
            
            # Try to get connected components and average clustering
            try:
                num_components = nx.number_connected_components(self.network)
                avg_clustering = nx.average_clustering(self.network)
            except:
                num_components = "N/A"
                avg_clustering = "N/A"
            
            # Get hub proteins
            hub_proteins = self.find_hub_proteins(top_n=20)
            
            # Get module information
            self.identify_modules()
            num_modules = len(self.modules)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>PPI Network Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2C3E50; }}
                    h2 {{ color: #2980B9; }}
                    h3 {{ color: #3498DB; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .summary {{ background-color: #eef6f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>Protein-Protein Interaction Network Analysis Report</h1>
                
                <div class="summary">
                    <h2>Network Summary</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Number of Proteins (Nodes)</td><td>{num_nodes}</td></tr>
                        <tr><td>Number of Interactions (Edges)</td><td>{num_edges}</td></tr>
                        <tr><td>Network Density</td><td>{density:.4f}</td></tr>
                        <tr><td>Number of Connected Components</td><td>{num_components}</td></tr>
                        <tr><td>Average Clustering Coefficient</td><td>{avg_clustering}</td></tr>
                        <tr><td>Number of Functional Modules</td><td>{num_modules}</td></tr>
                    </table>
                </div>
                
                <h2>Network Visualization</h2>
                <img src="{os.path.basename(output_file).replace('.html', '_network.png')}" alt="PPI Network">
                
                <h2>Hub Proteins</h2>
                <p>Hub proteins are highly connected nodes in the network that may play important roles in the biological system.</p>
            """
            
            # Add hub proteins table
            if not hub_proteins.empty:
                html_content += """
                <table>
                    <tr>
                        <th>Protein</th>
                        <th>Degree Centrality</th>
                        <th>Betweenness Centrality</th>
                        <th>Closeness Centrality</th>
                        <th>Eigenvector Centrality</th>
                        <th>Composite Score</th>
                    </tr>
                """
                
                for idx, row in hub_proteins.iterrows():
                    html_content += f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{row.get('Degree', 'N/A'):.4f}</td>
                        <td>{row.get('Betweenness', 'N/A'):.4f}</td>
                        <td>{row.get('Closeness', 'N/A'):.4f}</td>
                        <td>{row.get('Eigenvector', 'N/A'):.4f}</td>
                        <td>{row.get('Composite', 'N/A'):.4f}</td>
                    </tr>
                    """
                
                html_content += "</table>"
            else:
                html_content += "<p>No hub proteins identified.</p>"
                
            # Add module information
            html_content += """
                <h2>Functional Modules</h2>
                <p>Functional modules represent groups of proteins that are densely connected and may work together in biological processes.</p>
            """
            
            if self.modules:
                html_content += """
                <table>
                    <tr>
                        <th>Module ID</th>
                        <th>Number of Proteins</th>
                        <th>Member Proteins</th>
                    </tr>
                """
                
                for module_id, nodes in self.modules.items():
                    # Limit the number of proteins shown to avoid very long cells
                    if len(nodes) > 10:
                        shown_proteins = ", ".join(nodes[:10]) + f"... (+{len(nodes)-10} more)"
                    else:
                        shown_proteins = ", ".join(nodes)
                    
                    html_content += f"""
                    <tr>
                        <td>Module {module_id}</td>
                        <td>{len(nodes)}</td>
                        <td>{shown_proteins}</td>
                    </tr>
                    """
                
                html_content += "</table>"
            else:
                html_content += "<p>No functional modules identified.</p>"
            
            # Close HTML
            html_content += """
                <hr>
                <p><em>Report generated by PPI Network Analysis Pipeline</em></p>
            </body>
            </html>
            """
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logging.info(f"Network analysis report generated: {output_file}")
            
        except Exception as e:
            logging.error(f"Error generating network report: {str(e)}")

def main():
    try:
        # Get input and output files from Snakemake
        diff_expr_file = snakemake.input.diff_expr
        output_network_plot = snakemake.output.network_plot
        output_cytoscape = snakemake.output.cytoscape_json
        output_report = snakemake.output.report
        
        # Get parameters
        organism = snakemake.params.get("organism", "human")
        confidence_score = float(snakemake.params.get("confidence_score", 0.7))
        log2fc_cutoff = float(snakemake.params.get("log2fc_cutoff", 1.0))
        pval_cutoff = float(snakemake.params.get("pval_cutoff", 0.05))
        
        # Initialize PPI network analyzer
        ppi_analyzer = PPINetwork(organism=organism, confidence_score=confidence_score)
        
        # Build network
        ppi_analyzer.build_network(diff_expr_file, log2fc_cutoff=log2fc_cutoff, pval_cutoff=pval_cutoff)
        
        # Identify modules
        ppi_analyzer.identify_modules()
        
        # Find hub proteins
        ppi_analyzer.find_hub_proteins()
        
        # Visualize network
        ppi_analyzer.visualize_network(output_network_plot)
        
        # Export for Cytoscape
        ppi_analyzer.export_to_cytoscape(output_cytoscape)
        
        # Generate report
        ppi_analyzer.generate_network_report(output_report)
        
        logging.info("PPI network analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in PPI network analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
