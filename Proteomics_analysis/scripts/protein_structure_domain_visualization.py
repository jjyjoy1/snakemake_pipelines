#!/usr/bin/env python3
"""
Advanced Data Visualization Module for Proteomics Data

This script:
1. Creates interactive visualizations for proteomics data
2. Generates complex multi-dimensional plots
3. Produces publication-quality figures
4. Integrates results from multiple analysis types
5. Creates comprehensive dashboards

Author: Jiyang Jiang
Date: March 11, 2025
"""

import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union
import json
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import zscore
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(snakemake.log[0]),
        logging.StreamHandler()
    ]
)

class AdvancedVisualization:
    """Class for advanced proteomics data visualization"""
    
    def __init__(self, color_palette='viridis'):
        """
        Initialize the visualization tool
        
        Parameters:
        -----------
        color_palette : str
            Color palette for visualizations (default: 'viridis')
        """
        self.color_palette = color_palette
        self.data = {}
        self.figures = {}
        self.dash_components = {}
        
        # Set default plotting style
        sns.set(style="ticks", context="talk")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['svg.fonttype'] = 'none'  # Make text editable in SVG
        
    def load_data(self, data_files, data_types=None):
        """
        Load multiple data files for visualization
        
        Parameters:
        -----------
        data_files : dict
            Dictionary mapping data types to file paths
        data_types : list
            List of data types to load (default: all)
            
        Returns:
        --------
        dict
            Dictionary of loaded datasets
        """
        logging.info(f"Loading data from {len(data_files)} files")
        
        # If no specific data types, load all
        if data_types is None:
            data_types = list(data_files.keys())
        
        for data_type, file_path in data_files.items():
            if data_type not in data_types:
                continue
                
            try:
                # Check file extension to determine how to load
                if file_path.endswith('.csv'):
                    self.data[data_type] = pd.read_csv(file_path)
                elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
                    self.data[data_type] = pd.read_csv(file_path, sep='\t')
                elif file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        self.data[data_type] = json.load(f)
                else:
                    logging.warning(f"Unsupported file format for {file_path}")
                    continue
                
                logging.info(f"Loaded {data_type} data from {file_path}")
                
            except Exception as e:
                logging.error(f"Error loading {data_type} data from {file_path}: {str(e)}")
        
        return self.data
    
    def create_heatmap(self, data_type, output_file, 
                      row_cluster=True, col_cluster=True, 
                      z_score=True, cmap='viridis',
                      figsize=(12, 10), title=None,
                      sample_col=None, feature_col=None, value_col=None):
        """
        Create an enhanced heatmap visualization
        
        Parameters:
        -----------
        data_type : str
            Type of data to visualize (must be loaded)
        output_file : str
            Path to save the visualization
        row_cluster : bool
            Whether to cluster rows (default: True)
        col_cluster : bool
            Whether to cluster columns (default: True)
        z_score : bool
            Whether to Z-score normalize data (default: True)
        cmap : str
            Colormap for heatmap (default: 'viridis')
        figsize : tuple
            Figure size (default: (12, 10))
        title : str
            Plot title (default: None)
        sample_col : str
            Column name for samples (for long-format data)
        feature_col : str
            Column name for features (for long-format data) 
        value_col : str
            Column name for values (for long-format data)
            
        Returns:
        --------
        str
            Path to saved visualization
        """
        logging.info(f"Creating heatmap for {data_type} data")
        
        if data_type not in self.data:
            logging.error(f"Data type {data_type} not loaded")
            return None
        
        try:
            # Get data
            df = self.data[data_type].copy()
            
            # Handle long format data if specified
            if sample_col is not None and feature_col is not None and value_col is not None:
                # Convert long to wide format
                df = df.pivot(index=feature_col, columns=sample_col, values=value_col)
            
            # Apply Z-score normalization if requested
            if z_score:
                df = df.apply(zscore, axis=1, nan_policy='omit')
            
            # Create figure and axes
            fig, ax = plt.subplots(figsize=figsize)
            
            # Calculate row linkage if clustering
            if row_cluster:
                row_linkage = linkage(pdist(df.values), method='average')
                row_dendrogram = dendrogram(row_linkage, ax=ax, labels=df.index, orientation='right', no_plot=True)
                row_order = df.index[row_dendrogram['leaves']]
            else:
                row_order = df.index
            
            # Calculate column linkage if clustering
            if col_cluster:
                col_linkage = linkage(pdist(df.values.T), method='average')
                col_dendrogram = dendrogram(col_linkage, ax=ax, labels=df.columns, orientation='top', no_plot=True)
                col_order = df.columns[col_dendrogram['leaves']]
            else:
                col_order = df.columns
            
            # Reorder data frame
            df_ordered = df.loc[row_order, col_order]
            
            # Create heatmap
            sns.heatmap(df_ordered, cmap=cmap, ax=ax, cbar_kws={'label': 'Z-score' if z_score else 'Value'})
            
            # Set title and labels
            if title:
                ax.set_title(title, fontsize=14)
            ax.set_xlabel('Samples', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            # Adjust appearance
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store figure information
            self.figures[f"{data_type}_heatmap"] = {
                'path': output_file,
                'type': 'heatmap',
                'title': title or f"{data_type} Heatmap"
            }
            
            logging.info(f"Heatmap saved to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error creating heatmap: {str(e)}")
            return None
    
    def create_volcano_plot(self, data_type, output_file, 
                           x_col='log2FC', y_col='padj', 
                           log_transform=True, interactive=True,
                           fc_cutoff=1.0, pval_cutoff=0.05,
                           highlight_features=None, label_top_n=10,
                           figsize=(10, 8), title=None):
        """
        Create an enhanced volcano plot
        
        Parameters:
        -----------
        data_type : str
            Type of data to visualize (must be loaded)
        output_file : str
            Path to save the visualization
        x_col : str
            Column name for x-axis (default: 'log2FC')
        y_col : str
            Column name for y-axis (default: 'padj')
        log_transform : bool
            Whether to -log10 transform the y-axis (default: True)
        interactive : bool
            Whether to create an interactive plot with plotly (default: True)
        fc_cutoff : float
            Fold change cutoff for significance (default: 1.0)
        pval_cutoff : float
            P-value cutoff for significance (default: 0.05)
        highlight_features : list
            List of feature names to highlight (default: None)
        label_top_n : int
            Number of top features to label (default: 10)
        figsize : tuple
            Figure size (default: (10, 8))
        title : str
            Plot title (default: None)
            
        Returns:
        --------
        str
            Path to saved visualization
        """
        logging.info(f"Creating volcano plot for {data_type} data")
        
        if data_type not in self.data:
            logging.error(f"Data type {data_type} not loaded")
            return None
        
        try:
            # Get data
            df = self.data[data_type].copy()
            
            # Check if required columns exist
            if x_col not in df.columns:
                logging.error(f"X-axis column {x_col} not found in data")
                return None
            if y_col not in df.columns:
                logging.error(f"Y-axis column {y_col} not found in data")
                return None
            
            # Apply log transformation to p-values if needed
            if log_transform:
                # Add small value to avoid log(0)
                df[f'-log10({y_col})'] = -np.log10(df[y_col] + 1e-10)
                plot_y_col = f'-log10({y_col})'
                plot_y_label = f'-log10({y_col})'
            else:
                plot_y_col = y_col
                plot_y_label = y_col
            
            # Add significance categories
            df['Significance'] = 'Not Significant'
            
            # Upregulated
            df.loc[(df[x_col] >= fc_cutoff) & (df[y_col] < pval_cutoff), 'Significance'] = 'Upregulated'
            
            # Downregulated
            df.loc[(df[x_col] <= -fc_cutoff) & (df[y_col] < pval_cutoff), 'Significance'] = 'Downregulated'
            
            # Create color mapping
            color_map = {
                'Not Significant': 'gray',
                'Upregulated': 'red',
                'Downregulated': 'blue'
            }
            
            # Select top features to label
            if label_top_n > 0:
                # Prioritize significant features
                sig_features = df[df['Significance'] != 'Not Significant'].copy()
                sig_features['Score'] = sig_features[plot_y_col] * abs(sig_features[x_col])
                top_features = sig_features.sort_values('Score', ascending=False).head(label_top_n).index.tolist()
            else:
                top_features = []
            
            # Add user-specified features to highlight
            if highlight_features:
                top_features.extend([f for f in highlight_features if f in df.index and f not in top_features])
            
            # Create interactive plot with plotly
            if interactive:
                # Prepare hover text
                df['hover_text'] = df.index + '<br>' + \
                                  f"{x_col}: " + df[x_col].round(2).astype(str) + '<br>' + \
                                  f"{y_col}: " + df[y_col].round(4).astype(str)
                
                # Create plot
                fig = px.scatter(
                    df, x=x_col, y=plot_y_col,
                    color='Significance', color_discrete_map=color_map,
                    hover_data=[x_col, y_col],
                    labels={x_col: x_col, plot_y_col: plot_y_label},
                    title=title or f"Volcano Plot - {data_type}"
                )
                
                # Add threshold lines
                fig.add_shape(
                    type="line", line=dict(dash="dash", color="gray"),
                    x0=-fc_cutoff, y0=0, x1=-fc_cutoff, y1=df[plot_y_col].max()
                )
                fig.add_shape(
                    type="line", line=dict(dash="dash", color="gray"),
                    x0=fc_cutoff, y0=0, x1=fc_cutoff, y1=df[plot_y_col].max()
                )
                fig.add_shape(
                    type="line", line=dict(dash="dash", color="gray"),
                    x0=df[x_col].min(), y0=-np.log10(pval_cutoff), 
                    x1=df[x_col].max(), y1=-np.log10(pval_cutoff)
                )
                
                # Add labels for top features
                annotations = []
                for feature in top_features:
                    row = df.loc[feature]
                    annotations.append(
                        dict(
                            x=row[x_col],
                            y=row[plot_y_col],
                            text=feature,
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )
                    )
                fig.update_layout(annotations=annotations)
                
                # Update layout
                fig.update_layout(
                    font=dict(family="Arial", size=12),
                    legend_title_text="Significance",
                    height=600,
                    width=800
                )
                
                # Save interactive plot as HTML
                html_output = output_file.replace('.png', '.html')
                fig.write_html(html_output)
                
                # Also save as static image
                fig.write_image(output_file, scale=2)
                
            else:
                # Create static plot with matplotlib
                plt.figure(figsize=figsize)
                
                # Plot each significance category
                for category, color in color_map.items():
                    subset = df[df['Significance'] == category]
                    plt.scatter(subset[x_col], subset[plot_y_col], c=color, label=category, alpha=0.7)
                
                # Add threshold lines
                plt.axvline(-fc_cutoff, color='gray', linestyle='--', alpha=0.5)
                plt.axvline(fc_cutoff, color='gray', linestyle='--', alpha=0.5)
                plt.axhline(-np.log10(pval_cutoff), color='gray', linestyle='--', alpha=0.5)
                
                # Add labels for top features
                for feature in top_features:
                    row = df.loc[feature]
                    plt.annotate(feature, xy=(row[x_col], row[plot_y_col]),
                                xytext=(5, 5), textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
                
                # Set title and labels
                if title:
                    plt.title(title, fontsize=14)
                plt.xlabel(x_col, fontsize=12)
                plt.ylabel(plot_y_label, fontsize=12)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Store figure information
            self.figures[f"{data_type}_volcano"] = {
                'path': output_file,
                'type': 'volcano',
                'title': title or f"Volcano Plot - {data_type}"
            }
            
            logging.info(f"Volcano plot saved to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error creating volcano plot: {str(e)}")
            return None
    
    def create_enrichment_network(self, data_type, output_file,
                                 term_col='term', p_col='pvalue',
                                 gene_col='genes', similarity_cutoff=0.3,
                                 max_terms=50, interactive=True,
                                 figsize=(14, 12), title=None):
        """
        Create a network visualization of enrichment results
        
        Parameters:
        -----------
        data_type : str
            Type of data to visualize (must be loaded)
        output_file : str
            Path to save the visualization
        term_col : str
            Column name for enrichment terms (default: 'term')
        p_col : str
            Column name for p-values (default: 'pvalue')
        gene_col : str
            Column name for gene lists (default: 'genes')
        similarity_cutoff : float
            Jaccard similarity cutoff for drawing edges (default: 0.3)
        max_terms : int
            Maximum number of terms to include (default: 50)
        interactive : bool
            Whether to create an interactive plot with plotly (default: True)
        figsize : tuple
            Figure size (default: (14, 12))
        title : str
            Plot title (default: None)
            
        Returns:
        --------
        str
            Path to saved visualization
        """
        logging.info(f"Creating enrichment network for {data_type} data")
        
        if data_type not in self.data:
            logging.error(f"Data type {data_type} not loaded")
            return None
        
        try:
            # Get data
            df = self.data[data_type].copy()
            
            # Check if required columns exist
            if term_col not in df.columns:
                logging.error(f"Term column {term_col} not found in data")
                return None
            if p_col not in df.columns:
                logging.error(f"P-value column {p_col} not found in data")
                return None
            if gene_col not in df.columns:
                logging.error(f"Gene column {gene_col} not found in data")
                return None
            
            # Sort by p-value and take top terms
            df = df.sort_values(p_col).head(max_terms)
            
            # Create network
            G = nx.Graph()
            
            # Process gene lists and add nodes
            term_genes = {}
            for idx, row in df.iterrows():
                term = row[term_col]
                p_value = row[p_col]
                
                # Process gene list (could be string, list, or other format)
                if isinstance(row[gene_col], str):
                    # Try different separators
                    for sep in [',', ';', '|', ' ']:
                        if sep in row[gene_col]:
                            genes = set(g.strip() for g in row[gene_col].split(sep))
                            break
                    else:
                        # No separator found, treat as single gene
                        genes = {row[gene_col].strip()}
                elif isinstance(row[gene_col], list):
                    genes = set(row[gene_col])
                else:
                    genes = set()
                
                # Store gene set for term
                term_genes[term] = genes
                
                # Add node
                node_size = len(genes)
                node_color = -np.log10(p_value)
                
                G.add_node(term, size=node_size, color=node_color, p_value=p_value)
            
            # Add edges based on gene overlap
            for term1, genes1 in term_genes.items():
                for term2, genes2 in term_genes.items():
                    if term1 >= term2:
                        continue
                    
                    # Calculate Jaccard similarity
                    intersection = len(genes1.intersection(genes2))
                    union = len(genes1.union(genes2))
                    
                    if union > 0:
                        similarity = intersection / union
                        
                        # Add edge if similarity is above threshold
                        if similarity >= similarity_cutoff:
                            G.add_edge(term1, term2, weight=similarity)
            
            # Check if network has nodes
            if not G.nodes():
                logging.error("Network has no nodes")
                return None
            
            # Create interactive network visualization with plotly
            if interactive:
                # Use force-directed layout
                pos = nx.spring_layout(G, weight='weight', k=0.2, iterations=50)
                
                # Create edge trace
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines')
                
                # Create node trace
                node_x = []
                node_y = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                
                # Get node attributes
                node_sizes = [G.nodes[node]['size'] * 10 for node in G.nodes()]
                node_colors = [G.nodes[node]['color'] for node in G.nodes()]
                
                # Create hover text
                node_text = []
                for node in G.nodes():
                    p_value = G.nodes[node]['p_value']
                    genes = term_genes[node]
                    hover_text = f"{node}<br>p-value: {p_value:.2e}<br>Genes: {len(genes)}"
                    node_text.append(hover_text)
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='Viridis',
                        size=node_sizes,
                        color=node_colors,
                        colorbar=dict(
                            thickness=15,
                            title='-log10(p-value)',
                            xanchor='left',
                            titleside='right'
                        )
                    )
                )
                
                # Create figure
                fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title=title or f"Enrichment Network - {data_type}",
                                titlefont=dict(size=16),
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                             ))
                
                # Save interactive plot as HTML
                html_output = output_file.replace('.png', '.html')
                fig.write_html(html_output)
                
                # Also save as static image
                fig.write_image(output_file, scale=2)
                
            else:
                # Create static network visualization with matplotlib
                plt.figure(figsize=figsize)
                
                # Use spring layout for node positions
                pos = nx.spring_layout(G, weight='weight', k=0.2, iterations=50)
                
                # Get node attributes
                node_sizes = [G.nodes[node]['size'] * 20 for node in G.nodes()]
                node_colors = [G.nodes[node]['color'] for node in G.nodes()]
                
                # Get edge weights
                edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
                
                # Create colormap for nodes
                cmap = plt.cm.viridis
                
                # Draw network
                nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, alpha=0.8)
                edges = nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
                
                # Add labels to larger nodes
                larger_nodes = [node for node in G.nodes() if G.nodes[node]['size'] > 5]
                labels = {node: node if len(node) < 20 else node[:17] + '...' for node in larger_nodes}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(node_colors), max(node_colors)))
                sm.set_array([])
                cbar = plt.colorbar(sm)
                cbar.set_label('-log10(p-value)', rotation=270, labelpad=15)
                
                # Set title and remove axes
                if title:
                    plt.title(title, fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Store figure information
            self.figures[f"{data_type}_network"] = {
                'path': output_file,
                'type': 'network',
                'title': title or f"Enrichment Network - {data_type}"
            }
            
            logging.info(f"Enrichment network saved to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error creating enrichment network: {str(e)}")
            return None
    
    def create_pca_plot(self, data_type, output_file,
                       group_col=None, color_col=None, shape_col=None,
                       n_components=3, feature_filter=None,
                       interactive=True, figsize=(12, 10), title=None):
        """
        Create PCA visualization with 2D and 3D plots
        
        Parameters:
        -----------
        data_type : str
            Type of data to visualize (must be loaded)
        output_file : str
            Path to save the visualization
        group_col : str
            Column name for grouping/coloring (default: None)
        color_col : str
            Column name for color mapping (default: None)
        shape_col : str
            Column name for shape mapping (default: None)
        n_components : int
            Number of PCA components to calculate (default: 3)
        feature_filter : callable
            Function to filter features (default: None)
        interactive : bool
            Whether to create an interactive plot with plotly (default: True)
        figsize : tuple
            Figure size (default: (12, 10))
        title : str
            Plot title (default: None)
            
        Returns:
        --------
        str
            Path to saved visualization
        """
        logging.info(f"Creating PCA plot for {data_type} data")
        
        if data_type not in self.data:
            logging.error(f"Data type {data_type} not loaded")
            return None
        
        try:
            # Get data
            df = self.data[data_type].copy()
            
            # Determine if data needs transposing (features as rows or columns)
            if group_col and group_col in df.columns:
                # Samples as rows, features as columns
                metadata = df[[group_col]].copy() if group_col else None
                
                if color_col and color_col in df.columns:
                    metadata[color_col] = df[color_col]
                if shape_col and shape_col in df.columns:
                    metadata[shape_col] = df[shape_col]
                
                # Remove non-numeric columns
                feature_cols = [col for col in df.columns if col not in [group_col, color_col, shape_col] and np.issubdtype(df[col].dtype, np.number)]
                X = df[feature_cols]
            else:
                # Features as rows, samples as columns
                X = df.T
                
                # Check if group information is provided
                if group_col is None:
                    metadata = None
                else:
                    # Try to load metadata from another source
                    if 'metadata' in self.data and group_col in self.data['metadata'].columns:
                        metadata = self.data['metadata'][[group_col]].copy()
                        
                        if color_col and color_col in self.data['metadata'].columns:
                            metadata[color_col] = self.data['metadata'][color_col]
                        if shape_col and shape_col in self.data['metadata'].columns:
                            metadata[shape_col] = self.data['metadata'][shape_col]
                    else:
                        metadata = None
            
            # Apply feature filter if provided
            if feature_filter is not None:
                X = X.loc[:, feature_filter(X)]
            
            # Remove columns with all NaN
            X = X.dropna(axis=1, how='all')
            
            # Impute remaining NaN values with mean
            X = X.fillna(X.mean())
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Run PCA
            pca = PCA(n_components=min(n_components, min(X_scaled.shape)))
            pca_result = pca.fit_transform(X_scaled)
            
            # Create DataFrame with PCA results
            pca_df = pd.DataFrame(data=pca_result, 
                                 columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
                                 index=X.index)
            
            # Add metadata if available
            if metadata is not None:
                for col in metadata.columns:
                    pca_df[col] = metadata.loc[pca_df.index, col]
            
            # Calculate explained variance
            explained_var = pca.explained_variance_ratio_ * 100
            
            # Create interactive 3D plot with plotly
            if interactive:
                if pca_result.shape[1] >= 3:
                    # 3D plot with first three components
                    if group_col and group_col in pca_df.columns:
                        fig = px.scatter_3d(
                            pca_df, x='PC1', y='PC2', z='PC3',
                            color=group_col,
                            symbol=shape_col if shape_col in pca_df.columns else None,
                            hover_name=pca_df.index,
                            title=title or f"PCA Plot - {data_type}",
                            labels={
                                'PC1': f'PC1 ({explained_var[0]:.1f}%)',
                                'PC2': f'PC2 ({explained_var[1]:.1f}%)',
                                'PC3': f'PC3 ({explained_var[2]:.1f}%)'
                            }
                        )
                    else:
                        fig = px.scatter_3d(
                            pca_df, x='PC1', y='PC2', z='PC3',
                            hover_name=pca_df.index,
                            title=title or f"PCA Plot - {data_type}",
                            labels={
                                'PC1': f'PC1 ({explained_var[0]:.1f}%)',
                                'PC2': f'PC2 ({explained_var[1]:.1f}%)',
                                'PC3': f'PC3 ({explained_var[2]:.1f}%)'
                            }
                        )
                else:
                    # 2D plot for two components
                    if group_col and group_col in pca_df.columns:
                        fig = px.scatter(
                            pca_df, x='PC1', y='PC2',
                            color=group_col,
                            symbol=shape_col if shape_col in pca_df.columns else None,
                            hover_name=pca_df.index,
                            title=title or f"PCA Plot - {data_type}",
                            labels={
                                'PC1': f'PC1 ({explained_var[0]:.1f}%)',
                                'PC2': f'PC2 ({explained_var[1]:.1f}%)'
                            }
                        )
                    else:
                        fig = px.scatter(
                            pca_df, x='PC1', y='PC2',
                            hover_name=pca_df.index,
                            title=title or f"PCA Plot - {data_type}",
                            labels={
                                'PC1': f'PC1 ({explained_var[0]:.1f}%)',
                                'PC2': f'PC2 ({explained_var[1]:.1f}%)'
                            }
                        )
                
                # Update layout
                fig.update_layout(
                    font=dict(family="Arial", size=12),
                    legend_title_text=group_col if group_col else None,
                    height=600,
                    width=800
                )
                
                # Save interactive plot as HTML
                html_output = output_file.replace('.png', '.html')
                fig.write_html(html_output)
                
                # Also save as static image
                fig.write_image(output_file, scale=2)
                
            else:
                # Create static plots with matplotlib
                if pca_result.shape[1] >= 3:
                    # Create figure with two subplots (2D and 3D)
                    fig = plt.figure(figsize=figsize)
                    
                    # Add 2D subplot
                    ax1 = fig.add_subplot(121)
                    
                    # Set up color mapping
                    if group_col and group_col in pca_df.columns:
                        groups = pca_df[group_col].unique()
                        colors = plt.cm.get_cmap(self.color_palette, len(groups))
                        
                        for i, group in enumerate(groups):
                            subset = pca_df[pca_df[group_col] == group]
                            ax1.scatter(subset['PC1'], subset['PC2'], color=colors(i), label=group, alpha=0.7)
                        
                        ax1.legend(title=group_col)
                    else:
                        ax1.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
                    
                    # Set labels
                    ax1.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)', fontsize=12)
                    ax1.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)', fontsize=12)
                    ax1.set_title('PCA - 2D Projection', fontsize=14)
                    ax1.grid(alpha=0.3)
                    
                    # Add 3D subplot
                    ax2 = fig.add_subplot(122, projection='3d')
                    
                    # Set up color mapping
                    if group_col and group_col in pca_df.columns:
                        for i, group in enumerate(groups):
                            subset = pca_df[pca_df[group_col] == group]
                            ax2.scatter(subset['PC1'], subset['PC2'], subset['PC3'], color=colors(i), label=group, alpha=0.7)
                    else:
                        ax2.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], alpha=0.7)
                    
                    # Set labels
                    ax2.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)', fontsize=12)
                    ax2.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)', fontsize=12)
                    ax2.set_zlabel(f'PC3 ({explained_var[2]:.1f}%)', fontsize=12)
                    ax2.set_title('PCA - 3D Projection', fontsize=14)
                    
                else:
                    # Create single 2D plot
                    fig, ax = plt.subplots(figsize=figsize)
                    
                    # Set up color mapping
                    if group_col and group_col in pca_df.columns:
                        groups = pca_df[group_col].unique()
                        colors = plt.cm.get_cmap(self.color_palette, len(groups))
                        
                        for i, group in enumerate(groups):
                            subset = pca_df[pca_df[group_col] == group]
                            ax.scatter(subset['PC1'], subset['PC2'], color=colors(i), label=group, alpha=0.7)
                        
                        ax.legend(title=group_col)
                    else:
                        ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
                    
                    # Set labels
                    ax.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)', fontsize=12)
                    ax.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)', fontsize=12)
                    ax.set_title('PCA - 2D Projection', fontsize=14)
                    ax.grid(alpha=0.3)
                
                # Set overall title if provided
                if title:
                    plt.suptitle(title, fontsize=16)
                
                plt.tight_layout()
                
                # Save figure
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Store figure information
            self.figures[f"{data_type}_pca"] = {
                'path': output_file,
                'type': 'pca',
                'title': title or f"PCA Plot - {data_type}"
            }
            
            logging.info(f"PCA plot saved to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error creating PCA plot: {str(e)}")
            return None
    
    def create_integrated_visualization(self, output_file, figure_types=None, 
                                       max_figures=6, figsize=(18, 15), title=None):
        """
        Create integrated visualization dashboard combining multiple plots
        
        Parameters:
        -----------
        output_file : str
            Path to save the visualization
        figure_types : list
            List of figure types to include (default: all)
        max_figures : int
            Maximum number of figures to include (default: 6)
        figsize : tuple
            Figure size (default: (18, 15))
        title : str
            Plot title (default: None)
            
        Returns:
        --------
        str
            Path to saved visualization
        """
        logging.info("Creating integrated visualization dashboard")
        
        if not self.figures:
            logging.error("No figures available to create dashboard")
            return None
        
        try:
            # Filter figures by type if requested
            if figure_types:
                selected_figures = {k: v for k, v in self.figures.items() if v['type'] in figure_types}
            else:
                selected_figures = self.figures
            
            # Limit to max_figures
            if len(selected_figures) > max_figures:
                selected_figures = dict(list(selected_figures.items())[:max_figures])
            
            # Calculate grid dimensions
            n_figures = len(selected_figures)
            if n_figures <= 3:
                n_rows, n_cols = 1, n_figures
            else:
                n_rows = (n_figures + 1) // 2  # Ceiling division
                n_cols = 2
            
            # Create figure with gridspec for flexible layout
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
            
            # Add each figure
            for i, (fig_key, fig_info) in enumerate(selected_figures.items()):
                # Calculate position in grid
                row = i // n_cols
                col = i % n_cols
                
                # Create subplot
                ax = fig.add_subplot(gs[row, col])
                
                # Load figure
                img = plt.imread(fig_info['path'])
                
                # Display image
                ax.imshow(img)
                ax.set_title(fig_info['title'], fontsize=12)
                ax.axis('off')
            
            # Set overall title if provided
            if title:
                plt.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Integrated visualization dashboard saved to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error creating integrated visualization: {str(e)}")
            return None
    
    def create_html_dashboard(self, output_file, include_interactive=True, title=None):
        """
        Create an HTML dashboard with all visualizations
        
        Parameters:
        -----------
        output_file : str
            Path to save the HTML dashboard
        include_interactive : bool
            Whether to include interactive versions if available (default: True)
        title : str
            Dashboard title (default: None)
            
        Returns:
        --------
        str
            Path to saved HTML dashboard
        """
        logging.info("Creating HTML dashboard")
        
        if not self.figures:
            logging.error("No figures available to create dashboard")
            return None
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Begin HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
                    h2 {{ color: #444; margin-top: 30px; }}
                    .figure-grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }}
                    .figure-card {{ background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; width: 100%; }}
                    .figure-card img {{ width: 100%; height: auto; }}
                    .figure-card iframe {{ width: 100%; height: 600px; border: none; }}
                    .figure-content {{ padding: 15px; }}
                    .figure-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                    .figure-type {{ color: #666; margin-bottom: 5px; font-style: italic; }}
                    .nav {{ background-color: #333; color: white; padding: 10px 0; position: sticky; top: 0; z-index: 100; }}
                    .nav-container {{ display: flex; justify-content: center; gap: 15px; }}
                    .nav-item {{ cursor: pointer; padding: 5px 10px; }}
                    .nav-item:hover {{ background-color: #555; border-radius: 4px; }}
                    @media (min-width: 768px) {{
                        .figure-card {{ width: calc(50% - 20px); }}
                    }}
                    @media (min-width: 1024px) {{
                        .figure-card {{ width: calc(33.33% - 20px); }}
                    }}
                </style>
            </head>
            <body>
                <div class="nav">
                    <div class="nav-container">
                        <div class="nav-item" onclick="scrollToTop()">Top</div>
            """.format(title=title or "Proteomics Data Visualization Dashboard")
            
            # Add navigation items for each figure type
            figure_types = list(set(fig_info['type'] for fig_info in self.figures.values()))
            for fig_type in figure_types:
                html_content += f'<div class="nav-item" onclick="scrollToSection(\'{fig_type}\')">{fig_type.title()}</div>\n'
            
            html_content += """
                    </div>
                </div>
                <div class="container">
                    <h1>{title}</h1>
            """.format(title=title or "Proteomics Data Visualization Dashboard")
            
            # Group figures by type
            figures_by_type = {}
            for fig_key, fig_info in self.figures.items():
                fig_type = fig_info['type']
                if fig_type not in figures_by_type:
                    figures_by_type[fig_type] = []
                figures_by_type[fig_type].append((fig_key, fig_info))
            
            # Add sections for each figure type
            for fig_type, figures in figures_by_type.items():
                html_content += f"""
                    <h2 id="{fig_type}">{fig_type.title()} Visualizations</h2>
                    <div class="figure-grid">
                """
                
                for fig_key, fig_info in figures:
                    # Check if interactive version exists
                    interactive_path = fig_info['path'].replace('.png', '.html')
                    has_interactive = include_interactive and os.path.exists(interactive_path)
                    
                    # Get relative paths
                    rel_path = os.path.relpath(fig_info['path'], output_dir)
                    rel_interactive_path = os.path.relpath(interactive_path, output_dir) if has_interactive else None
                    
                    html_content += f"""
                        <div class="figure-card">
                            <div class="figure-content">
                                <div class="figure-title">{fig_info['title']}</div>
                                <div class="figure-type">{fig_type.title()}</div>
                            </div>
                    """
                    
                    if has_interactive:
                        html_content += f"""
                            <iframe src="{rel_interactive_path}" allowfullscreen></iframe>
                        """
                    else:
                        html_content += f"""
                            <img src="{rel_path}" alt="{fig_info['title']}">
                        """
                    
                    html_content += """
                        </div>
                    """
                
                html_content += """
                    </div>
                """
            
            # Close HTML
            html_content += """
                </div>
                <script>
                    function scrollToTop() {
                        window.scrollTo({top: 0, behavior: 'smooth'});
                    }
                    
                    function scrollToSection(sectionId) {
                        const section = document.getElementById(sectionId);
                        if (section) {
                            section.scrollIntoView({behavior: 'smooth'});
                        }
                    }
                </script>
            </body>
            </html>
            """
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logging.info(f"HTML dashboard saved to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error creating HTML dashboard: {str(e)}")
            return None
    
    def create_all_visualizations(self, data_types, output_dir, create_dashboard=True):
        """
        Create all visualizations for specified data types
        
        Parameters:
        -----------
        data_types : dict
            Dictionary mapping data types to visualization types
        output_dir : str
            Directory to save visualizations
        create_dashboard : bool
            Whether to create a dashboard (default: True)
            
        Returns:
        --------
        dict
            Dictionary of created visualizations
        """
        logging.info(f"Creating all visualizations for {len(data_types)} data types")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store all created visualizations
        created_viz = {}
        
        try:
            # For each data type, create specified visualizations
            for data_type, viz_types in data_types.items():
                if data_type not in self.data:
                    logging.error(f"Data type {data_type} not loaded")
                    continue
                
                # Create data type directory
                data_dir = os.path.join(output_dir, data_type)
                os.makedirs(data_dir, exist_ok=True)
                
                # Store visualizations for this data type
                created_viz[data_type] = {}
                
                # Create each visualization type
                for viz_type in viz_types:
                    if viz_type == 'heatmap':
                        output_file = os.path.join(data_dir, f"{data_type}_heatmap.png")
                        result = self.create_heatmap(data_type, output_file)
                        if result:
                            created_viz[data_type]['heatmap'] = result
                    
                    elif viz_type == 'volcano':
                        output_file = os.path.join(data_dir, f"{data_type}_volcano.png")
                        result = self.create_volcano_plot(data_type, output_file)
                        if result:
                            created_viz[data_type]['volcano'] = result
                    
                    elif viz_type == 'pca':
                        output_file = os.path.join(data_dir, f"{data_type}_pca.png")
                        result = self.create_pca_plot(data_type, output_file)
                        if result:
                            created_viz[data_type]['pca'] = result
                    
                    elif viz_type == 'network':
                        output_file = os.path.join(data_dir, f"{data_type}_network.png")
                        result = self.create_enrichment_network(data_type, output_file)
                        if result:
                            created_viz[data_type]['network'] = result
                    
                    else:
                        logging.warning(f"Unknown visualization type: {viz_type}")
            
            # Create integrated dashboard if requested
            if create_dashboard and self.figures:
                dashboard_file = os.path.join(output_dir, "dashboard.png")
                self.create_integrated_visualization(dashboard_file)
                
                # Create HTML dashboard
                html_dashboard_file = os.path.join(output_dir, "dashboard.html")
                self.create_html_dashboard(html_dashboard_file)
            
            logging.info(f"Created visualizations for {len(created_viz)} data types")
            return created_viz
            
        except Exception as e:
            logging.error(f"Error creating all visualizations: {str(e)}")
            return created_viz

def main():
    try:
        # Get inputs and outputs from Snakemake
        data_files = {}
        for input_type, input_file in snakemake.input.items():
            data_files[input_type] = input_file
        
        output_dir = snakemake.output.output_dir
        dashboard_file = snakemake.output.dashboard
        
        # Get parameters
        color_palette = snakemake.params.get("color_palette", "viridis")
        viz_config = snakemake.params.get("viz_config", {})
        
        # Initialize visualization tool
        viz_tool = AdvancedVisualization(color_palette=color_palette)
        
        # Load data
        viz_tool.load_data(data_files)
        
        # Create all visualizations
        viz_tool.create_all_visualizations(viz_config, output_dir)
        
        # Ensure dashboard is created
        if not os.path.exists(dashboard_file):
            viz_tool.create_html_dashboard(dashboard_file)
        
        logging.info("Advanced visualization workflow completed successfully")
        
    except Exception as e:
        logging.error(f"Error in advanced visualization workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()


