#!/usr/bin/env python3
"""
Visualization module for enrichment analysis results.

This module provides enhanced visualization capabilities for enrichment analysis results:
- Heatmap visualizations for GO terms
- Network visualizations for pathway interactions
- Bubble plots for enrichment results
- Interactive visualizations using plotly

Author: Jiyang Jiang
Date: March 11, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def create_enrichment_heatmap(enrichment_results: Dict[str, pd.DataFrame], 
                             output_dir: str,
                             top_n: int = 15,
                             width: int = 12,
                             height: int = 14) -> None:
    """
    Create heatmap visualizations for enrichment results.
    
    Parameters:
    -----------
    enrichment_results : Dict[str, pd.DataFrame]
        Dictionary with source as key and DataFrame of enrichment results as value
    output_dir : str
        Directory to save output files
    top_n : int
        Number of top terms to include in the heatmap
    width : int
        Width of the plot in inches
    height : int
        Height of the plot in inches
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each enrichment result
    for source, df in enrichment_results.items():
        if df is None or df.empty or len(df) < 3:
            continue
            
        # Get top terms
        top_terms = df.sort_values('p_value').head(top_n).copy()
        
        # Create -log10(p-value) for better visualization
        top_terms['-log10(p)'] = -np.log10(top_terms['p_value'])
        
        # Make term names shorter if needed
        top_terms['short_name'] = top_terms['name'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
        
        # Sort for visualization
        top_terms = top_terms.sort_values('-log10(p)', ascending=True)
        
        # Create heatmap based on term similarity
        # For simplicity, use -log10(p-value) for color coding
        plt.figure(figsize=(width, height))
        
        # Plot heatmap
        ax = sns.barh(y='short_name', x='-log10(p)', data=top_terms, 
                      color=sns.color_palette("viridis", n_colors=len(top_terms)))
        
        # Add gene counts
        for i, v in enumerate(top_terms['-log10(p)']):
            gene_count = len(top_terms.iloc[i]['intersection'].split(','))
            ax.text(v + 0.1, i, f"({gene_count} genes)", va='center')
        
        # Add title and labels
        plt.title(f'Top {top_n} Enriched Terms - {source}', fontsize=14)
        plt.xlabel('-log10(p-value)', fontsize=12)
        plt.ylabel('Term', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{output_dir}/{source}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_enrichment_network(enrichment_results: Dict[str, pd.DataFrame],
                             output_dir: str,
                             min_overlap_ratio: float = 0.5,
                             max_terms: int = 30) -> None:
    """
    Create a network visualization of the enrichment results based on term overlap.
    
    Parameters:
    -----------
    enrichment_results : Dict[str, pd.DataFrame]
        Dictionary with source as key and DataFrame of enrichment results as value
    output_dir : str
        Directory to save output files
    min_overlap_ratio : float
        Minimum Jaccard similarity to draw an edge between terms
    max_terms : int
        Maximum number of terms to include in the network
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each enrichment result
    for source, df in enrichment_results.items():
        if df is None or df.empty or len(df) < 5:
            continue
            
        # Get top terms
        top_terms = df.sort_values('p_value').head(max_terms).copy()
        
        # Extract gene sets for each term
        term_genes = {}
        for idx, row in top_terms.iterrows():
            genes = set(row['intersection'].split(','))
            term_genes[row['name']] = genes
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for term, genes in term_genes.items():
            # Use abbreviated term name for node label
            short_term = term[:40] + '...' if len(term) > 40 else term
            G.add_node(term, 
                      label=short_term,
                      size=len(genes),
                      genes=genes,
                      pvalue=top_terms[top_terms['name'] == term]['p_value'].values[0])
        
        # Add edges based on gene overlap
        for term1, genes1 in term_genes.items():
            for term2, genes2 in term_genes.items():
                if term1 >= term2:
                    continue
                
                # Calculate Jaccard similarity
                intersection = genes1.intersection(genes2)
                union = genes1.union(genes2)
                
                if len(union) > 0:
                    jaccard = len(intersection) / len(union)
                    
                    # Add edge if similarity is above threshold
                    if jaccard >= min_overlap_ratio:
                        G.add_edge(term1, term2, weight=jaccard, 
                                  label=f"{jaccard:.2f}")
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        # Skip if the graph is empty
        if len(G.nodes()) < 3:
            continue
        
        # Create plot
        plt.figure(figsize=(14, 12))
        
        # Node sizes based on gene count
        node_sizes = [G.nodes[node]['size'] * 20 for node in G.nodes()]
        
        # Node colors based on p-value (using log scale)
        cmap = plt.cm.viridis_r
        pvalues = np.array([G.nodes[node]['pvalue'] for node in G.nodes()])
        log_pvalues = -np.log10(pvalues)
        norm_pvalues = (log_pvalues - log_pvalues.min()) / (log_pvalues.max() - log_pvalues.min())
        node_colors = [cmap(p) for p in norm_pvalues]
        
        # Edge weights
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        
        # Use spring layout weighted by edge weights
        pos = nx.spring_layout(G, weight='weight', k=0.20, iterations=50)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
        
        # Add labels only to larger nodes (important terms)
        larger_nodes = [node for node in G.nodes() if G.nodes[node]['size'] > 5]
        labels = {node: G.nodes[node]['label'] for node in larger_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
        
        # Add title
        plt.title(f'Enrichment Term Network - {source}', fontsize=14)
        
        # Add colorbar for p-value
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(log_pvalues)
        cbar = plt.colorbar(sm)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=20)
        
        # Remove axes
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{output_dir}/{source}_network.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_bubble_plot(enrichment_results: Dict[str, pd.DataFrame],
                      output_dir: str,
                      max_terms: int = 20) -> None:
    """
    Create bubble plots for enrichment results.
    
    Parameters:
    -----------
    enrichment_results : Dict[str, pd.DataFrame]
        Dictionary with source as key and DataFrame of enrichment results as value
    output_dir : str
        Directory to save output files
    max_terms : int
        Maximum number of terms to include in the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each enrichment result
    for source, df in enrichment_results.items():
        if df is None or df.empty or len(df) < 5:
            continue
            
        # Get top terms
        top_terms = df.sort_values('p_value').head(max_terms).copy()
        
        # Create bubble plot data
        plot_data = []
        
        for idx, row in top_terms.iterrows():
            genes = row['intersection'].split(',')
            
            # Make term name shorter if needed
            short_name = row['name'][:40] + '...' if len(row['name']) > 40 else row['name']
            
            plot_data.append({
                'Term': short_name,
                'P-value': row['p_value'],
                'Gene Count': len(genes),
                'Source': source,
                '-log10(p)': -np.log10(row['p_value'])
            })
        
        # Create DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        # Sort by p-value
        plot_df = plot_df.sort_values('P-value')
        
        # Create bubble plot
        fig = px.scatter(
            plot_df,
            y='Term',
            x='-log10(p)',
            size='Gene Count',
            color='-log10(p)',
            color_continuous_scale='viridis',
            hover_name='Term',
            hover_data={
                'Term': False,
                '-log10(p)': True,
                'P-value': ':.2e',
                'Gene Count': True
            },
            title=f'Top Enriched Terms - {source}'
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            width=900,
            xaxis_title='-log10(p-value)',
            yaxis_title='',
            coloraxis_showscale=False,
            margin=dict(l=250)
        )
        
        # Save as both PNG and HTML for interactivity
        fig.write_image(f"{output_dir}/{source}_bubble_plot.png")
        fig.write_html(f"{output_dir}/{source}_bubble_plot.html")

def create_combined_enrichment_viz(all_enrichment_results: Dict[str, pd.DataFrame],
                                  output_dir: str,
                                  max_terms_per_source: int = 5) -> None:
    """
    Create a combined visualization of top enriched terms across all sources.
    
    Parameters:
    -----------
    all_enrichment_results : Dict[str, pd.DataFrame]
        Dictionary with source as key and DataFrame of enrichment results as value
    output_dir : str
        Directory to save output files
    max_terms_per_source : int
        Maximum number of terms to include per source
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter sources with enough results
    valid_sources = {}
    for source, df in all_enrichment_results.items():
        if df is not None and not df.empty and len(df) >= max_terms_per_source:
            valid_sources[source] = df
    
    if len(valid_sources) <= 1:
        return  # Not enough sources for comparison
    
    # Prepare data for plotting
    plot_data = []
    
    for source, df in valid_sources.items():
        # Get top terms
        top_terms = df.sort_values('p_value').head(max_terms_per_source)
        
        for idx, row in top_terms.iterrows():
            genes = row['intersection'].split(',') if 'intersection' in row else []
            
            # Make term name shorter if needed
            short_name = row['name'][:40] + '...' if len(row['name']) > 40 else row['name']
            
            plot_data.append({
                'Term': short_name,
                'Category': source,
                'P-value': row['p_value'],
                'Gene Count': len(genes),
                '-log10(p)': -np.log10(row['p_value'])
            })
    
    # Create DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Sort by category and p-value
    plot_df = plot_df.sort_values(['Category', 'P-value'])
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    
    # Assign colors to categories
    categories = plot_df['Category'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    category_colors = dict(zip(categories, colors))
    
    # Create scatter plot
    for i, category in enumerate(categories):
        cat_data = plot_df[plot_df['Category'] == category]
        plt.scatter(
            cat_data['-log10(p)'], 
            cat_data['Term'],
            s=cat_data['Gene Count'] * 20, 
            alpha=0.7,
            c=[category_colors[category]],
            label=category
        )
    
    # Add legend
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set title and labels
    plt.title('Top Enriched Terms Across Categories', fontsize=14)
    plt.xlabel('-log10(p-value)', fontsize=12)
    plt.ylabel('Term', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_dir}/combined_enrichment.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create an interactive version with plotly
    fig = px.scatter(
        plot_df,
        y='Term',
        x='-log10(p)',
        size='Gene Count',
        color='Category',
        hover_name='Term',
        hover_data={
            'Term': False,
            '-log10(p)': True,
            'P-value': ':.2e',
            'Gene Count': True,
            'Category': True
        },
        title='Top Enriched Terms Across Categories'
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        xaxis_title='-log10(p-value)',
        yaxis_title='',
        margin=dict(l=250)
    )
    
    # Save as both PNG and HTML for interactivity
    fig.write_image(f"{output_dir}/combined_enrichment_interactive.png")
    fig.write_html(f"{output_dir}/combined_enrichment_interactive.html")

def generate_all_visualizations(enrichment_results_file: str, output_dir: str) -> None:
    """
    Generate all visualizations for enrichment results.
    
    Parameters:
    -----------
    enrichment_results_file : str
        Path to JSON file with all enrichment results
    output_dir : str
        Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load enrichment results
    with open(enrichment_results_file, 'r') as f:
        all_results = json.load(f)
    
    # Extract GO results
    go_results = {}
    for key, value in all_results.get('go_terms', {}).items():
        if value:
            go_results[key] = pd.DataFrame(value)
    
    # Extract pathway results
    pathway_results = {}
    for key, value in all_results.get('pathways', {}).items():
        if value and not key.endswith('_GSEA'):
            pathway_results[key] = pd.DataFrame(value)
    
    # Generate visualizations
    # 1. Heatmaps
    create_enrichment_heatmap(go_results, f"{output_dir}/heatmaps")
    create_enrichment_heatmap(pathway_results, f"{output_dir}/heatmaps")
    
    # 2. Networks
    create_enrichment_network(go_results, f"{output_dir}/networks")
    create_enrichment_network(pathway_results, f"{output_dir}/networks")
    
    # 3. Bubble plots
    create_bubble_plot(go_results, f"{output_dir}/bubbles")
    create_bubble_plot(pathway_results, f"{output_dir}/bubbles")
    
    # 4. Combined visualization
    all_enrichment = {**go_results, **pathway_results}
    create_combined_enrichment_viz(all_enrichment, output_dir)

if __name__ == "__main__":
    # This can be run as a standalone script
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for enrichment results')
    parser.add_argument('--results', type=str, required=True, help='Path to JSON file with all enrichment results')
    parser.add_argument('--output', type=str, required=True, help='Directory to save output files')
    
    args = parser.parse_args()
    
    generate_all_visualizations(args.results, args.output)


