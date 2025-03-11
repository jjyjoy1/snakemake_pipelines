#!/usr/bin/env python3
"""
Comprehensive enrichment analysis for proteomics data.

This script performs:
1. Gene Ontology (GO) enrichment analysis (Biological Process, Molecular Function, Cellular Component)
2. Pathway analysis using multiple databases (KEGG, Reactome, WikiPathways)
3. Visualization of enrichment results
4. Export of results in various formats

Author: Claude
Date: March 11, 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import gseapy as gp
from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
from gprofiler import GProfiler
import argparse
from pathlib import Path
import json
from typing import List, Dict, Tuple, Set, Optional, Union
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(snakemake.log[0]),
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_gene_list(diff_expr_file: str, 
                   sig_threshold: float = 0.05, 
                   fc_threshold: float = 1.0) -> Tuple[List[str], List[str]]:
    """
    Parse differential expression results and extract significant gene/protein list.
    
    Parameters:
    -----------
    diff_expr_file : str
        Path to differential expression results file
    sig_threshold : float
        Significance threshold for adjusted p-value
    fc_threshold : float
        Fold change threshold (absolute log2FC)
        
    Returns:
    --------
    Tuple[List[str], List[str]]
        Tuple containing (significant_genes, background_genes)
    """
    logging.info(f"Reading differential expression results from {diff_expr_file}")
    
    try:
        df = pd.read_csv(diff_expr_file, sep='\t', index_col=0)
        
        # Check required columns
        required_cols = ['padj', 'log2FC']
        if not all(col in df.columns for col in required_cols):
            available_cols = list(df.columns)
            logging.error(f"Missing required columns. Available: {available_cols}")
            raise ValueError(f"Differential expression file must contain: {required_cols}")
        
        # Get significant genes based on thresholds
        sig_genes = df[(df['padj'] < sig_threshold) & 
                       (abs(df['log2FC']) > fc_threshold)].index.tolist()
        
        # Get background (all tested genes/proteins)
        background = df.index.tolist()
        
        logging.info(f"Found {len(sig_genes)} significant genes/proteins out of {len(background)} total")
        
        return sig_genes, background
        
    except Exception as e:
        logging.error(f"Error parsing gene list: {str(e)}")
        raise

def get_ranked_list(diff_expr_file: str) -> pd.Series:
    """
    Create a ranked list of genes for GSEA-like analysis.
    
    Parameters:
    -----------
    diff_expr_file : str
        Path to differential expression results file
        
    Returns:
    --------
    pd.Series
        Series with gene names as index and ranking metric as values
    """
    logging.info(f"Creating ranked gene list from {diff_expr_file}")
    
    try:
        df = pd.read_csv(diff_expr_file, sep='\t', index_col=0)
        
        # Check if file has log2FC column
        if 'log2FC' not in df.columns:
            available_cols = list(df.columns)
            logging.error(f"No log2FC column found. Available columns: {available_cols}")
            raise ValueError("Differential expression file must contain log2FC column")
        
        # Create ranking metric: -log10(padj) * sign(log2FC)
        if 'padj' in df.columns:
            # Add small value to avoid log(0)
            df['padj'] = df['padj'].replace(0, 1e-300)
            df['ranking_metric'] = -np.log10(df['padj']) * np.sign(df['log2FC'])
        else:
            # If no p-value, just use log2FC
            df['ranking_metric'] = df['log2FC']
        
        # Sort and return as Series
        ranked_list = df['ranking_metric'].sort_values(ascending=False)
        
        logging.info(f"Created ranked list with {len(ranked_list)} genes")
        
        return ranked_list
        
    except Exception as e:
        logging.error(f"Error creating ranked list: {str(e)}")
        raise

def perform_go_enrichment(sig_genes: List[str], 
                         background: List[str],
                         organism: str = 'hsapiens',
                         output_dir: str = './results') -> Dict[str, pd.DataFrame]:
    """
    Perform Gene Ontology enrichment analysis for biological process,
    molecular function, and cellular component categories.
    
    Parameters:
    -----------
    sig_genes : List[str]
        List of significant gene/protein IDs
    background : List[str]
        List of all tested gene/protein IDs
    organism : str
        Organism name (default: 'hsapiens')
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with GO categories as keys and result DataFrames as values
    """
    logging.info(f"Performing GO enrichment analysis for {len(sig_genes)} genes")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use gprofiler for GO enrichment
        gp_client = GProfiler(return_dataframe=True)
        
        # Configure query
        query = sig_genes
        
        # Run enrichment with background
        results = gp_client.profile(
            organism=organism,
            query=query, 
            domain_scope='annotated',  # Use only genes that have annotations
            sources=['GO:BP', 'GO:MF', 'GO:CC'],  # Gene Ontology categories
            background=background,
            no_evidences=False,
            user_threshold=0.05,  # FDR threshold
            all_results=False,
            ordered=False  # Unordered query
        )
        
        # If no results found
        if results.empty:
            logging.warning("No significant GO terms found")
            return {}
        
        # Split results by GO category
        go_results = {}
        
        # Process biological process terms
        bp_results = results[results['source'] == 'GO:BP'].copy()
        if not bp_results.empty:
            bp_results = bp_results.sort_values('p_value')
            go_results['GO_BP'] = bp_results
            bp_results.to_csv(f"{output_dir}/go_bp_enrichment.tsv", sep='\t', index=False)
            logging.info(f"Found {len(bp_results)} enriched biological process terms")
        
        # Process molecular function terms
        mf_results = results[results['source'] == 'GO:MF'].copy()
        if not mf_results.empty:
            mf_results = mf_results.sort_values('p_value')
            go_results['GO_MF'] = mf_results
            mf_results.to_csv(f"{output_dir}/go_mf_enrichment.tsv", sep='\t', index=False)
            logging.info(f"Found {len(mf_results)} enriched molecular function terms")
        
        # Process cellular component terms
        cc_results = results[results['source'] == 'GO:CC'].copy()
        if not cc_results.empty:
            cc_results = cc_results.sort_values('p_value')
            go_results['GO_CC'] = cc_results
            cc_results.to_csv(f"{output_dir}/go_cc_enrichment.tsv", sep='\t', index=False)
            logging.info(f"Found {len(cc_results)} enriched cellular component terms")
        
        return go_results
        
    except Exception as e:
        logging.error(f"Error in GO enrichment: {str(e)}")
        raise

def perform_pathway_analysis(sig_genes: List[str], 
                            background: List[str], 
                            ranked_list: Optional[pd.Series] = None,
                            organism: str = 'hsapiens',
                            output_dir: str = './results') -> Dict[str, pd.DataFrame]:
    """
    Perform pathway enrichment analysis using multiple databases.
    
    Parameters:
    -----------
    sig_genes : List[str]
        List of significant gene/protein IDs
    background : List[str]
        List of all tested gene/protein IDs
    ranked_list : Optional[pd.Series]
        Ranked gene list for GSEA-like analysis
    organism : str
        Organism name (default: 'hsapiens')
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with pathway databases as keys and result DataFrames as values
    """
    logging.info(f"Performing pathway enrichment analysis")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    pathway_results = {}
    
    try:
        # Use gprofiler for over-representation analysis
        gp_client = GProfiler(return_dataframe=True)
        
        # Run enrichment with multiple pathway databases
        results = gp_client.profile(
            organism=organism,
            query=sig_genes,
            domain_scope='annotated',
            sources=['KEGG', 'REAC', 'WP'],  # KEGG, Reactome, WikiPathways
            background=background,
            no_evidences=False,
            user_threshold=0.05,
            all_results=False
        )
        
        # If no results found
        if results.empty:
            logging.warning("No significant pathway terms found in ORA")
        else:
            # Split results by database
            # Process KEGG results
            kegg_results = results[results['source'] == 'KEGG'].copy()
            if not kegg_results.empty:
                kegg_results = kegg_results.sort_values('p_value')
                pathway_results['KEGG'] = kegg_results
                kegg_results.to_csv(f"{output_dir}/kegg_enrichment.tsv", sep='\t', index=False)
                logging.info(f"Found {len(kegg_results)} enriched KEGG pathways")
            
            # Process Reactome results
            reactome_results = results[results['source'] == 'REAC'].copy()
            if not reactome_results.empty:
                reactome_results = reactome_results.sort_values('p_value')
                pathway_results['Reactome'] = reactome_results
                reactome_results.to_csv(f"{output_dir}/reactome_enrichment.tsv", sep='\t', index=False)
                logging.info(f"Found {len(reactome_results)} enriched Reactome pathways")
            
            # Process WikiPathways results
            wp_results = results[results['source'] == 'WP'].copy()
            if not wp_results.empty:
                wp_results = wp_results.sort_values('p_value')
                pathway_results['WikiPathways'] = wp_results
                wp_results.to_csv(f"{output_dir}/wikipathways_enrichment.tsv", sep='\t', index=False)
                logging.info(f"Found {len(wp_results)} enriched WikiPathways")
        
        # If ranked list is provided, also perform GSEA
        if ranked_list is not None and len(ranked_list) > 15:
            logging.info("Performing GSEA-like analysis with ranked gene list")
            
            # For KEGG
            try:
                kegg_gsea = gp.gsea(
                    data=ranked_list,
                    gene_sets='KEGG_2021_Human',
                    permutation_num=1000,
                    outdir=f"{output_dir}/kegg_gsea",
                    seed=42,
                    no_plot=False
                )
                kegg_gsea_res = kegg_gsea.res2d
                if not kegg_gsea_res.empty:
                    pathway_results['KEGG_GSEA'] = kegg_gsea_res
                    kegg_gsea_res.to_csv(f"{output_dir}/kegg_gsea_results.tsv", sep='\t')
                    logging.info(f"Found {len(kegg_gsea_res)} GSEA enriched KEGG pathways")
            except Exception as e:
                logging.warning(f"KEGG GSEA analysis failed: {str(e)}")
            
            # For Reactome
            try:
                reactome_gsea = gp.gsea(
                    data=ranked_list,
                    gene_sets='Reactome_2022',
                    permutation_num=1000,
                    outdir=f"{output_dir}/reactome_gsea",
                    seed=42,
                    no_plot=False
                )
                reactome_gsea_res = reactome_gsea.res2d
                if not reactome_gsea_res.empty:
                    pathway_results['Reactome_GSEA'] = reactome_gsea_res
                    reactome_gsea_res.to_csv(f"{output_dir}/reactome_gsea_results.tsv", sep='\t')
                    logging.info(f"Found {len(reactome_gsea_res)} GSEA enriched Reactome pathways")
            except Exception as e:
                logging.warning(f"Reactome GSEA analysis failed: {str(e)}")
            
            # For WikiPathways
            try:
                wp_gsea = gp.gsea(
                    data=ranked_list,
                    gene_sets='WikiPathways_2021_Human',
                    permutation_num=1000,
                    outdir=f"{output_dir}/wikipathways_gsea",
                    seed=42,
                    no_plot=False
                )
                wp_gsea_res = wp_gsea.res2d
                if not wp_gsea_res.empty:
                    pathway_results['WikiPathways_GSEA'] = wp_gsea_res
                    wp_gsea_res.to_csv(f"{output_dir}/wikipathways_gsea_results.tsv", sep='\t')
                    logging.info(f"Found {len(wp_gsea_res)} GSEA enriched WikiPathways")
            except Exception as e:
                logging.warning(f"WikiPathways GSEA analysis failed: {str(e)}")
                
        return pathway_results
            
    except Exception as e:
        logging.error(f"Error in pathway analysis: {str(e)}")
        raise

def visualize_enrichment(go_results: Dict[str, pd.DataFrame], 
                        pathway_results: Dict[str, pd.DataFrame],
                        output_dir: str = './results') -> None:
    """
    Create visualization plots for enrichment results.
    
    Parameters:
    -----------
    go_results : Dict[str, pd.DataFrame]
        GO enrichment results
    pathway_results : Dict[str, pd.DataFrame]
        Pathway enrichment results
    output_dir : str
        Directory to save plots
    """
    logging.info("Creating visualization plots for enrichment results")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Plot GO terms (top 15 for each category)
        for category, results_df in go_results.items():
            if results_df is None or results_df.empty:
                continue
                
            # Get top 15 terms
            plot_df = results_df.head(15).copy()
            
            if len(plot_df) < 3:
                continue  # Skip if too few terms
                
            # Create -log10(p-value) for plotting
            plot_df['-log10(p_value)'] = -np.log10(plot_df['p_value'])
            
            # Sort for plotting
            plot_df = plot_df.sort_values('-log10(p_value)')
            
            # Make the term names shorter if needed
            plot_df['term_name_short'] = plot_df['name'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
            
            # Plot
            plt.figure(figsize=(10, 8))
            bar_plot = sns.barplot(x='-log10(p_value)', y='term_name_short', data=plot_df)
            
            # Customize plot
            plt.title(f'Top {len(plot_df)} {category} Terms', fontsize=14)
            plt.xlabel('-log10(p-value)', fontsize=12)
            plt.ylabel('GO Term', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f"{output_dir}/{category}_top_terms.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Created bar plot for {category} terms")
        
        # Plot pathway terms (top 15 for each database)
        for db, results_df in pathway_results.items():
            if results_df is None or results_df.empty:
                continue
                
            # Skip GSEA results for this plotting
            if db.endswith('_GSEA'):
                continue
                
            # Get top 15 terms
            plot_df = results_df.head(15).copy()
            
            if len(plot_df) < 3:
                continue  # Skip if too few terms
                
            # Create -log10(p-value) for plotting
            plot_df['-log10(p_value)'] = -np.log10(plot_df['p_value'])
            
            # Sort for plotting
            plot_df = plot_df.sort_values('-log10(p_value)')
            
            # Make the term names shorter if needed
            plot_df['term_name_short'] = plot_df['name'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
            
            # Plot
            plt.figure(figsize=(10, 8))
            bar_plot = sns.barplot(x='-log10(p_value)', y='term_name_short', data=plot_df)
            
            # Customize plot
            plt.title(f'Top {len(plot_df)} {db} Pathways', fontsize=14)
            plt.xlabel('-log10(p-value)', fontsize=12)
            plt.ylabel('Pathway', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f"{output_dir}/{db}_top_pathways.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Created bar plot for {db} pathways")
            
        # Create enrichment map for visual network representation
        # (This would require additional dependencies like networkx)
        
    except Exception as e:
        logging.error(f"Error creating visualization: {str(e)}")

def create_summary_report(go_results: Dict[str, pd.DataFrame], 
                        pathway_results: Dict[str, pd.DataFrame],
                        output_file: str = './results/enrichment_summary.html') -> None:
    """
    Create an HTML summary report of enrichment results.
    
    Parameters:
    -----------
    go_results : Dict[str, pd.DataFrame]
        GO enrichment results
    pathway_results : Dict[str, pd.DataFrame]
        Pathway enrichment results
    output_file : str
        Path to output HTML file
    """
    logging.info("Creating HTML summary report")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enrichment Analysis Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2C3E50; }
                h2 { color: #2980B9; }
                h3 { color: #3498DB; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .summary { background-color: #eef6f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Enrichment Analysis Summary Report</h1>
            <div class="summary">
                <h2>Analysis Overview</h2>
                <p>This report summarizes the results of Gene Ontology and Pathway enrichment analyses.</p>
        """
        
        # Add GO summary
        if go_results:
            html_content += "<h2>Gene Ontology Enrichment Results</h2>"
            
            for category, results_df in go_results.items():
                if results_df is None or results_df.empty:
                    html_content += f"<p>No significant terms found for {category}.</p>"
                    continue
                    
                # Get number of results
                num_results = len(results_df)
                
                html_content += f"""
                <h3>{category} ({num_results} significant terms)</h3>
                <p>Top 10 most significant terms:</p>
                <table>
                    <tr>
                        <th>Term Name</th>
                        <th>Term ID</th>
                        <th>P-value</th>
                        <th>FDR</th>
                        <th>Genes</th>
                    </tr>
                """
                
                # Add top 10 rows
                for idx, row in results_df.head(10).iterrows():
                    # Take only the first 5 genes if there are many
                    genes = row['intersection'].split(',')
                    genes_display = ', '.join(genes[:5])
                    if len(genes) > 5:
                        genes_display += f"... (+{len(genes)-5} more)"
                        
                    html_content += f"""
                    <tr>
                        <td>{row['name']}</td>
                        <td>{row['native']}</td>
                        <td>{row['p_value']:.2e}</td>
                        <td>{row['p_value']:.2e}</td>
                        <td>{genes_display}</td>
                    </tr>
                    """
                
                html_content += "</table>"
                
                # Add image if it exists
                image_path = f"{os.path.dirname(output_file)}/{category}_top_terms.png"
                if os.path.exists(image_path):
                    # Get relative path from the HTML file to the image
                    rel_path = os.path.basename(image_path)
                    html_content += f'<img src="{rel_path}" alt="{category} visualization">'
        
        # Add Pathway summary
        if pathway_results:
            html_content += "<h2>Pathway Enrichment Results</h2>"
            
            for db, results_df in pathway_results.items():
                if results_df is None or results_df.empty:
                    html_content += f"<p>No significant pathways found for {db}.</p>"
                    continue
                
                # Skip visualization for GSEA results
                if db.endswith('_GSEA'):
                    continue
                    
                # Get number of results
                num_results = len(results_df)
                
                html_content += f"""
                <h3>{db} ({num_results} significant pathways)</h3>
                <p>Top 10 most significant pathways:</p>
                <table>
                    <tr>
                        <th>Pathway Name</th>
                        <th>Pathway ID</th>
                        <th>P-value</th>
                        <th>FDR</th>
                        <th>Genes</th>
                    </tr>
                """
                
                # Add top 10 rows
                for idx, row in results_df.head(10).iterrows():
                    # Take only the first 5 genes if there are many
                    genes = row['intersection'].split(',')
                    genes_display = ', '.join(genes[:5])
                    if len(genes) > 5:
                        genes_display += f"... (+{len(genes)-5} more)"
                        
                    html_content += f"""
                    <tr>
                        <td>{row['name']}</td>
                        <td>{row['native']}</td>
                        <td>{row['p_value']:.2e}</td>
                        <td>{row['p_value']:.2e}</td>
                        <td>{genes_display}</td>
                    </tr>
                    """
                
                html_content += "</table>"
                
                # Add image if it exists
                image_path = f"{os.path.dirname(output_file)}/{db}_top_pathways.png"
                if os.path.exists(image_path):
                    # Get relative path from the HTML file to the image
                    rel_path = os.path.basename(image_path)
                    html_content += f'<img src="{rel_path}" alt="{db} visualization">'
        
        # Add GSEA results
        gsea_dbs = [db for db in pathway_results.keys() if db.endswith('_GSEA')]
        if gsea_dbs:
            html_content += "<h2>Gene Set Enrichment Analysis (GSEA) Results</h2>"
            
            for db in gsea_dbs:
                results_df = pathway_results[db]
                if results_df is None or results_df.empty:
                    html_content += f"<p>No significant pathways found for {db}.</p>"
                    continue
                
                # Get number of results
                num_results = len(results_df)
                db_name = db.replace('_GSEA', '')
                
                html_content += f"""
                <h3>{db_name} GSEA ({num_results} enriched pathways)</h3>
                <p>Top 10 most significant pathways:</p>
                <table>
                    <tr>
                        <th>Pathway</th>
                        <th>NES</th>
                        <th>P-value</th>
                        <th>FDR</th>
                        <th>Direction</th>
                    </tr>
                """
                
                # Add top 10 rows
                for idx, row in results_df.head(10).iterrows():
                    # Determine direction (up/down)
                    direction = "Upregulated" if row['nes'] > 0 else "Downregulated"
                        
                    html_content += f"""
                    <tr>
                        <td>{row['Term']}</td>
                        <td>{row['nes']:.3f}</td>
                        <td>{row['pval']:.2e}</td>
                        <td>{row['fdr']:.2e}</td>
                        <td>{direction}</td>
                    </tr>
                    """
                
                html_content += "</table>"
        
        # Close HTML
        html_content += """
            <hr>
            <p><em>Report generated by Proteomics Enrichment Analysis Pipeline</em></p>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        logging.info(f"HTML summary report created: {output_file}")
        
    except Exception as e:
        logging.error(f"Error creating summary report: {str(e)}")

def main():
    try:
        # Get inputs from Snakemake
        diff_expr_file = snakemake.input.diff_expr
        output_dir = os.path.dirname(snakemake.output[0])
        output_file = snakemake.output[0]
        
        # Get parameters
        organism = snakemake.params.organism
        sig_threshold = float(snakemake.params.significance_threshold)
        fc_threshold = float(snakemake.params.get('fc_threshold', 1.0))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse gene lists
        sig_genes, background = parse_gene_list(
            diff_expr_file, 
            sig_threshold=sig_threshold,
            fc_threshold=fc_threshold
        )
        
        # Create ranked list for GSEA
        ranked_list = get_ranked_list(diff_expr_file)
        
        # Perform GO enrichment analysis
        go_results = perform_go_enrichment(
            sig_genes=sig_genes,
            background=background,
            organism=organism,
            output_dir=output_dir
        )
        
        # Perform pathway analysis
        pathway_results = perform_pathway_analysis(
            sig_genes=sig_genes,
            background=background,
            ranked_list=ranked_list,
            organism=organism,
            output_dir=output_dir
        )
        
        # Visualize results
        visualize_enrichment(
            go_results=go_results,
            pathway_results=pathway_results,
            output_dir=output_dir
        )
        
        # Create summary report
        create_summary_report(
            go_results=go_results,
            pathway_results=pathway_results,
            output_file=output_file
        )
        
        # Also save all results to a consolidated JSON for easier access
        all_results = {
            "metadata": {
                "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "organism": organism,
                "significant_genes": len(sig_genes),
                "background_genes": len(background),
                "significance_threshold": sig_threshold,
                "fc_threshold": fc_threshold
            },
            "go_terms": {k: v.to_dict('records') for k, v in go_results.items() if v is not None and not v.empty},
            "pathways": {k: v.to_dict('records') for k, v in pathway_results.items() if v is not None and not v.empty}
        }
        
        # Save consolidated results
        with open(f"{output_dir}/all_enrichment_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info("Enrichment analysis completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred during enrichment analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
