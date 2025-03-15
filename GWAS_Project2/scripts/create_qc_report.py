# scripts/create_qc_report.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os
import logging
from io import BytesIO

# Configure logging
logging.basicConfig(filename=snakemake.log[0], level=logging.INFO,
                   format='%(asctime)s %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def main():
    # Set the plot style
    sns.set(style="whitegrid")
    
    # Load VCF stats
    logging.info(f"Loading VCF statistics from {snakemake.input.vcf_stats}")
    try:
        vcf_stats = pd.read_csv(snakemake.input.vcf_stats, sep='\t')
    except Exception as e:
        logging.warning(f"Could not load VCF stats: {e}")
        vcf_stats = pd.DataFrame({'Error': ['Could not load VCF statistics']})
    
    # Load QC directory for sample and variant metrics
    qc_dir = os.path.dirname(snakemake.input.hail_qc)
    
    # Try to load sample QC metrics
    sample_qc_file = os.path.join(qc_dir, "sample_qc_metrics.tsv")
    try:
        if os.path.exists(sample_qc_file):
            logging.info(f"Loading sample QC metrics from {sample_qc_file}")
            sample_qc = pd.read_csv(sample_qc_file, sep='\t')
        else:
            logging.warning(f"Sample QC metrics file not found: {sample_qc_file}")
            sample_qc = None
    except Exception as e:
        logging.warning(f"Could not load sample QC metrics: {e}")
        sample_qc = None
    
    # Try to load variant QC metrics
    variant_qc_file = os.path.join(qc_dir, "variant_qc_metrics.tsv")
    try:
        if os.path.exists(variant_qc_file):
            logging.info(f"Loading variant QC metrics from {variant_qc_file}")
            variant_qc = pd.read_csv(variant_qc_file, sep='\t')
        else:
            logging.warning(f"Variant QC metrics file not found: {variant_qc_file}")
            variant_qc = None
    except Exception as e:
        logging.warning(f"Could not load variant QC metrics: {e}")
        variant_qc = None
    
    # Create HTML report
    logging.info("Generating QC report")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GWAS Quality Control Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2980b9; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
            h3 { color: #3498db; }
            .container { max-width: 1200px; margin: 0 auto; }
            .summary-box { background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }
            .warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
            .success { background-color: #d4edda; border-left: 4px solid #28a745; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .image-container { margin: 20px 0; text-align: center; }
            .image-container img { max-width: 100%; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GWAS Rare Variant Discovery Pipeline QC Report</h1>
            
            <div class="summary-box">
                <h3>Project Summary</h3>
                <p>This report summarizes the quality control metrics for the GWAS rare variant discovery pipeline.</p>
                <p>Date: """ + pd.Timestamp.now().strftime('%Y-%m-%d') + """</p>
            </div>
    """
    
    # 1. Sample QC section
    html_content += """
            <h2>1. Sample Quality Control</h2>
    """
    
    if sample_qc is not None and not sample_qc.empty:
        # Calculate summary statistics
        n_samples = len(sample_qc)
        avg_call_rate = sample_qc['call_rate'].mean() if 'call_rate' in sample_qc.columns else "N/A"
        avg_depth = sample_qc['dp_stats.mean'].mean() if 'dp_stats.mean' in sample_qc.columns else "N/A"
        
        html_content += f"""
            <div class="summary-box success">
                <p><strong>Total samples analyzed:</strong> {n_samples}</p>
                <p><strong>Average call rate:</strong> {avg_call_rate:.4f}</p>
                <p><strong>Average sequencing depth:</strong> {avg_depth:.2f}X</p>
            </div>
            
            <h3>Sample QC Metrics</h3>
        """
        
        # Sample metrics table (first 10 rows)
        display_cols = ['s', 'call_rate', 'dp_stats.mean', 'gq_stats.mean', 'n_called', 'n_not_called']
        display_cols = [col for col in display_cols if col in sample_qc.columns]
        
        if display_cols:
            html_content += """
            <table>
                <tr>
            """
            
            # Table headers
            for col in display_cols:
                html_content += f"<th>{col}</th>"
            html_content += "</tr>"
            
            # Table rows
            for _, row in sample_qc.head(10).iterrows():
                html_content += "<tr>"
                for col in display_cols:
                    value = row[col]
                    if isinstance(value, float):
                        html_content += f"<td>{value:.4f}</td>"
                    else:
                        html_content += f"<td>{value}</td>"
                html_content += "</tr>"
            
            html_content += """
            </table>
            <p><em>Showing first 10 samples only. See full data for complete metrics.</em></p>
            """
            
            # Add some visualizations if we have the data
            if 'call_rate' in sample_qc.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(sample_qc['call_rate'], kde=True)
                plt.title('Distribution of Sample Call Rates')
                plt.xlabel('Call Rate')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
                
                # Convert plot to base64 for embedding
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                <div class="image-container">
                    <img src="data:image/png;base64,{img_str}" alt="Sample Call Rate Distribution">
                </div>
                """
    else:
        html_content += """
            <div class="summary-box warning">
                <p>No sample QC metrics available.</p>
            </div>
        """
    
    # 2. Variant QC section
    html_content += """
            <h2>2. Variant Quality Control</h2>
    """
    
    if variant_qc is not None and not variant_qc.empty:
        # Calculate summary statistics
        n_variants = len(variant_qc)
        avg_variant_call_rate = variant_qc['call_rate'].mean() if 'call_rate' in variant_qc.columns else "N/A"
        
        html_content += f"""
            <div class="summary-box success">
                <p><strong>Total variants analyzed:</strong> {n_variants}</p>
                <p><strong>Average variant call rate:</strong> {avg_variant_call_rate:.4f}</p>
            </div>
            
            <h3>Variant QC Metrics</h3>
        """
        
        # Create a summary table
        variant_summary = pd.DataFrame({
            'Metric': ['Total Variants', 'SNPs', 'Indels', 'Multiallelic Sites'],
            'Count': [
                n_variants,
                variant_qc['n_SNP'].sum() if 'n_SNP' in variant_qc.columns else "N/A",
                variant_qc['n_insertion'].sum() + variant_qc['n_deletion'].sum() 
                if 'n_insertion' in variant_qc.columns and 'n_deletion' in variant_qc.columns else "N/A",
                variant_qc['n_non_ref'].sum() - variant_qc['n_SNP'].sum() - variant_qc['n_insertion'].sum() - variant_qc['n_deletion'].sum()
                if all(col in variant_qc.columns for col in ['n_non_ref', 'n_SNP', 'n_insertion', 'n_deletion']) else "N/A"
            ]
        })
        
        html_content += """
        <table>
            <tr>
                <th>Metric</th>
                <th>Count</th>
            </tr>
        """
        
        for _, row in variant_summary.iterrows():
            html_content += f"""
            <tr>
                <td>{row['Metric']}</td>
                <td>{row['Count']}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
        
        # Plot variant quality distribution if available
        if 'call_rate' in variant_qc.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(variant_qc['call_rate'], kde=True)
            plt.title('Distribution of Variant Call Rates')
            plt.xlabel('Call Rate')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
            img_str = fig_to_base64(plt.gcf())
            plt.close()
            
            html_content += f"""
            <div class="image-container">
                <img src="data:image/png;base64,{img_str}" alt="Variant Call Rate Distribution">
            </div>
            """
    else:
        html_content += """
            <div class="summary-box warning">
                <p>No variant QC metrics available.</p>
            </div>
        """
    
    # 3. PCA Results
    html_content += """
        <h2>3. Population Structure (PCA)</h2>
    """
    
    if os.path.exists(snakemake.input.pca):
        html_content += f"""
        <div class="image-container">
            <img src="data:image/png;base64,{base64.b64encode(open(snakemake.input.pca, 'rb').read()).decode('utf-8')}" alt="PCA Clusters">
        </div>
        """
    else:
        html_content += """
        <div class="summary-box warning">
            <p>PCA visualization not available.</p>
        </div>
        """
    
    # 4. Rare Variant Analysis Results
    html_content += """
        <h2>4. Rare Variant Association Results</h2>
    """
    
    if os.path.exists(snakemake.input.manhattan):
        html_content += f"""
        <div class="image-container">
            <img src="data:image/png;base64,{base64.b64encode(open(snakemake.input.manhattan, 'rb').read()).decode('utf-8')}" alt="Manhattan Plot">
        </div>
        """
    else:
        html_content += """
        <div class="summary-box warning">
            <p>Manhattan plot not available.</p>
        </div>
        """
    
    # Close HTML document
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write the HTML report
    logging.info(f"Writing QC report to {snakemake.output.report}")
    with open(snakemake.output.report, 'w') as f:
        f.write(html_content)
    
    logging.info("QC report generation completed successfully")

if __name__ == "__main__":
    main()

