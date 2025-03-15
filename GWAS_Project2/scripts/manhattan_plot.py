# scripts/manhattan_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from matplotlib import rcParams

# Configure logging
logging.basicConfig(filename=snakemake.log[0], level=logging.INFO,
                   format='%(asctime)s %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def main():
    # Set plot style
    sns.set(style="whitegrid")
    rcParams['figure.figsize'] = 12, 6
    
    # Load the rare variant burden test results
    logging.info(f"Loading rare variant burden test results from {snakemake.input.results}")
    results = pd.read_csv(snakemake.input.results, sep='\t')
    
    # Check if we have the required columns
    required_columns = ['gene', 'p_value', 'chromosome', 'position']
    for col in required_columns:
        if col not in results.columns:
            # If the specific chromosome/position columns are missing,
            # try to parse from gene column (often in format chr:pos:gene)
            if col in ['chromosome', 'position'] and 'gene' in results.columns:
                try:
                    logging.info("Trying to parse chromosome and position from gene column")
                    results['gene_parts'] = results['gene'].str.split(':')
                    results['chromosome'] = results['gene_parts'].apply(lambda x: x[0] if len(x) > 0 else None)
                    results['position'] = results['gene_parts'].apply(lambda x: int(x[1]) if len(x) > 1 else None)
                except Exception as e:
                    logging.error(f"Error parsing chromosome/position from gene column: {e}")
                    raise ValueError(f"Required column {col} not found in results file and could not be parsed")
            else:
                logging.error(f"Required column {col} not found in results file")
                raise ValueError(f"Required column {col} not found in results file")
    
    # Convert chromosome to string and handle 'X', 'Y', 'MT' etc.
    results['chromosome'] = results['chromosome'].astype(str)
    chrom_to_int = {str(i): i for i in range(1, 23)}
    chrom_to_int.update({'X': 23, 'Y': 24, 'MT': 25, 'M': 25})
    
    # Add numeric chromosome for sorting
    results['chrom_num'] = results['chromosome'].map(chrom_to_int)
    results = results.sort_values(['chrom_num', 'position'])
    
    # Calculate -log10(p-value)
    results['log_p'] = -np.log10(results['p_value'])
    
    # Create a column for chromosome alternating colors
    results['color'] = ['#1F77B4' if x % 2 == 0 else '#FF7F0E' for x in results['chrom_num']]
    
    # Get positions for x-axis chromosome labels
    chrom_df = results.groupby('chromosome')['position'].mean().reset_index()
    chrom_df = chrom_df.sort_values('chromosome', key=lambda x: x.map(chrom_to_int))
    
    # Prepare the Manhattan plot
    logging.info("Creating Manhattan plot")
    plt.figure(figsize=(14, 8))
    
    # Plot the points
    plt.scatter(results.index, results['log_p'], c=results['color'], alpha=0.8, s=30)
    
    # Add chromosome labels on x-axis
    plt.xticks(chrom_df.index, chrom_df['chromosome'], rotation=45)
    
    # Add threshold line for significance (Bonferroni or custom)
    if len(results) > 0:
        # Bonferroni threshold (0.05 / number of tests)
        bonferroni_threshold = 0.05 / len(results)
        plt.axhline(y=-np.log10(bonferroni_threshold), color='red', linestyle='--', 
                   label=f'Bonferroni (p={bonferroni_threshold:.2e})')
        
        # Add suggestive threshold (often 1e-5)
        plt.axhline(y=-np.log10(1e-5), color='blue', linestyle='--', 
                   label='Suggestive (p=1e-5)')
    
    # Annotate top significant hits
    sig_threshold = 0.05 / len(results) if len(results) > 0 else 0.05
    top_hits = results[results['p_value'] < sig_threshold].head(10)
    
    for idx, row in top_hits.iterrows():
        plt.annotate(row['gene'], 
                    xy=(idx, row['log_p']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    # Set plot labels and title
    plt.xlabel('Chromosome')
    plt.ylabel('-log10(p-value)')
    plt.title('Manhattan Plot of Rare Variant Burden Test Results')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    logging.info(f"Saving Manhattan plot to {snakemake.output.plot}")
    plt.savefig(snakemake.output.plot, dpi=300)
    
    logging.info("Manhattan plot created successfully")

if __name__ == "__main__":
    main()


