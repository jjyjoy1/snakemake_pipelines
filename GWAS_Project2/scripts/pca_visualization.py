# scripts/pca_visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(filename=snakemake.log[0], level=logging.INFO,
                   format='%(asctime)s %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def main():
    # Set the plot style
    sns.set(style="whitegrid")
    
    # Load PCA eigenvalues and eigenvectors
    logging.info(f"Loading PCA eigenvectors from {snakemake.input.eigenvec}")
    eigenvec = pd.read_csv(snakemake.input.eigenvec, sep=r'\s+', header=None)
    
    # PLINK outputs the first two columns as FID and IID
    # Renaming columns for clarity
    col_names = ['FID', 'IID'] + [f'PC{i+1}' for i in range(eigenvec.shape[1]-2)]
    eigenvec.columns = col_names
    
    logging.info(f"Loading PCA eigenvalues from {snakemake.input.eigenval}")
    eigenval = pd.read_csv(snakemake.input.eigenval, sep=r'\s+', header=None)
    eigenval.columns = ['Eigenvalue']
    
    # Calculate proportion of variance explained by each PC
    logging.info("Calculating proportion of variance explained")
    total_variance = eigenval['Eigenvalue'].sum()
    eigenval['Proportion'] = eigenval['Eigenvalue'] / total_variance
    eigenval['Cumulative'] = eigenval['Proportion'].cumsum()
    
    # Load phenotype data
    logging.info(f"Loading phenotype data from {snakemake.input.phenotype}")
    phenotype = pd.read_csv(snakemake.input.phenotype, sep='\t')
    
    # Ensure we have a sample_id column to merge with PCA results
    if 'sample_id' not in phenotype.columns:
        # Try to find a column that might be the sample ID
        id_columns = [col for col in phenotype.columns if 'id' in col.lower()]
        if id_columns:
            logging.info(f"Using '{id_columns[0]}' as the sample ID column")
            phenotype.rename(columns={id_columns[0]: 'sample_id'}, inplace=True)
        else:
            logging.error("No sample ID column found in phenotype data")
            raise ValueError("No sample ID column found in phenotype data")
    
    # Merge PCA results with phenotype data
    logging.info("Merging PCA results with phenotype data")
    pca_data = pd.merge(eigenvec, phenotype, left_on='IID', right_on='sample_id', how='inner')
    
    # Check if we have case/control information
    case_control_columns = [col for col in pca_data.columns if 'case' in col.lower() or 'status' in col.lower()]
    if case_control_columns:
        case_col = case_control_columns[0]
        logging.info(f"Using '{case_col}' as the case/control column")
    else:
        # Default to 'is_case' column or create one if we have phenotype information
        if 'phenotype' in pca_data.columns:
            logging.info("Creating case/control column from 'phenotype'")
            # Assuming phenotype is binary or can be converted to binary
            pca_data['is_case'] = pca_data['phenotype'].apply(lambda x: 1 if x > 0 else 0)
            case_col = 'is_case'
        else:
            logging.warning("No case/control column found, creating a dummy column")
            pca_data['is_case'] = 0
            case_col = 'is_case'
    
    # Create PCA scatter plot
    logging.info("Creating PCA scatter plot")
    plt.figure(figsize=(12, 10))
    
    # Plot PC1 vs PC2 colored by case/control status
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue=case_col,
        palette=['#1F77B4', '#FF7F0E'],  # Blue for controls, orange for cases
        s=80,
        alpha=0.7,
        data=pca_data
    )
    
    # Add plot annotations
    plt.title('PCA Plot of Study Samples', fontsize=16)
    plt.xlabel(f'PC1 ({eigenval.iloc[0]["Proportion"]*100:.2f}% variance)', fontsize=14)
    plt.ylabel(f'PC2 ({eigenval.iloc[1]["Proportion"]*100:.2f}% variance)', fontsize=14)
    plt.legend(title='Case/Control Status', fontsize=12, title_fontsize=13)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create a subplot for the scree plot (variance explained)
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(eigenval)+1), eigenval['Proportion'], alpha=0.7)
    plt.plot(range(1, len(eigenval)+1), eigenval['Cumulative'], 'ro-')
    plt.axhline(y=0.8, color='r', linestyle='--')  # Often 80% variance is considered a good cutoff
    
    plt.title('Scree Plot - Variance Explained by PCs', fontsize=16)
    plt.xlabel('Principal Component', fontsize=14)
    plt.ylabel('Proportion of Variance Explained', fontsize=14)
    plt.xticks(range(1, len(eigenval)+1))
    plt.tight_layout()
    
    # Create a 2x2 grid of PC plots
    plt.figure(figsize=(16, 16))
    
    # PC1 vs PC2
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='PC1', y='PC2', hue=case_col, palette=['#1F77B4', '#FF7F0E'], s=50, alpha=0.7, data=pca_data)
    plt.title('PC1 vs PC2', fontsize=14)
    
    # PC1 vs PC3
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='PC1', y='PC3', hue=case_col, palette=['#1F77B4', '#FF7F0E'], s=50, alpha=0.7, data=pca_data)
    plt.title('PC1 vs PC3', fontsize=14)
    
    # PC2 vs PC3
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='PC2', y='PC3', hue=case_col, palette=['#1F77B4', '#FF7F0E'], s=50, alpha=0.7, data=pca_data)
    plt.title('PC2 vs PC3', fontsize=14)
    
    # PC3 vs PC4
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='PC3', y='PC4', hue=case_col, palette=['#1F77B4', '#FF7F0E'], s=50, alpha=0.7, data=pca_data)
    plt.title('PC3 vs PC4', fontsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    logging.info(f"Saving PCA visualization to {snakemake.output.plot}")
    plt.savefig(snakemake.output.plot, dpi=300)
    
    # Save additional plots in the same directory
    plot_dir = os.path.dirname(snakemake.output.plot)
    plt.figure(1)
    plt.savefig(os.path.join(plot_dir, "pca_main.png"), dpi=300)
    
    plt.figure(2)
    plt.savefig(os.path.join(plot_dir, "pca_scree.png"), dpi=300)
    
    logging.info("PCA visualization completed successfully")

if __name__ == "__main__":
    main()


