#!/usr/bin/env python
# ChIP-seq and Gene Expression Integration Analysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import pyBigWig
import pybedtools
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# Set this to your project directory
project_dir = "/path/to/your/project"
os.chdir(project_dir)

# Create output directory
os.makedirs("results/integration", exist_ok=True)

###########################################
# Step 1: Load and process ChIP-seq data #
###########################################

def load_peaks(peak_file):
    """Load peak file in narrowPeak format."""
    try:
        peaks = pd.read_csv(peak_file, sep='\t', header=None,
                           names=['chr', 'start', 'end', 'name', 'score', 
                                 'strand', 'signalValue', 'pValue', 
                                 'qValue', 'peak'])
        return peaks
    except Exception as e:
        print(f"Error loading {peak_file}: {e}")
        return None

# Load all peak files
peak_files = [f for f in os.listdir("results/peaks") if f.endswith("_peaks.narrowPeak")]
peak_list = {}

for file_name in peak_files:
    sample_name = file_name.replace("_peaks.narrowPeak", "")
    full_path = os.path.join("results/peaks", file_name)
    peaks = load_peaks(full_path)
    if peaks is not None:
        peak_list[sample_name] = peaks
        print(f"Loaded {len(peaks)} peaks from {sample_name}")

#################################################
# Step 2: Load gene annotations                 #
#################################################

# Load gene annotations (BED format with gene information)
# Format: chr, start, end, name, score, strand, gene_id
# Modify the file path as needed
genes = pd.read_csv("data/gene_annotations.bed", sep='\t', 
                   names=['chr', 'start', 'end', 'name', 'score', 'strand', 'gene_id'])

# Convert to BedTool
genes_bed = pybedtools.BedTool.from_dataframe(genes)

# If you need to create promoter regions (e.g., TSS ± 2kb)
def create_promoters(genes, upstream=2000, downstream=500):
    """Create promoter regions around TSS."""
    promoters = genes.copy()
    # Adjust start and end based on strand
    for i, row in promoters.iterrows():
        if row['strand'] == '+':
            promoters.at[i, 'start'] = max(0, row['start'] - upstream)
            promoters.at[i, 'end'] = row['start'] + downstream
        else:  # strand is '-'
            promoters.at[i, 'start'] = max(0, row['end'] - downstream)
            promoters.at[i, 'end'] = row['end'] + upstream
    return promoters

# Create promoter regions
promoters = create_promoters(genes)
promoters_bed = pybedtools.BedTool.from_dataframe(promoters)

#################################################
# Step 3: Associate peaks with genes            #
#################################################

def associate_peaks_with_genes(peaks, gene_regions, method='closest'):
    """Associate peaks with genes based on specified method."""
    # Convert peaks to BedTool
    peaks_df = peaks.copy()
    if 'name' not in peaks_df.columns:
        peaks_df['name'] = [f"peak_{i}" for i in range(len(peaks_df))]
    if 'score' not in peaks_df.columns and 'signalValue' in peaks_df.columns:
        peaks_df['score'] = peaks_df['signalValue']
    if 'score' not in peaks_df.columns:
        peaks_df['score'] = 1
    if 'strand' not in peaks_df.columns:
        peaks_df['strand'] = '.'
        
    peaks_bed = pybedtools.BedTool.from_dataframe(
        peaks_df[['chr', 'start', 'end', 'name', 'score', 'strand']])
    
    if method == 'closest':
        # Find closest gene for each peak
        closest = peaks_bed.closest(gene_regions, d=True)
        result = closest.to_dataframe(names=['peak_chr', 'peak_start', 'peak_end', 
                                            'peak_name', 'peak_score', 'peak_strand',
                                            'gene_chr', 'gene_start', 'gene_end', 
                                            'gene_name', 'gene_score', 'gene_strand', 
                                            'gene_id', 'distance'])
        return result
    
    elif method == 'intersect':
        # Find overlapping genes for each peak
        intersect = peaks_bed.intersect(gene_regions, wb=True)
        if len(intersect) > 0:
            result = intersect.to_dataframe(names=['peak_chr', 'peak_start', 'peak_end', 
                                                 'peak_name', 'peak_score', 'peak_strand',
                                                 'gene_chr', 'gene_start', 'gene_end', 
                                                 'gene_name', 'gene_score', 'gene_strand', 
                                                 'gene_id'])
            return result
        else:
            return pd.DataFrame()
    
    else:
        raise ValueError(f"Unknown method: {method}")

# Process each peak file and associate with promoters
peak_gene_assoc = {}

for sample_name, peaks in peak_list.items():
    # For promoter regions
    assoc = associate_peaks_with_genes(peaks, promoters_bed, method='intersect')
    if len(assoc) > 0:
        # If a gene has multiple peaks, take the one with the highest score
        if len(assoc) > 0:
            assoc = assoc.sort_values('peak_score', ascending=False)
            assoc = assoc.drop_duplicates('gene_id', keep='first')
        peak_gene_assoc[sample_name] = assoc
    else:
        print(f"No peaks associated with promoters for {sample_name}, trying closest method.")
        # If no intersection, try closest method
        assoc = associate_peaks_with_genes(peaks, genes_bed, method='closest')
        # Filter to keep only peaks within a reasonable distance (e.g., 5kb)
        assoc = assoc[assoc['distance'] <= 5000]
        if len(assoc) > 0:
            assoc = assoc.sort_values('peak_score', ascending=False)
            assoc = assoc.drop_duplicates('gene_id', keep='first')
        peak_gene_assoc[sample_name] = assoc
    
    print(f"Associated {len(peak_gene_assoc[sample_name])} peaks with genes for {sample_name}")

#################################################
# Step 4: Create peak-to-gene mapping matrix    #
#################################################

# Create a matrix of peak signals per gene
def create_peak_gene_matrix(peak_gene_associations, sample_names):
    """Create a matrix of peak scores for each gene across samples."""
    # Initialize with empty dataframe
    all_genes = set()
    for sample in sample_names:
        if sample in peak_gene_associations and len(peak_gene_associations[sample]) > 0:
            all_genes.update(peak_gene_associations[sample]['gene_id'].unique())
    
    all_genes = sorted(list(all_genes))
    peak_matrix = pd.DataFrame({'gene_id': all_genes})
    
    # Add each sample's peak scores
    for sample in sample_names:
        if sample in peak_gene_associations and len(peak_gene_associations[sample]) > 0:
            # Extract gene_id and peak_score
            sample_data = peak_gene_associations[sample][['gene_id', 'peak_score']]
            # Rename peak_score column to sample name
            sample_data = sample_data.rename(columns={'peak_score': sample})
            # Merge with the matrix
            peak_matrix = peak_matrix.merge(sample_data, on='gene_id', how='left')
        else:
            # If no data for this sample, add column of NAs
            peak_matrix[sample] = np.nan
    
    # Replace NA with 0 (genes with no peaks)
    peak_matrix = peak_matrix.fillna(0)
    
    # Add gene symbols if possible
    if 'gene_name' in peak_gene_associations[sample_names[0]].columns:
        # Get gene names from the first association
        gene_names = {}
        for sample in sample_names:
            if sample in peak_gene_associations and len(peak_gene_associations[sample]) > 0:
                for _, row in peak_gene_associations[sample].iterrows():
                    gene_names[row['gene_id']] = row['gene_name']
        
        # Add gene symbols
        peak_matrix['gene_symbol'] = peak_matrix['gene_id'].map(gene_names)
    
    return peak_matrix

# Create the peak-gene matrix
sample_names = list(peak_list.keys())
peak_gene_matrix = create_peak_gene_matrix(peak_gene_assoc, sample_names)

# Write the matrix to file
peak_gene_matrix.to_csv("results/integration/peak_gene_matrix.csv", index=False)

print(f"Created peak-gene matrix with {len(peak_gene_matrix)} genes and {len(sample_names)} ChIP samples.")

#################################################
# Step 5: Load and process gene expression data #
#################################################

# Load expression data (modify path as needed)
# This assumes your expression data is in a CSV file with gene IDs or symbols
gene_expression = pd.read_csv("data/gene_expression_matrix.csv")

# Check if we need to map identifiers for merging
if 'gene_id' in gene_expression.columns and 'gene_id' in peak_gene_matrix.columns:
    # Both matrices have gene IDs, we can merge directly
    merge_col_expr = 'gene_id'
    merge_col_peak = 'gene_id'
elif 'gene_symbol' in gene_expression.columns and 'gene_symbol' in peak_gene_matrix.columns:
    # Both matrices have gene symbols, we can merge directly
    merge_col_expr = 'gene_symbol'
    merge_col_peak = 'gene_symbol'
else:
    # Need conversion - this is complex in Python and depends on what gene ID mapping resources you have
    print("Warning: Gene identifier conversion needed but not implemented. Please ensure gene IDs match.")
    # If you have a mapping file, you could load it here
    # mapping = pd.read_csv("path/to/gene_id_mapping.csv")
    # And then apply the mapping to either peak_gene_matrix or gene_expression

##################################################
# Step 6: Integrate ChIP-seq and expression data #
##################################################

# Merge the peak-gene matrix with expression data
integrated_data = pd.merge(peak_gene_matrix, gene_expression, 
                          left_on=merge_col_peak, right_on=merge_col_expr)

# Write the integrated data to file
integrated_data.to_csv("results/integration/integrated_peak_expression_data.csv", index=False)

print(f"Created integrated dataset with {len(integrated_data)} genes.")

##################################################
# Step 7: Correlation analysis                   #
##################################################

# Identify the expression columns (modify based on your data format)
# This assumes expression columns start with "expr_" - adapt to your naming convention
expr_cols = [col for col in integrated_data.columns if col.startswith('expr_')]

# Identify the ChIP peak columns (all columns in peak_gene_matrix except gene IDs and symbols)
peak_cols = [col for col in peak_gene_matrix.columns 
           if col not in [merge_col_peak, 'gene_id', 'gene_symbol']]

# Create a correlation matrix between peak scores and expression
corr_matrix = pd.DataFrame(index=peak_cols, columns=expr_cols)

# Fill the correlation matrix
for peak_col in peak_cols:
    for expr_col in expr_cols:
        # Calculate Spearman correlation
        corr, p_value = spearmanr(integrated_data[peak_col], integrated_data[expr_col], 
                                 nan_policy='omit')
        corr_matrix.loc[peak_col, expr_col] = corr

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation between ChIP-seq Peaks and Gene Expression')
plt.tight_layout()
plt.savefig("results/integration/peak_expression_correlation_heatmap.pdf")
plt.close()

# Write correlation matrix to file
corr_matrix.to_csv("results/integration/peak_expression_correlation.csv")

# Calculate p-values for correlations and identify significant relationships
p_values = pd.DataFrame(index=peak_cols, columns=expr_cols)
for peak_col in peak_cols:
    for expr_col in expr_cols:
        _, p = spearmanr(integrated_data[peak_col], integrated_data[expr_col], nan_policy='omit')
        p_values.loc[peak_col, expr_col] = p

# Adjust p-values for multiple testing
flat_p_values = p_values.values.flatten()
_, adjusted_p_flat, _, _ = multipletests(flat_p_values, method='fdr_bh')
adjusted_p_values = pd.DataFrame(
    adjusted_p_flat.reshape(p_values.shape),
    index=p_values.index,
    columns=p_values.columns
)

# Identify significant correlations
significant_threshold = 0.05
significant_correlations = (adjusted_p_values < significant_threshold) & (abs(corr_matrix) > 0.3)

# Write significant correlations to file
significant_corr_df = pd.DataFrame({
    'peak_feature': [],
    'expression_feature': [],
    'correlation': [],
    'p_value': [],
    'adjusted_p_value': []
})

for peak_col in peak_cols:
    for expr_col in expr_cols:
        if significant_correlations.loc[peak_col, expr_col]:
            significant_corr_df = significant_corr_df.append({
                'peak_feature': peak_col,
                'expression_feature': expr_col,
                'correlation': corr_matrix.loc[peak_col, expr_col],
                'p_value': p_values.loc[peak_col, expr_col],
                'adjusted_p_value': adjusted_p_values.loc[peak_col, expr_col]
            }, ignore_index=True)

significant_corr_df = significant_corr_df.sort_values('correlation', ascending=False)
significant_corr_df.to_csv("results/integration/significant_correlations.csv", index=False)

##################################################
# Step 8: Predictive modeling                    #
##################################################

def build_predictive_model(data, expr_col, feature_cols, train_size=0.7, 
                         alpha=0.5, l1_ratio=0.5, random_state=42):
    """Build an ElasticNet model to predict gene expression from ChIP-seq data."""
    # Prepare the data
    X = data[feature_cols].values
    y = data[expr_col].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    
    # Find optimal regularization parameters using cross-validation
    model_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                          alphas=[0.1, 0.5, 1, 5, 10, 50, 100],
                          max_iter=1000,
                          cv=5,
                          random_state=random_state)
    model_cv.fit(X_train, y_train)
    
    # Get best parameters
    best_alpha = model_cv.alpha_
    best_l1_ratio = model_cv.l1_ratio_
    
    # Train final model with best parameters
    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, 
                     max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=False)
    
    # Create result object
    result = {
        'model': model,
        'best_alpha': best_alpha,
        'best_l1_ratio': best_l1_ratio,
        'rmse': rmse,
        'r2': r2,
        'feature_importance': feature_importance,
        'predictions': pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    }
    
    return result

# Build predictive models for each expression column
model_results = {}

for expr_col in expr_cols:
    print(f"Building predictive model for {expr_col}")
    # Clean data: remove rows with NaN values
    clean_data = integrated_data.dropna(subset=[expr_col] + peak_cols)
    
    if len(clean_data) > 50:  # Only build model if enough data points
        model_results[expr_col] = build_predictive_model(clean_data, expr_col, peak_cols)
        
        # Plot predictions vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(model_results[expr_col]['predictions']['actual'], 
                   model_results[expr_col]['predictions']['predicted'], alpha=0.5)
        plt.plot([min(model_results[expr_col]['predictions']['actual']), 
                 max(model_results[expr_col]['predictions']['actual'])],
                [min(model_results[expr_col]['predictions']['actual']), 
                 max(model_results[expr_col]['predictions']['actual'])], 
                'r--')
        plt.title(f"Predicted vs Actual Expression - {expr_col}")
        plt.xlabel("Actual Expression")
        plt.ylabel("Predicted Expression")
        plt.annotate(f"R² = {model_results[expr_col]['r2']:.3f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction')
        plt.annotate(f"RMSE = {model_results[expr_col]['rmse']:.3f}", 
                    xy=(0.05, 0.90), xycoords='axes fraction')
        plt.savefig(f"results/integration/prediction_model_{expr_col}.pdf")
        plt.close()
        
        # Save feature importance
        model_results[expr_col]['feature_importance'].to_csv(
            f"results/integration/feature_importance_{expr_col}.csv", index=False)
    else:
        print(f"Not enough data points for {expr_col} after removing NaNs.")

##################################################
# Step 9: Visualization of key findings          #
##################################################

# Function to identify genes with strong correlation between ChIP and expression
def find_correlated_genes(data, peak_col, expr_col, top_n=20):
    """Find genes with strongest correlation between peak score and expression."""
    # Calculate correlation for each gene
    correlations = []
    gene_symbols = []
    
    # Get gene symbol column
    if 'gene_symbol' in data.columns:
        symbol_col = 'gene_symbol'
    elif 'gene_name' in data.columns:
        symbol_col = 'gene_name'
    else:
        symbol_col = merge_col_peak
    
    # Group by gene
    for gene, group in data.groupby(symbol_col):
        # Skip genes with only one sample
        if len(group) < 2:
            continue
        
        # Get peak scores and expression values
        peak_values = group[peak_col].values
        expr_values = group[expr_col].values
        
        # Calculate correlation if there's variation
        if np.std(peak_values) > 0 and np.std(expr_values) > 0:
            corr, _ = spearmanr(peak_values, expr_values)
            correlations.append(corr)
            gene_symbols.append(gene)
    
    # Create dataframe with correlations
    corr_df = pd.DataFrame({
        'gene': gene_symbols,
        'correlation': correlations
    }).sort_values('correlation')
    
    # Get top positive and negative correlations
    top_negative = corr_df.head(top_n)
    top_positive = corr_df.tail(top_n).iloc[::-1]  # Reverse to get highest first
    
    return top_positive, top_negative

# For the first expression and peak column, get top correlated genes
if len(expr_cols) > 0 and len(peak_cols) > 0:
    top_pos, top_neg = find_correlated_genes(integrated_data, peak_cols[0], expr_cols[0])
    
    # Write to file
    pd.concat([top_pos, top_neg]).to_csv("results/integration/top_correlated_genes.csv", index=False)
    
    # Create scatter plots for top positively correlated genes
    for i, row in top_pos.head(5).iterrows():
        gene = row['gene']
        gene_data = integrated_data[integrated_data['gene_symbol'] == gene]
        
        if len(gene_data) > 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(gene_data[peak_cols[0]], gene_data[expr_cols[0]], color='blue', s=30)
            plt.title(f"Positive Correlation for {gene}")
            plt.xlabel("ChIP-seq Peak Score")
            plt.ylabel("Gene Expression Level")
            plt.savefig(f"results/integration/scatter_positive_{i}_{gene}.pdf")
            plt.close()
    
    # Create scatter plots for top negatively correlated genes
    for i, row in top_neg.head(5).iterrows():
        gene = row['gene']
        gene_data = integrated_data[integrated_data['gene_symbol'] == gene]
        
        if len(gene_data) > 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(gene_data[peak_cols[0]], gene_data[expr_cols[0]], color='red', s=30)
            plt.title(f"Negative Correlation for {gene}")
            plt.xlabel("ChIP-seq Peak Score")
            plt.ylabel("Gene Expression Level")
            plt.savefig(f"results/integration/scatter_negative_{i}_{gene}.pdf")
            plt.close()

##################################################
# Step 10: Generate summary report               #
##################################################

def generate_summary_report():
    """Generate a summary report of the analysis."""
    with open("results/integration/summary_report.md", "w") as f:
        f.write("# ChIP-seq and Gene Expression Integration Analysis\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This report summarizes the integration analysis of ChIP-seq and gene expression data.\n\n")
        
        f.write("## Data Summary\n\n")
        f.write(f"- Number of ChIP-seq samples: {len(peak_cols)}\n")
        f.write(f"- Number of expression datasets: {len(expr_cols)}\n")
        f.write(f"- Number of genes in the integrated dataset: {len(integrated_data)}\n\n")
        
        f.write("## Correlation Analysis\n\n")
        f.write("The following table shows the Spearman correlation between ChIP-seq peaks and gene expression:\n\n")
        f.write(corr_matrix.to_markdown())
        f.write("\n\n")
        
        f.write("### Significant Correlations\n\n")
        if os.path.exists("results/integration/significant_correlations.csv"):
            sig_corr = pd.read_csv("results/integration/significant_correlations.csv")
            if len(sig_corr) > 0:
                f.write(f"Found {len(sig_corr)} significant correlations.\n\n")
                f.write(sig_corr.head(10).to_markdown())
                f.write("\n\n")
            else:
                f.write("No significant correlations found.\n\n")
        
        f.write("## Predictive Modeling\n\n")
        f.write("Results of predicting gene expression from ChIP-seq data:\n\n")
        model_summary = pd.DataFrame({
            'Expression': [],
            'R-squared': [],
            'RMSE': [],
            'Top predictors': []
        })
        
        for expr_col, result in model_results.items():
            top_predictors = ", ".join(result['feature_importance'].head(3)['feature'].tolist())
            model_summary = model_summary.append({
                'Expression': expr_col,
                'R-squared': round(result['r2'], 3),
                'RMSE': round(result['rmse'], 3),
                'Top predictors': top_predictors
            }, ignore_index=True)
        
        if len(model_summary) > 0:
            f.write(model_summary.to_markdown())
            f.write("\n\n")
        
        f.write("## Top Correlated Genes\n\n")
        if os.path.exists("results/integration/top_correlated_genes.csv"):
            top_genes = pd.read_csv("results/integration/top_correlated_genes.csv")
            f.write("### Top Positively Correlated Genes\n\n")
            f.write(top_genes[top_genes['correlation'] > 0].head(10).to_markdown())
            f.write("\n\n")
            f.write("### Top Negatively Correlated Genes\n\n")
            f.write(top_genes[top_genes['correlation'] < 0].head(10).to_markdown())
            f.write("\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This analysis has identified relationships between ChIP-seq peaks and gene expression patterns.\n")
        
        # Add potentially interesting findings
        if len(significant_corr_df) > 0:
            strongest_corr = significant_corr_df.iloc[0]
            f.write(f"The strongest correlation was found between {strongest_corr['peak_feature']} and {strongest_corr['expression_feature']} (r = {strongest_corr['correlation']:.3f}).\n")
        
        # If we have good predictive models, mention that
        good_models = [expr for expr, res in model_results.items() if res['r2'] > 0.3]
        if good_models:
            f.write(f"The expression of {', '.join(good_models)} could be predicted from ChIP-seq data with reasonable accuracy.\n")

# Generate the summary report
generate_summary_report()

print("Analysis completed successfully!")
