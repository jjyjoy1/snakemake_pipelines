import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Step 1: Create synthetic RNA-seq data
def create_synthetic_rnaseq_data(n_samples=50, n_genes=500):
    """Create a simple RNA-seq count matrix with two condition groups"""
    # Generate random counts (negative binomial is common for RNA-seq count data)
    counts = np.random.negative_binomial(n=20, p=0.3, size=(n_samples, n_genes))
    
    # Create sample metadata (two conditions)
    conditions = ['Treatment'] * (n_samples // 2) + ['Control'] * (n_samples // 2)
    
    # Make some genes differentially expressed in the treatment group
    informative_genes = np.random.choice(range(n_genes), size=50, replace=False)
    
    # Increase expression of informative genes in the treatment group
    for i in range(n_samples // 2):
        counts[i, informative_genes] = counts[i, informative_genes] * 5
    
    # Create DataFrame with sample and gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    sample_names = [f"sample_{i}" for i in range(n_samples)]
    
    count_matrix = pd.DataFrame(counts, index=sample_names, columns=gene_names)
    metadata = pd.DataFrame({'condition': conditions}, index=sample_names)
    
    # Create gene lengths for TPM calculation
    gene_lengths = pd.Series(np.random.randint(500, 5000, size=n_genes), index=gene_names)
    
    return count_matrix, metadata, gene_lengths, informative_genes

# Main example script
def main():
    print("Bioinformatics Pipeline Example")
    print("-------------------------------\n")
    
    # Generate synthetic data
    print("Generating synthetic RNA-seq data...")
    count_matrix, metadata, gene_lengths, informative_genes = create_synthetic_rnaseq_data()
    print(f"Generated data with {count_matrix.shape[0]} samples and {count_matrix.shape[1]} genes")
    print(f"Sample distribution: {metadata['condition'].value_counts()}")
    print("\nRaw count matrix preview:")
    print(count_matrix.iloc[:3, :5])
    
    # Split data for demonstration
    X_train, X_test, y_train, y_test = train_test_split(
        count_matrix, 
        metadata['condition'],
        test_size=0.3, 
        random_state=42,
        stratify=metadata['condition']
    )
    
    # Create our bioinformatics preprocessor
    print("\n1. Preprocessing Data")
    print("--------------------")
    from BioinformaticsPreprocessor import BioinformaticsPreprocessor  # Import from your module
    
    preprocessor = BioinformaticsPreprocessor(data_type='rna_seq')
    
    # Show different normalization methods
    print("\nApplying different normalization methods:")
    
    # 1. Simple log normalization
    X_train_log = preprocessor.preprocess(X_train, method='log')
    print("\nLog normalization result:")
    print(X_train_log.iloc[:2, :3])
    
    # 2. TPM normalization (requires gene lengths)
    X_train_tpm = preprocessor.preprocess(X_train, method='tpm', gene_lengths=gene_lengths)
    print("\nTPM normalization result:")
    print(X_train_tpm.iloc[:2, :3])
    
    # 3. DESeq-like normalization 
    X_train_deseq = preprocessor.preprocess(X_train, method='deseq')
    print("\nDESeq normalization result:")
    print(X_train_deseq.iloc[:2, :3])
    
    # Use log-normalized data for feature selection
    X_norm = X_train_log
    
    # Create our feature selector
    print("\n2. Feature Selection")
    print("------------------")
    from BioinformaticsFeatureSelector import BioinformaticsFeatureSelector  # Import from your module
    
    # Example 1: Statistical feature selection
    print("\n2.1 Statistical Feature Selection (F-test)")
    stat_selector = BioinformaticsFeatureSelector(method='statistical')
    stat_selector.fit(X_norm, y_train)
    
    # Get selected features and their importance scores
    stat_features = stat_selector.get_selected_features()
    stat_importance = stat_selector.get_feature_importances()
    
    print(f"Selected {len(stat_features)} features")
    print("\nTop 10 features by importance:")
    print(stat_importance.sort_values(ascending=False).head(10))
    
    # Transform the data to include only selected features
    X_stat_selected = stat_selector.transform(X_norm)
    print(f"\nTransformed data shape: {X_stat_selected.shape}")
    
    # Example 2: Variance-based feature selection
    print("\n2.2 Variance-based Feature Selection")
    var_selector = BioinformaticsFeatureSelector(method='variance', task_type='classification')
    var_selector.fit(X_norm, threshold=0.5)
    
    var_features = var_selector.get_selected_features()
    print(f"Selected {len(var_features)} features with variance > 0.5")
    
    # Example 3: Tree-based feature selection
    print("\n2.3 Tree-based Feature Selection (Random Forest)")
    tree_selector = BioinformaticsFeatureSelector(method='tree', task_type='classification')
    tree_selector.fit(X_norm, y_train, model_type='rf', n_estimators=100)
    
    tree_features = tree_selector.get_selected_features()
    tree_importance = tree_selector.get_feature_importances()
    
    print(f"Selected {len(tree_features)} features")
    print("\nTop 10 features by importance:")
    print(pd.Series(tree_importance, index=X_norm.columns).sort_values(ascending=False).head(10))
    
    # Check overlap between feature selection methods
    stat_set = set(stat_features)
    var_set = set(var_features)
    tree_set = set(tree_features)
    
    print("\nFeature selection overlap:")
    print(f"Statistical ∩ Variance: {len(stat_set.intersection(var_set))}")
    print(f"Statistical ∩ Tree: {len(stat_set.intersection(tree_set))}")
    print(f"Variance ∩ Tree: {len(var_set.intersection(tree_set))}")
    print(f"All methods: {len(stat_set.intersection(var_set).intersection(tree_set))}")
    
    # Check if our methods found the informative genes
    informative_gene_names = [f"gene_{i}" for i in informative_genes]
    
    stat_recall = len(set(informative_gene_names).intersection(stat_set)) / len(informative_genes)
    var_recall = len(set(informative_gene_names).intersection(var_set)) / len(informative_genes)
    tree_recall = len(set(informative_gene_names).intersection(tree_set)) / len(informative_genes)
    
    print("\nRecall of truly informative genes:")
    print(f"Statistical: {stat_recall:.2f}")
    print(f"Variance: {var_recall:.2f}")
    print(f"Tree: {tree_recall:.2f}")
    
    # Visualize the impact of feature selection with PCA
    print("\n3. Visualizing Results")
    print("--------------------")
    
    # Use the statistically selected features
    X_selected = stat_selector.transform(X_norm)
    
    # Run PCA on both the full and selected feature sets
    pca_full = PCA(n_components=2).fit_transform(X_norm)
    pca_selected = PCA(n_components=2).fit_transform(X_selected)
    
    # Create a function to plot PCA results
    def plot_pca(pca_data, title):
        plt.figure(figsize=(8, 6))
        colors = {'Treatment': 'red', 'Control': 'blue'}
        for condition in ['Treatment', 'Control']:
            mask = y_train == condition
            plt.scatter(
                pca_data[mask, 0], 
                pca_data[mask, 1],
                c=colors[condition],
                label=condition,
                alpha=0.7
            )
        plt.title(title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
    print("\nPCA visualization before and after feature selection:")
    plot_pca(pca_full, "PCA on all features")
    plot_pca(pca_selected, "PCA on selected features")
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()

