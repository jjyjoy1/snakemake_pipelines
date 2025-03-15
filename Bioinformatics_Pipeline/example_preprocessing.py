import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from Bioinformatics_Pipeline.processor import BioinformaticsPreprocessor
from Bioinformatics_Pipeline.feature_selection import BioinformaticsFeatureSelector


# Import our custom bioinformatics classes
# In a real project, you would import from your module
# from bioinformatics_pipeline import BioinformaticsPreprocessor, BioinformaticsFeatureSelector

# -------------------------------------------------------------------------------
# EXAMPLE 1: RNA-seq Data Analysis
# -------------------------------------------------------------------------------

def create_synthetic_rnaseq_data(n_samples=100, n_genes=1000, n_informative=50):
    """
    Create synthetic RNA-seq count data with class labels.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_genes : int
        Number of genes
    n_informative : int
        Number of informative genes that separate the classes
        
    Returns:
    --------
    tuple
        (X, y) where X is the count matrix and y is the class labels
    """
    print(f"Creating synthetic RNA-seq data with {n_samples} samples and {n_genes} genes...")
    
    # Generate random counts following negative binomial distribution (common for RNA-seq)
    counts = np.random.negative_binomial(n=20, p=0.3, size=(n_samples, n_genes))
    
    # Create two sample groups
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1  # Second half of samples are class 1
    
    # Make informative genes differential between groups
    fold_changes = np.ones(n_genes)
    informative_indices = np.random.choice(n_genes, n_informative, replace=False)
    fold_changes[informative_indices] = np.random.choice([0.1, 5], size=n_informative)
    
    # Apply fold changes to the second group
    for i in range(n_samples):
        if y[i] == 1:  # If sample is in second group
            counts[i, :] = counts[i, :] * fold_changes
    
    # Convert to DataFrame with gene and sample names
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]
    X = pd.DataFrame(counts, index=sample_names, columns=gene_names)
    y = pd.Series(y, index=sample_names).map({0: 'Control', 1: 'Treatment'})
    
    # Create gene lengths for TPM normalization
    gene_lengths = pd.Series(np.random.randint(500, 5000, size=n_genes), index=gene_names)
    
    return X, y, gene_lengths, informative_indices

def example_rnaseq_workflow():
    """Example workflow for RNA-seq data processing."""
    
    # Generate synthetic RNA-seq data
    X, y, gene_lengths, informative_indices = create_synthetic_rnaseq_data(
        n_samples=100, n_genes=1000, n_informative=50
    )
    
    print(f"Original data shape: {X.shape}")
    print(f"Class distribution: {y.value_counts()}")
    print(f"Sample of count data:\n{X.iloc[:5, :5]}")
    
    # Split data for demonstration
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # -------------------------------------------------------------------------------
    # Step 1: Preprocessing & Normalization
    # -------------------------------------------------------------------------------
    print("\nStep 1: Preprocessing & Normalization")
    
    # Initialize the preprocessor for RNA-seq data
    preprocessor = BioinformaticsPreprocessor(data_type='rna_seq')
    
    # Try different normalization methods
    methods = ['log', 'vst', 'deseq', 'tmm']
    normalized_data = {}
    
    for method in methods:
        print(f"Applying {method} normalization...")
        if method == 'tpm':
            # TPM normalization requires gene lengths
            normalized_data[method] = preprocessor.preprocess(
                X_train, method=method, gene_lengths=gene_lengths
            )
        else:
            normalized_data[method] = preprocessor.preprocess(X_train, method=method)
        
        print(f"  Normalized data shape: {normalized_data[method].shape}")
        print(f"  Sample of normalized data:\n{normalized_data[method].iloc[:2, :3]}")
    
    # Visualize the normalized data distribution for the first 5 genes
    plt.figure(figsize=(15, 10))
    for i, (method, data) in enumerate(normalized_data.items()):
        plt.subplot(2, 2, i+1)
        sns.histplot(data=data.iloc[:, :5].melt(), x='value', hue='variable', bins=30, kde=True)
        plt.title(f"{method.upper()} Normalized Distribution")
        plt.xlabel("Normalized Value")
        plt.tight_layout()
    
    # Use log normalization for the next steps
    X_train_norm = normalized_data['log']
    
    # Apply same normalization to test data
    X_test_norm = preprocessor.transform(X_test, method='log')
    
    # -------------------------------------------------------------------------------
    # Step 2: Feature Selection
    # -------------------------------------------------------------------------------
    print("\nStep 2: Feature Selection")
    
    # Encode class labels for supervised methods
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Try different feature selection methods
    selection_methods = [
        ('statistical', {'k': 100, 'test_type': 'f_test'}),
        ('variance', {'threshold': 0.1}),
        ('regularization', {'model_type': 'lasso', 'alpha': 0.01}),
        ('tree', {'model_type': 'rf', 'n_estimators': 100, 'threshold': 'mean'}),
        ('rfe', {'n_features_to_select': 100, 'step': 10}),
        ('autoencoder', {'encoding_dim': 50, 'epochs': 50, 'n_features_to_select': 100})
    ]
    
    selected_features = {}
    
    for method, params in selection_methods:
        print(f"Applying {method} feature selection with params: {params}")
        
        # Initialize feature selector
        if method in ['statistical', 'regularization', 'tree', 'rfe']:
            # These methods need class labels
            selector = BioinformaticsFeatureSelector(method=method, task_type='classification')
            selector.fit(X_train_norm, y_train_encoded, **params)
        else:
            # Unsupervised methods
            selector = BioinformaticsFeatureSelector(method=method)
            selector.fit(X_train_norm, **params)
        
        # Get selected features
        selected_features[method] = selector.get_selected_features()
        
        # Get feature importances
        feature_importances = selector.get_feature_importances()
        top_features = feature_importances.sort_values(ascending=False).head(10)
        
        print(f"  Number of selected features: {len(selected_features[method])}")
        print(f"  Top 10 features by importance:\n{top_features}")
        
        # Apply feature selection to training and test data
        X_train_selected = selector.transform(X_train_norm)
        X_test_selected = selector.transform(X_test_norm)
        
        print(f"  Selected data shape: {X_train_selected.shape}")
        
        # Visualize the distribution of feature importances
        plt.figure(figsize=(10, 6))
        sns.histplot(feature_importances, bins=30)
        plt.title(f"Feature Importance Distribution ({method})")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        
        # Check if the method found the informative genes
        informative_gene_names = [f"Gene_{i}" for i in informative_indices]
        found_informative = set(informative_gene_names).intersection(set(selected_features[method]))
        
        print(f"  Features correctly identified as informative: {len(found_informative)}/{len(informative_indices)}")
        recall = len(found_informative) / len(informative_indices)
        print(f"  Recall of informative features: {recall:.2f}")
    
    # Compare the overlap between selected features from different methods
    overlap_matrix = np.zeros((len(selection_methods), len(selection_methods)))
    
    for i, (method1, _) in enumerate(selection_methods):
        for j, (method2, _) in enumerate(selection_methods):
            overlap = len(set(selected_features[method1]).intersection(set(selected_features[method2])))
            overlap_matrix[i, j] = overlap
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        overlap_matrix, 
        annot=True, 
        fmt='g', 
        xticklabels=[m[0] for m in selection_methods],
        yticklabels=[m[0] for m in selection_methods]
    )
    plt.title('Overlap in Selected Features Between Methods')
    plt.tight_layout()
    
    print("\nFeature selection comparison complete!")

# -------------------------------------------------------------------------------
# EXAMPLE 2: ChIP-seq Data Analysis
# -------------------------------------------------------------------------------

def example_chipseq_workflow():
    """Example workflow for ChIP-seq data processing."""
    # Similar to RNA-seq workflow but with ChIP-seq specific preprocessing
    # This is a placeholder - would implement similar to the RNA-seq example
    pass

# -------------------------------------------------------------------------------
# RUN THE EXAMPLES
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    example_rnaseq_workflow()
    # example_chipseq_workflow()  # Uncomment to run ChIP-seq example


