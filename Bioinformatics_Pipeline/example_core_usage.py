# Initialize the preprocessor
preprocessor = BioinformaticsPreprocessor(data_type='rna_seq')

# Normalize RNA-seq data using log transformation
normalized_counts = preprocessor.preprocess(count_matrix, method='log')

# Initialize the feature selector with statistical method
selector = BioinformaticsFeatureSelector(method='statistical', task_type='classification')

# Fit the selector to identify differentially expressed genes
selector.fit(normalized_counts, condition_labels, test_type='f_test', k=50)

# Get selected features and their importance scores
selected_features = selector.get_selected_features()
feature_importances = selector.get_feature_importances()

# Apply selection to get reduced dataset
reduced_data = selector.transform(normalized_counts)
