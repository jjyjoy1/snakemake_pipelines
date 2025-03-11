# Load libraries
library(limma)

# Read expression data
expression_data <- read.table("gene_expression.txt", header=TRUE, row.names=1)

# Basic QC
# Check for missing values
missing_rate <- colSums(is.na(expression_data))/nrow(expression_data)
genes_to_keep <- names(missing_rate[missing_rate < 0.05])  # Keep genes with <5% missing values
expression_data <- expression_data[, genes_to_keep]

# Normalize expression data (common methods include log2 transformation and quantile normalization)
expression_data <- log2(expression_data + 1)  # Add 1 to handle zeros
normalized_expression <- normalizeBetweenArrays(expression_data, method="quantile")

# Get sample IDs from genotype data
genotype_samples <- read.table("genotype_data_qc.fam", header=FALSE)[,2]

# Find common samples
common_samples <- intersect(rownames(normalized_expression), genotype_samples)
print(paste("Number of samples with both genotype and expression data:", length(common_samples)))

# Subset expression data to common samples
expression_subset <- normalized_expression[common_samples,]

# Create a file mapping sample IDs to their respective genotype and expression data rows
sample_mapping <- data.frame(
  sample_id = common_samples,
  genotype_index = match(common_samples, genotype_samples),
  expression_index = match(common_samples, rownames(normalized_expression))
)
write.table(sample_mapping, "sample_mapping.txt", row.names=FALSE, quote=FALSE)
