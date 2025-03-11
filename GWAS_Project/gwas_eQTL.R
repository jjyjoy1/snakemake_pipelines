# Using MatrixEQTL in R
library(MatrixEQTL)

# Set file paths
SNP_file_name = "genotype_data_qc.matrixeqtl.gz"
expression_file_name = "gene_expression.txt"
covariates_file_name = "covariates.txt"
output_file_name = "eqtl_results.txt"

# Run eQTL analysis
eqtl <- Matrix_eQTL_main(
  snps_filename = SNP_file_name,
  gene_expression_filename = expression_file_name,
  covariates_filename = covariates_file_name,
  output_file_name = output_file_name,
  pvOutputThreshold = 1e-5,
  useModel = modelLINEAR,
  errorCovariance = numeric())

