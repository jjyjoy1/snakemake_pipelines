library(MatrixEQTL)

# Set file paths
SNP_file_name = "genotype_data_qc.geno"  # Genotype file in MatrixEQTL format
expression_file_name = "expression_data.txt"  # Expression file in MatrixEQTL format
covariates_file_name = "covariates.txt"  # Covariates file (e.g., age, sex, PCs)
output_file_name = "eqtl_results.txt"

# Prepare file formats (MatrixEQTL requires specific formats)
# Convert your genotype data to MatrixEQTL format using PLINK
system("plink --bfile genotype_data_qc --recode A --out genotype_data_qc")
# This creates a raw file which needs conversion to MatrixEQTL format

# Convert expression data to MatrixEQTL format
write.table(t(expression_subset), "expression_data.txt", quote=FALSE, sep="\t")

# Load data for MatrixEQTL
snps = SlicedData$new()
snps$fileDelimiter = "\t"     # the TAB character
snps$fileOmitCharacters = "NA" # denote missing values
snps$fileSkipRows = 1         # one row of column labels
snps$fileSkipColumns = 1      # one column of row labels
snps$fileSliceSize = 2000     # read file in pieces of 2,000 rows
snps$LoadFile(SNP_file_name)

gene = SlicedData$new()
gene$fileDelimiter = "\t"      # the TAB character
gene$fileOmitCharacters = "NA" # denote missing values
gene$fileSkipRows = 1          # one row of column labels
gene$fileSkipColumns = 1       # one column of row labels
gene$fileSliceSize = 2000      # read file in pieces of 2,000 rows
gene$LoadFile(expression_file_name)

# Load covariates (optional)
cvrt = SlicedData$new()
cvrt$fileDelimiter = "\t"      # the TAB character
cvrt$fileOmitCharacters = "NA" # denote missing values
cvrt$fileSkipRows = 1          # one row of column labels
cvrt$fileSkipColumns = 1       # one column of row labels
cvrt$LoadFile(covariates_file_name)

# Define model
useModel = modelLINEAR # modelANOVA or modelLINEAR

# Set parameters for cis-eQTL analysis
cisDist = 1e6  # Distance defining cis-eQTLs (1Mb)

# Set p-value thresholds
pvOutputThreshold_cis = 1e-5
pvOutputThreshold_trans = 1e-7

# Run the analysis
me = Matrix_eQTL_main(
  snps = snps,
  gene = gene,
  cvrt = cvrt,
  output_file_name = output_file_name,
  pvOutputThreshold = pvOutputThreshold_trans,
  useModel = useModel,
  errorCovariance = numeric(),
  verbose = TRUE,
  output_file_name.cis = "eqtl_cis_results.txt",
  pvOutputThreshold.cis = pvOutputThreshold_cis,
  snpspos = read.table("snp_positions.txt", header=TRUE, stringsAsFactors=FALSE),
  genepos = read.table("gene_positions.txt", header=TRUE, stringsAsFactors=FALSE),
  cisDist = cisDist,
  pvalue.hist = TRUE,
  min.pv.by.genesnp = FALSE,
  noFDRsaveMemory = FALSE)
