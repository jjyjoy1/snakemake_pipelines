# Step 1: Convert genotype to MatrixEQTL format
system("plink --bfile genotype_data_qc --recode A --out genotype_data_qc")
genotype_raw <- read.table("genotype_data_qc.raw", header=TRUE)
genotype_only <- genotype_raw[, 7:ncol(genotype_raw)]
genotype_t <- t(genotype_only)
rownames(genotype_t) <- sub("_[ACGT]$", "", rownames(genotype_t))  # Clean SNP names
write.table(genotype_t, "genotype_data_qc.geno", quote=FALSE, sep="\t")

# Step 2: Prepare expression data for MatrixEQTL
expression_data <- read.table("gene_expression.txt", header=TRUE, row.names=1)
expression_t <- t(expression_data)  # MatrixEQTL needs samples as columns
write.table(expression_t, "expression_data.matrixeqtl", quote=FALSE, sep="\t")

# Step 3: Prepare SNP and gene position files
snp_pos <- read.table("genotype_data_qc.bim", header=FALSE)
snp_pos_formatted <- data.frame(
  snpid = snp_pos$V2,
  chr = snp_pos$V1,
  pos = snp_pos$V4
)
write.table(snp_pos_formatted, "snp_positions.txt", quote=FALSE, sep="\t", row.names=FALSE)

gene_pos <- read.table("gene_positions.txt", header=TRUE)
gene_pos_formatted <- data.frame(
  geneid = gene_pos$gene_id,
  chr = gene_pos$chr,
  left = gene_pos$start,
  right = gene_pos$end
)
write.table(gene_pos_formatted, "gene_positions.txt", quote=FALSE, sep="\t", row.names=FALSE)

# Step 4: Run MatrixEQTL
# (Use the MatrixEQTL code from my previous response)
