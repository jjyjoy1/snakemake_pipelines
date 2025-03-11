# Start with your PLINK binary files (.bed/.bim/.fam) after QC
# First convert to additive coding format using PLINK
system("plink --bfile genotype_data_qc --recode A --out genotype_data_qc")
# This creates genotype_data_qc.raw file

# Now read and reformat this for MatrixEQTL
raw_data <- read.table("genotype_data_qc.raw", header=TRUE)

# Extract just the genotype columns (columns 7 onwards in .raw files)
genotypes <- raw_data[, 7:ncol(raw_data)]

# Transpose it (MatrixEQTL needs SNPs as rows, samples as columns)
genotypes_t <- t(genotypes)

# Write out in the format needed
write.table(genotypes_t, "genotype_data_qc.geno", quote=FALSE, sep="\t", row.names=TRUE, col.names=TRUE)

