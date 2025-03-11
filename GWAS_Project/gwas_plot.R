# R script for Manhattan and QQ plots
library(qqman)

# Read GWAS results
gwas_results <- read.table("gwas_bmi.BMI.glm.linear", header=TRUE)

# Manhattan plot
manhattan(gwas_results, chr="CHROM", bp="POS", snp="ID", p="P")

# QQ plot
qq(gwas_results$P)
