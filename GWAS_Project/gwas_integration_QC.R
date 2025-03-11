library(snpStats)
library(GenABEL)
library(gwasTools)
library(qqman)

# Load data
geno <- read.plink("genotype_data")

# Quality control
qc <- snpStats::snpgdsQC(geno, maf=0.01, missing.rate=0.05, hwe=1e-6)

# Association testing
gwas <- GenABEL::qtscore(phenotype ~ genotype + covariate1 + covariate2, data=phenoData)

# Visualization
manhattan(gwas)
qq(gwas$P)




