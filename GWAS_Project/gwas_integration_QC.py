import pygwas

# Load data
gwas = pygwas.GWAS()
gwas.load_genotype("genotype_data.bed")
gwas.load_phenotype("phenotype.txt")

# QC
gwas.run_qc(maf=0.01, geno=0.02, mind=0.02, hwe=1e-6)

# PCA for population structure
gwas.run_pca(n_components=10)

# Association testing
gwas.run_association(phenotype="BMI", covariates=["Age", "Sex", "PC1", "PC2"])

# Visualization
gwas.manhattan_plot()
gwas.qq_plot()
