module load GWAS_Proj/v1.0.0
#Loading all required bioinformatics tools in one module

#Genotype Data Processing, convert VCF file to PLINK format for downstream analysis
# Convert VCF to PLINK format
bcftools norm -m -any input.vcf.gz | bcftools annotate --set-id +'%CHROM:%POS:%REF:%ALT' | plink2 --vcf - --make-bed --out genotype_data

# Perform basic QC
# Merge phenotype data, phenotype TSV file should be first column matching the IDs in the input VCF file.
plink2 --bfile genotype_data \
  --maf 0.01 \
  --geno 0.02 \
  --mind 0.02 \
  --hwe 1e-6 \
  --make-bed \
  --out genotype_data_qc


#For continuous traits
plink2 --bfile genotype_data_qc \
  --pheno phenotype.tsv \
  --pheno-name BMI \
  --covar phenotype.tsv \
  --covar-name Age,Sex \
  --glm \
  --out gwas_bmi

#For binary traits
plink2 --bfile genotype_data_qc \
  --pheno phenotype.tsv \
  --pheno-name Hypertension \
  --covar phenotype.tsv \
  --covar-name Age,Sex,BMI \
  --glm firth-fallback \
  --out gwas_hypertension

# Visualize GWAS Results
Rscript gwas_plot.R

#Fine Mapping of Significant Loci, for a significant locus identified in your GWAS.

# Extract region of interest (e.g., chromosome 2, positions 25MB to 26MB)
plink2 --bfile genotype_data_qc \
  --chr 2 \
  --from-bp 25000000 \
  --to-bp 26000000 \
  --make-bed \
  --out locus_chr2

# Calculate LD matrix
plink2 --bfile locus_chr2 \
  --r square \
  --out locus_chr2_ld

#Then use FINEMAP or SuSiE
# FINEMAP example
finemap --sss \
  --in-files locus_chr2.finemap.z \
  --in-ld locus_chr2_ld.ld \
  --n-samples 5000 \
  --n-causal-max 5 \
  --out-files locus_chr2.finemap.results

#Combined with gene expression data
#expression data QC, normalization, and mapping samples with genotype data
Rscript expression_preprocess.R

#Using Run eQTL Analysis Using MatrixEQTL
#MatrixEQTL is one of the most efficient tools for eQTL analysi
Rscript convert_format_matrixeqtl.R
Rscript run_matrixEQTL.R

#FastQTL is another popular tool, especially good for permutation-based p-values
#FastQTL requires specific formats: VCF for genotypes and BED for expression data.
plink --bfile genotype_data_qc --recode vcf-4.2 --out genotype_data_qc

# Compress and index the VCF
bgzip genotype_data_qc.vcf
tabix -p vcf genotype_data_qc.vcf.gz

Rscript convert_format_fastqtl.R

# Run FastQTL
fastQTL --vcf genotypes.vcf.gz \
        --bed expression.bed.gz \
        --cov covariates.txt \
        --window 1e6 \
        --out eqtl_results \
        --permute 1000 10000


