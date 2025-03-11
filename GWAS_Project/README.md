I created a bash script pipeline for GWAS studies, which can perform genomic analyses including GWAS, eQTL mapping, fine mapping, and polygenic risk score modeling. 
I listed the key tools and a roadmap for developing this analysis pipeline.

Analysis Roadmap

Data Acquisition & QC

Genotype QC: Remove low-quality variants and samples
Check for population stratification using PCA
Imputation if necessary using Michigan/TOPMed Imputation Server


GWAS Analysis

Run association testing with appropriate covariates
Generate Manhattan/QQ plots to visualize results
Identify genome-wide significant loci


eQTL Analysis

Process gene expression data, normalize and adjust for covariates
Link expression data with genotype data
Perform cis/trans-eQTL mapping


Fine Mapping

For each significant locus from GWAS, perform fine mapping
Integrate with functional annotation data
Identify causal variants with highest posterior probabilities


Polygenic Risk Score Development

Split data into training/validation/testing sets
Select optimal p-value thresholds and weighting methods
Evaluate prediction performance


Integration & Functional Interpretation

Integrate GWAS and eQTL results
Pathway/gene-set enrichment analysis
Explore gene-environment interactions if data available


Validation & Replication

Validate findings in independent datasets
Perform meta-analysis with published studies
