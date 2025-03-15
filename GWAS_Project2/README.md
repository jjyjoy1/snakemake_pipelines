I created a Snakemake pipeline for GWAS project that integrates GATK, Hail, PLINK, and visualization tools to identify rare variants associated with disease in a cohort study. Here's a summary of what's included:

Snakefile: The core workflow definition with rules for each step of the analysis:

GATK variant calling pipeline (alignment, BQSR, HaplotypeCaller, joint genotyping)
Hail-based QC, annotation, and rare variant burden testing
PLINK/PCA analysis for population structure
Visualization and reporting

Hail Analysis Scripts:

hail_import_qc.py: Imports VCF data into Hail and performs QC on samples and variants
hail_vep_annotate.py: Annotates variants using VEP through Hail
hail_rare_variant_test.py: Performs rare variant burden testing (SKAT/Burden/SKAT-O)


Visualization Scripts:

manhattan_plot.py: Creates Manhattan plots of rare variant association results
pca_visualization.py: Visualizes population structure using PCA results
create_qc_report.py: Generates a comprehensive HTML QC report


Configuration:

config.yaml: Template configuration file for customizing the pipeline

Key Features:

Modular design: Each step is implemented as a separate rule, making it easy to modify or extend
Reproducibility: Workflow captures all dependencies and parameters
Scalability: Can run on a local machine or high-performance computing cluster
Comprehensive QC: Multiple QC steps with visual reporting
Flexible analysis options: Configurable parameters for rare variant discovery


Citations
If you use this pipeline, please cite the following tools:

GATK: Van der Auwera et al. (2013) https://doi.org/10.1002/0471250953.bi1110s43
Hail: https://github.com/hail-is/hail
PLINK: Purcell et al. (2007) https://doi.org/10.1086/519795
Snakemake: Köster and Rahmann (2012) https://doi.org/10.1093/bioinformatics/bts480









