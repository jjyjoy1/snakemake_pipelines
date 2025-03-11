# Add these rules to your Snakemake workflow for comprehensive enrichment analysis

# Update the all rule to include new outputs
rule all:
    input:
        # Previous outputs
        # ...
        # Enrichment analysis outputs
        "results/enrichment/go_bp_enrichment.tsv",
        "results/enrichment/go_mf_enrichment.tsv",
        "results/enrichment/go_cc_enrichment.tsv",
        "results/enrichment/kegg_enrichment.tsv",
        "results/enrichment/reactome_enrichment.tsv",
        "results/enrichment/wikipathways_enrichment.tsv",
        "results/enrichment/enrichment_summary.html"

# Comprehensive enrichment analysis including GO and pathway analysis
rule enrichment_analysis:
    input:
        diff_expr = "results/differential_expression/diff_expr_results.tsv"
    output:
        go_bp = "results/enrichment/go_bp_enrichment.tsv",
        go_mf = "results/enrichment/go_mf_enrichment.tsv",
        go_cc = "results/enrichment/go_cc_enrichment.tsv",
        kegg = "results/enrichment/kegg_enrichment.tsv",
        reactome = "results/enrichment/reactome_enrichment.tsv",
        wikipathways = "results/enrichment/wikipathways_enrichment.tsv",
        summary = "results/enrichment/enrichment_summary.html",
        all_results = "results/enrichment/all_enrichment_results.json"
    params:
        organism = config["organism"],
        significance_threshold = config["significance_threshold"],
        fc_threshold = 1.0  # Log2 fold change threshold
    conda:
        "envs/enrichment.yaml"
    log:
        "logs/enrichment/enrichment_analysis.log"
    script:
        "scripts/enrichment_analysis.py"

# Create a conda environment file for enrichment analysis
# Add this as envs/enrichment.yaml
"""
name: enrichment
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - scipy
  - matplotlib
  - seaborn
  - statsmodels
  - gseapy
  - goatools
  - pip
  - pip:
    - gprofiler-official
"""

