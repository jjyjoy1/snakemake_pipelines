# Advanced Proteomics Workflow with Snakemake
# This file extends the basic proteomics workflow with advanced analyses:
# - Protein-Protein Interaction Networks
# - Protein Structure and Domain Analysis
# - Biomarker Discovery and Machine Learning
# - Advanced Data Visualization

# Configuration file
configfile: "config.yaml"

# Define wildcards for sample handling
SAMPLES = config["samples"]
CONDITION = config["conditions"]

# Final target rule that collects all final output files
rule all:
    input:
        # Basic proteomics outputs
        "results/qc/quality_report.html",
        expand("results/identification/{sample}/proteins.txt", sample=SAMPLES),
        "results/quantification/protein_quantities.tsv",
        "results/differential_expression/diff_expr_results.tsv",
        "results/ptm/ptm_sites.tsv",
        "results/pathways/enrichment_results.tsv",
        
        # Enrichment analysis outputs
        "results/enrichment/go_bp_enrichment.tsv",
        "results/enrichment/go_mf_enrichment.tsv",
        "results/enrichment/go_cc_enrichment.tsv",
        "results/enrichment/kegg_enrichment.tsv",
        "results/enrichment/reactome_enrichment.tsv",
        "results/enrichment/wikipathways_enrichment.tsv",
        "results/enrichment/enrichment_summary.html",
        "results/enrichment/visualizations/combined_enrichment_interactive.html",
        
        # Protein-Protein Interaction Network outputs
        "results/ppi_network/network_plot.png",
        "results/ppi_network/cytoscape.json",
        "results/ppi_network/network_report.html",
        
        # Protein Structure and Domain Analysis outputs
        "results/structure/domain_analysis_report.html",
        
        # Biomarker Discovery and Machine Learning outputs
        "results/biomarkers/biomarker_report.html",
        
        # Advanced Visualization outputs
        "results/visualizations/dashboard.html",
        
        # Final integrated report
        "results/report/proteomics_report.html"

# Include basic proteomics workflow rules
include: "basic_proteomics_workflow.smk"

# Protein-Protein Interaction Network Analysis
rule ppi_network_analysis:
    input:
        diff_expr = "results/differential_expression/diff_expr_results.tsv"
    output:
        output_dir = directory("results/ppi_network"),
        network_plot = "results/ppi_network/network_plot.png",
        cytoscape_json = "results/ppi_network/cytoscape.json",
        report = "results/ppi_network/network_report.html"
    params:
        organism = config["organism"],
        confidence_score = config.get("ppi", {}).get("confidence_score", 0.7),
        log2fc_cutoff = config.get("ppi", {}).get("log2fc_cutoff", 1.0),
        pval_cutoff = config.get("ppi", {}).get("pval_cutoff", 0.05),
        biogrid_api_key = config.get("ppi", {}).get("biogrid_api_key", "")
    conda:
        "envs/ppi.yaml"
    log:
        "logs/ppi/network_analysis.log"
    script:
        "scripts/ppi_network_analysis.py"

# Protein Structure and Domain Analysis
rule protein_structure_analysis:
    input:
        diff_expr = "results/differential_expression/diff_expr_results.tsv"
    output:
        output_dir = directory("results/structure"),
        report = "results/structure/domain_analysis_report.html"
    params:
        organism = config["organism"],
        log2fc_cutoff = config.get("structure", {}).get("log2fc_cutoff", 1.0),
        pval_cutoff = config.get("structure", {}).get("pval_cutoff", 0.05),
        max_proteins = config.get("structure", {}).get("max_proteins", 10)
    conda:
        "envs/structure.yaml"
    log:
        "logs/structure/domain_analysis.log"
    script:
        "scripts/protein_structure_analysis.py"

# Biomarker Discovery and Machine Learning
rule biomarker_discovery:
    input:
        quantification = "results/quantification/protein_quantities.tsv",
        metadata = config["sample_metadata"]
    output:
        output_dir = directory("results/biomarkers"),
        report = "results/biomarkers/biomarker_report.html"
    params:
        sample_col = config.get("biomarkers", {}).get("sample_col", "sample_id"),
        group_col = config.get("biomarkers", {}).get("group_col", "condition"),
        feature_selection = config.get("biomarkers", {}).get("feature_selection", "rfe"),
        n_features = config.get("biomarkers", {}).get("n_features", 20)
    conda:
        "envs/ml.yaml"
    log:
        "logs/biomarkers/discovery.log"
    script:
        "scripts/biomarker_discovery.py"

# Advanced Visualization
rule advanced_visualization:
    input:
        diff_expr = "results/differential_expression/diff_expr_results.tsv",
        quantification = "results/quantification/protein_quantities.tsv",
        enrichment = "results/enrichment/all_enrichment_results.json",
        ppi = "results/ppi_network/cytoscape.json",
        metadata = config["sample_metadata"]
    output:
        output_dir = directory("results/visualizations"),
        dashboard = "results/visualizations/dashboard.html"
    params:
        color_palette = config.get("visualization", {}).get("color_palette", "viridis"),
        viz_config = config.get("visualization", {}).get("viz_config", {
            "diff_expr": ["volcano", "heatmap", "pca"],
            "enrichment": ["network", "heatmap"],
            "quantification": ["heatmap", "pca"]
        })
    conda:
        "envs/visualization.yaml"
    log:
        "logs/visualization/advanced_viz.log"
    script:
        "scripts/advanced_visualization.py"

# Integrated Final Report
rule generate_integrated_report:
    input:
        qc = "results/qc/quality_report.html",
        identification = expand("results/identification/{sample}/proteins.txt", sample=SAMPLES),
        quantification = "results/quantification/protein_quantities.tsv",
        diff_expr = "results/differential_expression/diff_expr_results.tsv",
        ptm = "results/ptm/ptm_sites.tsv",
        pathways = "results/pathways/enrichment_results.tsv",
        enrichment = "results/enrichment/enrichment_summary.html",
        ppi = "results/ppi_network/network_report.html",
        structure = "results/structure/domain_analysis_report.html",
        biomarkers = "results/biomarkers/biomarker_report.html",
        visualizations = "results/visualizations/dashboard.html"
    output:
        "results/report/proteomics_report.html"
    conda:
        "envs/report.yaml"
    log:
        "logs/report/integrated_report.log"
    script:
        "scripts/generate_integrated_report.py"

# Conda environment specifications
# Create conda environment files for each specialized task

# PPI Network Analysis environment
# File: envs/ppi.yaml
"""
name: ppi
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - networkx
  - requests
  - python-louvain  # For community detection
  - ipywidgets
"""

# Protein Structure Analysis environment
# File: envs/structure.yaml
"""
name: structure
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - requests
  - biopython
  - ipywidgets
"""

# Biomarker Discovery and Machine Learning environment
# File: envs/ml.yaml
"""
name: ml
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - imbalanced-learn
  - shap
  - ipywidgets
  - umap-learn
"""

# Advanced Visualization environment
# File: envs/visualization.yaml
"""
name: visualization
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - networkx
  - scikit-learn
  - ipywidgets
  - kaleido  # For plotly static image export
  - umap-learn
"""

# Updated Report Generation environment
# File: envs/report.yaml
"""
name: report
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - jinja2
  - markdown
  - ipywidgets
  - plotly
  - kaleido
"""


