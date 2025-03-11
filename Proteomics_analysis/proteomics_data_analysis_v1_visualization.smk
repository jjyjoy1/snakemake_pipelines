# Add this rule to your Snakemake workflow for advanced enrichment visualizations

# Update the all rule to include visualization outputs
rule all:
    input:
        # Previous outputs
        # ...
        # Visualization outputs
        "results/enrichment/visualizations/combined_enrichment.png",
        "results/enrichment/visualizations/combined_enrichment_interactive.html"

# Enhanced visualization of enrichment results
rule visualize_enrichment:
    input:
        enrichment_results = "results/enrichment/all_enrichment_results.json"
    output:
        combined_viz = "results/enrichment/visualizations/combined_enrichment.png",
        interactive_viz = "results/enrichment/visualizations/combined_enrichment_interactive.html",
        heatmap_dir = directory("results/enrichment/visualizations/heatmaps"),
        network_dir = directory("results/enrichment/visualizations/networks"),
        bubble_dir = directory("results/enrichment/visualizations/bubbles")
    conda:
        "envs/visualization.yaml"
    log:
        "logs/enrichment/visualization.log"
    shell:
        """
        mkdir -p {output.heatmap_dir}
        mkdir -p {output.network_dir}
        mkdir -p {output.bubble_dir}
        
        python scripts/enrichment_visualization.py \
            --results {input.enrichment_results} \
            --output results/enrichment/visualizations \
            > {log} 2>&1
        """

# Create a conda environment file for visualization
# Add this as envs/visualization.yaml
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
  - networkx
  - plotly
  - kaleido  # For plotly static image export
  - pip
  - pip:
    - gprofiler-official
"""
