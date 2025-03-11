# Snakemake workflow for proteomics data analysis
# Author: Claude
# Date: March 11, 2025

# Configuration file
configfile: "config.yaml"

# Define wildcards for sample handling
SAMPLES = config["samples"]
CONDITION = config["conditions"]

# Final target rule that collects all final output files
rule all:
    input:
        # Data quality control
        "results/qc/quality_report.html",
        # Protein identification 
        expand("results/identification/{sample}/proteins.txt", sample=SAMPLES),
        # Protein quantification
        "results/quantification/protein_quantities.tsv",
        # Differential expression analysis
        "results/differential_expression/diff_expr_results.tsv",
        # PTM analysis
        "results/ptm/ptm_sites.tsv",
        # Pathway analysis
        "results/pathways/enrichment_results.tsv",
        # Final report
        "results/report/proteomics_report.html"

# Data Acquisition and Pre-processing
rule convert_raw_to_mzml:
    input:
        "data/raw/{sample}.raw"
    output:
        "data/mzml/{sample}.mzML"
    log:
        "logs/msconvert/{sample}.log"
    shell:
        """
        msconvert {input} --outdir data/mzml/ --mzML --filter "peakPicking true 1-" > {log} 2>&1
        """

rule quality_control:
    input:
        expand("data/mzml/{sample}.mzML", sample=SAMPLES)
    output:
        report = "results/qc/quality_report.html"
    conda:
        "envs/proteomics.yaml"
    script:
        "scripts/quality_control.py"

# Protein Identification
rule database_search:
    input:
        spectra = "data/mzml/{sample}.mzML",
        database = config["protein_database"]
    output:
        "results/identification/{sample}/search_results.idXML"
    params:
        precursor_mass_tolerance = config["precursor_mass_tolerance"],
        fragment_mass_tolerance = config["fragment_mass_tolerance"],
        enzyme = config["enzyme"],
        missed_cleavages = config["missed_cleavages"]
    conda:
        "envs/proteomics.yaml"
    log:
        "logs/database_search/{sample}.log"
    shell:
        """
        SearchEngine -in {input.spectra} \
                     -out {output} \
                     -database {input.database} \
                     -precursor_mass_tolerance {params.precursor_mass_tolerance} \
                     -fragment_mass_tolerance {params.fragment_mass_tolerance} \
                     -enzyme {params.enzyme} \
                     -missed_cleavages {params.missed_cleavages} > {log} 2>&1
        """

rule peptide_fdr_control:
    input:
        "results/identification/{sample}/search_results.idXML"
    output:
        "results/identification/{sample}/filtered_results.idXML"
    params:
        fdr_threshold = config["peptide_fdr"]
    conda:
        "envs/proteomics.yaml"
    log:
        "logs/fdr_control/{sample}.log"
    shell:
        """
        FalseDiscoveryRate -in {input} \
                          -out {output} \
                          -FDR {params.fdr_threshold} \
                          -method target_decoy > {log} 2>&1
        """

rule protein_inference:
    input:
        "results/identification/{sample}/filtered_results.idXML"
    output:
        "results/identification/{sample}/proteins.txt"
    conda:
        "envs/proteomics.yaml"
    log:
        "logs/protein_inference/{sample}.log"
    shell:
        """
        ProteinInference -in {input} \
                        -out {output} \
                        -protein_fdr {config[protein_fdr]} > {log} 2>&1
        """

# Quantitative Proteomics
rule extract_ion_chromatograms:
    input:
        spectra = "data/mzml/{sample}.mzML",
        identifications = "results/identification/{sample}/proteins.txt"
    output:
        "results/quantification/{sample}/ion_chromatograms.tsv"
    conda:
        "envs/proteomics.yaml"
    log:
        "logs/quantification/{sample}.log"
    shell:
        """
        FeatureFinderCentroided -in {input.spectra} \
                               -out {output} \
                               -id {input.identifications} > {log} 2>&1
        """

rule map_features:
    input:
        expand("results/quantification/{sample}/ion_chromatograms.tsv", sample=SAMPLES)
    output:
        "results/quantification/feature_map.consensusXML"
    conda:
        "envs/proteomics.yaml"
    log:
        "logs/feature_mapping/mapping.log"
    shell:
        """
        FeatureLinkerUnlabeledQT -in {input} \
                                -out {output} \
                                -rt_tol {config[rt_tolerance]} \
                                -mz_tol {config[mz_tolerance]} > {log} 2>&1
        """

rule protein_quantification:
    input:
        "results/quantification/feature_map.consensusXML"
    output:
        "results/quantification/protein_quantities.tsv"
    conda:
        "envs/proteomics.yaml"
    log:
        "logs/protein_quantification/quant.log"
    script:
        "scripts/protein_quantification.py"

# Post-translational Modification (PTM) Analysis
rule ptm_search:
    input:
        spectra = "data/mzml/{sample}.mzML",
        database = config["protein_database"]
    output:
        "results/ptm/{sample}/ptm_search_results.idXML"
    params:
        ptm_list = config["ptm_list"]
    conda:
        "envs/proteomics.yaml"
    log:
        "logs/ptm/{sample}.log"
    shell:
        """
        PTMSearchEngine -in {input.spectra} \
                       -out {output} \
                       -database {input.database} \
                       -modifications {params.ptm_list} > {log} 2>&1
        """

rule ptm_localization:
    input:
        expand("results/ptm/{sample}/ptm_search_results.idXML", sample=SAMPLES)
    output:
        "results/ptm/ptm_sites.tsv"
    conda:
        "envs/proteomics.yaml"
    log:
        "logs/ptm/localization.log"
    script:
        "scripts/ptm_localization.py"

# Differential Expression Analysis
rule differential_expression:
    input:
        quantities = "results/quantification/protein_quantities.tsv",
        metadata = config["sample_metadata"]
    output:
        "results/differential_expression/diff_expr_results.tsv"
    params:
        control = config["control_condition"],
        treatment = config["treatment_condition"],
        significance_threshold = config["significance_threshold"]
    conda:
        "envs/stats.yaml"
    log:
        "logs/differential_expression/diff_expr.log"
    script:
        "scripts/differential_expression.py"

# Pathway and Functional Analysis
rule pathway_analysis:
    input:
        diff_expr = "results/differential_expression/diff_expr_results.tsv",
        pathway_db = config["pathway_database"]
    output:
        "results/pathways/enrichment_results.tsv"
    params:
        organism = config["organism"],
        enrichment_method = config["enrichment_method"]
    conda:
        "envs/pathway.yaml"
    log:
        "logs/pathway/enrichment.log"
    script:
        "scripts/pathway_enrichment.py"

# Final Report Generation
rule generate_report:
    input:
        qc = "results/qc/quality_report.html",
        identification = expand("results/identification/{sample}/proteins.txt", sample=SAMPLES),
        quantification = "results/quantification/protein_quantities.tsv",
        diff_expr = "results/differential_expression/diff_expr_results.tsv",
        ptm = "results/ptm/ptm_sites.tsv",
        pathways = "results/pathways/enrichment_results.tsv"
    output:
        "results/report/proteomics_report.html"
    conda:
        "envs/report.yaml"
    log:
        "logs/report/final_report.log"
    script:
        "scripts/generate_report.py"

