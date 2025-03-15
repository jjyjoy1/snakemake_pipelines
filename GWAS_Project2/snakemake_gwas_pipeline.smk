# Snakemake workflow for GWAS analysis focusing on rare variant discovery
# File: Snakefile

import os
from snakemake.utils import min_version

# Set minimum Snakemake version
min_version("6.0")

# Configuration file with all parameters
configfile: "config/config.yaml"

# Directory structure
RESULTS_DIR = config["results_dir"]
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
QC_DIR = os.path.join(RESULTS_DIR, "qc")
VCF_DIR = os.path.join(RESULTS_DIR, "vcf")
HAIL_DIR = os.path.join(RESULTS_DIR, "hail")
PLINK_DIR = os.path.join(RESULTS_DIR, "plink")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# Input samples and reference data
SAMPLES = config["samples"]
REFERENCE = config["reference"]["genome"]
KNOWN_SITES = config["reference"]["known_sites"]
CHROMOSOMES = config["chromosomes"]

# Final output files
rule all:
    input:
        # Final QC report
        os.path.join(QC_DIR, "final_qc_report.html"),
        # Cohort VCF from GATK
        os.path.join(VCF_DIR, "cohort.vcf.gz"),
        # Hail rare variant results
        os.path.join(HAIL_DIR, "rare_variant_burden.tsv"),
        # PLINK PCA results
        os.path.join(PLINK_DIR, "pca.eigenvec"),
        # Manhattan plot
        os.path.join(PLOT_DIR, "manhattan_plot.png"),
        # PCA visualization
        os.path.join(PLOT_DIR, "pca_clusters.png")

# 1. GATK Variant Calling Pipeline

# Perform alignment and sort BAM files
rule align_and_sort:
    input:
        r1 = lambda wildcards: SAMPLES[wildcards.sample]["r1"],
        r2 = lambda wildcards: SAMPLES[wildcards.sample]["r2"],
        ref = REFERENCE
    output:
        bam = os.path.join(RESULTS_DIR, "bam/{sample}.sorted.bam"),
        bai = os.path.join(RESULTS_DIR, "bam/{sample}.sorted.bam.bai")
    log:
        os.path.join(LOG_DIR, "align/{sample}.log")
    threads: 8
    shell:
        """
        bwa mem -t {threads} {input.ref} {input.r1} {input.r2} 2> {log} | \
        samtools sort -@ {threads} -o {output.bam} - && \
        samtools index {output.bam}
        """

# Mark duplicates
rule mark_duplicates:
    input:
        bam = os.path.join(RESULTS_DIR, "bam/{sample}.sorted.bam")
    output:
        bam = os.path.join(RESULTS_DIR, "bam/{sample}.dedup.bam"),
        metrics = os.path.join(QC_DIR, "dedup/{sample}.metrics.txt")
    log:
        os.path.join(LOG_DIR, "dedup/{sample}.log")
    shell:
        """
        gatk MarkDuplicates \
            -I {input.bam} \
            -O {output.bam} \
            -M {output.metrics} \
            --CREATE_INDEX true \
            2> {log}
        """

# Perform base recalibration
rule base_recalibration:
    input:
        bam = os.path.join(RESULTS_DIR, "bam/{sample}.dedup.bam"),
        ref = REFERENCE,
        known_sites = KNOWN_SITES
    output:
        table = os.path.join(RESULTS_DIR, "bqsr/{sample}.recal_data.table"),
        bam = os.path.join(RESULTS_DIR, "bam/{sample}.recal.bam")
    log:
        os.path.join(LOG_DIR, "bqsr/{sample}.log")
    shell:
        """
        gatk BaseRecalibrator \
            -I {input.bam} \
            -R {input.ref} \
            --known-sites {input.known_sites} \
            -O {output.table} \
            2> {log}
        
        gatk ApplyBQSR \
            -I {input.bam} \
            -R {input.ref} \
            --bqsr-recal-file {output.table} \
            -O {output.bam} \
            2>> {log}
        """

# HaplotypeCaller per sample per chromosome
rule haplotype_caller:
    input:
        bam = os.path.join(RESULTS_DIR, "bam/{sample}.recal.bam"),
        ref = REFERENCE
    output:
        gvcf = os.path.join(VCF_DIR, "gvcf/{sample}.{chrom}.g.vcf.gz")
    log:
        os.path.join(LOG_DIR, "gatk/{sample}.{chrom}.hc.log")
    resources:
        mem_mb = 8000
    shell:
        """
        gatk HaplotypeCaller \
            -I {input.bam} \
            -R {input.ref} \
            -L {wildcards.chrom} \
            -O {output.gvcf} \
            -ERC GVCF \
            2> {log}
        """

# Consolidate GVCFs per chromosome
rule consolidate_gvcfs:
    input:
        gvcfs = expand(os.path.join(VCF_DIR, "gvcf/{sample}.{{chrom}}.g.vcf.gz"), sample=SAMPLES),
        ref = REFERENCE
    output:
        db = directory(os.path.join(VCF_DIR, "genomicsdb/{chrom}")),
    log:
        os.path.join(LOG_DIR, "gatk/{chrom}.consolidate.log")
    params:
        # Create a variable-length argument of "-V path" for each input file
        gvcfs_list = lambda wildcards, input: [f"-V {gvcf}" for gvcf in input.gvcfs]
    shell:
        """
        gatk GenomicsDBImport \
            {params.gvcfs_list} \
            --genomicsdb-workspace-path {output.db} \
            -L {wildcards.chrom} \
            --reader-threads 5 \
            2> {log}
        """

# GenotypeGVCFs per chromosome
rule genotype_gvcfs:
    input:
        db = os.path.join(VCF_DIR, "genomicsdb/{chrom}"),
        ref = REFERENCE
    output:
        vcf = os.path.join(VCF_DIR, "raw/{chrom}.vcf.gz")
    log:
        os.path.join(LOG_DIR, "gatk/{chrom}.genotype.log")
    shell:
        """
        gatk GenotypeGVCFs \
            -R {input.ref} \
            -V gendb://{input.db} \
            -O {output.vcf} \
            2> {log}
        """

# Merge VCFs from all chromosomes
rule merge_vcfs:
    input:
        vcfs = expand(os.path.join(VCF_DIR, "raw/{chrom}.vcf.gz"), chrom=CHROMOSOMES)
    output:
        vcf = os.path.join(VCF_DIR, "cohort.vcf.gz")
    log:
        os.path.join(LOG_DIR, "gatk/merge_vcfs.log")
    params:
        vcfs_list = lambda wildcards, input: [f"-I {vcf}" for vcf in input.vcfs]
    shell:
        """
        gatk MergeVcfs \
            {params.vcfs_list} \
            -O {output.vcf} \
            2> {log}
        """

# 2. Hail Analysis

# Import VCF, perform QC, and annotate with VEP
rule hail_import_and_qc:
    input:
        vcf = os.path.join(VCF_DIR, "cohort.vcf.gz"),
        phenotype = config["phenotype_file"]
    output:
        mt = os.path.join(HAIL_DIR, "cohort.qc.mt"),
        qc_report = os.path.join(QC_DIR, "hail_qc_report.html")
    log:
        os.path.join(LOG_DIR, "hail/import_qc.log")
    script:
        "scripts/hail_import_qc.py"

# VEP annotation through Hail
rule hail_vep_annotate:
    input:
        mt = os.path.join(HAIL_DIR, "cohort.qc.mt"),
        vep_config = config["vep_config"]
    output:
        mt = os.path.join(HAIL_DIR, "cohort.vep.mt")
    log:
        os.path.join(LOG_DIR, "hail/vep_annotate.log")
    script:
        "scripts/hail_vep_annotate.py"

# Rare variant burden testing
rule hail_rare_variant_test:
    input:
        mt = os.path.join(HAIL_DIR, "cohort.vep.mt"),
        phenotype = config["phenotype_file"],
        gene_list = config.get("gene_list", None)
    output:
        results = os.path.join(HAIL_DIR, "rare_variant_burden.tsv"),
        mt = os.path.join(HAIL_DIR, "rare_variants.mt")
    log:
        os.path.join(LOG_DIR, "hail/rare_variant_test.log")
    params:
        maf_threshold = config.get("maf_threshold", 0.01),
        burden_test = config.get("burden_test", "skat")
    script:
        "scripts/hail_rare_variant_test.py"

# 3. PLINK/PCA Analysis

# Convert VCF to PLINK format
rule vcf_to_plink:
    input:
        vcf = os.path.join(VCF_DIR, "cohort.vcf.gz")
    output:
        bed = os.path.join(PLINK_DIR, "cohort.bed"),
        bim = os.path.join(PLINK_DIR, "cohort.bim"),
        fam = os.path.join(PLINK_DIR, "cohort.fam")
    log:
        os.path.join(LOG_DIR, "plink/vcf_to_plink.log")
    params:
        out_prefix = os.path.join(PLINK_DIR, "cohort")
    shell:
        """
        plink2 --vcf {input.vcf} \
            --double-id \
            --make-bed \
            --out {params.out_prefix} \
            2> {log}
        """

# LD pruning for PCA
rule ld_pruning:
    input:
        bed = os.path.join(PLINK_DIR, "cohort.bed"),
        bim = os.path.join(PLINK_DIR, "cohort.bim"),
        fam = os.path.join(PLINK_DIR, "cohort.fam")
    output:
        prune_in = os.path.join(PLINK_DIR, "pruned.prune.in"),
        prune_out = os.path.join(PLINK_DIR, "pruned.prune.out"),
        pruned_bed = os.path.join(PLINK_DIR, "pruned.bed"),
        pruned_bim = os.path.join(PLINK_DIR, "pruned.bim"),
        pruned_fam = os.path.join(PLINK_DIR, "pruned.fam")
    log:
        os.path.join(LOG_DIR, "plink/ld_pruning.log")
    params:
        in_prefix = os.path.join(PLINK_DIR, "cohort"),
        out_prefix = os.path.join(PLINK_DIR, "pruned"),
        window_size = config.get("ld_window_size", 50),
        step_size = config.get("ld_step_size", 5),
        r2_threshold = config.get("ld_r2_threshold", 0.2)
    shell:
        """
        plink2 --bfile {params.in_prefix} \
            --indep-pairwise {params.window_size} {params.step_size} {params.r2_threshold} \
            --out {params.out_prefix} \
            2> {log}
            
        plink2 --bfile {params.in_prefix} \
            --extract {output.prune_in} \
            --make-bed \
            --out {params.out_prefix} \
            2>> {log}
        """

# Run PCA
rule run_pca:
    input:
        bed = os.path.join(PLINK_DIR, "pruned.bed"),
        bim = os.path.join(PLINK_DIR, "pruned.bim"),
        fam = os.path.join(PLINK_DIR, "pruned.fam")
    output:
        eigenvec = os.path.join(PLINK_DIR, "pca.eigenvec"),
        eigenval = os.path.join(PLINK_DIR, "pca.eigenval")
    log:
        os.path.join(LOG_DIR, "plink/pca.log")
    params:
        in_prefix = os.path.join(PLINK_DIR, "pruned"),
        out_prefix = os.path.join(PLINK_DIR, "pca"),
        pcs = config.get("num_pcs", 10)
    shell:
        """
        plink2 --bfile {params.in_prefix} \
            --pca {params.pcs} \
            --out {params.out_prefix} \
            2> {log}
        """

# 4. Visualization

# Create Manhattan plot
rule manhattan_plot:
    input:
        results = os.path.join(HAIL_DIR, "rare_variant_burden.tsv")
    output:
        plot = os.path.join(PLOT_DIR, "manhattan_plot.png")
    log:
        os.path.join(LOG_DIR, "plots/manhattan.log")
    script:
        "scripts/manhattan_plot.py"

# Visualize PCA clusters
rule pca_visualization:
    input:
        eigenvec = os.path.join(PLINK_DIR, "pca.eigenvec"),
        eigenval = os.path.join(PLINK_DIR, "pca.eigenval"),
        phenotype = config["phenotype_file"]
    output:
        plot = os.path.join(PLOT_DIR, "pca_clusters.png")
    log:
        os.path.join(LOG_DIR, "plots/pca_viz.log")
    script:
        "scripts/pca_visualization.py"

# Create final QC report
rule final_qc_report:
    input:
        vcf_stats = os.path.join(QC_DIR, "vcf_stats.txt"),
        hail_qc = os.path.join(QC_DIR, "hail_qc_report.html"),
        pca = os.path.join(PLOT_DIR, "pca_clusters.png"),
        manhattan = os.path.join(PLOT_DIR, "manhattan_plot.png")
    output:
        report = os.path.join(QC_DIR, "final_qc_report.html")
    log:
        os.path.join(LOG_DIR, "qc/final_report.log")
    script:
        "scripts/create_qc_report.py"




