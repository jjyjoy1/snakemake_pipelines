# Snakefile for methylation sequencing data analysis
# Usage: snakemake --configfile config.yaml --cores N

import os
from os.path import join

# Load configuration
configfile: "config.yaml"

# Get sample names from config
SAMPLES = config["samples"]
GENOME_DIR = config["genome_dir"]
GENOME_FASTA = config["genome_fasta"]
OUTPUT_DIR = config["output_dir"]
BISMARK_GENOME_DIR = join(OUTPUT_DIR, "bismark_genome")

# Define output directories
QC_DIR = join(OUTPUT_DIR, "fastqc")
TRIM_DIR = join(OUTPUT_DIR, "trimmed")
ALIGN_DIR = join(OUTPUT_DIR, "aligned")
METH_DIR = join(OUTPUT_DIR, "methylation")
REPORT_DIR = join(OUTPUT_DIR, "reports")
DMRS_DIR = join(OUTPUT_DIR, "dmrs")

# Final output rule
rule all:
    input:
        # QC reports
        expand(join(QC_DIR, "{sample}_R{read}_fastqc.html"), sample=SAMPLES, read=[1, 2]),
        # Trimmed reads
        expand(join(TRIM_DIR, "{sample}_R{read}_trimmed.fastq.gz"), sample=SAMPLES, read=[1, 2]),
        # Alignment reports
        expand(join(ALIGN_DIR, "{sample}_bismark_bt2_PE_report.txt"), sample=SAMPLES),
        # Methylation extraction
        expand(join(METH_DIR, "{sample}_bismark_bt2_pe.deduplicated.bismark.cov.gz"), sample=SAMPLES),
        # MultiQC report
        join(REPORT_DIR, "multiqc_report.html"),
        # Methylation summary report
        expand(join(METH_DIR, "{sample}_bismark_bt2_pe.deduplicated.M-bias.txt"), sample=SAMPLES),
        # Differential methylation report
        join(DMRS_DIR, "differential_methylation_report.html")

# Prepare bismark genome
rule prepare_genome:
    input:
        genome = join(GENOME_DIR, GENOME_FASTA)
    output:
        directory(BISMARK_GENOME_DIR)
    log:
        "logs/prepare_genome.log"
    threads: config["threads"]["prepare_genome"]
    shell:
        """
        bismark_genome_preparation --parallel {threads} --path_to_aligner bowtie2 --bowtie2 --verbose --output_dir {output} {GENOME_DIR} &> {log}
        """

# Run FastQC on raw reads
rule fastqc_raw:
    input:
        r1 = lambda wildcards: config["samples"][wildcards.sample]["r1"],
        r2 = lambda wildcards: config["samples"][wildcards.sample]["r2"]
    output:
        html_r1 = join(QC_DIR, "{sample}_R1_fastqc.html"),
        html_r2 = join(QC_DIR, "{sample}_R2_fastqc.html"),
        zip_r1 = join(QC_DIR, "{sample}_R1_fastqc.zip"),
        zip_r2 = join(QC_DIR, "{sample}_R2_fastqc.zip")
    params:
        outdir = QC_DIR
    log:
        "logs/fastqc/{sample}.log"
    threads: config["threads"]["fastqc"]
    shell:
        """
        fastqc --threads {threads} --outdir {params.outdir} {input.r1} {input.r2} &> {log}
        """

# Trim adapters and low quality bases
rule trim_reads:
    input:
        r1 = lambda wildcards: config["samples"][wildcards.sample]["r1"],
        r2 = lambda wildcards: config["samples"][wildcards.sample]["r2"]
    output:
        r1 = join(TRIM_DIR, "{sample}_R1_trimmed.fastq.gz"),
        r2 = join(TRIM_DIR, "{sample}_R2_trimmed.fastq.gz"),
        unpaired_r1 = join(TRIM_DIR, "{sample}_R1_unpaired.fastq.gz"),
        unpaired_r2 = join(TRIM_DIR, "{sample}_R2_unpaired.fastq.gz")
    params:
        trim_opts = config["trim_options"]
    log:
        "logs/trim/{sample}.log"
    threads: config["threads"]["trim"]
    shell:
        """
        trimmomatic PE -threads {threads} \
            {input.r1} {input.r2} \
            {output.r1} {output.unpaired_r1} \
            {output.r2} {output.unpaired_r2} \
            {params.trim_opts} &> {log}
        """

# Run FastQC on trimmed reads
rule fastqc_trimmed:
    input:
        r1 = join(TRIM_DIR, "{sample}_R1_trimmed.fastq.gz"),
        r2 = join(TRIM_DIR, "{sample}_R2_trimmed.fastq.gz")
    output:
        html_r1 = join(QC_DIR, "{sample}_R1_trimmed_fastqc.html"),
        html_r2 = join(QC_DIR, "{sample}_R2_trimmed_fastqc.html"),
        zip_r1 = join(QC_DIR, "{sample}_R1_trimmed_fastqc.zip"),
        zip_r2 = join(QC_DIR, "{sample}_R2_trimmed_fastqc.zip")
    params:
        outdir = QC_DIR
    log:
        "logs/fastqc_trimmed/{sample}.log"
    threads: config["threads"]["fastqc"]
    shell:
        """
        fastqc --threads {threads} --outdir {params.outdir} {input.r1} {input.r2} &> {log}
        """

# Align reads with Bismark
rule bismark_align:
    input:
        r1 = join(TRIM_DIR, "{sample}_R1_trimmed.fastq.gz"),
        r2 = join(TRIM_DIR, "{sample}_R2_trimmed.fastq.gz"),
        genome_dir = BISMARK_GENOME_DIR
    output:
        bam = join(ALIGN_DIR, "{sample}_bismark_bt2_pe.bam"),
        report = join(ALIGN_DIR, "{sample}_bismark_bt2_PE_report.txt")
    params:
        outdir = ALIGN_DIR,
        bismark_opts = config["bismark_align_options"]
    log:
        "logs/bismark_align/{sample}.log"
    threads: config["threads"]["bismark_align"]
    shell:
        """
        bismark --genome {input.genome_dir} \
            -1 {input.r1} -2 {input.r2} \
            --output_dir {params.outdir} \
            --parallel {threads} \
            {params.bismark_opts} &> {log}
        """

# Deduplicate aligned reads
rule deduplicate:
    input:
        bam = join(ALIGN_DIR, "{sample}_bismark_bt2_pe.bam")
    output:
        bam = join(ALIGN_DIR, "{sample}_bismark_bt2_pe.deduplicated.bam"),
        report = join(ALIGN_DIR, "{sample}_bismark_bt2_pe.deduplication_report.txt")
    params:
        outdir = ALIGN_DIR
    log:
        "logs/deduplicate/{sample}.log"
    shell:
        """
        deduplicate_bismark --paired \
            --output_dir {params.outdir} \
            {input.bam} &> {log}
        """

# Extract methylation information
rule methylation_extraction:
    input:
        bam = join(ALIGN_DIR, "{sample}_bismark_bt2_pe.deduplicated.bam")
    output:
        cov = join(METH_DIR, "{sample}_bismark_bt2_pe.deduplicated.bismark.cov.gz"),
        bedGraph = join(METH_DIR, "{sample}_bismark_bt2_pe.deduplicated.bedGraph.gz"),
        mbias = join(METH_DIR, "{sample}_bismark_bt2_pe.deduplicated.M-bias.txt")
    params:
        outdir = METH_DIR,
        meth_extract_opts = config["meth_extract_options"]
    log:
        "logs/methylation_extraction/{sample}.log"
    threads: config["threads"]["methylation_extraction"]
    shell:
        """
        bismark_methylation_extractor --paired-end \
            --gzip \
            --bedGraph \
            --cytosine_report \
            --genome_folder {GENOME_DIR} \
            --output {params.outdir} \
            --parallel {threads} \
            {params.meth_extract_opts} \
            {input.bam} &> {log}
        """

# Generate bigWig files for visualization
rule bedgraph_to_bigwig:
    input:
        bedGraph = join(METH_DIR, "{sample}_bismark_bt2_pe.deduplicated.bedGraph.gz"),
        genome_sizes = join(GENOME_DIR, config["genome_sizes"])
    output:
        bigwig = join(METH_DIR, "{sample}_methylation.bw")
    shell:
        """
        gunzip -c {input.bedGraph} | sort -k1,1 -k2,2n > {wildcards.sample}_sorted.bedGraph
        bedGraphToBigWig {wildcards.sample}_sorted.bedGraph {input.genome_sizes} {output.bigwig}
        rm {wildcards.sample}_sorted.bedGraph
        """

# Differential methylation analysis with methylKit
rule differential_methylation:
    input:
        covs = expand(join(METH_DIR, "{sample}_bismark_bt2_pe.deduplicated.bismark.cov.gz"), sample=SAMPLES)
    output:
        dmrs = join(DMRS_DIR, "dmrs.csv"),
        report = join(DMRS_DIR, "differential_methylation_report.html")
    params:
        sample_groups = config["sample_groups"],
        dmr_params = config["dmr_parameters"],
        outdir = DMRS_DIR
    log:
        "logs/differential_methylation.log"
    script:
        "scripts/methylation_analysis.R"

# Generate MultiQC report
rule multiqc:
    input:
        fastqc = expand(join(QC_DIR, "{sample}_R{read}_fastqc.zip"), sample=SAMPLES, read=[1, 2]),
        fastqc_trimmed = expand(join(QC_DIR, "{sample}_R{read}_trimmed_fastqc.zip"), sample=SAMPLES, read=[1, 2]),
        bismark_reports = expand(join(ALIGN_DIR, "{sample}_bismark_bt2_PE_report.txt"), sample=SAMPLES),
        dedup_reports = expand(join(ALIGN_DIR, "{sample}_bismark_bt2_pe.deduplication_report.txt"), sample=SAMPLES),
        mbias = expand(join(METH_DIR, "{sample}_bismark_bt2_pe.deduplicated.M-bias.txt"), sample=SAMPLES)
    output:
        report = join(REPORT_DIR, "multiqc_report.html")
    params:
        outdir = REPORT_DIR
    log:
        "logs/multiqc.log"
    shell:
        """
        multiqc --force --outdir {params.outdir} \
            {QC_DIR} {ALIGN_DIR} {METH_DIR} \
            -f &> {log}
        """


