# Snakefile for ChIP-seq analysis pipeline
import pandas as pd
import os

# Configuration
configfile: "config.yaml"

# Read manifest file
manifest = pd.read_csv(config["manifest_file"], sep="\t")
SAMPLES = manifest["sample_id"].tolist()
CONDITIONS = manifest["condition"].tolist()
REPLICATES = manifest["replicate"].tolist()

# Set up pairs for differential analysis
CONTROLS = manifest[manifest["sample_type"] == "control"]["sample_id"].tolist()
TREATMENTS = manifest[manifest["sample_type"] == "treatment"]["sample_id"].tolist()

# Output directories
FASTQC_DIR = "results/fastqc"
TRIMMED_DIR = "results/trimmed"
MAPPED_DIR = "results/mapped"
FILTERED_DIR = "results/filtered"
PEAKS_DIR = "results/peaks"
BIGWIG_DIR = "results/bigwig"
DIFF_DIR = "results/differential"

# Additional directories for annotation and functional analysis
ANNOTATION_DIR = "results/annotation"
FUNCTION_DIR = "results/functional"

# Target rule
rule all:
    input:
        # Quality control
        expand(FASTQC_DIR + "/{sample}_fastqc.html", sample=SAMPLES),
        # Trimmed reads
        expand(TRIMMED_DIR + "/{sample}_trimmed.fastq.gz", sample=SAMPLES),
        # Mapped reads
        expand(MAPPED_DIR + "/{sample}.bam", sample=SAMPLES),
        expand(MAPPED_DIR + "/{sample}.bam.bai", sample=SAMPLES),
        # Filtered and deduplicated reads
        expand(FILTERED_DIR + "/{sample}.filtered.bam", sample=SAMPLES),
        expand(FILTERED_DIR + "/{sample}.filtered.bam.bai", sample=SAMPLES),
        # Peak calling
        expand(PEAKS_DIR + "/{treatment}_vs_{control}_peaks.narrowPeak", 
               treatment=TREATMENTS, control=CONTROLS),
        # BigWig files for visualization
        expand(BIGWIG_DIR + "/{sample}.bw", sample=SAMPLES),
        # Peak annotations and functional analysis
        expand(ANNOTATION_DIR + "/{treatment}_vs_{control}_ChIPseeker_annotation.csv", 
               treatment=TREATMENTS, control=CONTROLS),
        expand(ANNOTATION_DIR + "/{treatment}_vs_{control}_GREAT_GO_BP.csv", 
               treatment=TREATMENTS, control=CONTROLS),
        expand(FUNCTION_DIR + "/{treatment}_vs_{control}_GO_BP.csv", 
               treatment=TREATMENTS, control=CONTROLS),
        expand(FUNCTION_DIR + "/{treatment}_vs_{control}_KEGG_pathways.csv", 
               treatment=TREATMENTS, control=CONTROLS),
        # Differential binding analysis
        DIFF_DIR + "/differential_binding_results.csv",
        # Final report
        "results/final_report.html"

# Quality control
rule fastqc:
    input:
        "data/raw/{sample}.fastq.gz"
    output:
        html=FASTQC_DIR + "/{sample}_fastqc.html",
        zip=FASTQC_DIR + "/{sample}_fastqc.zip"
    params:
        outdir=FASTQC_DIR
    threads: 2
    shell:
        "fastqc {input} --outdir={params.outdir} --threads={threads}"

# Trim adapters and low-quality reads
rule trim_reads:
    input:
        "data/raw/{sample}.fastq.gz"
    output:
        trimmed=TRIMMED_DIR + "/{sample}_trimmed.fastq.gz",
        trimlog=TRIMMED_DIR + "/{sample}_trimmed.log"
    threads: 4
    params:
        adapter=config["adapter_sequence"],
        quality=config["trim_quality_threshold"]
    shell:
        """
        trim_galore --quality {params.quality} --adapter {params.adapter} \
        --output_dir {TRIMMED_DIR} --cores {threads} {input}
        """

# Align reads to reference genome
rule map_reads:
    input:
        TRIMMED_DIR + "/{sample}_trimmed.fastq.gz"
    output:
        bam=MAPPED_DIR + "/{sample}.bam",
        stats=MAPPED_DIR + "/{sample}.stats.txt"
    params:
        genome=config["reference_genome"],
        prefix=MAPPED_DIR + "/{sample}"
    threads: 8
    shell:
        """
        bowtie2 -p {threads} -x {params.genome} -U {input} | \
        samtools view -bS - | samtools sort -o {output.bam}
        samtools flagstat {output.bam} > {output.stats}
        """

# Index BAM files
rule index_bam:
    input:
        MAPPED_DIR + "/{sample}.bam"
    output:
        MAPPED_DIR + "/{sample}.bam.bai"
    shell:
        "samtools index {input}"

# Filter and remove duplicates
rule filter_bam:
    input:
        bam=MAPPED_DIR + "/{sample}.bam",
        bai=MAPPED_DIR + "/{sample}.bam.bai"
    output:
        filtered=FILTERED_DIR + "/{sample}.filtered.bam",
        metrics=FILTERED_DIR + "/{sample}.metrics.txt"
    params:
        mapq=config["mapping_quality"],
        blacklist=config["blacklist_regions"]
    shell:
        """
        # Filter by mapping quality and remove blacklisted regions
        samtools view -b -q {params.mapq} {input.bam} | \
        bedtools intersect -v -a stdin -b {params.blacklist} > {FILTERED_DIR}/{wildcards.sample}.tmp.bam
        
        # Remove duplicates
        picard MarkDuplicates \
        INPUT={FILTERED_DIR}/{wildcards.sample}.tmp.bam \
        OUTPUT={output.filtered} \
        METRICS_FILE={output.metrics} \
        REMOVE_DUPLICATES=true \
        VALIDATION_STRINGENCY=LENIENT
        
        # Clean up temp files
        rm {FILTERED_DIR}/{wildcards.sample}.tmp.bam
        """

# Index filtered BAM files
rule index_filtered_bam:
    input:
        FILTERED_DIR + "/{sample}.filtered.bam"
    output:
        FILTERED_DIR + "/{sample}.filtered.bam.bai"
    shell:
        "samtools index {input}"

# Call peaks using MACS2
rule call_peaks:
    input:
        treatment=FILTERED_DIR + "/{treatment}.filtered.bam",
        control=FILTERED_DIR + "/{control}.filtered.bam"
    output:
        narrowPeak=PEAKS_DIR + "/{treatment}_vs_{control}_peaks.narrowPeak",
        xls=PEAKS_DIR + "/{treatment}_vs_{control}_peaks.xls",
        bed=PEAKS_DIR + "/{treatment}_vs_{control}_summits.bed"
    params:
        outdir=PEAKS_DIR,
        name="{treatment}_vs_{control}",
        genome=config["genome_size"],
        qvalue=config["peak_qvalue"]
    shell:
        """
        macs2 callpeak -t {input.treatment} -c {input.control} \
        -n {params.name} --outdir {params.outdir} \
        -g {params.genome} -q {params.qvalue} \
        --keep-dup auto
        """

# Create BigWig files for visualization
rule create_bigwig:
    input:
        bam=FILTERED_DIR + "/{sample}.filtered.bam",
        bai=FILTERED_DIR + "/{sample}.filtered.bam.bai"
    output:
        BIGWIG_DIR + "/{sample}.bw"
    params:
        genome=config["effective_genome_size"],
        binSize=config["bigwig_bin_size"]
    shell:
        """
        bamCoverage -b {input.bam} -o {output} \
        --effectiveGenomeSize {params.genome} \
        --binSize {params.binSize} \
        --normalizeUsing CPM
        """

# Differential binding analysis using DiffBind
rule differential_binding:
    input:
        peaks=expand(PEAKS_DIR + "/{treatment}_vs_{control}_peaks.narrowPeak", 
                     treatment=TREATMENTS, control=CONTROLS),
        bams=expand(FILTERED_DIR + "/{sample}.filtered.bam", sample=SAMPLES)
    output:
        csv=DIFF_DIR + "/differential_binding_results.csv",
        rdata=DIFF_DIR + "/dba_object.RData",
        plots=directory(DIFF_DIR + "/plots")
    params:
        manifest=config["manifest_file"],
        fdr=config["diff_binding_fdr"]
    script:
        "scripts/differential_binding.R"

# ChIPseeker annotation
rule chipseeker_annotation:
    input:
        peaks=PEAKS_DIR + "/{treatment}_vs_{control}_peaks.narrowPeak"
    output:
        anno=ANNOTATION_DIR + "/{treatment}_vs_{control}_ChIPseeker_annotation.csv",
        plots=ANNOTATION_DIR + "/{treatment}_vs_{control}_annotation_plots.pdf"
    params:
        outdir=ANNOTATION_DIR,
        organism=config["organism"],
        genome=config["genome_version"]
    conda:
        "envs/chipseeker.yaml"
    script:
        "scripts/chipseeker_annotation.R"

# GREAT analysis
rule great_analysis:
    input:
        peaks=PEAKS_DIR + "/{treatment}_vs_{control}_peaks.narrowPeak"
    output:
        go_bp=ANNOTATION_DIR + "/{treatment}_vs_{control}_GREAT_GO_BP.csv",
        assoc=ANNOTATION_DIR + "/{treatment}_vs_{control}_GREAT_gene_associations.csv"
    params:
        outdir=ANNOTATION_DIR,
        species=config["genome_version"],
        background="wholeGenome"
    conda:
        "envs/great.yaml"
    script:
        "scripts/great_analysis.R"

# GO and pathway enrichment analysis
rule functional_analysis:
    input:
        peaks=PEAKS_DIR + "/{treatment}_vs_{control}_peaks.narrowPeak",
        anno=ANNOTATION_DIR + "/{treatment}_vs_{control}_ChIPseeker_annotation.csv"
    output:
        go_bp=FUNCTION_DIR + "/{treatment}_vs_{control}_GO_BP.csv",
        go_mf=FUNCTION_DIR + "/{treatment}_vs_{control}_GO_MF.csv",
        go_cc=FUNCTION_DIR + "/{treatment}_vs_{control}_GO_CC.csv",
        kegg=FUNCTION_DIR + "/{treatment}_vs_{control}_KEGG_pathways.csv",
        reactome=FUNCTION_DIR + "/{treatment}_vs_{control}_Reactome_pathways.csv"
    params:
        outdir=FUNCTION_DIR,
        organism=config["organism"]
    conda:
        "envs/functional.yaml"
    script:
        "scripts/functional_analysis.R"

# Generate final report
rule final_report:
    input:
        fastqc=expand(FASTQC_DIR + "/{sample}_fastqc.html", sample=SAMPLES),
        peaks=expand(PEAKS_DIR + "/{treatment}_vs_{control}_peaks.narrowPeak", 
                     treatment=TREATMENTS, control=CONTROLS),
        anno=expand(ANNOTATION_DIR + "/{treatment}_vs_{control}_ChIPseeker_annotation.csv", 
                    treatment=TREATMENTS, control=CONTROLS),
        go=expand(FUNCTION_DIR + "/{treatment}_vs_{control}_GO_BP.csv", 
                  treatment=TREATMENTS, control=CONTROLS),
        diff_results=DIFF_DIR + "/differential_binding_results.csv"
    output:
        "results/final_report.html"
    script:
        "scripts/generate_report.R"
