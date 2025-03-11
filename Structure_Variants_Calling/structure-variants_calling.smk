# Snakefile for Structural Variant Calling Pipeline
# Usage: snakemake --cores <threads> --configfile config.yaml

import os
from os.path import join

configfile: "config.yaml"

# Define global variables
SAMPLES = config["samples"]
WORKDIR = config["workdir"]
REFERENCE = config["reference"]["genome"]
THREADS = config["threads"]

# Define output directories
ALIGN_DIR = join(WORKDIR, "alignment")
SV_DIR = join(WORKDIR, "sv_calls")
FILT_DIR = join(WORKDIR, "filtered_calls")
ANNOT_DIR = join(WORKDIR, "annotated_calls")
LOG_DIR = join(WORKDIR, "logs")

# Create output directories
for dir in [ALIGN_DIR, SV_DIR, FILT_DIR, ANNOT_DIR, LOG_DIR]:
    os.makedirs(dir, exist_ok=True)

# Final output rule
rule all:
    input:
        # Final annotated SV calls for each sample and each caller
        expand(join(ANNOT_DIR, "{sample}.delly.annotated.vcf"), sample=SAMPLES),
        expand(join(ANNOT_DIR, "{sample}.manta.annotated.vcf"), sample=SAMPLES),
        expand(join(ANNOT_DIR, "{sample}.smoove.annotated.vcf"), sample=SAMPLES),
        # Merged and annotated SV calls
        expand(join(ANNOT_DIR, "{sample}.merged.annotated.vcf"), sample=SAMPLES)

# Alignment with BWA-MEM2
rule bwa_mem2_align:
    input:
        r1 = lambda wildcards: config["samples"][wildcards.sample]["fastq_1"],
        r2 = lambda wildcards: config["samples"][wildcards.sample]["fastq_2"],
        ref = REFERENCE
    output:
        bam = join(ALIGN_DIR, "{sample}.bam"),
        bai = join(ALIGN_DIR, "{sample}.bam.bai")
    threads: THREADS
    log:
        join(LOG_DIR, "alignment", "{sample}.bwa_mem2.log")
    shell:
        """
        # Align reads with BWA-MEM2
        bwa-mem2 mem -t {threads} -R '@RG\\tID:{wildcards.sample}\\tSM:{wildcards.sample}' {input.ref} {input.r1} {input.r2} 2> {log} | \
        samtools sort -@ {threads} -o {output.bam} -
        
        # Index BAM file
        samtools index -@ {threads} {output.bam}
        """

# Mark duplicates with GATK
rule mark_duplicates:
    input:
        bam = join(ALIGN_DIR, "{sample}.bam")
    output:
        bam = join(ALIGN_DIR, "{sample}.md.bam"),
        bai = join(ALIGN_DIR, "{sample}.md.bam.bai"),
        metrics = join(ALIGN_DIR, "{sample}.md.metrics.txt")
    threads: THREADS
    log:
        join(LOG_DIR, "alignment", "{sample}.markdup.log")
    shell:
        """
        # Mark duplicates
        gatk MarkDuplicates \
            -I {input.bam} \
            -O {output.bam} \
            -M {output.metrics} \
            --CREATE_INDEX true \
            --VALIDATION_STRINGENCY LENIENT 2> {log}
        """

# Run Delly for SV calling
rule delly_call:
    input:
        bam = join(ALIGN_DIR, "{sample}.md.bam"),
        ref = REFERENCE
    output:
        vcf = join(SV_DIR, "{sample}.delly.vcf")
    threads: THREADS
    log:
        join(LOG_DIR, "sv_calling", "{sample}.delly.log")
    shell:
        """
        # Call SVs with Delly
        delly call -g {input.ref} -o {output.vcf} {input.bam} 2> {log}
        """

# Run Manta for SV calling
rule manta_call:
    input:
        bam = join(ALIGN_DIR, "{sample}.md.bam"),
        ref = REFERENCE
    output:
        vcf = join(SV_DIR, "{sample}.manta.vcf"),
        dir = directory(join(SV_DIR, "{sample}_manta_workdir"))
    threads: THREADS
    log:
        join(LOG_DIR, "sv_calling", "{sample}.manta.log")
    shell:
        """
        # Configure Manta
        configManta.py \
            --bam {input.bam} \
            --referenceFasta {input.ref} \
            --runDir {output.dir} 2> {log}
        
        # Run Manta
        {output.dir}/runWorkflow.py -j {threads} 2>> {log}
        
        # Copy results
        cp {output.dir}/results/variants/diploidSV.vcf.gz {output.vcf}.gz
        gunzip -f {output.vcf}.gz
        """

# Run Smoove for SV calling
rule smoove_call:
    input:
        bam = join(ALIGN_DIR, "{sample}.md.bam"),
        ref = REFERENCE
    output:
        vcf = join(SV_DIR, "{sample}.smoove.vcf")
    threads: THREADS
    log:
        join(LOG_DIR, "sv_calling", "{sample}.smoove.log")
    shell:
        """
        # Call SVs with Smoove
        smoove call --outdir {SV_DIR} \
            --name {wildcards.sample} \
            --fasta {input.ref} \
            -p {threads} \
            --genotype {input.bam} 2> {log}
        
        # Extract and rename the VCF
        gunzip -c {SV_DIR}/{wildcards.sample}-smoove.genotyped.vcf.gz > {output.vcf}
        """

# Filter Delly calls
rule filter_delly:
    input:
        vcf = join(SV_DIR, "{sample}.delly.vcf")
    output:
        vcf = join(FILT_DIR, "{sample}.delly.filtered.vcf")
    log:
        join(LOG_DIR, "filtering", "{sample}.delly.filter.log")
    shell:
        """
        # Filter Delly calls (PASS only, minimum size 50bp)
        bcftools filter -i 'FILTER="PASS" && ABS(SVLEN)>=50' {input.vcf} > {output.vcf} 2> {log}
        """

# Filter Manta calls
rule filter_manta:
    input:
        vcf = join(SV_DIR, "{sample}.manta.vcf")
    output:
        vcf = join(FILT_DIR, "{sample}.manta.filtered.vcf")
    log:
        join(LOG_DIR, "filtering", "{sample}.manta.filter.log")
    shell:
        """
        # Filter Manta calls (PASS only, minimum size 50bp)
        bcftools filter -i 'FILTER="PASS" && (INFO/SVLEN>=50 || INFO/SVLEN<=-50 || INFO/SVTYPE="BND")' {input.vcf} > {output.vcf} 2> {log}
        """

# Filter Smoove calls
rule filter_smoove:
    input:
        vcf = join(SV_DIR, "{sample}.smoove.vcf")
    output:
        vcf = join(FILT_DIR, "{sample}.smoove.filtered.vcf")
    log:
        join(LOG_DIR, "filtering", "{sample}.smoove.filter.log")
    shell:
        """
        # Filter Smoove calls (high-quality calls with minimum evidence)
        bcftools filter -i 'FILTER="PASS" && INFO/SU>=3' {input.vcf} > {output.vcf} 2> {log}
        """

# Annotate SVs with AnnotSV
rule annotate_sv:
    input:
        vcf = join(FILT_DIR, "{sample}.{caller}.filtered.vcf")
    output:
        vcf = join(ANNOT_DIR, "{sample}.{caller}.annotated.vcf"),
        tsv = join(ANNOT_DIR, "{sample}.{caller}.annotated.tsv")
    params:
        annotsv_dir = config["annotsv_dir"],
        build = config["reference"]["build"]
    log:
        join(LOG_DIR, "annotation", "{sample}.{caller}.annotsv.log")
    shell:
        """
        # Annotate SVs with AnnotSV
        AnnotSV \
            -SVinputFile {input.vcf} \
            -outputFile {output.tsv} \
            -genomeBuild {params.build} \
            -SVminSize 50 \
            -annotationMode both \
            -outputDir {ANNOT_DIR} 2> {log}
            
        # Create annotated VCF (for compatibility)
        cp {input.vcf} {output.vcf}
        """

# Merge SV calls with SURVIVOR
rule merge_sv_calls:
    input:
        delly = join(FILT_DIR, "{sample}.delly.filtered.vcf"),
        manta = join(FILT_DIR, "{sample}.manta.filtered.vcf"),
        smoove = join(FILT_DIR, "{sample}.smoove.filtered.vcf")
    output:
        list = join(FILT_DIR, "{sample}.vcf.list"),
        merged = join(FILT_DIR, "{sample}.merged.vcf")
    params:
        max_dist = 1000,  # max distance between breakpoints
        min_overlap = 0.8,  # minimum reciprocal overlap
        min_callers = 2  # minimum number of callers to support a variant
    log:
        join(LOG_DIR, "merging", "{sample}.survivor.log")
    shell:
        """
        # Create list of VCF files for SURVIVOR
        echo {input.delly} > {output.list}
        echo {input.manta} >> {output.list}
        echo {input.smoove} >> {output.list}
        
        # Merge calls with SURVIVOR
        SURVIVOR merge {output.list} {params.max_dist} {params.min_callers} 1 1 0 {params.min_overlap} {output.merged} 2> {log}
        """

# Annotate merged SV calls
rule annotate_merged:
    input:
        vcf = join(FILT_DIR, "{sample}.merged.vcf")
    output:
        vcf = join(ANNOT_DIR, "{sample}.merged.annotated.vcf"),
        tsv = join(ANNOT_DIR, "{sample}.merged.annotated.tsv")
    params:
        annotsv_dir = config["annotsv_dir"],
        build = config["reference"]["build"]
    log:
        join(LOG_DIR, "annotation", "{sample}.merged.annotsv.log")
    shell:
        """
        # Annotate merged SVs with AnnotSV
        AnnotSV \
            -SVinputFile {input.vcf} \
            -outputFile {output.tsv} \
            -genomeBuild {params.build} \
            -SVminSize 50 \
            -annotationMode both \
            -outputDir {ANNOT_DIR} 2> {log}
            
        # Create annotated VCF (for compatibility)
        cp {input.vcf} {output.vcf}
        """
