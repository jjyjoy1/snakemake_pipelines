Structural Variant Calling Pipeline
This Snakemake pipeline performs structural variant (SV) calling from short paired-end reads, including alignment, SV detection with multiple callers, filtering, merging, and annotation.
Overview
The pipeline includes the following steps:

Alignment: Map reads to reference genome using BWA-MEM2
Preprocessing: Mark duplicates using GATK
SV Calling: Call structural variants using three complementary tools:

Delly2 (paired-end, split-read, read-depth)
Manta (paired-end + assembly)
Smoove (Lumpy wrapper with improved workflow)


Filtering: Filter low-quality and small variants
Merging: Combine calls from multiple tools using SURVIVOR
Annotation: Annotate SVs with AnnotSV

Requirements
Software Dependencies

Snakemake (≥6.0.0)
BWA-MEM2 (≥2.0)
Samtools (≥1.13)
GATK (≥4.2.0)
Delly (≥0.8.7)
Manta (≥1.6.0)
Smoove (≥0.2.5)
SURVIVOR (≥1.0.7)
AnnotSV (≥3.1.0)
Bcftools (≥1.13)


Pipeline Features
The pipeline implements a complete workflow including:

Alignment with BWA-MEM2 and duplicate marking with GATK
Multi-caller approach using three complementary tools:

Delly2 (paired-end + split-read + read-depth)
Manta (assembly-based)
Smoove (Lumpy wrapper with improved filtering)


Filtering of low-quality and small variants
Merging of calls using SURVIVOR for consensus detection
Annotation with AnnotSV for functional interpretation

Key Advantages

Robust detection: By using multiple callers, the pipeline improves sensitivity and specificity
Flexibility: Easily configurable for different projects
Reproducibility: Snakemake ensures consistent workflow execution
Scalability: Works with single samples or large cohorts

Getting Started

Install required dependencies through conda (as detailed in the README)
Customize the config.yaml file with your sample information and paths
Run with snakemake --cores <threads> -s structure-variants_calling.smk
