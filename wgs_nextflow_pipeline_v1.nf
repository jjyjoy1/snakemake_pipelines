#!/usr/bin/env nextflow

// Define pipeline parameters
params.input_fastq = "input.fastq.gz"
params.reference_genome = "reference_genome.fa"
params.output_dir = "results"

// Define the modules to load for each tool
def fastqc_module = "fastqc/0.11.9"
def trimmomatic_module = "trimmomatic/0.39"
def bwa_module = "bwa/0.7.17"
def samtools_module = "samtools/1.12"
def picard_module = "picard/2.25.0"
def gatk_module = "gatk/4.2.0.0"
def htslib_module = "htslib/1.12"

// Define the process for running FastQC
process fastqc {
    module fastqc_module

    input:
    path fastq

    output:
    path "fastqc_output/*"

    script:
    """
    mkdir -p fastqc_output
    fastqc $fastq -o fastqc_output
    """
}

// Define the process for trimming reads with Trimmomatic
process trimmomatic {
    module trimmomatic_module

    input:
    path fastq

    output:
    tuple path("output_forward_paired.fq.gz"), path("output_forward_unpaired.fq.gz"),
          path("output_reverse_paired.fq.gz"), path("output_reverse_unpaired.fq.gz")

    script:
    """
    java -jar \$TRIMMOMATIC_HOME/trimmomatic.jar PE -phred33 $fastq \
        output_forward_paired.fq.gz output_forward_unpaired.fq.gz \
        output_reverse_paired.fq.gz output_reverse_unpaired.fq.gz \
        ILLUMINACLIP:TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36
    """
}

// Define the process for aligning reads with BWA
process bwa_align {
    module bwa_module

    input:
    path reference
    tuple path(forward), path(reverse)

    output:
    path "aligned_reads.sam"

    script:
    """
    bwa mem -t 4 $reference $forward $reverse > aligned_reads.sam
    """
}

// Define the process for converting SAM to BAM and sorting
process sam_to_bam {
    module samtools_module

    input:
    path sam

    output:
    path "sorted_reads.bam"

    script:
    """
    samtools view -Sb $sam | samtools sort -o sorted_reads.bam
    """
}

// Define the process for marking duplicates with Picard
process mark_duplicates {
    module picard_module

    input:
    path bam

    output:
    tuple path("marked_duplicates.bam"), path("marked_dup_metrics.txt")

    script:
    """
    java -jar \$PICARD_HOME/picard.jar MarkDuplicates I=$bam O=marked_duplicates.bam M=marked_dup_metrics.txt
    """
}

// Define the process for variant calling with GATK HaplotypeCaller
process gatk_haplotype_caller {
    module gatk_module

    input:
    path reference
    path bam

    output:
    path "raw_variants.vcf"

    script:
    """
    gatk HaplotypeCaller -R $reference -I $bam -O raw_variants.vcf
    """
}

// Define the process for filtering variants with GATK VariantFiltration
process gatk_variant_filtration {
    module gatk_module

    input:
    path reference
    path vcf

    output:
    path "filtered_variants.vcf"

    script:
    """
    gatk VariantFiltration -R $reference -V $vcf -O filtered_variants.vcf \
        --filter-expression "QD < 2.0 || FS > 60.0 || MQ < 40.0" --filter-name "my_filter"
    """
}

// Define the process for compressing and indexing the final VCF
process compress_index_vcf {
    module htslib_module

    input:
    path vcf

    output:
    tuple path("filtered_variants.vcf.gz"), path("filtered_variants.vcf.gz.tbi")

    script:
    """
    bgzip $vcf
    tabix -p vcf ${vcf}.gz
    """
}

// Define the workflow
workflow {
    // Step 1: Run FastQC
    fastqc(params.input_fastq)

    // Step 2: Trim reads with Trimmomatic
    trimmomatic(params.input_fastq)

    // Step 3: Align reads with BWA
    bwa_align(params.reference_genome, trimmomatic.out)

    // Step 4: Convert SAM to BAM and sort
    sam_to_bam(bwa_align.out)

    // Step 5: Mark duplicates with Picard
    mark_duplicates(sam_to_bam.out)

    // Step 6: Call variants with GATK HaplotypeCaller
    gatk_haplotype_caller(params.reference_genome, mark_duplicates.out)

    // Step 7: Filter variants with GATK VariantFiltration
    gatk_variant_filtration(params.reference_genome, gatk_haplotype_caller.out)

    // Step 8: Compress and index the final VCF
    compress_index_vcf(gatk_variant_filtration.out)

    // Publish results to the output directory
    compress_index_vcf.out.view().collect { file ->
        publishDir(params.output_dir, mode: 'copy') {
            file
        }
    }
}





