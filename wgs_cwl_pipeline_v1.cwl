cwlVersion: v1.2
class: Workflow
id: wgs_pipeline
label: "Whole Genome Sequencing Variant Calling Pipeline"
inputs:
  fastq1: File
  fastq2: File
  reference: File
  known_sites: File

outputs:
  vcf_output:
    type: File
    outputSource: variant_calling/vcf_output

steps:
  fastqc:
    run: fastqc.cwl
    in:
      input_fastq: fastq1
    out: [fastqc_report]

  trimming:
    run: trimmomatic.cwl
    in:
      fastq1: fastq1
      fastq2: fastq2
    out: [trimmed_fastq1, trimmed_fastq2]

  alignment:
    run: bwa_mem.cwl
    in:
      fastq1: trimming/trimmed_fastq1
      fastq2: trimming/trimmed_fastq2
      reference: reference
    out: [aligned_bam]

  sorting:
    run: samtools_sort.cwl
    in:
      bam: alignment/aligned_bam
    out: [sorted_bam]

  mark_duplicates:
    run: picard_markdup.cwl
    in:
      bam: sorting/sorted_bam
    out: [dedup_bam]

  base_recalibration:
    run: gatk_bqsr.cwl
    in:
      bam: mark_duplicates/dedup_bam
      reference: reference
      known_sites: known_sites
    out: [recalibrated_bam]

  variant_calling:
    run: gatk_haplotypecaller.cwl
    in:
      bam: base_recalibration/recalibrated_bam
      reference: reference
    out: [vcf_output]

  annotation:
    run: vep_annotation.cwl
    in:
      vcf: variant_calling/vcf_output
    out: [annotated_vcf]

