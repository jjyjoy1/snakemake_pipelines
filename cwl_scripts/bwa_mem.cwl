cwlVersion: v1.2
class: CommandLineTool
id: bwa_mem
label: "BWA-MEM Alignment"
baseCommand: ["bash", "-c"]
inputs:
  fastq1: File
  fastq2: File
  reference: File
outputs:
  aligned_bam:
    type: File
    outputBinding:
      glob: "*.bam"
arguments:
  - "module load bwa/0.7.17 && bwa mem $(inputs.reference.path) $(inputs.fastq1.path) $(inputs.fastq2.path) \
      | samtools view -Sb - > aligned_reads.bam && module unload bwa/0.7.17"

