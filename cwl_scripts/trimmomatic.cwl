cwlVersion: v1.2
class: CommandLineTool
id: trimmomatic
label: "Trimmomatic Adapter and Quality Trimming"
baseCommand: ["bash", "-c"]
inputs:
  fastq1: File
  fastq2: File
outputs:
  trimmed_fastq1:
    type: File
    outputBinding:
      glob: "*_paired_1.fq.gz"
  trimmed_fastq2:
    type: File
    outputBinding:
      glob: "*_paired_2.fq.gz"
arguments:
  - "module load trimmomatic/0.39 && trimmomatic PE $(inputs.fastq1.path) $(inputs.fastq2.path) \
      trimmed_paired_1.fq.gz trimmed_unpaired_1.fq.gz trimmed_paired_2.fq.gz trimmed_unpaired_2.fq.gz \
      SLIDINGWINDOW:4:20 MINLEN:50 && module unload trimmomatic/0.39"

