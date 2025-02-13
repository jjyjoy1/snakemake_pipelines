cwlVersion: v1.2
class: CommandLineTool
id: samtools_sort
label: "Sort BAM using Samtools"
baseCommand: ["bash", "-c"]
inputs:
  bam: File
outputs:
  sorted_bam:
    type: File
    outputBinding:
      glob: "*.sorted.bam"
arguments:
  - "module load samtools/1.15 && samtools sort $(inputs.bam.path) -o sorted_reads.bam && module unload samtools/1.15"



