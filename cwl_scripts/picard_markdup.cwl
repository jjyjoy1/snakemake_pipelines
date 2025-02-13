cwlVersion: v1.2
class: CommandLineTool
id: picard_markdup
label: "Mark Duplicates using Picard"
baseCommand: ["bash", "-c"]
inputs:
  bam: File
outputs:
  dedup_bam:
    type: File
    outputBinding:
      glob: "*.dedup.bam"
arguments:
  - "module load picard/2.26.10 && picard MarkDuplicates I=$(inputs.bam.path) O=dedup_reads.bam M=marked_dup_metrics.txt REMOVE_DUPLICATES=true && module unload picard/2.26.10"

