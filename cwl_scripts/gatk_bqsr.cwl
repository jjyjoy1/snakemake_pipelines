cwlVersion: v1.2
class: CommandLineTool
id: gatk_bqsr
label: "Base Quality Score Recalibration (BQSR) using GATK"
baseCommand: ["bash", "-c"]
inputs:
  bam: File
  reference: File
  known_sites: File
outputs:
  recalibrated_bam:
    type: File
    outputBinding:
      glob: "*.recal.bam"
arguments:
  - "module load gatk/4.2.0.0 && gatk BaseRecalibrator -I $(inputs.bam.path) -R $(inputs.reference.path) \
      --known-sites $(inputs.known_sites.path) -O recal_data.table && \
      gatk ApplyBQSR -R $(inputs.reference.path) -I $(inputs.bam.path) \
      --bqsr-recal-file recal_data.table -O recalibrated_reads.bam && module unload gatk/4.2.0.0"



